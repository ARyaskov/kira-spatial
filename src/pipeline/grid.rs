use std::collections::HashMap;

use kira_spatial_core as kcore;
use kira_spatial_field::gene_field;
use kira_spatial_io::Dataset;
use rayon::prelude::*;

use crate::error::OrchestratorError;

fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        if v == 0.0 { 0.0 } else { v }
    } else {
        0.0
    }
}

pub(crate) fn domain_id(len: usize) -> u64 {
    // Deterministic non-cryptographic domain id used by kira-spatial-field contract.
    0x4B49_5241_0000_0000_u64.wrapping_add(len as u64)
}

pub(crate) struct FieldCsrAdapter {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f32>,
    gene_index: HashMap<String, usize>,
}

impl FieldCsrAdapter {
    pub(crate) fn from_dataset(dataset: &Dataset) -> Result<Self, OrchestratorError> {
        let csr = dataset.expression_csr();
        let mut indptr = Vec::with_capacity(csr.indptr.len());
        for &p in &csr.indptr {
            indptr.push(
                usize::try_from(p).map_err(|_| {
                    OrchestratorError::InvalidInput("csr indptr does not fit usize")
                })?,
            );
        }

        let indices = csr.indices.iter().map(|&v| v as usize).collect::<Vec<_>>();
        let data = csr.data.clone();

        let mut gene_index = HashMap::with_capacity(dataset.features().rows.len());
        for row in &dataset.features().rows {
            gene_index.insert(row.gene_name.clone(), row.gene_id as usize);
        }

        Ok(Self {
            indptr,
            indices,
            data,
            gene_index,
        })
    }
}

impl gene_field::ExpressionCsrView for FieldCsrAdapter {
    fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    fn indices(&self) -> &[usize] {
        &self.indices
    }

    fn data(&self) -> &[f32] {
        &self.data
    }

    fn gene_index(&self) -> &HashMap<String, usize> {
        &self.gene_index
    }
}

pub(crate) struct FieldDomainAdapter {
    pub(crate) id: u64,
    pub(crate) bins: usize,
}

impl gene_field::SpatialDomainView for FieldDomainAdapter {
    fn id(&self) -> u64 {
        self.id
    }

    fn bin_count(&self) -> usize {
        self.bins
    }
}

pub(crate) struct GridMapping {
    rows: Vec<u32>,
    cols: Vec<u32>,
    min_row: u32,
    min_col: u32,
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) origin_x: f32,
    pub(crate) origin_y: f32,
}

impl GridMapping {
    pub(crate) fn from_io_domain(
        domain: &kira_spatial_io::SpatialDomain,
    ) -> Result<Self, OrchestratorError> {
        let rows = domain
            .grid_row
            .as_ref()
            .ok_or(OrchestratorError::InvalidInput(
                "grid_row is required for regular-grid projection",
            ))?;
        let cols = domain
            .grid_col
            .as_ref()
            .ok_or(OrchestratorError::InvalidInput(
                "grid_col is required for regular-grid projection",
            ))?;
        if rows.len() != cols.len() || rows.len() != domain.len() {
            return Err(OrchestratorError::InvalidInput(
                "grid_row/grid_col lengths are inconsistent",
            ));
        }

        let mut min_row = u32::MAX;
        let mut max_row = 0_u32;
        let mut min_col = u32::MAX;
        let mut max_col = 0_u32;
        for i in 0..rows.len() {
            min_row = min_row.min(rows[i]);
            max_row = max_row.max(rows[i]);
            min_col = min_col.min(cols[i]);
            max_col = max_col.max(cols[i]);
        }

        let nx_u32 = max_col
            .checked_sub(min_col)
            .and_then(|d| d.checked_add(1))
            .ok_or(OrchestratorError::InvalidInput("invalid grid_col range"))?;
        let ny_u32 = max_row
            .checked_sub(min_row)
            .and_then(|d| d.checked_add(1))
            .ok_or(OrchestratorError::InvalidInput("invalid grid_row range"))?;

        Ok(Self {
            rows: rows.clone(),
            cols: cols.clone(),
            min_row,
            min_col,
            nx: nx_u32 as usize,
            ny: ny_u32 as usize,
            origin_x: min_col as f32,
            origin_y: min_row as f32,
        })
    }
    pub(crate) fn to_core_domain(
        &self,
        io_domain: &kira_spatial_io::SpatialDomain,
    ) -> Result<kcore::SpatialDomain, OrchestratorError> {
        let width = u32::try_from(self.nx)
            .map_err(|_| OrchestratorError::InvalidInput("nx does not fit u32"))?;
        let height = u32::try_from(self.ny)
            .map_err(|_| OrchestratorError::InvalidInput("ny does not fit u32"))?;

        let mut x_q = Vec::with_capacity(self.cols.len());
        let mut y_q = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            x_q.push(
                i32::try_from(self.cols[i])
                    .map_err(|_| OrchestratorError::InvalidInput("grid_col does not fit i32"))?,
            );
            y_q.push(
                i32::try_from(self.rows[i])
                    .map_err(|_| OrchestratorError::InvalidInput("grid_row does not fit i32"))?,
            );
        }

        let mut mask = Vec::with_capacity(io_domain.len());
        for i in 0..io_domain.len() {
            mask.push(u8::from(io_domain.tissue_mask[i]));
        }

        let grid = kcore::GridSpec::new(
            i32::try_from(self.min_col)
                .map_err(|_| OrchestratorError::InvalidInput("min_col does not fit i32"))?,
            i32::try_from(self.min_row)
                .map_err(|_| OrchestratorError::InvalidInput("min_row does not fit i32"))?,
            1,
            width,
            height,
        )?;

        Ok(kcore::SpatialDomain::new_quantized(
            x_q,
            y_q,
            Some(mask),
            Some(grid),
        )?)
    }
    pub(crate) fn scatter_to_dense(&self, sparse: &[f32]) -> Result<Vec<f32>, OrchestratorError> {
        if sparse.len() != self.rows.len() {
            return Err(OrchestratorError::InvalidInput(
                "sparse field length mismatches spatial domain",
            ));
        }
        let mut mapped: Vec<(usize, f32)> = (0..sparse.len())
            .into_par_iter()
            .map(|i| {
                let gx = (self.cols[i] - self.min_col) as usize;
                let gy = (self.rows[i] - self.min_row) as usize;
                let idx = gy * self.nx + gx;
                (idx, sanitize(sparse[i]))
            })
            .collect();

        mapped.par_sort_unstable_by_key(|(idx, _)| *idx);
        let mut dense = vec![0.0_f32; self.nx * self.ny];
        let mut prev = None::<usize>;
        for (idx, value) in mapped {
            if prev == Some(idx) {
                return Err(OrchestratorError::InvalidInput(
                    "multiple bins map to the same grid cell",
                ));
            }
            dense[idx] = value;
            prev = Some(idx);
        }

        Ok(dense)
    }
}
