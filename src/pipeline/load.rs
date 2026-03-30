use std::time::Instant;

use kira_spatial_core as kcore;
use kira_spatial_field::Field as KField;
use kira_spatial_io::{Dataset, LoadConfig};
use rayon::prelude::*;

use crate::cli::{RunArgs, SignalKind};
use crate::error::OrchestratorError;
use crate::feature_slice::{
    FeatureSliceCache, detect_feature_slice_paths, feature_slice_cache_path,
    load_feature_slice_cache, load_feature_slice_values_from_h5, load_or_build_feature_index,
    open_h5, write_feature_slice_cache,
};
use crate::pipeline::grid::{FieldCsrAdapter, FieldDomainAdapter, GridMapping, domain_id};
use crate::pipeline::log_timing;
use crate::pipeline::signal::{compute_signal, find_gene_id};

pub(crate) fn is_feature_slice_matrix_missing(err: &kira_spatial_io::SpatialIoError) -> bool {
    err.to_string().contains("missing /matrix/barcodes dataset")
}

pub(crate) fn compute_signal_dense_from_dataset(
    args: &RunArgs,
    dataset: &Dataset,
) -> Result<(GridMapping, Vec<f32>, Vec<(String, Vec<f32>)>), OrchestratorError> {
    let t_total = Instant::now();
    let io_domain = dataset.spatial_domain();
    let io_expr = dataset.expression_csr();
    let features = &dataset.features().rows;

    let field_csr = FieldCsrAdapter::from_dataset(dataset)?;
    let field_domain = FieldDomainAdapter {
        id: domain_id(io_domain.len()),
        bins: io_domain.len(),
    };
    for gene in &args.genes {
        let _ = KField::from_gene(&field_csr, gene, &field_domain)?;
    }

    let grid = GridMapping::from_io_domain(io_domain)?;
    let core_domain = grid.to_core_domain(io_domain)?;

    let mut core_indptr = Vec::with_capacity(io_expr.indptr.len());
    for &p in &io_expr.indptr {
        core_indptr.push(
            u32::try_from(p)
                .map_err(|_| OrchestratorError::InvalidInput("csr indptr does not fit u32"))?,
        );
    }
    let core_expr_view = kcore::BinMajorCsrView::new(
        io_expr.n_bins,
        io_expr.n_genes,
        &core_indptr,
        &io_expr.indices,
        &io_expr.data,
    );

    let mut core_values = vec![0.0_f32; core_domain.len()];
    let mut per_gene_fields = Vec::<(String, kcore::Field<'_>)>::new();
    for gene in &args.genes {
        let gene_id = find_gene_id(features, gene)?;
        let gene_field = kcore::build_gene_field(
            &core_domain,
            &core_expr_view,
            gene_id,
            kcore::GeneFieldConfig::default(),
        )?;
        core_values
            .par_iter_mut()
            .zip(gene_field.values().par_iter())
            .for_each(|(dst, &v)| *dst += v as f32);
        per_gene_fields.push((gene.clone(), gene_field));
    }
    let core_gene = kcore::Field::new(&core_domain, core_values)?;

    let signal_sparse = if args.signal == SignalKind::CrossGradient {
        if per_gene_fields.len() < 2 {
            return Err(OrchestratorError::InvalidInput(
                "cross-gradient requires at least two values in --genes",
            ));
        }
        compute_signal(
            args.signal,
            &core_domain,
            &per_gene_fields[0].1,
            Some(&per_gene_fields[1].1),
        )?
    } else {
        compute_signal(args.signal, &core_domain, &core_gene, None)?
    };
    let signal_dense = grid.scatter_to_dense(&signal_sparse)?;

    let mut per_gene_signal_dense = Vec::<(String, Vec<f32>)>::new();
    for (gene, gene_field) in &per_gene_fields {
        let gene_signal_sparse = if args.signal == SignalKind::CrossGradient {
            compute_signal(args.signal, &core_domain, gene_field, Some(&core_gene))?
        } else {
            compute_signal(args.signal, &core_domain, gene_field, None)?
        };
        let gene_signal_dense = grid.scatter_to_dense(&gene_signal_sparse)?;
        per_gene_signal_dense.push((gene.clone(), gene_signal_dense));
    }
    log_timing("dataset_path_total", t_total);
    Ok((grid, signal_dense, per_gene_signal_dense))
}

pub(crate) fn compute_signal_dense_from_feature_slice(
    args: &RunArgs,
) -> Result<(GridMapping, Vec<f32>, Vec<(String, Vec<f32>)>), OrchestratorError> {
    let t_total = Instant::now();
    let Some(paths) = detect_feature_slice_paths(&args.h5) else {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "feature_slice.h5 layout not detected at {}",
            args.h5.display()
        )));
    };

    let mut genes = args.genes.iter();
    let first_gene = genes.next().ok_or(OrchestratorError::InvalidInput(
        "at least one gene is required in --genes",
    ))?;

    let cache_path = feature_slice_cache_path(&paths.parquet_path);
    eprintln!(
        "[timing] feature_slice_cache path={} exists={}",
        cache_path.display(),
        cache_path.is_file()
    );
    let cache = load_feature_slice_cache(&cache_path).ok();
    eprintln!("[timing] feature_slice_cache loaded={}", cache.is_some());

    let fallback_io_path = || compute_signal_dense_from_feature_slice_via_io(args);

    let (domain, mut summed, cache_for_rest, h5_and_index) = if let Some(cache) = cache {
        let t_fast = Instant::now();
        let h5 = open_h5(&paths.h5_path)?;
        let feature_index = match load_or_build_feature_index(&paths.h5_path, &h5) {
            Ok(v) => v,
            Err(_) => return fallback_io_path(),
        };
        let first_idx = *feature_index.get(first_gene).ok_or_else(|| {
            OrchestratorError::InvalidInputOwned(format!(
                "gene not found in /features/name: {first_gene}"
            ))
        })?;
        let domain = cache.domain.clone();
        let first_vals =
            load_feature_slice_values_from_h5(&h5, first_idx, domain.len(), &cache.coord_to_bin)?;
        log_timing("feature_slice_fast_first_gene", t_fast);
        (domain, first_vals, Some(cache), Some((h5, feature_index)))
    } else {
        let t_slow = Instant::now();
        let first =
            kira_spatial_io::load_feature_slice_gene(&args.h5, first_gene, LoadConfig::default())?;
        let built_cache = FeatureSliceCache::from_domain(&first.spatial_domain)?;
        if let Err(err) = write_feature_slice_cache(&cache_path, &built_cache) {
            eprintln!(
                "warning: failed writing feature-slice mmap cache {}: {}",
                cache_path.display(),
                err
            );
        }
        log_timing("feature_slice_io_first_gene", t_slow);
        let h5_and_index = match open_h5(&paths.h5_path)
            .and_then(|h5| load_or_build_feature_index(&paths.h5_path, &h5).map(|idx| (h5, idx)))
        {
            Ok(v) => Some(v),
            Err(err) => {
                eprintln!(
                    "warning: failed switching to fast path after cache build; staying io-only: {}",
                    err
                );
                None
            }
        };
        (
            first.spatial_domain,
            first.values,
            Some(built_cache),
            h5_and_index,
        )
    };
    let mut per_gene_sparse = Vec::<(String, Vec<f32>)>::new();
    per_gene_sparse.push((first_gene.clone(), summed.clone()));

    if let (Some(cache), Some((h5, feature_index))) = (cache_for_rest, h5_and_index) {
        for gene in genes {
            let t_gene = Instant::now();
            let idx = *feature_index.get(gene).ok_or_else(|| {
                OrchestratorError::InvalidInputOwned(format!(
                    "gene not found in /features/name: {gene}"
                ))
            })?;
            let values =
                load_feature_slice_values_from_h5(&h5, idx, domain.len(), &cache.coord_to_bin)?;
            per_gene_sparse.push((gene.clone(), values.clone()));
            summed
                .par_iter_mut()
                .zip(values.par_iter())
                .for_each(|(dst, &v)| *dst += v);
            eprintln!(
                "[timing] feature_slice_fast_gene {} {:?}",
                gene,
                t_gene.elapsed()
            );
        }
    } else {
        for gene in genes {
            let t_gene = Instant::now();
            let loaded =
                kira_spatial_io::load_feature_slice_gene(&args.h5, gene, LoadConfig::default())?;
            if loaded.spatial_domain.len() != domain.len() || loaded.values.len() != summed.len() {
                return Err(OrchestratorError::InvalidInput(
                    "feature-slice domains mismatch across requested genes",
                ));
            }
            per_gene_sparse.push((gene.clone(), loaded.values.clone()));
            summed
                .par_iter_mut()
                .zip(loaded.values.par_iter())
                .for_each(|(dst, &v)| *dst += v);
            eprintln!(
                "[timing] feature_slice_io_gene {} {:?}",
                gene,
                t_gene.elapsed()
            );
        }
    }

    let t_post = Instant::now();
    let grid = GridMapping::from_io_domain(&domain)?;
    let core_domain = grid.to_core_domain(&domain)?;
    let core_gene = kcore::Field::new(&core_domain, summed)?;

    let signal_sparse = if args.signal == SignalKind::CrossGradient {
        if per_gene_sparse.len() < 2 {
            return Err(OrchestratorError::InvalidInput(
                "cross-gradient requires at least two values in --genes",
            ));
        }
        let a = kcore::Field::new(&core_domain, per_gene_sparse[0].1.clone())?;
        let b = kcore::Field::new(&core_domain, per_gene_sparse[1].1.clone())?;
        compute_signal(args.signal, &core_domain, &a, Some(&b))?
    } else {
        compute_signal(args.signal, &core_domain, &core_gene, None)?
    };
    let signal_dense = grid.scatter_to_dense(&signal_sparse)?;
    let mut per_gene_signal_dense = Vec::<(String, Vec<f32>)>::with_capacity(per_gene_sparse.len());
    for (gene, sparse_vals) in per_gene_sparse {
        let gf = kcore::Field::new(&core_domain, sparse_vals)?;
        let sig_sparse = if args.signal == SignalKind::CrossGradient {
            compute_signal(args.signal, &core_domain, &gf, Some(&core_gene))?
        } else {
            compute_signal(args.signal, &core_domain, &gf, None)?
        };
        let sig_dense = grid.scatter_to_dense(&sig_sparse)?;
        per_gene_signal_dense.push((gene, sig_dense));
    }
    log_timing("feature_slice_postprocess", t_post);
    log_timing("feature_slice_total", t_total);
    Ok((grid, signal_dense, per_gene_signal_dense))
}

pub(crate) fn compute_signal_dense_from_feature_slice_via_io(
    args: &RunArgs,
) -> Result<(GridMapping, Vec<f32>, Vec<(String, Vec<f32>)>), OrchestratorError> {
    let t_total = Instant::now();
    eprintln!("[timing] fallback=io_only");
    let mut genes = args.genes.iter();
    let first_gene = genes.next().ok_or(OrchestratorError::InvalidInput(
        "at least one gene is required in --genes",
    ))?;
    let first =
        kira_spatial_io::load_feature_slice_gene(&args.h5, first_gene, LoadConfig::default())?;

    let mut summed = first.values;
    let domain = first.spatial_domain;
    let mut per_gene_sparse = Vec::<(String, Vec<f32>)>::new();
    per_gene_sparse.push((first_gene.clone(), summed.clone()));

    for gene in genes {
        let loaded =
            kira_spatial_io::load_feature_slice_gene(&args.h5, gene, LoadConfig::default())?;
        if loaded.spatial_domain.len() != domain.len() || loaded.values.len() != summed.len() {
            return Err(OrchestratorError::InvalidInput(
                "feature-slice domains mismatch across requested genes",
            ));
        }
        per_gene_sparse.push((gene.clone(), loaded.values.clone()));
        summed
            .par_iter_mut()
            .zip(loaded.values.par_iter())
            .for_each(|(dst, &v)| *dst += v);
    }

    let grid = GridMapping::from_io_domain(&domain)?;
    let core_domain = grid.to_core_domain(&domain)?;
    let core_gene = kcore::Field::new(&core_domain, summed)?;

    let signal_sparse = if args.signal == SignalKind::CrossGradient {
        if per_gene_sparse.len() < 2 {
            return Err(OrchestratorError::InvalidInput(
                "cross-gradient requires at least two values in --genes",
            ));
        }
        let a = kcore::Field::new(&core_domain, per_gene_sparse[0].1.clone())?;
        let b = kcore::Field::new(&core_domain, per_gene_sparse[1].1.clone())?;
        compute_signal(args.signal, &core_domain, &a, Some(&b))?
    } else {
        compute_signal(args.signal, &core_domain, &core_gene, None)?
    };
    let signal_dense = grid.scatter_to_dense(&signal_sparse)?;
    let mut per_gene_signal_dense = Vec::<(String, Vec<f32>)>::with_capacity(per_gene_sparse.len());
    for (gene, sparse_vals) in per_gene_sparse {
        let gf = kcore::Field::new(&core_domain, sparse_vals)?;
        let sig_sparse = if args.signal == SignalKind::CrossGradient {
            compute_signal(args.signal, &core_domain, &gf, Some(&core_gene))?
        } else {
            compute_signal(args.signal, &core_domain, &gf, None)?
        };
        let sig_dense = grid.scatter_to_dense(&sig_sparse)?;
        per_gene_signal_dense.push((gene, sig_dense));
    }
    log_timing("feature_slice_io_total", t_total);
    Ok((grid, signal_dense, per_gene_signal_dense))
}
