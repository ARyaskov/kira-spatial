use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::{Instant, UNIX_EPOCH};

use bitvec::vec::BitVec;
use hdf5::File as H5File;
use kira_spatial_io::CoordSystem;
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};

use crate::error::OrchestratorError;
use crate::pipeline::log_timing;
const FS_CACHE_MAGIC: &[u8; 8] = b"KSPFSMM1";
const FS_CACHE_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub(crate) struct FeatureSlicePaths {
    pub(crate) h5_path: PathBuf,
    pub(crate) parquet_path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct FeatureSliceCache {
    pub(crate) domain: kira_spatial_io::SpatialDomain,
    pub(crate) coord_to_bin: Vec<(u64, u32)>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FeatureIndexCacheFile {
    h5_len: u64,
    h5_mtime_unix: u64,
    map: HashMap<String, usize>,
}
impl FeatureSliceCache {
    pub(crate) fn from_domain(
        domain: &kira_spatial_io::SpatialDomain,
    ) -> Result<Self, OrchestratorError> {
        let rows = domain
            .grid_row
            .as_ref()
            .ok_or(OrchestratorError::InvalidInput(
                "grid_row is required in feature-slice domain",
            ))?;
        let cols = domain
            .grid_col
            .as_ref()
            .ok_or(OrchestratorError::InvalidInput(
                "grid_col is required in feature-slice domain",
            ))?;
        if rows.len() != domain.len() || cols.len() != domain.len() {
            return Err(OrchestratorError::InvalidInput(
                "feature-slice domain coordinate arrays have inconsistent lengths",
            ));
        }

        let mut coord_to_bin = Vec::<(u64, u32)>::with_capacity(domain.len());
        for (idx, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
            coord_to_bin.push((coord_key(row, col), idx as u32));
        }
        coord_to_bin.sort_unstable_by_key(|(key, _)| *key);
        for pair in coord_to_bin.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(OrchestratorError::InvalidInput(
                    "feature-slice domain has duplicate (grid_row, grid_col)",
                ));
            }
        }

        Ok(Self {
            domain: domain.clone(),
            coord_to_bin,
        })
    }
}

pub(crate) fn detect_feature_slice_paths(path: &Path) -> Option<FeatureSlicePaths> {
    let (root_dir, h5_path) = if path.is_dir() {
        let h5 = path.join("feature_slice.h5");
        if !h5.is_file() {
            return None;
        }
        (path.to_path_buf(), h5)
    } else {
        if !is_feature_slice_h5_name(path.file_name()?.to_str()?) {
            return None;
        }
        let root = path.parent()?.to_path_buf();
        (root, path.to_path_buf())
    };

    let parquet_path = parquet_candidates(&h5_path, &root_dir)
        .into_iter()
        .find(|p| p.is_file())?;

    Some(FeatureSlicePaths {
        h5_path,
        parquet_path,
    })
}

fn is_feature_slice_h5_name(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    lower == "feature_slice.h5" || lower.ends_with("_feature_slice.h5")
}

fn paired_barcode_mapping_prefix(h5_path: &Path) -> Option<String> {
    let stem = h5_path.file_stem()?.to_str()?;
    let lower = stem.to_ascii_lowercase();
    let suffix = "_feature_slice";
    if lower == "feature_slice" {
        return None;
    }
    if !lower.ends_with(suffix) {
        return None;
    }
    let prefix_len = stem.len().checked_sub(suffix.len())?;
    if prefix_len == 0 {
        return None;
    }
    Some(stem[..prefix_len].to_string())
}

fn parquet_candidates(h5_path: &Path, root_dir: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::with_capacity(6);
    if let Some(prefix) = paired_barcode_mapping_prefix(h5_path) {
        candidates.push(root_dir.join(format!("{prefix}_barcode_mappings.parquet")));
        candidates.push(root_dir.join(format!("{prefix}_barcode_mapping.parquet")));
        candidates.push(
            root_dir
                .join("spatial")
                .join(format!("{prefix}_barcode_mappings.parquet")),
        );
        candidates.push(
            root_dir
                .join("spatial")
                .join(format!("{prefix}_barcode_mapping.parquet")),
        );
    }
    candidates.push(root_dir.join("barcode_mappings.parquet"));
    candidates.push(root_dir.join("barcode_mapping.parquet"));
    candidates.push(root_dir.join("spatial").join("barcode_mappings.parquet"));
    candidates.push(root_dir.join("spatial").join("barcode_mapping.parquet"));
    candidates
}

pub(crate) fn feature_slice_cache_path(parquet_path: &Path) -> PathBuf {
    let mut p = parquet_path.to_path_buf();
    p.set_extension("kira_spatial.mmap.bin");
    p
}

pub(crate) fn write_feature_slice_cache(
    path: &Path,
    cache: &FeatureSliceCache,
) -> Result<(), OrchestratorError> {
    let n = cache.domain.len();
    let rows = cache
        .domain
        .grid_row
        .as_ref()
        .ok_or(OrchestratorError::InvalidInput(
            "missing grid_row in cache domain",
        ))?;
    let cols = cache
        .domain
        .grid_col
        .as_ref()
        .ok_or(OrchestratorError::InvalidInput(
            "missing grid_col in cache domain",
        ))?;

    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(FS_CACHE_MAGIC)?;
    out.write_all(&FS_CACHE_VERSION.to_le_bytes())?;
    out.write_all(&[coord_system_to_tag(cache.domain.coord_system)])?;
    out.write_all(&[cache.domain.bin_level])?;
    out.write_all(&[0_u8; 2])?;
    out.write_all(&(n as u64).to_le_bytes())?;
    out.write_all(&(cache.coord_to_bin.len() as u64).to_le_bytes())?;

    for &v in &cache.domain.x {
        out.write_all(&v.to_le_bytes())?;
    }
    for &v in &cache.domain.y {
        out.write_all(&v.to_le_bytes())?;
    }
    for &v in rows {
        out.write_all(&v.to_le_bytes())?;
    }
    for &v in cols {
        out.write_all(&v.to_le_bytes())?;
    }
    for i in 0..cache.domain.tissue_mask.len() {
        out.write_all(&[u8::from(cache.domain.tissue_mask[i])])?;
    }
    for &(key, bin_idx) in &cache.coord_to_bin {
        let row = (key >> 32) as u32;
        let col = key as u32;
        out.write_all(&row.to_le_bytes())?;
        out.write_all(&col.to_le_bytes())?;
        out.write_all(&bin_idx.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

pub(crate) fn load_feature_slice_cache(
    path: &Path,
) -> Result<FeatureSliceCache, OrchestratorError> {
    let file = File::open(path)?;
    let mmap = {
        // Read-only mmap cache for fast domain reconstruction.
        unsafe { MmapOptions::new().map(&file)? }
    };
    let data: &[u8] = &mmap;
    let mut off = 0_usize;

    let magic = take_bytes(data, &mut off, FS_CACHE_MAGIC.len())?;
    if magic != FS_CACHE_MAGIC {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "invalid feature-slice cache magic in {}",
            path.display()
        )));
    }
    let version = read_u32_le(data, &mut off)?;
    if version != FS_CACHE_VERSION {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "unsupported feature-slice cache version {} in {}",
            version,
            path.display()
        )));
    }
    let coord_tag = read_u8(data, &mut off)?;
    let bin_level = read_u8(data, &mut off)?;
    let _reserved = take_bytes(data, &mut off, 2)?;
    let n = read_u64_le(data, &mut off)? as usize;
    let n_entries = read_u64_le(data, &mut off)? as usize;

    let x = read_f32_vec(data, &mut off, n)?;
    let y = read_f32_vec(data, &mut off, n)?;
    let grid_row = read_u32_vec(data, &mut off, n)?;
    let grid_col = read_u32_vec(data, &mut off, n)?;
    let tissue_u8 = take_bytes(data, &mut off, n)?.to_vec();
    let mut tissue_mask = BitVec::with_capacity(n);
    tissue_mask.extend(tissue_u8.into_iter().map(|v| v != 0));

    let mut coord_to_bin = Vec::<(u64, u32)>::with_capacity(n_entries);
    for _ in 0..n_entries {
        let row = read_u32_le(data, &mut off)?;
        let col = read_u32_le(data, &mut off)?;
        let bin = read_u32_le(data, &mut off)?;
        coord_to_bin.push((coord_key(row, col), bin));
    }

    if off != data.len() {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "feature-slice cache has trailing bytes in {}",
            path.display()
        )));
    }

    let bin_id = (0..(n as u32)).collect::<Vec<_>>();
    let domain = kira_spatial_io::SpatialDomain::new(
        x,
        y,
        Some(grid_row),
        Some(grid_col),
        bin_id,
        tissue_mask,
        coord_system_from_tag(coord_tag)?,
        bin_level,
    )?;
    Ok(FeatureSliceCache {
        domain,
        coord_to_bin,
    })
}

pub(crate) fn open_h5(path: &Path) -> Result<H5File, OrchestratorError> {
    H5File::open(path).map_err(|e| {
        OrchestratorError::InvalidInputOwned(format!("failed to open h5 {}: {e}", path.display()))
    })
}

fn build_feature_index(h5: &H5File) -> Result<HashMap<String, usize>, OrchestratorError> {
    let ds = h5.dataset("/features/name").map_err(|_| {
        OrchestratorError::InvalidInputOwned("missing /features/name dataset".to_string())
    })?;
    let names = read_string_dataset_any(&ds)?;
    let mut out = HashMap::<String, usize>::with_capacity(names.len());
    for (idx, name) in names.into_iter().enumerate() {
        out.insert(name, idx);
    }
    Ok(out)
}

pub(crate) fn load_or_build_feature_index(
    h5_path: &Path,
    h5: &H5File,
) -> Result<HashMap<String, usize>, OrchestratorError> {
    let t_total = Instant::now();
    if let Ok(map) = load_feature_index_cache(h5_path) {
        eprintln!(
            "[timing] feature_index cache=hit entries={} {:?}",
            map.len(),
            t_total.elapsed()
        );
        return Ok(map);
    }

    let built = match build_feature_index(h5) {
        Ok(v) => {
            eprintln!(
                "[timing] feature_index source=native entries={} {:?}",
                v.len(),
                t_total.elapsed()
            );
            v
        }
        Err(_) => {
            let t_dump = Instant::now();
            let v = build_feature_index_h5dump(h5_path)?;
            eprintln!(
                "[timing] feature_index source=h5dump entries={} {:?}",
                v.len(),
                t_dump.elapsed()
            );
            v
        }
    };
    if let Err(err) = write_feature_index_cache(h5_path, &built) {
        eprintln!(
            "warning: failed writing feature-index cache {}: {}",
            feature_index_cache_path(h5_path).display(),
            err
        );
    }
    eprintln!("[timing] feature_index total {:?}", t_total.elapsed());
    Ok(built)
}

fn feature_index_cache_path(h5_path: &Path) -> PathBuf {
    let mut p = h5_path.to_path_buf();
    p.set_extension("features.index.cache.json");
    p
}

fn load_feature_index_cache(h5_path: &Path) -> Result<HashMap<String, usize>, OrchestratorError> {
    let cache_path = feature_index_cache_path(h5_path);
    let raw = std::fs::read(&cache_path)?;
    let cached: FeatureIndexCacheFile = serde_json::from_slice(&raw)?;

    let meta = std::fs::metadata(h5_path)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    if cached.h5_len != meta.len() || cached.h5_mtime_unix != mtime {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "stale feature-index cache {}",
            cache_path.display()
        )));
    }
    Ok(cached.map)
}

fn write_feature_index_cache(
    h5_path: &Path,
    map: &HashMap<String, usize>,
) -> Result<(), OrchestratorError> {
    let cache_path = feature_index_cache_path(h5_path);
    let meta = std::fs::metadata(h5_path)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());

    let payload = FeatureIndexCacheFile {
        h5_len: meta.len(),
        h5_mtime_unix: mtime,
        map: map.clone(),
    };
    let f = File::create(cache_path)?;
    serde_json::to_writer(BufWriter::new(f), &payload)?;
    Ok(())
}

fn build_feature_index_h5dump(h5_path: &Path) -> Result<HashMap<String, usize>, OrchestratorError> {
    let out = ProcessCommand::new("h5dump")
        .arg("-d")
        .arg("/features/name")
        .arg(h5_path)
        .output()
        .map_err(|e| {
            OrchestratorError::InvalidInputOwned(format!(
                "failed to run h5dump for /features/name: {e}"
            ))
        })?;
    if !out.status.success() {
        return Err(OrchestratorError::InvalidInputOwned(
            "h5dump failed reading /features/name".to_string(),
        ));
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut map = HashMap::<String, usize>::new();
    for line in stdout.lines() {
        if let Some((idx, name)) = parse_h5dump_feature_line(line) {
            map.entry(name).or_insert(idx);
        }
    }
    if map.is_empty() {
        return Err(OrchestratorError::InvalidInputOwned(
            "failed to parse /features/name from h5dump output".to_string(),
        ));
    }
    Ok(map)
}

fn parse_h5dump_feature_line(line: &str) -> Option<(usize, String)> {
    let start = line.find('(')?;
    let end = line[start + 1..].find(')')? + start + 1;
    let idx = line[start + 1..end].trim().parse::<usize>().ok()?;

    let quote_start = line[end..].find('"')? + end + 1;
    let rest = &line[quote_start..];
    let quote_end = rest.find('"')?;
    let mut value = rest[..quote_end].to_string();

    if let Some(null_pos) = value.find("\\000") {
        value.truncate(null_pos);
    }
    Some((idx, value))
}

pub(crate) fn load_feature_slice_values_from_h5(
    h5: &H5File,
    feature_index: usize,
    n_bins: usize,
    coord_to_bin: &[(u64, u32)],
) -> Result<Vec<f32>, OrchestratorError> {
    let t_total = Instant::now();
    let slice_base = format!("/feature_slices/{feature_index}");
    let row_ds = h5.dataset(&(slice_base.clone() + "/row")).map_err(|_| {
        OrchestratorError::InvalidInputOwned(format!("missing {}/row dataset", slice_base))
    })?;
    let col_ds = h5.dataset(&(slice_base.clone() + "/col")).map_err(|_| {
        OrchestratorError::InvalidInputOwned(format!("missing {}/col dataset", slice_base))
    })?;
    let data_ds = h5.dataset(&(slice_base.clone() + "/data")).map_err(|_| {
        OrchestratorError::InvalidInputOwned(format!("missing {}/data dataset", slice_base))
    })?;

    let t_read = Instant::now();
    let rows = read_u32_dataset_any_with_fallback(h5, &row_ds)?;
    let cols = read_u32_dataset_any_with_fallback(h5, &col_ds)?;
    let data = read_u32_dataset_any_with_fallback(h5, &data_ds)?;
    log_timing("feature_slice_row_col_data_read", t_read);
    if rows.len() != cols.len() || rows.len() != data.len() {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "feature-slice array length mismatch in {}",
            slice_base
        )));
    }

    let mut out = vec![0.0_f32; n_bins];
    let t_scatter = Instant::now();
    for ((&row, &col), &count) in rows.iter().zip(cols.iter()).zip(data.iter()) {
        let key = coord_key(row, col);
        if let Ok(pos) = coord_to_bin.binary_search_by_key(&key, |(k, _)| *k) {
            let idx = coord_to_bin[pos].1 as usize;
            out[idx] += count as f32;
        }
    }
    log_timing("feature_slice_row_col_data_scatter", t_scatter);
    log_timing("feature_slice_values_total", t_total);
    Ok(out)
}

fn read_u32_dataset_any(ds: &hdf5::Dataset) -> Result<Vec<u32>, OrchestratorError> {
    if let Ok(v) = ds.read_raw::<u32>() {
        return Ok(v);
    }
    if let Ok(v) = ds.read_raw::<u16>() {
        return Ok(v.into_iter().map(u32::from).collect());
    }
    if let Ok(v) = ds.read_raw::<u8>() {
        return Ok(v.into_iter().map(u32::from).collect());
    }
    if let Ok(v) = ds.read_raw::<u64>() {
        let mut out = Vec::with_capacity(v.len());
        for x in v {
            out.push(u32::try_from(x).map_err(|_| {
                OrchestratorError::InvalidInput(
                    "u64 value does not fit u32 in feature-slice dataset",
                )
            })?);
        }
        return Ok(out);
    }
    if let Ok(v) = ds.read_raw::<i32>() {
        let mut out = Vec::with_capacity(v.len());
        for x in v {
            out.push(u32::try_from(x).map_err(|_| {
                OrchestratorError::InvalidInput("negative i32 value in feature-slice dataset")
            })?);
        }
        return Ok(out);
    }
    if let Ok(v) = ds.read_raw::<i64>() {
        let mut out = Vec::with_capacity(v.len());
        for x in v {
            out.push(u32::try_from(x).map_err(|_| {
                OrchestratorError::InvalidInput("negative i64 value in feature-slice dataset")
            })?);
        }
        return Ok(out);
    }
    if let Ok(v) = ds.read_raw::<i16>() {
        let mut out = Vec::with_capacity(v.len());
        for x in v {
            out.push(u32::try_from(x).map_err(|_| {
                OrchestratorError::InvalidInput("negative i16 value in feature-slice dataset")
            })?);
        }
        return Ok(out);
    }
    if let Ok(v) = ds.read_raw::<i8>() {
        let mut out = Vec::with_capacity(v.len());
        for x in v {
            out.push(u32::try_from(x).map_err(|_| {
                OrchestratorError::InvalidInput("negative i8 value in feature-slice dataset")
            })?);
        }
        return Ok(out);
    }
    if let Ok(v) = ds.read_raw::<f32>() {
        let mut out = Vec::with_capacity(v.len());
        for x in v {
            if !x.is_finite() || x < 0.0 {
                return Err(OrchestratorError::InvalidInput(
                    "invalid f32 value in feature-slice dataset",
                ));
            }
            let r = x.round();
            if (r - x).abs() > 1e-3 || r > u32::MAX as f32 {
                return Err(OrchestratorError::InvalidInput(
                    "non-integer f32 value in feature-slice dataset",
                ));
            }
            out.push(r as u32);
        }
        return Ok(out);
    }
    if let Ok(v) = ds.read_raw::<f64>() {
        let mut out = Vec::with_capacity(v.len());
        for x in v {
            if !x.is_finite() || x < 0.0 {
                return Err(OrchestratorError::InvalidInput(
                    "invalid f64 value in feature-slice dataset",
                ));
            }
            let r = x.round();
            if (r - x).abs() > 1e-6 || r > u32::MAX as f64 {
                return Err(OrchestratorError::InvalidInput(
                    "non-integer f64 value in feature-slice dataset",
                ));
            }
            out.push(r as u32);
        }
        return Ok(out);
    }
    Err(OrchestratorError::InvalidInputOwned(format!(
        "unsupported numeric dataset type at {}",
        ds.name()
    )))
}

fn read_u32_dataset_any_with_fallback(
    h5: &H5File,
    ds: &hdf5::Dataset,
) -> Result<Vec<u32>, OrchestratorError> {
    match read_u32_dataset_any(ds) {
        Ok(v) => Ok(v),
        Err(_) => match read_u32_dataset_h5dump_binary(h5.filename().as_ref(), ds) {
            Ok(v) => Ok(v),
            Err(_) => read_u32_dataset_h5dump(h5.filename().as_ref(), &ds.name()),
        },
    }
}

fn read_u32_dataset_h5dump_binary(
    h5_path: &Path,
    ds: &hdf5::Dataset,
) -> Result<Vec<u32>, OrchestratorError> {
    let t_total = Instant::now();
    let dataset_path = ds.name();
    let n = ds.size();
    if n == 0 {
        return Ok(Vec::new());
    }

    if let Some(cache_path) = h5dump_binary_cache_path(h5_path, &dataset_path)
        && cache_path.is_file()
    {
        let raw = std::fs::read(&cache_path)?;
        let out_vals = parse_native_numeric_bytes_to_u32(&raw, n, &dataset_path)?;
        eprintln!(
            "[timing] h5dump-binary-cache-hit {} {:?}",
            dataset_path,
            t_total.elapsed()
        );
        return Ok(out_vals);
    }

    let tmp_name = format!("kira_spatial_h5dump_{}_{}.bin", std::process::id(), n);
    let tmp_path = std::env::temp_dir().join(tmp_name);

    let out = ProcessCommand::new("h5dump")
        .arg("-d")
        .arg(&dataset_path)
        .arg("-b")
        .arg("NATIVE")
        .arg("-o")
        .arg(&tmp_path)
        .arg(h5_path)
        .output()
        .map_err(|e| {
            OrchestratorError::InvalidInputOwned(format!(
                "failed to run h5dump binary for {}: {e}",
                dataset_path
            ))
        })?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "h5dump binary failed reading {}",
            dataset_path
        )));
    }

    let raw = std::fs::read(&tmp_path)?;
    let _ = std::fs::remove_file(&tmp_path);
    if let Some(cache_path) = h5dump_binary_cache_path(h5_path, &dataset_path) {
        if let Some(parent) = cache_path.parent()
            && let Err(err) = std::fs::create_dir_all(parent)
        {
            eprintln!(
                "warning: failed creating h5dump binary cache dir {}: {}",
                parent.display(),
                err
            );
        }
        if let Err(err) = std::fs::write(&cache_path, &raw) {
            eprintln!(
                "warning: failed writing h5dump binary cache {}: {}",
                cache_path.display(),
                err
            );
        }
    }
    if raw.is_empty() {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "empty h5dump binary output for {}",
            dataset_path
        )));
    }
    let out_vals = parse_native_numeric_bytes_to_u32(&raw, n, &dataset_path)?;
    eprintln!(
        "[timing] h5dump-binary {} {:?}",
        dataset_path,
        t_total.elapsed()
    );
    Ok(out_vals)
}

fn parse_native_numeric_bytes_to_u32(
    raw: &[u8],
    n: usize,
    dataset_path: &str,
) -> Result<Vec<u32>, OrchestratorError> {
    if raw.len() % n != 0 {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "unexpected h5dump binary byte length for {}: bytes={} elements={}",
            dataset_path,
            raw.len(),
            n
        )));
    }
    let item_size = raw.len() / n;
    let mut out_vals = Vec::<u32>::with_capacity(n);
    match item_size {
        1 => {
            out_vals.extend(raw.iter().copied().map(u32::from));
        }
        2 => {
            for chunk in raw.chunks_exact(2) {
                let v = u16::from_ne_bytes([chunk[0], chunk[1]]);
                out_vals.push(u32::from(v));
            }
        }
        4 => {
            for chunk in raw.chunks_exact(4) {
                let mut b = [0_u8; 4];
                b.copy_from_slice(chunk);
                let u = u32::from_ne_bytes(b);
                if u <= i32::MAX as u32 {
                    out_vals.push(u);
                } else {
                    let i = i32::from_ne_bytes(b);
                    out_vals.push(u32::try_from(i).map_err(|_| {
                        OrchestratorError::InvalidInputOwned(format!(
                            "negative value in binary fallback for {}",
                            dataset_path
                        ))
                    })?);
                }
            }
        }
        8 => {
            for chunk in raw.chunks_exact(8) {
                let mut b = [0_u8; 8];
                b.copy_from_slice(chunk);
                let u = u64::from_ne_bytes(b);
                if u <= u32::MAX as u64 {
                    out_vals.push(u as u32);
                } else {
                    let i = i64::from_ne_bytes(b);
                    out_vals.push(u32::try_from(i).map_err(|_| {
                        OrchestratorError::InvalidInputOwned(format!(
                            "out-of-range value in binary fallback for {}",
                            dataset_path
                        ))
                    })?);
                }
            }
        }
        _ => {
            return Err(OrchestratorError::InvalidInputOwned(format!(
                "unsupported binary element size {} for {}",
                item_size, dataset_path
            )));
        }
    }
    Ok(out_vals)
}

fn h5dump_binary_cache_path(h5_path: &Path, dataset_path: &str) -> Option<PathBuf> {
    let meta = std::fs::metadata(h5_path).ok()?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let stem = h5_path.file_stem()?.to_string_lossy();
    let ds = dataset_path
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    let file_name = format!("{}.{}.{}.{}.bin", stem, meta.len(), mtime, ds);
    Some(
        h5_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(".kira_spatial_h5dump_cache")
            .join(file_name),
    )
}

fn read_u32_dataset_h5dump(
    h5_path: &Path,
    dataset_path: &str,
) -> Result<Vec<u32>, OrchestratorError> {
    let t_total = Instant::now();
    let out = ProcessCommand::new("h5dump")
        .arg("-d")
        .arg(dataset_path)
        .arg(h5_path)
        .output()
        .map_err(|e| {
            OrchestratorError::InvalidInputOwned(format!(
                "failed to run h5dump for {}: {e}",
                dataset_path
            ))
        })?;
    if !out.status.success() {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "h5dump failed reading {}",
            dataset_path
        )));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut values = Vec::<u32>::new();
    for line in stdout.lines() {
        let Some(colon) = line.find(':') else {
            continue;
        };
        let payload = &line[colon + 1..];
        for token in payload.split(|ch: char| !(ch.is_ascii_digit() || ch == '-')) {
            if token.is_empty() {
                continue;
            }
            let parsed = token.parse::<i64>().map_err(|_| {
                OrchestratorError::InvalidInputOwned(format!(
                    "invalid numeric token in h5dump output for {}",
                    dataset_path
                ))
            })?;
            values.push(u32::try_from(parsed).map_err(|_| {
                OrchestratorError::InvalidInputOwned(format!(
                    "out-of-range numeric token in h5dump output for {}",
                    dataset_path
                ))
            })?);
        }
    }
    if values.is_empty() {
        return Err(OrchestratorError::InvalidInputOwned(format!(
            "no values parsed from h5dump output for {}",
            dataset_path
        )));
    }
    eprintln!("[timing] h5dump {} {:?}", dataset_path, t_total.elapsed());
    Ok(values)
}

fn read_string_dataset_any(ds: &hdf5::Dataset) -> Result<Vec<String>, OrchestratorError> {
    if let Ok(vals) = ds.read_raw::<hdf5::types::VarLenUnicode>() {
        return Ok(vals.into_iter().map(|v| v.to_string()).collect());
    }
    if let Ok(vals) = ds.read_raw::<hdf5::types::VarLenAscii>() {
        return Ok(vals.into_iter().map(|v| v.to_string()).collect());
    }

    macro_rules! try_fixed {
        ($n:expr) => {
            if let Ok(vals) = ds.read_raw::<hdf5::types::FixedAscii<$n>>() {
                return Ok(vals
                    .into_iter()
                    .map(|v| v.as_str().trim_matches(char::from(0)).to_string())
                    .collect());
            }
            if let Ok(vals) = ds.read_raw::<hdf5::types::FixedUnicode<$n>>() {
                return Ok(vals
                    .into_iter()
                    .map(|v| v.as_str().trim_matches(char::from(0)).to_string())
                    .collect());
            }
            if let Ok(vals) = ds.read_raw::<[u8; $n]>() {
                return Ok(vals
                    .into_iter()
                    .map(|buf| {
                        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
                        String::from_utf8_lossy(&buf[..end]).trim().to_string()
                    })
                    .collect());
            }
        };
    }
    try_fixed!(16);
    try_fixed!(24);
    try_fixed!(32);
    try_fixed!(48);
    try_fixed!(64);
    try_fixed!(96);
    try_fixed!(128);
    try_fixed!(192);
    try_fixed!(256);
    try_fixed!(384);
    try_fixed!(512);
    try_fixed!(768);
    try_fixed!(1024);
    try_fixed!(1536);
    try_fixed!(2048);

    let n = ds.size();
    if n == 0 {
        return Ok(Vec::new());
    }
    if let Ok(raw) = ds.read_raw::<u8>() {
        if raw.len() % n == 0 {
            let width = raw.len() / n;
            if width > 0 {
                let mut out = Vec::with_capacity(n);
                for chunk in raw.chunks_exact(width) {
                    let end = chunk.iter().position(|&b| b == 0).unwrap_or(chunk.len());
                    out.push(String::from_utf8_lossy(&chunk[..end]).trim().to_string());
                }
                return Ok(out);
            }
        }
    }

    Err(OrchestratorError::InvalidInputOwned(format!(
        "unsupported string dataset type at {}",
        ds.name()
    )))
}

fn coord_key(row: u32, col: u32) -> u64 {
    ((row as u64) << 32) | (col as u64)
}

fn coord_system_to_tag(v: CoordSystem) -> u8 {
    match v {
        CoordSystem::Grid => 0,
        CoordSystem::Pixel => 1,
        CoordSystem::Micron => 2,
    }
}

fn coord_system_from_tag(tag: u8) -> Result<CoordSystem, OrchestratorError> {
    match tag {
        0 => Ok(CoordSystem::Grid),
        1 => Ok(CoordSystem::Pixel),
        2 => Ok(CoordSystem::Micron),
        _ => Err(OrchestratorError::InvalidInputOwned(format!(
            "invalid coord-system tag in feature-slice cache: {tag}"
        ))),
    }
}

fn take_bytes<'a>(
    data: &'a [u8],
    off: &mut usize,
    len: usize,
) -> Result<&'a [u8], OrchestratorError> {
    let end = off
        .checked_add(len)
        .ok_or(OrchestratorError::InvalidInput("cache offset overflow"))?;
    if end > data.len() {
        return Err(OrchestratorError::InvalidInput(
            "truncated feature-slice cache",
        ));
    }
    let out = &data[*off..end];
    *off = end;
    Ok(out)
}

fn read_u8(data: &[u8], off: &mut usize) -> Result<u8, OrchestratorError> {
    Ok(take_bytes(data, off, 1)?[0])
}

fn read_u32_le(data: &[u8], off: &mut usize) -> Result<u32, OrchestratorError> {
    let mut arr = [0_u8; 4];
    arr.copy_from_slice(take_bytes(data, off, 4)?);
    Ok(u32::from_le_bytes(arr))
}

fn read_u64_le(data: &[u8], off: &mut usize) -> Result<u64, OrchestratorError> {
    let mut arr = [0_u8; 8];
    arr.copy_from_slice(take_bytes(data, off, 8)?);
    Ok(u64::from_le_bytes(arr))
}

fn read_f32_vec(data: &[u8], off: &mut usize, n: usize) -> Result<Vec<f32>, OrchestratorError> {
    let bytes = take_bytes(
        data,
        off,
        n.checked_mul(4)
            .ok_or(OrchestratorError::InvalidInput("cache length overflow"))?,
    )?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        let mut arr = [0_u8; 4];
        arr.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

fn read_u32_vec(data: &[u8], off: &mut usize, n: usize) -> Result<Vec<u32>, OrchestratorError> {
    let bytes = take_bytes(
        data,
        off,
        n.checked_mul(4)
            .ok_or(OrchestratorError::InvalidInput("cache length overflow"))?,
    )?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        let mut arr = [0_u8; 4];
        arr.copy_from_slice(chunk);
        out.push(u32::from_le_bytes(arr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{is_feature_slice_h5_name, paired_barcode_mapping_prefix};
    use std::path::Path;

    #[test]
    fn accepts_prefixed_feature_slice_name() {
        assert!(is_feature_slice_h5_name(
            "Visium_HD_3prime_Human_Pancreatic_Cancer_feature_slice.h5"
        ));
        assert!(is_feature_slice_h5_name("feature_slice.h5"));
        assert!(!is_feature_slice_h5_name(
            "Visium_HD_3prime_Human_Pancreatic_Cancer.h5"
        ));
    }

    #[test]
    fn derives_prefixed_barcode_mapping_name() {
        let path = Path::new("Visium_HD_3prime_Human_Pancreatic_Cancer_feature_slice.h5");
        assert_eq!(
            paired_barcode_mapping_prefix(path).as_deref(),
            Some("Visium_HD_3prime_Human_Pancreatic_Cancer")
        );
        assert_eq!(
            paired_barcode_mapping_prefix(Path::new("feature_slice.h5")),
            None
        );
    }
}
