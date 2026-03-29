mod grid;
mod load;
mod render;
mod signal;

use std::fs::create_dir_all;
use std::time::Instant;

use crate::cli::RunArgs;
use crate::error::OrchestratorError;
use crate::image_render::write_per_gene_signal_webps;
use kira_spatial_io::{Dataset, LoadConfig};
use load::{
    compute_signal_dense_from_dataset, compute_signal_dense_from_feature_slice,
    is_feature_slice_matrix_missing,
};
use render::{render_outputs, resolve_contour_levels};

pub(crate) use grid::GridMapping;
pub(crate) use signal::{build_height_spec, percentile_from_sorted};

pub(crate) fn run_pipeline(args: RunArgs) -> Result<(), OrchestratorError> {
    let t_total = Instant::now();
    if args.gene.is_empty() {
        return Err(OrchestratorError::InvalidInput(
            "at least one --gene is required",
        ));
    }

    if !args.quantize_grid.is_finite() || args.quantize_grid <= 0.0 {
        return Err(OrchestratorError::InvalidInput(
            "quantize_grid must be finite and > 0",
        ));
    }
    if !args.laplacian_sigma.is_finite() || args.laplacian_sigma < 0.0 {
        return Err(OrchestratorError::InvalidInput(
            "laplacian_sigma must be finite and >= 0",
        ));
    }

    create_dir_all(&args.out_dir)?;

    let t_load_signal = Instant::now();
    let (grid, signal_dense, per_gene_signal_dense) =
        if crate::feature_slice::detect_feature_slice_paths(&args.h5).is_some() {
            compute_signal_dense_from_feature_slice(&args)?
        } else {
            match Dataset::open_h5(&args.h5, LoadConfig::default()) {
                Ok(dataset) => compute_signal_dense_from_dataset(&args, &dataset)?,
                Err(err) if is_feature_slice_matrix_missing(&err) => {
                    compute_signal_dense_from_feature_slice(&args)?
                }
                Err(err) => return Err(err.into()),
            }
        };
    log_timing("load+signal", t_load_signal);

    let t_webp = Instant::now();
    write_per_gene_signal_webps(&args, &grid, &per_gene_signal_dense)?;
    log_timing("per_gene_webps", t_webp);

    if !args.extended {
        log_timing("run_pipeline_total", t_total);
        return Ok(());
    }

    let contour_levels = resolve_contour_levels(&args, &per_gene_signal_dense)?;

    let t_render = Instant::now();
    let out = render_outputs(&args, &grid, &signal_dense, &contour_levels);
    log_timing("render_outputs", t_render);
    log_timing("run_pipeline_total", t_total);
    out
}

pub(crate) fn log_timing(label: &str, start: Instant) {
    eprintln!("[timing] {} {:?}", label, start.elapsed());
}
