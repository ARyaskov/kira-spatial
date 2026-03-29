use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use kira_spatial_3d::{
    ContourMetaInput, ExportBundleOptions, FloatFmt, Quantize, ScalarField, Spatial3dMetadata,
    SpatialDomain, StitchOptions, TsvOptions, build_heightmap_mesh_mapped, compute_ridge_metrics,
    extract_contours, fmt_f32, save_metadata_json, save_polylines_json, save_polylines_tsv,
    save_ridge_metrics_json, save_ridge_metrics_tsv, stitch_contours,
};
use serde::Serialize;

use crate::cli::RunArgs;
use crate::error::OrchestratorError;
use crate::pipeline::grid::GridMapping;
use crate::pipeline::log_timing;
use crate::pipeline::signal::{
    build_height_spec, median_or_default, normalize_for_contours, percentile_from_sorted,
};

#[derive(Serialize)]
struct ContourIndex {
    version: &'static str,
    files: Vec<String>,
}

pub(crate) fn render_outputs(
    args: &RunArgs,
    grid: &GridMapping,
    signal_dense: &[f32],
    contour_levels: &[f32],
) -> Result<(), OrchestratorError> {
    let t_total = Instant::now();
    let mesh_domain = SpatialDomain::new(grid.nx, grid.ny, grid.origin_x, grid.origin_y, 1.0, 1.0)?;

    let height_spec = build_height_spec(args)?;

    let dense_field = ScalarField::new(mesh_domain, signal_dense)?;
    let t_mesh = Instant::now();
    let mesh = build_heightmap_mesh_mapped(&dense_field, height_spec)?;
    log_timing("mesh_build", t_mesh);

    let t_export = Instant::now();
    kira_spatial_3d::export_bundle(
        &args.out_dir,
        Some(&mesh),
        None,
        None,
        None,
        ExportBundleOptions {
            float: FloatFmt {
                decimals: args.float_decimals,
            },
            write_obj: args.write_obj_ply,
            write_ply: args.write_obj_ply,
            obj_normals: true,
            ply_normals: true,
            write_k3d: true,
        },
    )?;
    log_timing("mesh_export_bundle", t_export);

    let contour_enabled = !args.no_contours && !contour_levels.is_empty();
    if contour_enabled {
        let t_contours = Instant::now();
        let normed = normalize_for_contours(signal_dense, height_spec);
        let normed_field = ScalarField::new(mesh_domain, &normed)?;
        let multi = extract_contours(&normed_field, contour_levels)?;
        let float_fmt = FloatFmt {
            decimals: args.float_decimals,
        };

        let mut files = Vec::<String>::new();
        for set in multi.contours {
            let poly = stitch_contours(
                &set,
                StitchOptions {
                    quantize: Quantize {
                        grid: args.quantize_grid,
                    },
                },
            )?;
            let metrics = compute_ridge_metrics(&poly);

            let level_tag = fmt_f32(set.level, float_fmt);
            let poly_json = format!("polylines.level_{level_tag}.json");
            let poly_tsv = format!("polylines.level_{level_tag}.tsv");
            let metrics_json = format!("ridge_metrics.level_{level_tag}.json");
            let metrics_tsv = format!("ridge_metrics.level_{level_tag}.tsv");

            save_polylines_json(&poly, args.out_dir.join(&poly_json))?;
            save_polylines_tsv(
                &poly,
                args.out_dir.join(&poly_tsv),
                TsvOptions { float: float_fmt },
            )?;
            save_ridge_metrics_json(&metrics, args.out_dir.join(&metrics_json))?;
            save_ridge_metrics_tsv(
                &metrics,
                args.out_dir.join(&metrics_tsv),
                TsvOptions { float: float_fmt },
            )?;

            files.push(poly_json);
            files.push(poly_tsv);
            files.push(metrics_json);
            files.push(metrics_tsv);
        }

        if contour_levels.len() > 1 {
            let index = ContourIndex {
                version: "kira-spatial/index/v1",
                files,
            };
            let index_file = File::create(args.out_dir.join("index.json"))?;
            serde_json::to_writer(BufWriter::new(index_file), &index)?;
        }
        log_timing("contour_pipeline", t_contours);
    }

    let contour_meta = contour_enabled.then_some(ContourMetaInput {
        levels: contour_levels,
        quantize: Quantize {
            grid: args.quantize_grid,
        },
    });
    let metadata = Spatial3dMetadata::from_specs(mesh_domain, height_spec, contour_meta);
    save_metadata_json(&metadata, args.out_dir.join("metadata.json"))?;

    log_timing("render_outputs_total", t_total);
    Ok(())
}

pub(crate) fn resolve_contour_levels(
    args: &RunArgs,
    per_gene_signal_dense: &[(String, Vec<f32>)],
) -> Result<Vec<f32>, OrchestratorError> {
    let raw = args.contour_levels.trim();
    if raw.eq_ignore_ascii_case("auto") {
        return resolve_auto_contour_levels(args, per_gene_signal_dense);
    }
    if raw.is_empty() {
        return Ok(Vec::new());
    }

    let mut levels = Vec::<f32>::new();
    for part in raw.split(',') {
        let token = part.trim();
        if token.is_empty() {
            continue;
        }
        let parsed = token.parse::<f32>().map_err(|_| {
            OrchestratorError::InvalidInputOwned(format!("invalid contour level value: {token}"))
        })?;
        if !parsed.is_finite() {
            return Err(OrchestratorError::InvalidInput(
                "all contour levels must be finite",
            ));
        }
        levels.push(parsed);
    }
    if levels.is_empty() {
        return Err(OrchestratorError::InvalidInput(
            "contour levels are empty; use \"auto\" or provide numeric values",
        ));
    }
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup_by(|a, b| (*a - *b).abs() <= 1e-6);
    Ok(levels)
}

pub(crate) fn resolve_auto_contour_levels(
    args: &RunArgs,
    per_gene_signal_dense: &[(String, Vec<f32>)],
) -> Result<Vec<f32>, OrchestratorError> {
    let height_spec = build_height_spec(args)?;
    let mut p70 = Vec::<f32>::new();
    let mut p80 = Vec::<f32>::new();
    let mut p90 = Vec::<f32>::new();

    for (_, signal_dense) in per_gene_signal_dense {
        let normed = normalize_for_contours(signal_dense, height_spec);
        let mut finite_pos = normed
            .into_iter()
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect::<Vec<_>>();
        if finite_pos.is_empty() {
            finite_pos = signal_dense
                .iter()
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .collect::<Vec<_>>();
        }
        if finite_pos.is_empty() {
            continue;
        }
        finite_pos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        p70.push(percentile_from_sorted(&finite_pos, 70.0));
        p80.push(percentile_from_sorted(&finite_pos, 80.0));
        p90.push(percentile_from_sorted(&finite_pos, 90.0));
    }

    let mut levels = vec![
        median_or_default(&mut p70, 0.2),
        median_or_default(&mut p80, 0.4),
        median_or_default(&mut p90, 0.6),
    ];
    if levels.iter().all(|v| !v.is_finite() || *v <= 0.0) {
        levels = vec![0.2, 0.4, 0.6];
    }
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup_by(|a, b| (*a - *b).abs() <= 1e-6);
    eprintln!(
        "[timing] contour_levels_auto levels={}",
        levels
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(levels)
}
