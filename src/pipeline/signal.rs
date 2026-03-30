use kira_spatial_3d::{
    ComputeBackend, ComputeConfig, HeightMapSpec, HeightMode, Normalization, NormalizeOptions,
    normalize,
};
use kira_spatial_core as kcore;
use rayon::prelude::*;

use crate::cli::{HeightModeArg, NormalizationArg, RunArgs, SignalKind};
use crate::error::OrchestratorError;

fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        if v == 0.0 { 0.0 } else { v }
    } else {
        0.0
    }
}

pub(crate) fn percentile_from_sorted(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let n = sorted.len();
    let idx = ((pct.clamp(0.0, 100.0) / 100.0) * (n.saturating_sub(1) as f32)).round() as usize;
    sorted[idx.min(n - 1)]
}

pub(crate) fn median_or_default(values: &mut [f32], default: f32) -> f32 {
    if values.is_empty() {
        return default;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

pub(crate) fn build_height_spec(args: &RunArgs) -> Result<HeightMapSpec, OrchestratorError> {
    let compute = if args.scalar {
        ComputeConfig {
            backend: ComputeBackend::Scalar,
        }
    } else {
        ComputeConfig::default()
    };
    Ok(HeightMapSpec {
        mode: map_height_mode(args.height_mode),
        normalization: map_normalization(args)?,
        z_scale: args.z_scale,
        z_offset: args.z_offset,
        compute,
    })
}
pub(crate) fn find_gene_id(
    features: &[kira_spatial_io::FeatureRow],
    gene_name: &str,
) -> Result<u32, OrchestratorError> {
    features
        .iter()
        .find(|f| f.gene_name == gene_name)
        .map(|f| f.gene_id)
        .ok_or_else(|| {
            OrchestratorError::InvalidInputOwned(format!(
                "gene not found in feature table: {gene_name}"
            ))
        })
}

pub(crate) fn compute_signal(
    signal: SignalKind,
    domain: &kcore::SpatialDomain,
    gene_field: &kcore::Field<'_>,
    other_field: Option<&kcore::Field<'_>>,
) -> Result<Vec<f32>, OrchestratorError> {
    match signal {
        SignalKind::Raw => Ok(gene_field
            .values()
            .par_iter()
            .copied()
            .map(sanitize)
            .collect()),
        SignalKind::Gradient | SignalKind::GradMag => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let grad = gene_field.gradient(
                &grid,
                &plan,
                kcore::GradientConfig::default(),
                kcore::OpsConfig::default(),
            )?;

            Ok((0..gene_field.len())
                .into_par_iter()
                .map(|i| {
                    let gx = grad.gx.values()[i] as f32;
                    let gy = grad.gy.values()[i] as f32;
                    let m = (gx.mul_add(gx, gy * gy)).sqrt();
                    sanitize(m)
                })
                .collect())
        }
        SignalKind::Laplacian => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let lap = gene_field.laplacian(
                &grid,
                &plan,
                kcore::LaplacianConfig::default(),
                kcore::OpsConfig::default(),
            )?;
            Ok(lap.values().par_iter().copied().map(sanitize).collect())
        }
        SignalKind::StructureTensor => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let st = gene_field.structure_tensor(
                &grid,
                &plan,
                kcore::StructureTensorConfig::default(),
                kcore::OpsConfig::default(),
            )?;
            Ok(st
                .coherence
                .values()
                .par_iter()
                .copied()
                .map(sanitize)
                .collect())
        }
        SignalKind::Divergence => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let div = gene_field.divergence_of_gradient(
                &grid,
                &plan,
                kcore::EdgeMode::OneSided,
                kcore::OpsConfig::default(),
            )?;
            Ok(div.values().par_iter().copied().map(sanitize).collect())
        }
        SignalKind::Curl => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let curl = gene_field.curl_of_gradient(
                &grid,
                &plan,
                kcore::EdgeMode::OneSided,
                kcore::OpsConfig::default(),
            )?;
            Ok(curl.values().par_iter().copied().map(sanitize).collect())
        }
        SignalKind::DistanceTransform => {
            let grid = kcore::GridIndex::build(domain)?;
            let dt = gene_field.distance_transform(
                &grid,
                kcore::DistanceThreshold::Quantile(0.5),
                false,
            )?;
            Ok(dt.values().par_iter().copied().map(sanitize).collect())
        }
        SignalKind::Curvature => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let curv = gene_field.boundary_curvature(
                &grid,
                &plan,
                kcore::EdgeMode::OneSided,
                1e-6,
                kcore::OpsConfig::default(),
            )?;
            Ok(curv.values().par_iter().copied().map(sanitize).collect())
        }
        SignalKind::HessianRidge | SignalKind::HessianValley => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let feature = match signal {
                SignalKind::HessianRidge => kcore::HessianFeature::Ridge,
                SignalKind::HessianValley => kcore::HessianFeature::Valley,
                _ => unreachable!(),
            };
            let h = gene_field.hessian_feature(
                &grid,
                &plan,
                kcore::HessianConfig {
                    edge_mode: kcore::EdgeMode::OneSided,
                    feature,
                },
                kcore::OpsConfig::default(),
            )?;
            Ok(h.values().par_iter().copied().map(sanitize).collect())
        }
        SignalKind::FractalDimension => {
            let grid = kcore::GridIndex::build(domain)?;
            let frac = gene_field.fractal_dimension(
                &grid,
                kcore::FractalDimensionConfig {
                    threshold: kcore::DistanceThreshold::Quantile(0.5),
                    include_masked: false,
                },
            )?;
            Ok(frac
                .field
                .values()
                .par_iter()
                .copied()
                .map(sanitize)
                .collect())
        }
        SignalKind::Skeletonization => {
            let grid = kcore::GridIndex::build(domain)?;
            let sk = gene_field.skeletonize(
                &grid,
                kcore::SkeletonConfig {
                    threshold: kcore::DistanceThreshold::Quantile(0.5),
                    include_masked: false,
                },
            )?;
            Ok(sk
                .image
                .values()
                .par_iter()
                .copied()
                .map(sanitize)
                .collect())
        }
        SignalKind::Diffusion => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let out = gene_field.diffuse(
                &grid,
                &plan,
                kcore::DiffusionConfig {
                    alpha: 0.2,
                    steps: 8,
                    edge_mode: kcore::EdgeMode::OneSided,
                    include_masked: false,
                },
            )?;
            Ok(out.values().par_iter().copied().map(sanitize).collect())
        }
        SignalKind::MultiscaleLog => {
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let log = gene_field.multiscale_log(
                &grid,
                &plan,
                &[1, 2, 4, 8],
                kcore::MultiScaleLogConfig::default(),
            )?;
            let mut out = vec![0.0_f32; gene_field.len()];
            for (i, dst) in out.iter_mut().enumerate() {
                let mut best = 0.0_f32;
                for resp in &log.responses {
                    let v = (resp.values()[i] as f32).abs();
                    if v > best {
                        best = v;
                    }
                }
                *dst = sanitize(best);
            }
            Ok(out)
        }
        SignalKind::CrossGradient => {
            let other = other_field.ok_or(OrchestratorError::InvalidInput(
                "cross-gradient requires two fields (provide at least two values in --genes)",
            ))?;
            let grid = kcore::GridIndex::build(domain)?;
            let plan = kcore::TilePlan::new(
                grid.spec().width,
                grid.spec().height,
                kcore::TileOrder::RowMajor,
            )?;
            let cg = gene_field.cross_gradient(
                other,
                &grid,
                &plan,
                kcore::EdgeMode::OneSided,
                kcore::OpsConfig::default(),
            )?;
            Ok(cg.values().par_iter().copied().map(sanitize).collect())
        }
    }
}

pub(crate) fn normalize_for_contours(signal_dense: &[f32], spec: HeightMapSpec) -> Vec<f32> {
    let transformed: Vec<f32> = signal_dense
        .par_iter()
        .map(|&v| {
            if !v.is_finite() {
                return f32::NAN;
            }
            match spec.mode {
                HeightMode::Raw | HeightMode::Signed => v,
                HeightMode::Abs => v.abs(),
            }
        })
        .collect();

    normalize(
        &transformed,
        NormalizeOptions {
            policy: spec.normalization,
        },
    )
}

fn map_height_mode(mode: HeightModeArg) -> HeightMode {
    match mode {
        HeightModeArg::Raw => HeightMode::Raw,
        HeightModeArg::Abs => HeightMode::Abs,
        HeightModeArg::Signed => HeightMode::Signed,
    }
}

fn map_normalization(args: &RunArgs) -> Result<Normalization, OrchestratorError> {
    let policy = match args.normalization {
        NormalizationArg::None => Normalization::None,
        NormalizationArg::Minmax => Normalization::MinMax { clip: None },
        NormalizationArg::Robustz => Normalization::RobustZ { clip_z: None },
        NormalizationArg::Percentile => {
            if !args.percentile_lo.is_finite() || !args.percentile_hi.is_finite() {
                return Err(OrchestratorError::InvalidInput(
                    "percentile_lo/percentile_hi must be finite",
                ));
            }
            if !(0.0..=100.0).contains(&args.percentile_lo)
                || !(0.0..=100.0).contains(&args.percentile_hi)
                || args.percentile_lo >= args.percentile_hi
            {
                return Err(OrchestratorError::InvalidInput(
                    "percentile bounds must satisfy 0<=lo<hi<=100",
                ));
            }
            Normalization::Percentile {
                lo: args.percentile_lo,
                hi: args.percentile_hi,
            }
        }
    };
    kira_spatial_3d::validate_normalization(&policy)?;
    Ok(policy)
}
