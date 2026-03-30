use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "kira-spatial",
    version,
    about = "Deterministic orchestrator: io -> field -> core -> 3d"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) cmd: Command,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum Command {
    /// Run full deterministic pipeline from H5 to 3D exports.
    Run(RunArgs),
}

#[derive(Debug, Parser)]
pub(crate) struct RunArgs {
    /// Path to input .h5 file or directory containing feature_slice.h5 + spatial/
    #[arg(long)]
    pub(crate) h5: PathBuf,
    /// Comma-separated gene symbols to aggregate, for example: EPCAM,KRT19,MSLN.
    #[arg(long, required = true, value_delimiter = ',')]
    pub(crate) genes: Vec<String>,
    /// Output directory for all generated artifacts.
    #[arg(long)]
    pub(crate) out_dir: PathBuf,
    /// Derived scalar used for 3D projection.
    #[arg(long, value_enum, default_value = "grad-mag")]
    pub(crate) signal: SignalKind,
    /// Height interpretation mode before normalization.
    #[arg(long, value_enum, default_value = "abs")]
    pub(crate) height_mode: HeightModeArg,
    /// Normalization policy.
    #[arg(long, value_enum, default_value = "percentile")]
    pub(crate) normalization: NormalizationArg,
    /// Lower percentile for percentile normalization.
    #[arg(long, default_value_t = 5.0)]
    pub(crate) percentile_lo: f32,
    /// Upper percentile for percentile normalization.
    #[arg(long, default_value_t = 95.0)]
    pub(crate) percentile_hi: f32,
    /// Final z scale.
    #[arg(long, default_value_t = 20.0)]
    pub(crate) z_scale: f32,
    /// Final z offset.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) z_offset: f32,
    /// Contour levels computed on normalized field (pre-affine):
    /// "auto" (default) or comma-separated list like "0.2,0.4,0.6,0.8".
    #[arg(long, default_value = "auto")]
    pub(crate) contour_levels: String,
    /// Quantization grid for deterministic contour stitching.
    #[arg(long, default_value_t = 0.01)]
    pub(crate) quantize_grid: f32,
    /// Float decimals for text exports.
    #[arg(long, default_value_t = 6)]
    pub(crate) float_decimals: usize,
    /// Force scalar backend (disable SIMD path).
    #[arg(long)]
    pub(crate) scalar: bool,
    /// Disable contour extraction even if levels are provided.
    #[arg(long)]
    pub(crate) no_contours: bool,
    /// Write heavy surface.obj and surface.ply artifacts.
    #[arg(long, default_value_t = false)]
    pub(crate) write_obj_ply: bool,
    /// Extended output mode: in addition to per-gene images, write mesh/contour/json artifacts.
    #[arg(long, default_value_t = false)]
    pub(crate) extended: bool,
    /// Laplacian visualization style for per-gene images.
    #[arg(long, value_enum, default_value = "diverging")]
    pub(crate) laplacian_viz: LaplacianVizMode,
    /// Optional Gaussian sigma for Laplacian image smoothing (LoG-like).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) laplacian_sigma: f32,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub(crate) enum SignalKind {
    Raw,
    Gradient,
    GradMag,
    Laplacian,
    HessianRidge,
    HessianValley,
    StructureTensor,
    Divergence,
    Curl,
    DistanceTransform,
    Curvature,
    FractalDimension,
    Skeletonization,
    Diffusion,
    MultiscaleLog,
    CrossGradient,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub(crate) enum LaplacianVizMode {
    Diverging,
    ZeroCrossings,
    Both,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum HeightModeArg {
    Raw,
    Abs,
    Signed,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum NormalizationArg {
    None,
    Minmax,
    Robustz,
    Percentile,
}
