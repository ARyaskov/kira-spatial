use thiserror::Error;

use kira_spatial_3d::Error as Spatial3dError;
use kira_spatial_core as kcore;

#[derive(Debug, Error)]
pub(crate) enum OrchestratorError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("io-loader error: {0}")]
    SpatialIo(#[from] kira_spatial_io::SpatialIoError),
    #[error("field error: {0}")]
    Field(#[from] kira_spatial_field::error::FieldError),
    #[error("core error: {0}")]
    Core(#[from] kcore::CoreError),
    #[error("3d error: {0}")]
    Spatial3d(#[from] Spatial3dError),
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    #[error("invalid input: {0}")]
    InvalidInputOwned(String),
}
