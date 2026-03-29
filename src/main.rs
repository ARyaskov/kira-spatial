mod cli;
mod error;
mod feature_slice;
mod image_render;
mod pipeline;

use clap::Parser;

use crate::cli::{Cli, Command};
use crate::error::OrchestratorError;

fn main() -> Result<(), OrchestratorError> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Run(args) => pipeline::run_pipeline(args),
    }
}
