use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use eyre::{Context, Result};

mod pyproject;
use pyproject::{clean_pyproject, create_pyproject};

mod config;
use config::{v3, Build, BuildCompat};

mod util;
use util::{check_or_infer_kernel_dir, parse_and_validate};

mod version;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Generate CMake files for a kernel extension build.
    CreatePyproject {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        /// The directory to write the generated files to
        /// (directory of `BUILD_TOML` when absent).
        #[arg(name = "TARGET_DIR")]
        target_dir: Option<PathBuf>,

        /// Force-overwrite existing files.
        #[arg(short, long)]
        force: bool,

        /// This is an optional unique identifier that is suffixed to the
        /// kernel name to avoid name collisions. (e.g. Git SHA)
        #[arg(long)]
        ops_id: Option<String>,
    },

    /// Update a `build.toml` to the current format.
    UpdateBuild {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,
    },

    /// Validate the build.toml file.
    Validate {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,
    },

    /// Clean generated artifacts.
    CleanPyproject {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        /// The directory to clean from (directory of `BUILD_TOML` when absent).
        #[arg(name = "TARGET_DIR")]
        target_dir: Option<PathBuf>,

        /// Show what would be deleted without actually deleting.
        #[arg(short, long)]
        dry_run: bool,

        /// Force deletion without confirmation.
        #[arg(short, long)]
        force: bool,

        /// This is an optional unique identifier that is suffixed to the
        /// kernel name to avoid name collisions. (e.g. Git SHA)
        #[arg(long)]
        ops_id: Option<String>,
    },
}

fn main() -> Result<()> {
    let args = Cli::parse();
    match args.command {
        Commands::CreatePyproject {
            kernel_dir,
            force,
            target_dir,
            ops_id,
        } => create_pyproject(kernel_dir, target_dir, force, ops_id),
        Commands::UpdateBuild { kernel_dir } => update_build(kernel_dir),
        Commands::Validate { kernel_dir } => {
            validate(kernel_dir)?;
            Ok(())
        }
        Commands::CleanPyproject {
            kernel_dir,
            target_dir,
            dry_run,
            force,
            ops_id,
        } => clean_pyproject(kernel_dir, target_dir, dry_run, force, ops_id),
    }
}

fn validate(kernel_dir: Option<PathBuf>) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    parse_and_validate(kernel_dir)?;
    Ok(())
}

fn update_build(kernel_dir: Option<PathBuf>) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let build_compat: BuildCompat = parse_and_validate(&kernel_dir)?;

    if matches!(build_compat, BuildCompat::V3(_)) {
        return Ok(());
    }

    let build: Build = build_compat
        .try_into()
        .context("Cannot update build configuration")?;
    let v3_build: v3::Build = build.into();
    let pretty_toml = toml::to_string_pretty(&v3_build)?;

    let build_toml = kernel_dir.join("build.toml");
    let mut writer =
        BufWriter::new(File::create(&build_toml).wrap_err_with(|| {
            format!("Cannot open {} for writing", build_toml.to_string_lossy())
        })?);
    writer
        .write_all(pretty_toml.as_bytes())
        .wrap_err_with(|| format!("Cannot write to {}", build_toml.to_string_lossy()))?;

    Ok(())
}
