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
use util::parse_and_validate;

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
        #[arg(name = "BUILD_TOML")]
        build_toml: PathBuf,

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
        #[arg(name = "BUILD_TOML")]
        build_toml: PathBuf,
    },

    /// Validate the build.toml file.
    Validate {
        #[arg(name = "BUILD_TOML")]
        build_toml: PathBuf,
    },

    /// Clean generated artifacts.
    CleanPyproject {
        #[arg(name = "BUILD_TOML")]
        build_toml: PathBuf,

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
            build_toml,
            force,
            target_dir,
            ops_id,
        } => create_pyproject(build_toml, target_dir, force, ops_id),
        Commands::UpdateBuild { build_toml } => update_build(build_toml),
        Commands::Validate { build_toml } => {
            parse_and_validate(build_toml)?;
            Ok(())
        }
        Commands::CleanPyproject {
            build_toml,
            target_dir,
            dry_run,
            force,
            ops_id,
        } => clean_pyproject(build_toml, target_dir, dry_run, force, ops_id),
    }
}

fn update_build(build_toml: PathBuf) -> Result<()> {
    let build_compat: BuildCompat = parse_and_validate(&build_toml)?;

    if matches!(build_compat, BuildCompat::V3(_)) {
        return Ok(());
    }

    let build: Build = build_compat
        .try_into()
        .context("Cannot update build configuration")?;
    let v3_build: v3::Build = build.into();
    let pretty_toml = toml::to_string_pretty(&v3_build)?;

    let mut writer =
        BufWriter::new(File::create(&build_toml).wrap_err_with(|| {
            format!("Cannot open {} for writing", build_toml.to_string_lossy())
        })?);
    writer
        .write_all(pretty_toml.as_bytes())
        .wrap_err_with(|| format!("Cannot write to {}", build_toml.to_string_lossy()))?;

    Ok(())
}
