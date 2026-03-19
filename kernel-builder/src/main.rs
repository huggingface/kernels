use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Args, CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use eyre::{Context, Result};

mod completions;
use completions::print_completions;

mod develop;
use develop::{devshell, testshell};

mod hf;

mod init;
use init::{run_init, InitArgs};

mod upload;
use upload::{run_upload, UploadArgs};

mod pyproject;
use pyproject::{clean_pyproject, create_pyproject};

use kernels_data::config::{v3, Build, BuildCompat};

mod nix;

mod util;
use util::{check_or_infer_kernel_dir, parse_and_validate};

#[derive(Args, Debug)]
struct NixArgs {
    /// Maximum number of parallel Nix build jobs.
    #[arg(long)]
    max_jobs: Option<u32>,

    /// Number of CPU cores to use for each build job.
    #[arg(long)]
    cores: Option<u32>,

    /// Print full build logs on standard error.
    #[arg(short = 'L', long)]
    print_build_logs: bool,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Generate shell completions.
    Completions { shell: Shell },

    /// Initialize a new kernel project from template.
    Init(InitArgs),

    /// Upload kernel build artifacts to the Hugging Face Hub.
    Upload(UploadArgs),

    #[command(hide = true)]
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

    /// Spawn a kernel development shell.
    Devshell {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        #[command(flatten)]
        nix_args: NixArgs,
    },

    /// Spawn a kernel test shell.
    Testshell {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        #[command(flatten)]
        nix_args: NixArgs,
    },

    #[command(hide = true)]
    /// Update a `build.toml` to the current format.
    UpdateBuild {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,
    },

    #[command(hide = true)]
    /// Validate the build.toml file.
    Validate {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,
    },

    #[command(hide = true)]
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
        Commands::Completions { shell } => {
            print_completions(&mut Cli::command(), shell);
            Ok(())
        }
        Commands::Init(args) => run_init(args),
        Commands::Upload(args) => run_upload(args),
        Commands::CreatePyproject {
            kernel_dir,
            force,
            target_dir,
            ops_id,
        } => create_pyproject(kernel_dir, target_dir, force, ops_id),
        Commands::Devshell {
            kernel_dir,
            nix_args,
        } => devshell(
            kernel_dir,
            nix_args.max_jobs,
            nix_args.cores,
            nix_args.print_build_logs,
        ),
        Commands::Testshell {
            kernel_dir,
            nix_args,
        } => testshell(
            kernel_dir,
            nix_args.max_jobs,
            nix_args.cores,
            nix_args.print_build_logs,
        ),
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
