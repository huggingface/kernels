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

mod build;
use build::{run_build, run_build_and_copy};

mod hf;

mod init;
use init::{run_init, InitArgs};

mod list_variants;
use list_variants::list_variants;

mod upload;
use upload::{run_upload, RepoTypeArg, UploadArgs};

mod pyproject;
use pyproject::{clean_pyproject, create_pyproject};

use kernels_data::config::{v3, Build, BuildCompat};

mod nix;

mod skills;

mod util;
use util::{check_or_infer_kernel_dir, parse_and_validate};

#[derive(Args, Debug)]
struct NixArgs {
    /// Maximum number of parallel Nix build jobs.
    #[arg(long)]
    pub max_jobs: Option<u32>,

    /// Number of CPU cores to use for each build job.
    #[arg(long)]
    pub cores: Option<u32>,

    /// Print full build logs on standard error.
    #[arg(short = 'L', long)]
    pub print_build_logs: bool,
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

    /// Build the kernel locally (alias for build-and-copy).
    Build {
        /// Directory of the kernel project (defaults to current directory).
        #[arg(value_name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        /// Build a specific variant.
        #[arg(long)]
        variant: Option<String>,

        #[command(flatten)]
        nix_args: NixArgs,
    },

    /// Build the kernel and copy artifacts locally.
    BuildAndCopy {
        /// Directory of the kernel project (defaults to current directory).
        #[arg(value_name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        #[command(flatten)]
        nix_args: NixArgs,
    },

    /// Build the kernel and upload to Hugging Face Hub.
    BuildAndUpload {
        /// Directory of the kernel project (defaults to current directory).
        #[arg(value_name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        /// Build a specific variant.
        #[arg(long)]
        variant: Option<String>,

        #[command(flatten)]
        nix_args: NixArgs,

        /// Repository ID on the Hugging Face Hub (e.g. `user/my-kernel`).
        #[arg(long)]
        repo_id: Option<String>,

        /// Upload to a specific branch (defaults to `v{version}` from metadata).
        #[arg(long)]
        branch: Option<String>,

        /// Create the repository as private.
        #[arg(long)]
        private: bool,

        /// Repository type on Hugging Face Hub (`kernel` by default, or `model` for legacy repos).
        #[arg(long, value_enum, default_value_t = RepoTypeArg::Kernel)]
        repo_type: RepoTypeArg,
    },

    /// Upload kernel build artifacts to the Hugging Face Hub.
    Upload(UploadArgs),

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

        /// Use a specific variant.
        #[arg(long)]
        variant: Option<String>,

        #[command(flatten)]
        nix_args: NixArgs,
    },

    /// List build variants.
    ListVariants {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        /// Only list variants for the current architecture.
        #[arg(long)]
        arch: bool,
    },

    /// Spawn a kernel test shell.
    Testshell {
        #[arg(name = "KERNEL_DIR")]
        kernel_dir: Option<PathBuf>,

        /// Use a specific variant.
        #[arg(long)]
        variant: Option<String>,

        #[command(flatten)]
        nix_args: NixArgs,
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

    /// Install skills for AI coding assistants (Claude, Codex, OpenCode).
    Skills {
        #[command(subcommand)]
        command: SkillsCommands,
    },

    /// Generate Markdown documentation for the CLI.
    #[command(hide = true)]
    GenerateDocs,

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

#[derive(Debug, Subcommand)]
enum SkillsCommands {
    /// Install a kernels skill for an AI assistant.
    Add {
        /// Skill to install.
        #[arg(long, value_enum, default_value_t = skills::SkillId::CudaKernels)]
        skill: skills::SkillId,

        /// Install for Claude.
        #[arg(long)]
        claude: bool,

        /// Install for Codex.
        #[arg(long)]
        codex: bool,

        /// Install for OpenCode.
        #[arg(long)]
        opencode: bool,

        /// Install globally (user-level) instead of in the current project directory.
        #[arg(short, long)]
        global: bool,

        /// Install into a custom destination (path to skills directory).
        #[arg(long)]
        dest: Option<PathBuf>,

        /// Overwrite existing skills in the destination.
        #[arg(long)]
        force: bool,
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
        Commands::Build {
            kernel_dir,
            variant,
            nix_args,
        } => run_build(
            kernel_dir,
            nix_args.max_jobs,
            nix_args.cores,
            nix_args.print_build_logs,
            variant,
        ),
        Commands::BuildAndCopy {
            kernel_dir,
            nix_args,
            ..
        } => run_build_and_copy(
            kernel_dir,
            nix_args.max_jobs,
            nix_args.cores,
            nix_args.print_build_logs,
        ),
        Commands::BuildAndUpload {
            kernel_dir,
            variant,
            nix_args,
            repo_id,
            branch,
            private,
            repo_type,
        } => {
            run_build(
                kernel_dir.clone(),
                nix_args.max_jobs,
                nix_args.cores,
                nix_args.print_build_logs,
                variant,
            )?;
            run_upload(UploadArgs {
                kernel_dir,
                repo_id,
                branch,
                private,
                repo_type,
            })
        }
        Commands::CreatePyproject {
            kernel_dir,
            force,
            target_dir,
            ops_id,
        } => create_pyproject(kernel_dir, target_dir, force, ops_id),
        Commands::Devshell {
            kernel_dir,
            variant,
            nix_args,
        } => devshell(
            kernel_dir,
            nix_args.max_jobs,
            nix_args.cores,
            nix_args.print_build_logs,
            variant,
        ),
        Commands::ListVariants { kernel_dir, arch } => list_variants(kernel_dir, arch),
        Commands::Testshell {
            kernel_dir,
            variant,
            nix_args,
        } => testshell(
            kernel_dir,
            nix_args.max_jobs,
            nix_args.cores,
            nix_args.print_build_logs,
            variant,
        ),
        Commands::UpdateBuild { kernel_dir } => update_build(kernel_dir),
        Commands::Validate { kernel_dir } => {
            validate(kernel_dir)?;
            Ok(())
        }
        Commands::GenerateDocs => {
            let markdown = clap_markdown::help_markdown::<Cli>();
            print!("{}", markdown);
            Ok(())
        }
        Commands::Skills { command } => match command {
            SkillsCommands::Add {
                skill,
                claude,
                codex,
                opencode,
                global,
                dest,
                force,
            } => skills::add_skill(skill, claude, codex, opencode, global, dest, force),
        },
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
