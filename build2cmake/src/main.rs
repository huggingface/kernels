use std::{
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use clap::{Parser, Subcommand};
use eyre::{bail, ensure, Context, Result};
use minijinja::Environment;

mod torch;
use torch::{write_torch_ext, write_torch_ext_noarch};

mod config;
use config::{v3, Backend, Build, BuildCompat};

mod fileset;
use fileset::FileSet;

mod metadata;

mod version;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Generate CMake files for Torch extension builds.
    GenerateTorch {
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

        #[arg(long)]
        backend: Option<Backend>,
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
    Clean {
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
        Commands::GenerateTorch {
            backend,
            build_toml,
            force,
            target_dir,
            ops_id,
        } => generate_torch(backend, build_toml, target_dir, force, ops_id),
        Commands::UpdateBuild { build_toml } => update_build(build_toml),
        Commands::Validate { build_toml } => {
            parse_and_validate(build_toml)?;
            Ok(())
        }
        Commands::Clean {
            build_toml,
            target_dir,
            dry_run,
            force,
            ops_id,
        } => clean(build_toml, target_dir, dry_run, force, ops_id),
    }
}

fn generate_torch(
    backend: Option<Backend>,
    build_toml: PathBuf,
    target_dir: Option<PathBuf>,
    force: bool,
    ops_id: Option<String>,
) -> Result<()> {
    let target_dir = check_or_infer_target_dir(&build_toml, target_dir)?;

    let build_compat = parse_and_validate(build_toml)?;

    if matches!(build_compat, BuildCompat::V1(_) | BuildCompat::V2(_)) {
        eprintln!(
            "build.toml is in the deprecated V1 or V2 format, use `build2cmake update-build` to update."
        )
    }

    let build: Build = build_compat
        .try_into()
        .context("Cannot update build configuration")?;

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    minijinja_embed::load_templates!(&mut env);

    let backend = match backend {
        Some(backend) => {
            if !build.supports_backend(&backend) {
                bail!("Kernel does not support backend: {}", backend);
            }

            backend
        }
        None => {
            let kernel_backends = &build.general.backends;

            if kernel_backends.len() > 1 {
                let mut kernel_backends = kernel_backends
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                kernel_backends.sort();
                bail!(
                    "Multiple supported backends found in build.toml: {}. Please specify one with --backend.",
                    kernel_backends.join(", ")
                );
            }

            if let Some(backend) = kernel_backends.first() {
                *backend
            } else {
                bail!("No backends are specified in build.toml");
            }
        }
    };

    let file_set = if build.is_noarch() {
        write_torch_ext_noarch(&env, backend, &build, target_dir.clone(), ops_id)?
    } else {
        write_torch_ext(&env, backend, &build, target_dir.clone(), ops_id)?
    };
    file_set.write(&target_dir, force)?;

    Ok(())
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

fn check_or_infer_target_dir(
    build_toml: impl AsRef<Path>,
    target_dir: Option<PathBuf>,
) -> Result<PathBuf> {
    let build_toml = build_toml.as_ref();
    match target_dir {
        Some(target_dir) => {
            ensure!(
                target_dir.is_dir(),
                "`{}` is not a directory",
                target_dir.to_string_lossy()
            );
            Ok(target_dir)
        }
        None => {
            let absolute = std::path::absolute(build_toml)?;
            match absolute.parent() {
                Some(parent) => Ok(parent.to_owned()),
                None => bail!(
                    "Cannot get parent path of `{}`",
                    build_toml.to_string_lossy()
                ),
            }
        }
    }
}

fn parse_and_validate(build_toml: impl AsRef<Path>) -> Result<BuildCompat> {
    let build_toml = build_toml.as_ref();
    let mut toml_data = String::new();
    File::open(build_toml)
        .wrap_err_with(|| format!("Cannot open {} for reading", build_toml.to_string_lossy()))?
        .read_to_string(&mut toml_data)
        .wrap_err_with(|| format!("Cannot read from {}", build_toml.to_string_lossy()))?;

    let build_compat: BuildCompat = toml::from_str(&toml_data)
        .wrap_err_with(|| format!("Cannot parse TOML in {}", build_toml.to_string_lossy()))?;

    Ok(build_compat)
}

fn clean(
    build_toml: PathBuf,
    target_dir: Option<PathBuf>,
    dry_run: bool,
    force: bool,
    ops_id: Option<String>,
) -> Result<()> {
    let target_dir = check_or_infer_target_dir(&build_toml, target_dir)?;

    let build_compat = parse_and_validate(build_toml)?;

    if matches!(build_compat, BuildCompat::V1(_) | BuildCompat::V2(_)) {
        eprintln!(
            "build.toml is in the deprecated V1 or V2 format, use `build2cmake update-build` to update."
        )
    }

    let build: Build = build_compat
        .try_into()
        .context("Cannot update build configuration")?;

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    minijinja_embed::load_templates!(&mut env);

    let generated_files = get_generated_files(&env, &build, target_dir.clone(), ops_id)?;

    if generated_files.is_empty() {
        eprintln!("No generated artifacts found to clean.");
        return Ok(());
    }

    if dry_run {
        println!("Files that would be deleted:");
        for file in &generated_files {
            if file.exists() {
                println!("  {}", file.to_string_lossy());
            }
        }
        return Ok(());
    }

    let existing_files: Vec<_> = generated_files.iter().filter(|f| f.exists()).collect();

    if existing_files.is_empty() {
        eprintln!("No generated artifacts found to clean.");
        return Ok(());
    }

    if !force {
        println!("Files to be deleted:");
        for file in &existing_files {
            println!("  {}", file.to_string_lossy());
        }
        print!("Continue? [y/N] ");
        std::io::stdout().flush()?;

        let mut response = String::new();
        std::io::stdin().read_line(&mut response)?;
        let response = response.trim().to_lowercase();

        if response != "y" && response != "yes" {
            eprintln!("Aborted.");
            return Ok(());
        }
    }

    let mut deleted_count = 0;
    let mut errors = Vec::new();

    for file in existing_files {
        match fs::remove_file(file) {
            Ok(_) => {
                deleted_count += 1;
                println!("Deleted: {}", file.to_string_lossy());
            }
            Err(e) => {
                errors.push(format!(
                    "Failed to delete {}: {}",
                    file.to_string_lossy(),
                    e
                ));
            }
        }
    }

    // Clean up empty directories
    let dirs_to_check = [
        target_dir.join("cmake"),
        target_dir
            .join("torch-ext")
            .join(build.general.python_name()),
        target_dir.join("torch-ext"),
    ];

    for dir in dirs_to_check {
        if dir.exists() && is_empty_dir(&dir)? {
            match fs::remove_dir(&dir) {
                Ok(_) => println!("Removed empty directory: {}", dir.to_string_lossy()),
                Err(e) => eyre::bail!("Failed to remove directory `{}`: {e:?}", dir.display()),
            }
        }
    }

    if !errors.is_empty() {
        for error in errors {
            eprintln!("Error: {error}");
        }
        bail!("Some files could not be deleted");
    }

    println!("Cleaned {deleted_count} generated artifacts.");
    Ok(())
}

fn get_generated_files(
    env: &Environment,
    build: &Build,
    target_dir: PathBuf,
    ops_id: Option<String>,
) -> Result<Vec<PathBuf>> {
    let mut all_set = FileSet::new();

    for backend in &build.general.backends {
        let set = if build.is_noarch() {
            write_torch_ext_noarch(env, *backend, build, target_dir.clone(), ops_id.clone())?
        } else {
            write_torch_ext(env, *backend, build, target_dir.clone(), ops_id.clone())?
        };
        all_set.extend(set);
    }

    Ok(all_set.into_names())
}

fn is_empty_dir(dir: &Path) -> Result<bool> {
    if !dir.is_dir() {
        return Ok(false);
    }

    let mut entries = fs::read_dir(dir)?;
    Ok(entries.next().is_none())
}
