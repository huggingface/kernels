use std::{
    ffi::OsStr,
    fmt::{Display, Formatter},
    path::PathBuf,
    str::FromStr,
};

use clap::Args;
use eyre::{Context, Result};
use git2::{IndexAddOption, Repository};
use minijinja::{context, Environment};

use crate::config::Backend;
use crate::hf;
use crate::pyproject::FileSet;

fn to_camel_case(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '-' || c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c.to_ascii_lowercase());
        }
    }
    result
}

#[derive(Debug, Args)]
pub struct InitArgs {
    /// Directory to initialize (defaults to current directory).
    #[arg(value_name = "PATH")]
    pub path: Option<PathBuf>,

    /// Name of the kernel repo (e.g. `drbh/my-kernel`).
    #[arg(long, value_name = "OWNER/REPO")]
    pub name: Option<RepoInfo>,

    /// Backends to enable (`all`, `cpu`, `cuda`, `metal`, `neuron`, `rocm`, `xpu`).
    #[arg(long, num_args = 1.., default_values_t = default_init_backends())]
    pub backends: Vec<BackendSelection>,

    /// Overwrite existing scaffold files (preserves other files).
    #[arg(long)]
    pub overwrite: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendSelection {
    All,
    Backend(Backend),
}

impl Display for BackendSelection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendSelection::All => write!(f, "all"),
            BackendSelection::Backend(backend) => backend.fmt(f),
        }
    }
}

impl FromStr for BackendSelection {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("all") {
            Ok(Self::All)
        } else {
            Backend::from_str(value).map(Self::Backend)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepoInfo {
    pub name: String,
    pub normalized_name: String,
    pub class_name: String,
    pub owner: String,
    pub repo_id: String,
}

impl FromStr for RepoInfo {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let (owner, name) = value.split_once('/').ok_or("must be <owner>/<repo>")?;

        if owner.is_empty() || name.is_empty() || name.contains('/') {
            return Err("must be <owner>/<repo>".to_owned());
        }

        let display_name = name.replace('_', "-");
        let normalized_name = name.to_lowercase().replace('-', "_");
        let class_name = to_camel_case(name);
        let repo_id = format!("{owner}/{display_name}");

        Ok(Self {
            name: display_name,
            normalized_name,
            class_name,
            owner: owner.to_owned(),
            repo_id,
        })
    }
}

pub fn run_init(args: InitArgs) -> Result<()> {
    let cwd = std::env::current_dir().context("Cannot determine current directory")?;

    // Resolve repo info and target directory together, since they can inform each other.
    let (repo_info, target_dir) = match (args.path, args.name) {
        (Some(path), Some(info)) => {
            let dir = if path.is_absolute() {
                path
            } else {
                cwd.join(path)
            };
            (info, dir)
        }
        (None, Some(info)) => {
            let dir = cwd.join(&info.name);
            (info, dir)
        }
        (Some(path), None) => {
            let path_str = path.to_string_lossy();

            let info = if let Ok(info) = RepoInfo::from_str(&path_str) {
                info
            } else {
                let bare_name = path.file_name().and_then(OsStr::to_str).ok_or_else(|| {
                    eyre::eyre!("Cannot determine directory name from `{path_str}`")
                })?;
                let owner = hf::whoami_username()?;
                RepoInfo::from_str(&format!("{owner}/{bare_name}"))
                    .map_err(|e| eyre::eyre!("{e}"))?
            };

            let dir = cwd.join(&info.name);
            (info, dir)
        }
        (None, None) => {
            let dir = cwd.clone();
            let dir_name = dir
                .file_name()
                .and_then(OsStr::to_str)
                .ok_or_else(|| eyre::eyre!("Cannot determine directory name"))?;

            let info = match RepoInfo::from_str(dir_name) {
                Ok(info) => info,
                Err(_) => {
                    let owner = hf::whoami_username()?;
                    RepoInfo::from_str(&format!("{owner}/{dir_name}"))
                        .map_err(|e| eyre::eyre!("{e}"))?
                }
            };
            (info, dir)
        }
    };

    let enabled_backends = resolve_backends(&args.backends);

    // Set up minijinja environment
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    load_init_templates(&mut env);

    // Build FileSet in memory (atomic preparation)
    let file_set = build_init_fileset(&env, &repo_info, &enabled_backends)?;

    // Atomic write - validates first, then writes all files
    file_set.write(&target_dir, args.overwrite)?;

    initialize_git_repo(&target_dir)?;

    println!(
        "Initialized `{}` at {}",
        repo_info.repo_id,
        target_dir.display()
    );

    Ok(())
}

fn default_init_backends() -> Vec<BackendSelection> {
    if cfg!(target_os = "macos") {
        vec![BackendSelection::Backend(Backend::Metal)]
    } else {
        vec![BackendSelection::Backend(Backend::Cuda)]
    }
}

fn resolve_backends(backends: &[BackendSelection]) -> Vec<Backend> {
    if backends
        .iter()
        .any(|backend| matches!(backend, BackendSelection::All))
    {
        return Backend::all().into_iter().collect();
    }

    let mut unique = Vec::new();
    for backend in backends {
        let BackendSelection::Backend(backend) = backend else {
            continue;
        };
        if !unique.contains(backend) {
            unique.push(*backend);
        }
    }

    unique
}

/// Load all init templates into the environment
fn load_init_templates(env: &mut Environment) {
    // Static files (no templating needed)
    env.add_template_owned(
        "static/.gitignore",
        include_str!("init/templates/.gitignore").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "static/.gitattributes",
        include_str!("init/templates/.gitattributes").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "static/flake.nix",
        include_str!("init/templates/flake.nix").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "static/tests/__init__.py",
        include_str!("init/templates/tests/__init__.py").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "static/CARD.md",
        include_str!("init/templates/CARD.md").to_owned(),
    )
    .unwrap();

    // Templated files
    env.add_template_owned(
        "build.toml",
        include_str!("init/templates/build.toml").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "example.py",
        include_str!("init/templates/example.py").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "torch-ext/__init__.py",
        include_str!("init/templates/torch-ext/__init__.py").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "torch-ext/torch_binding.cpp",
        include_str!("init/templates/torch-ext/torch_binding.cpp").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "torch-ext/torch_binding.h",
        include_str!("init/templates/torch-ext/torch_binding.h").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "tests/test_kernel.py",
        include_str!("init/templates/tests/test_kernel.py").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "benchmarks/benchmark.py",
        include_str!("init/templates/benchmarks/benchmark.py").to_owned(),
    )
    .unwrap();

    // Backend-specific kernel sources
    env.add_template_owned(
        "kernel_cpu/kernel_cpu.cpp",
        include_str!("init/templates/kernel_cpu/kernel_cpu.cpp").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "kernel_cuda/kernel.cu",
        include_str!("init/templates/kernel_cuda/kernel.cu").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "kernel_metal/kernel.metal",
        include_str!("init/templates/kernel_metal/kernel.metal").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "kernel_metal/kernel.mm",
        include_str!("init/templates/kernel_metal/kernel.mm").to_owned(),
    )
    .unwrap();
    env.add_template_owned(
        "kernel_xpu/kernel.cpp",
        include_str!("init/templates/kernel_xpu/kernel.cpp").to_owned(),
    )
    .unwrap();
}

/// Build a FileSet with all init templates rendered in memory
fn build_init_fileset(
    env: &Environment,
    repo_info: &RepoInfo,
    enabled_backends: &[Backend],
) -> Result<FileSet> {
    let has_cpu = enabled_backends.contains(&Backend::Cpu);
    let has_cuda = enabled_backends.contains(&Backend::Cuda);
    let has_rocm = enabled_backends.contains(&Backend::Rocm);
    let has_metal = enabled_backends.contains(&Backend::Metal);
    let has_xpu = enabled_backends.contains(&Backend::Xpu);

    let backend_strings: Vec<String> = enabled_backends
        .iter()
        .map(|b| b.as_str().to_owned())
        .collect();

    let ctx = context! {
        kernel_name => &repo_info.name,
        kernel_name_normalized => &repo_info.normalized_name,
        kernel_name_class => &repo_info.class_name,
        repo_id => &repo_info.repo_id,
        backends => &backend_strings,
        has_cpu => has_cpu,
        has_cuda => has_cuda,
        has_rocm => has_rocm,
        has_metal => has_metal,
        has_xpu => has_xpu,
    };

    let mut file_set = FileSet::new();

    // Static files (no templating)
    render_template(env, &mut file_set, "static/.gitignore", ".gitignore", ())?;
    render_template(
        env,
        &mut file_set,
        "static/.gitattributes",
        ".gitattributes",
        (),
    )?;
    render_template(env, &mut file_set, "static/flake.nix", "flake.nix", ())?;
    render_template(
        env,
        &mut file_set,
        "static/tests/__init__.py",
        "tests/__init__.py",
        (),
    )?;
    render_template(env, &mut file_set, "static/CARD.md", "CARD.md", ())?;

    // Templated files
    render_template(env, &mut file_set, "build.toml", "build.toml", &ctx)?;
    render_template(env, &mut file_set, "example.py", "example.py", &ctx)?;
    render_template(
        env,
        &mut file_set,
        "benchmarks/benchmark.py",
        "benchmarks/benchmark.py",
        &ctx,
    )?;

    // torch-ext files
    let torch_ext_dir = format!("torch-ext/{}", repo_info.normalized_name);
    render_template(
        env,
        &mut file_set,
        "torch-ext/__init__.py",
        &format!("{torch_ext_dir}/__init__.py"),
        &ctx,
    )?;
    render_template(
        env,
        &mut file_set,
        "torch-ext/torch_binding.cpp",
        "torch-ext/torch_binding.cpp",
        &ctx,
    )?;
    render_template(
        env,
        &mut file_set,
        "torch-ext/torch_binding.h",
        "torch-ext/torch_binding.h",
        &ctx,
    )?;

    // Test file
    let test_file = format!("tests/test_{}.py", repo_info.normalized_name);
    render_template(env, &mut file_set, "tests/test_kernel.py", &test_file, &ctx)?;

    // Backend-specific kernel sources
    if has_cpu {
        let cpu_file = format!(
            "{}_cpu/{}_cpu.cpp",
            repo_info.normalized_name, repo_info.normalized_name
        );
        render_template(
            env,
            &mut file_set,
            "kernel_cpu/kernel_cpu.cpp",
            &cpu_file,
            &ctx,
        )?;
    }

    if has_cuda || has_rocm {
        let cuda_file = format!(
            "{}_cuda/{}.cu",
            repo_info.normalized_name, repo_info.normalized_name
        );
        render_template(
            env,
            &mut file_set,
            "kernel_cuda/kernel.cu",
            &cuda_file,
            &ctx,
        )?;
    }

    if has_metal {
        let metal_dir = format!("{}_metal", repo_info.normalized_name);
        render_template(
            env,
            &mut file_set,
            "kernel_metal/kernel.metal",
            &format!("{metal_dir}/{}.metal", repo_info.normalized_name),
            &ctx,
        )?;
        render_template(
            env,
            &mut file_set,
            "kernel_metal/kernel.mm",
            &format!("{metal_dir}/{}.mm", repo_info.normalized_name),
            &ctx,
        )?;
    }

    if has_xpu {
        let xpu_file = format!(
            "{}_xpu/{}.cpp",
            repo_info.normalized_name, repo_info.normalized_name
        );
        render_template(env, &mut file_set, "kernel_xpu/kernel.cpp", &xpu_file, &ctx)?;
    }

    Ok(file_set)
}

/// Render a template into the FileSet
fn render_template(
    env: &Environment,
    file_set: &mut FileSet,
    template_name: &str,
    output_path: &str,
    ctx: impl serde::Serialize,
) -> Result<()> {
    let template = env
        .get_template(template_name)
        .wrap_err_with(|| format!("Cannot get template `{template_name}`"))?;

    template
        .render_captured_to(ctx, file_set.entry(output_path))
        .wrap_err_with(|| format!("Cannot render template `{template_name}`"))?;

    Ok(())
}

fn initialize_git_repo(target_dir: &std::path::Path) -> Result<()> {
    let repo = Repository::init(target_dir).wrap_err_with(|| {
        format!(
            "Cannot initialize git repository in `{}`",
            target_dir.display()
        )
    })?;
    let mut index = repo.index().context("Cannot access git index")?;
    index
        .add_all(["."], IndexAddOption::DEFAULT, None)
        .context("Cannot stage scaffolded files")?;
    index.write().context("Cannot write git index")?;
    Ok(())
}
