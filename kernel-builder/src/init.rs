use std::{
    ffi::OsStr,
    fmt::{Display, Formatter},
    fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::atomic::{AtomicU32, Ordering},
};

use clap::Args;
use eyre::{bail, Context, Result};
use git2::{IndexAddOption, Repository};

use crate::config::Backend;
use crate::hf;

const DEFAULT_TEMPLATE_REPO: &str = "kernels-community/template";

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

    /// Hugging Face repo ID or local directory path for the template.
    #[arg(long, default_value = DEFAULT_TEMPLATE_REPO)]
    pub template: String,

    /// Backends to enable (`all`, `cpu`, `cuda`, `metal`, `neuron`, `rocm`, `xpu`).
    #[arg(long, num_args = 1.., default_values_t = default_init_backends())]
    pub backends: Vec<BackendSelection>,

    /// Overwrite the target directory if it already exists.
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
    //
    // Priority:
    //   1. Explicit --name always wins for repo identity.
    //   2. Positional arg as "owner/repo" sets both identity and directory.
    //   3. Positional arg as bare name (e.g. "my-kernel") + whoami → "owner/my-kernel".
    //   4. No positional arg + --name → derive directory from repo name.
    //   5. No positional arg, no --name → use cwd, try to infer identity.
    let (repo_info, target_dir) = match (args.path, args.name) {
        // --name provided: use it for identity, derive directory if needed.
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

        // No --name: infer identity from the positional arg.
        (Some(path), None) => {
            let path_str = path.to_string_lossy();

            // Try parsing the positional arg directly as "owner/repo".
            let info = if let Ok(info) = RepoInfo::from_str(&path_str) {
                info
            } else {
                // Bare name like "my-kernel" — look up the HF user as the owner.
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

        // Nothing provided — use cwd, try to infer identity.
        (None, None) => {
            let dir = cwd.clone();
            let dir_name = dir
                .file_name()
                .and_then(OsStr::to_str)
                .ok_or_else(|| eyre::eyre!("Cannot determine directory name"))?;

            let info = match RepoInfo::from_str(dir_name) {
                Ok(info) => info,
                Err(_) => bail!(
                    "Cannot infer kernel name from directory `{dir_name}`. \
                     Pass a name (e.g. `kernel-builder init my-kernel`) or use \
                     --name <owner/repo>."
                ),
            };
            (info, dir)
        }
    };

    if args.overwrite && target_dir.exists() {
        fs::remove_dir_all(&target_dir).wrap_err_with(|| {
            format!(
                "Cannot remove existing directory `{}`",
                target_dir.to_string_lossy()
            )
        })?;
    }

    if target_dir.exists() && !is_dir_empty(&target_dir)? {
        bail!(
            "Directory already exists and is not empty: {}",
            target_dir.to_string_lossy()
        );
    }

    let template_source = resolve_template_source(&args.template)?;
    copy_template(
        template_source.path(),
        &target_dir,
        &repo_info.name,
        &repo_info.normalized_name,
        &repo_info.class_name,
        &repo_info.repo_id,
    )?;

    let enabled_backends = resolve_backends(&args.backends);
    update_build_backends(&target_dir.join("build.toml"), &enabled_backends)?;
    update_torch_binding(
        &target_dir.join("torch-ext/torch_binding.cpp"),
        &enabled_backends,
    )?;
    remove_backend_dirs(&target_dir, &enabled_backends)?;
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

fn is_dir_empty(path: &Path) -> Result<bool> {
    if !path.is_dir() {
        return Ok(false);
    }

    let mut entries = fs::read_dir(path)?;
    Ok(entries.next().is_none())
}

enum TemplateSource {
    Local(PathBuf),
    Remote(TempDir),
}

impl TemplateSource {
    fn path(&self) -> &Path {
        match self {
            TemplateSource::Local(path) => path,
            TemplateSource::Remote(temp_dir) => temp_dir.path(),
        }
    }
}

fn resolve_template_source(template: &str) -> Result<TemplateSource> {
    let path = Path::new(template);
    if path.is_dir() {
        eprintln!("Using local template from {}...", path.display());
        return Ok(TemplateSource::Local(path.to_path_buf()));
    }

    eprintln!("Downloading template from {template}...");
    let tmp_dir = TempDir::new("kernel-builder-init-template")?;
    let repo_url = format!("https://huggingface.co/{template}");

    // Set up fetch options with HF token credentials when available.
    let mut fetch_opts = git2::FetchOptions::new();
    let mut callbacks = git2::RemoteCallbacks::new();

    let token = hf::token();
    callbacks.credentials(move |_url, _username, _allowed| match &token {
        Some(token) => git2::Cred::userpass_plaintext("hf_user", token),
        None => Err(git2::Error::from_str(
            "authentication required but no Hugging Face token found",
        )),
    });
    fetch_opts.remote_callbacks(callbacks);

    let mut builder = git2::build::RepoBuilder::new();
    builder.fetch_options(fetch_opts);

    builder.clone(&repo_url, tmp_dir.path()).map_err(|e| {
        if e.code() == git2::ErrorCode::Auth {
            eyre::eyre!(
                "Cannot clone template `{template}`: authentication failed.\n\
                 Run `hf auth login` or set the HF_TOKEN environment variable."
            )
        } else {
            eyre::eyre!("Cannot clone template `{template}` from `{repo_url}`: {e}")
        }
    })?;

    Ok(TemplateSource::Remote(tmp_dir))
}

fn copy_template(
    template_dir: &Path,
    target_dir: &Path,
    kernel_name: &str,
    normalized_name: &str,
    class_name: &str,
    repo_id: &str,
) -> Result<()> {
    let replacements = [
        ("__KERNEL_NAME__", kernel_name),
        ("__KERNEL_NAME_NORMALIZED__", normalized_name),
        ("__KERNEL_NAME_CLASS__", class_name),
        ("__REPO_ID__", repo_id),
    ];

    fs::create_dir_all(target_dir)
        .wrap_err_with(|| format!("Cannot create target directory `{}`", target_dir.display()))?;

    copy_template_dir(template_dir, template_dir, target_dir, &replacements)
}

fn copy_template_dir(
    root: &Path,
    source_dir: &Path,
    target_dir: &Path,
    replacements: &[(&str, &str)],
) -> Result<()> {
    for entry in fs::read_dir(source_dir)
        .wrap_err_with(|| format!("Cannot read template directory `{}`", source_dir.display()))?
    {
        let entry = entry?;
        let entry_path = entry.path();

        if entry.file_name() == OsStr::new(".git") {
            continue;
        }

        let relative_path = entry_path
            .strip_prefix(root)
            .wrap_err("Cannot determine template relative path")?;
        let destination = target_dir.join(apply_replacements(relative_path, replacements));

        if entry.file_type()?.is_dir() {
            fs::create_dir_all(&destination)
                .wrap_err_with(|| format!("Cannot create directory `{}`", destination.display()))?;
            copy_template_dir(root, &entry_path, target_dir, replacements)?;
            continue;
        }

        let contents = fs::read(&entry_path)
            .wrap_err_with(|| format!("Cannot read template file `{}`", entry_path.display()))?;

        match String::from_utf8(contents) {
            Ok(mut text) => {
                for (from, to) in replacements {
                    text = text.replace(from, to);
                }
                fs::write(&destination, text).wrap_err_with(|| {
                    format!("Cannot write scaffold file `{}`", destination.display())
                })?;
            }
            Err(err) => {
                // Not valid UTF-8, copy as binary
                fs::write(&destination, err.into_bytes()).wrap_err_with(|| {
                    format!("Cannot write binary file `{}`", destination.display())
                })?;
            }
        }
    }

    Ok(())
}

fn apply_replacements(path: &Path, replacements: &[(&str, &str)]) -> PathBuf {
    let mut rendered = path.to_string_lossy().into_owned();
    for (from, to) in replacements {
        rendered = rendered.replace(from, to);
    }
    PathBuf::from(rendered)
}

fn update_build_backends(build_toml_path: &Path, enabled_backends: &[Backend]) -> Result<()> {
    if !build_toml_path.exists() {
        return Ok(());
    }

    let build_toml = fs::read_to_string(build_toml_path)
        .wrap_err_with(|| format!("Cannot read `{}`", build_toml_path.display()))?;
    let mut document: toml::Value = toml::from_str(&build_toml)
        .wrap_err_with(|| format!("Cannot parse TOML in `{}`", build_toml_path.display()))?;

    // Update [general].backends
    if let Some(general) = document.get_mut("general").and_then(|v| v.as_table_mut()) {
        let mut backends_array = Vec::with_capacity(enabled_backends.len());
        for backend in enabled_backends {
            backends_array.push(toml::Value::String(backend.as_str().to_owned()));
        }
        general.insert("backends".to_owned(), toml::Value::Array(backends_array));
    }

    // Remove kernels for disabled backends
    if let Some(kernels) = document.get_mut("kernel").and_then(|v| v.as_table_mut()) {
        let mut names_to_remove = Vec::new();
        for (name, config) in kernels.iter() {
            let Some(table) = config.as_table() else {
                continue;
            };
            let Some(backend_val) = table.get("backend") else {
                continue;
            };
            let Some(backend_str) = backend_val.as_str() else {
                continue;
            };

            // Check if this backend is enabled (simple slice iteration, no allocation)
            let is_enabled = enabled_backends.iter().any(|b| b.as_str() == backend_str);
            if !is_enabled {
                names_to_remove.push(name.clone());
            }
        }
        for name in names_to_remove {
            kernels.remove(&name);
        }
    }

    let pretty_toml = toml::to_string_pretty(&document)
        .wrap_err_with(|| format!("Cannot serialize `{}`", build_toml_path.display()))?;
    fs::write(build_toml_path, pretty_toml)
        .wrap_err_with(|| format!("Cannot write `{}`", build_toml_path.display()))?;

    Ok(())
}

fn update_torch_binding(binding_path: &Path, enabled_backends: &[Backend]) -> Result<()> {
    if !binding_path.exists() {
        return Ok(());
    }

    let content = fs::read_to_string(binding_path)
        .wrap_err_with(|| format!("Cannot read `{}`", binding_path.display()))?;

    let has_cpu = enabled_backends.contains(&Backend::Cpu);
    let has_cuda = enabled_backends.contains(&Backend::Cuda);
    let has_rocm = enabled_backends.contains(&Backend::Rocm);
    let has_metal = enabled_backends.contains(&Backend::Metal);
    let has_xpu = enabled_backends.contains(&Backend::Xpu);

    // Build the new #if/#elif chain based on enabled backends
    let mut conditions: Vec<(&str, &str, &str)> = Vec::new();

    if has_cpu {
        conditions.push(("defined(CPU_KERNEL)", "torch::kCPU", "&"));
    }
    if has_cuda || has_rocm {
        conditions.push((
            "defined(CUDA_KERNEL) || defined(ROCM_KERNEL)",
            "torch::kCUDA",
            "&",
        ));
    }
    if has_metal {
        conditions.push(("defined(METAL_KERNEL)", "torch::kMPS", ""));
    }
    if has_xpu {
        conditions.push(("defined(XPU_KERNEL)", "torch::kXPU", "&"));
    }

    if conditions.is_empty() {
        return Ok(());
    }

    // Find function name: look for ops.def("funcname(...) -> ...") without regex
    let func_name = extract_func_name(&content).unwrap_or("__KERNEL_NAME_NORMALIZED__");

    // Build impl lines
    let mut impl_block = String::new();
    for (i, (condition, device, ref_prefix)) in conditions.iter().enumerate() {
        let directive = if i == 0 { "#if" } else { "#elif" };
        impl_block.push_str(directive);
        impl_block.push(' ');
        impl_block.push_str(condition);
        impl_block.push_str("\n  ops.impl(\"");
        impl_block.push_str(func_name);
        impl_block.push_str("\", ");
        impl_block.push_str(device);
        impl_block.push_str(", ");
        impl_block.push_str(ref_prefix);
        impl_block.push_str(func_name);
        impl_block.push_str(");\n");
    }
    impl_block.push_str("#endif");

    // Replace the #if block: find start and end without regex
    let new_content = replace_ifdef_block(&content, &impl_block);

    fs::write(binding_path, new_content)
        .wrap_err_with(|| format!("Cannot write `{}`", binding_path.display()))?;

    Ok(())
}

/// Extract function name from `ops.def("funcname(...)` without regex
fn extract_func_name(content: &str) -> Option<&str> {
    let marker = "ops.def(\"";
    let start = content.find(marker)? + marker.len();
    let rest = &content[start..];
    let end = rest.find('(')?;
    Some(&rest[..end])
}

/// Replace `#if defined(CPU_KERNEL)...#endif` block with new content
fn replace_ifdef_block(content: &str, replacement: &str) -> String {
    const START_MARKER: &str = "#if defined(CPU_KERNEL)";
    const END_MARKER: &str = "#endif";

    let Some(start) = content.find(START_MARKER) else {
        return content.to_owned();
    };

    let search_region = &content[start..];
    let Some(end_offset) = search_region.find(END_MARKER) else {
        return content.to_owned();
    };

    let end = start + end_offset + END_MARKER.len();

    let mut result = String::with_capacity(content.len());
    result.push_str(&content[..start]);
    result.push_str(replacement);
    result.push_str(&content[end..]);
    result
}

fn remove_backend_dirs(target_dir: &Path, enabled_backends: &[Backend]) -> Result<()> {
    // Check if a backend should be kept (simple slice search, max 6 items)
    let should_keep = |backend: &str| -> bool {
        enabled_backends.iter().any(|b| b.as_str() == backend)
            || (backend == "cuda" && enabled_backends.iter().any(|b| b.as_str() == "rocm"))
    };

    // Pre-compute suffixes to remove
    let mut suffixes_to_remove = Vec::with_capacity(6);
    for backend in Backend::all() {
        if !should_keep(backend.as_str()) {
            suffixes_to_remove.push(format!("_{}", backend.as_str()));
        }
    }

    for entry in fs::read_dir(target_dir)
        .wrap_err_with(|| format!("Cannot read target directory `{}`", target_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };

        for suffix in &suffixes_to_remove {
            if file_name.ends_with(suffix) {
                fs::remove_dir_all(&path)
                    .wrap_err_with(|| format!("Cannot remove `{}`", path.display()))?;
                break;
            }
        }
    }

    Ok(())
}

fn initialize_git_repo(target_dir: &Path) -> Result<()> {
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

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Result<Self> {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let pid = std::process::id();
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("{prefix}-{pid}-{count}");
        let path = std::env::temp_dir().join(name);

        fs::create_dir(&path)
            .wrap_err_with(|| format!("Cannot create temp dir `{}`", path.display()))?;

        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
