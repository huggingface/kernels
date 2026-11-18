use super::{Facts, check_include_dir, comma_list, glob_list, source_files, toml_str_list};
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Context, Result, bail};

// build.toml is generated, never overlaid. A field this op cannot emit is a
// reason to extend the op.
#[derive(Debug)]
pub struct Manifest {
    name: String,
    version: Option<String>,
    license: Option<String>,
    edition: Option<String>,
    upstream: Option<String>,
    source: Option<String>,
    backends: Vec<String>,
    repo_id: Option<String>,
    hub_branch: Option<String>,
    python_depends: Vec<String>,
    kernel_depends: Vec<KernelDepend>,
    cuda_minver: Option<String>,
    cuda_maxver: Option<String>,
    cuda_python_depends: Vec<String>,
    kind: ManifestKind,
}

#[derive(Debug)]
enum ManifestKind {
    Noarch {
        pyext: Vec<String>,
        rocm_archs: Vec<String>,
    },
    Torch {
        torch_src: Vec<Pattern>,
        pyext: Vec<String>,
        include: Vec<String>,
        follow_includes: bool,
        stable_abi_version: Option<String>,
        stable_abi: Vec<(String, String)>,
    },
}

impl Manifest {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let backends = comma_list(&args.take("backends")?);
        if backends.is_empty() {
            bail!("backends must list at least one backend");
        }
        let version = args.take_opt("version");
        if let Some(v) = &version {
            v.parse::<u64>()
                .with_context(|| format!("version must be an integer, got {v:?}"))?;
        }
        let edition = args.take_opt("edition");
        if let Some(e) = &edition {
            e.parse::<u64>()
                .with_context(|| format!("edition must be an integer, got {e:?}"))?;
        }
        let noarch = match args.take_opt("noarch").as_deref() {
            None => false,
            Some("true") => true,
            Some(other) => bail!("noarch must be true if given, got {other:?}"),
        };
        let kind = if noarch {
            if args.take_opt("torch_src").is_some() {
                bail!("torch_src does not apply when noarch=true");
            }
            if args.take_opt("stable_abi").is_some() {
                bail!("stable_abi does not apply when noarch=true");
            }
            if args.take_opt("stable_abi_version").is_some() {
                bail!("stable_abi_version does not apply when noarch=true");
            }
            if args.take_opt("torch_include").is_some() {
                bail!("torch_include does not apply when noarch=true");
            }
            if args.take_opt("torch_follow_includes").is_some() {
                bail!("torch_follow_includes does not apply when noarch=true");
            }
            ManifestKind::Noarch {
                pyext: comma_list(&args.take_opt("noarch_pyext").unwrap_or_default()),
                rocm_archs: comma_list(&args.take_opt("rocm_archs").unwrap_or_default()),
            }
        } else {
            if args.take_opt("noarch_pyext").is_some() {
                bail!("noarch_pyext requires noarch=true");
            }
            if args.take_opt("rocm_archs").is_some() {
                bail!(
                    "rocm_archs applies to noarch=true; for a torch extension, set rocm_archs on its `kernel` statements"
                );
            }
            let stable_abi_version = args.take_opt("stable_abi_version");
            let pyext = comma_list(&args.take_opt("torch_pyext").unwrap_or_default());
            let stable_abi = comma_list(&args.take_opt("stable_abi").unwrap_or_default())
                .iter()
                .map(|pair| {
                    pair.split_once('=')
                        .map(|(b, v)| (b.to_string(), v.to_string()))
                        .with_context(|| {
                            format!("stable_abi entry {pair:?} must be backend=version")
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            if stable_abi_version.is_some() && !stable_abi.is_empty() {
                bail!("stable_abi_version and stable_abi are mutually exclusive");
            }
            ManifestKind::Torch {
                torch_src: glob_list(&args.take("torch_src")?, "torch_src")?,
                pyext,
                include: comma_list(&args.take_opt("torch_include").unwrap_or_default()),
                follow_includes: args.take_bool("torch_follow_includes")?,
                stable_abi_version,
                stable_abi,
            }
        };
        Ok(Self {
            name: args.take("name")?,
            version,
            license: args.take_opt("license"),
            edition,
            upstream: args.take_opt("upstream"),
            source: args.take_opt("source"),
            backends,
            repo_id: args.take_opt("repo_id"),
            hub_branch: args.take_opt("hub_branch"),
            python_depends: comma_list(&args.take_opt("python_depends").unwrap_or_default()),
            kernel_depends: comma_list(&args.take_opt("kernel_depends").unwrap_or_default())
                .iter()
                .map(|entry| KernelDepend::parse(entry))
                .collect::<Result<Vec<_>>>()?,
            cuda_minver: args.take_opt("cuda_minver"),
            cuda_maxver: args.take_opt("cuda_maxver"),
            cuda_python_depends: comma_list(
                &args.take_opt("cuda_python_depends").unwrap_or_default(),
            ),
            kind,
        })
    }

    fn toml_list(key: &str, items: &[String]) -> String {
        if items.len() == 1 {
            format!("{key} = [{}]\n", toml_str_list(items))
        } else {
            let mut block = format!("{key} = [\n");
            for item in items {
                block.push_str(&format!("    {item:?},\n"));
            }
            block.push_str("]\n");
            block
        }
    }

    fn src_block(files: &[String]) -> String {
        let mut block = String::from("src = [\n");
        for f in files {
            block.push_str(&format!("    {f:?},\n"));
        }
        block.push(']');
        block
    }

    fn general_section(&self) -> String {
        let mut toml = String::from("[general]\n");
        toml.push_str(&format!("name = {:?}\n", self.name));
        if let Some(version) = &self.version {
            toml.push_str(&format!("version = {version}\n"));
        }
        // Same key order as the kernels-common config schema.
        if let Some(edition) = &self.edition {
            toml.push_str(&format!("edition = {edition}\n"));
        }
        if let Some(license) = &self.license {
            toml.push_str(&format!("license = {license:?}\n"));
        }
        if let Some(upstream) = &self.upstream {
            toml.push_str(&format!("upstream = {upstream:?}\n"));
        }
        if let Some(source) = &self.source {
            toml.push_str(&format!("source = {source:?}\n"));
        }
        toml.push_str(&Self::toml_list("backends", &self.backends));
        if !self.python_depends.is_empty() {
            toml.push_str(&Self::toml_list("python-depends", &self.python_depends));
        }
        if !self.kernel_depends.is_empty() {
            toml.push_str("kernel-depends = [\n");
            for dep in &self.kernel_depends {
                toml.push_str(&format!("    {},\n", dep.toml()));
            }
            toml.push_str("]\n");
        }
        if self.cuda_minver.is_some()
            || self.cuda_maxver.is_some()
            || !self.cuda_python_depends.is_empty()
        {
            toml.push_str("\n[general.cuda]\n");
            if let Some(minver) = &self.cuda_minver {
                toml.push_str(&format!("minver = {minver:?}\n"));
            }
            if let Some(maxver) = &self.cuda_maxver {
                toml.push_str(&format!("maxver = {maxver:?}\n"));
            }
            if !self.cuda_python_depends.is_empty() {
                toml.push_str(&Self::toml_list(
                    "python-depends",
                    &self.cuda_python_depends,
                ));
            }
        }
        if self.repo_id.is_some() || self.hub_branch.is_some() {
            toml.push_str("\n[general.hub]\n");
            if let Some(repo_id) = &self.repo_id {
                toml.push_str(&format!("repo-id = {repo_id:?}\n"));
            }
            if let Some(branch) = &self.hub_branch {
                toml.push_str(&format!("branch = {branch:?}\n"));
            }
        }
        toml
    }

    pub(super) fn apply(&self, ws: &mut Workspace, facts: &Facts) -> Result<String> {
        let mut toml = self.general_section();

        let summary = match &self.kind {
            ManifestKind::Noarch { pyext, rocm_archs } => {
                if !facts.kernels.is_empty() {
                    bail!("noarch=true but kernel sections were declared");
                }
                toml.push_str("\n[torch-noarch]\n");
                if !pyext.is_empty() {
                    toml.push_str(&format!("pyext = [{}]\n", toml_str_list(pyext)));
                }
                if !rocm_archs.is_empty() {
                    toml.push_str(&format!("rocm_archs = [{}]\n", toml_str_list(rocm_archs)));
                }
                toml.push_str("\n[kernel]\n");
                "wrote build.toml (noarch)".to_string()
            }
            ManifestKind::Torch {
                torch_src,
                pyext,
                include,
                follow_includes,
                stable_abi_version,
                stable_abi,
            } => {
                let torch_files =
                    source_files(ws, torch_src, "torch_src", include, *follow_includes)?;
                if facts.kernels.is_empty() {
                    bail!("no kernel sections declared; add `kernel` statements before manifest");
                }
                toml.push_str("\n[torch]\n");
                if let Some(version) = stable_abi_version {
                    toml.push_str(&format!("stable-abi = {version:?}\n"));
                }
                if !pyext.is_empty() {
                    toml.push_str(&Self::toml_list("pyext", pyext));
                }
                for dir in include {
                    check_include_dir(ws, dir)?;
                }
                if !include.is_empty() {
                    toml.push_str(&format!("include = [{}]\n", toml_str_list(include)));
                }
                toml.push_str(&Self::src_block(&torch_files));
                toml.push('\n');
                if !stable_abi.is_empty() {
                    toml.push_str("\n[torch.stable-abi]\n");
                    for (backend, version) in stable_abi {
                        toml.push_str(&format!("{backend} = {version:?}\n"));
                    }
                }
                for k in &facts.kernels {
                    toml.push_str(&format!("\n[kernel.{}]\n", k.name));
                    // TODO: update when supporting non-C++ backends
                    // Edition 6 tags kernels with their language and backend.
                    if self.edition.as_deref().and_then(|e| e.parse::<u32>().ok()) >= Some(6) {
                        toml.push_str(&format!("language = \"cpp-{}\"\n", k.backend));
                    } else {
                        toml.push_str(&format!("backend = {:?}\n", k.backend));
                    }
                    if !k.cxx_flags.is_empty() {
                        toml.push_str("cxx-flags = [\n");
                        for f in &k.cxx_flags {
                            toml.push_str(&format!("    {f:?},\n"));
                        }
                        toml.push_str("]\n");
                    }
                    if !k.capabilities.is_empty() {
                        toml.push_str(&format!(
                            "cuda-capabilities = [{}]\n",
                            toml_str_list(&k.capabilities)
                        ));
                    }
                    if !k.cuda_flags.is_empty() {
                        toml.push_str("cuda-flags = [\n");
                        for f in &k.cuda_flags {
                            toml.push_str(&format!("    {f:?},\n"));
                        }
                        toml.push_str("]\n");
                    }
                    if let Some(minver) = &k.cuda_minver {
                        toml.push_str(&format!("cuda-minver = {minver:?}\n"));
                    }
                    toml.push_str(&format!("depends = [{}]\n", toml_str_list(&k.depends)));
                    if !k.rocm_archs.is_empty() {
                        toml.push_str(&Self::toml_list("rocm-archs", &k.rocm_archs));
                    }
                    if !k.include.is_empty() {
                        toml.push_str(&format!("include = [{}]\n", toml_str_list(&k.include)));
                    }
                    toml.push_str(&Self::src_block(&k.src));
                    toml.push('\n');
                }
                format!(
                    "wrote build.toml ({} torch src, {} kernel section(s))",
                    torch_files.len(),
                    facts.kernels.len()
                )
            }
        };

        ws.set_text("build.toml", toml);
        Ok(summary)
    }
}

/// A kernel dependency.
#[derive(Debug)]
struct KernelDepend {
    repo_id: String,
    pin: KernelPin,
}

#[derive(Debug)]
enum KernelPin {
    Version(usize),
    Revision(String),
}

impl KernelDepend {
    fn parse(entry: &str) -> Result<Self> {
        // Parse a kernel dependency entry.
        //
        // The possible formats are: `repo_id=version` and `repo_id@revision`.
        let (repo_id, pin) = if let Some((repo_id, revision)) = entry.split_once('@') {
            if revision.is_empty() {
                bail!("kernel_depends entry {entry:?} has an empty revision");
            }
            (repo_id, KernelPin::Revision(revision.to_string()))
        } else if let Some((repo_id, version)) = entry.split_once('=') {
            let version = version.parse().with_context(|| {
                format!("kernel_depends entry {entry:?}: version must be an integer")
            })?;
            (repo_id, KernelPin::Version(version))
        } else {
            bail!("kernel_depends entry {entry:?} must be org/name=VERSION or org/name@REVISION");
        };
        if repo_id.split('/').count() != 2 || repo_id.split('/').any(str::is_empty) {
            bail!("kernel_depends entry {entry:?}: repository must be org/name");
        }
        Ok(Self {
            repo_id: repo_id.to_string(),
            pin,
        })
    }

    fn toml(&self) -> String {
        match &self.pin {
            KernelPin::Version(version) => {
                format!("{{ repo-id = {:?}, version = {version} }}", self.repo_id)
            }
            KernelPin::Revision(revision) => {
                format!(
                    "{{ repo-id = {:?}, revision = {revision:?} }}",
                    self.repo_id
                )
            }
        }
    }
}
