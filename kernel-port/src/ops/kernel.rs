use super::{Facts, KernelSection, check_include_dir, comma_list, glob_list, glob_union};
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct Kernel {
    name: String,
    backend: String,
    src: Vec<Pattern>,
    include: Vec<String>,
    depends: Vec<String>,
    capabilities: Vec<String>,
    cxx_flags: Vec<String>,
    cuda_flags: Vec<String>,
    cuda_minver: Option<String>,
    rocm_archs: Vec<String>,
    repeat_src: Vec<String>,
}

impl Kernel {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            name: args.take("name")?,
            backend: args.take("backend")?,
            src: glob_list(&args.take("src")?, "src")?,
            include: comma_list(&args.take_opt("include").unwrap_or_default()),
            depends: comma_list(&args.take_opt("depends").unwrap_or_else(|| "torch".into())),
            capabilities: comma_list(&args.take_opt("capabilities").unwrap_or_default()),
            cxx_flags: comma_list(&args.take_opt("cxx_flags").unwrap_or_default()),
            cuda_flags: comma_list(&args.take_opt("cuda_flags").unwrap_or_default()),
            cuda_minver: args.take_opt("cuda_minver"),
            rocm_archs: comma_list(&args.take_opt("rocm_archs").unwrap_or_default()),
            repeat_src: comma_list(&args.take_opt("repeat_src").unwrap_or_default()),
        })
    }

    pub(super) fn apply(&self, ws: &Workspace, facts: &mut Facts) -> Result<String> {
        if facts.kernels.iter().any(|k| k.name == self.name) {
            bail!("kernel section {:?} already declared", self.name);
        }
        let mut src = glob_union(ws, &self.src, "src")?;
        for path in &self.repeat_src {
            if !src.contains(path) {
                bail!("repeat_src path {path:?} is not selected by src globs");
            }
            src.push(path.clone());
        }
        src.sort();
        for dir in &self.include {
            check_include_dir(ws, dir)?;
        }
        let n = src.len();
        facts.kernels.push(KernelSection {
            name: self.name.clone(),
            backend: self.backend.clone(),
            cxx_flags: self.cxx_flags.clone(),
            cuda_flags: self.cuda_flags.clone(),
            cuda_minver: self.cuda_minver.clone(),
            rocm_archs: self.rocm_archs.clone(),
            depends: self.depends.clone(),
            include: self.include.clone(),
            capabilities: self.capabilities.clone(),
            src,
        });
        Ok(format!(
            "declared [kernel.{}] with {n} src file(s)",
            self.name
        ))
    }
}
