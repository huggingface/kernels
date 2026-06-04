use std::path::PathBuf;

use eyre::Result;
use itertools::Itertools;

use kernels_data::config::{Backend, General, TorchNoarch};
use kernels_data::metadata::{BackendInfo, Metadata};

use crate::pyproject::ops_identifier::KernelIdentifier;
use crate::pyproject::FileSet;

static COMPAT_PY: &str = include_str!("templates/compat.py");
static ADD_BUILD_METADATA_PY: &str = include_str!("templates/torch/add_build_metadata.py");

pub fn write_compat_py(file_set: &mut FileSet) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("compat.py");
    file_set.entry(path).extend_from_slice(COMPAT_PY.as_bytes());

    Ok(())
}

pub fn write_metadata(
    general: &General,
    kernel_id: &KernelIdentifier,
    file_set: &mut FileSet,
) -> Result<()> {
    write_metadata_impl(general, None, kernel_id, file_set)
}

/// Like [`write_metadata`], but also exports the GPU architectures that a
/// noarch kernel declares in its `[torch-noarch]` section. Arch kernels
/// detect their architectures at build time instead, so they use
/// [`write_metadata`].
pub fn write_noarch_metadata(
    general: &General,
    noarch: &TorchNoarch,
    kernel_id: &KernelIdentifier,
    file_set: &mut FileSet,
) -> Result<()> {
    write_metadata_impl(general, Some(noarch), kernel_id, file_set)
}

fn write_metadata_impl(
    general: &General,
    noarch: Option<&TorchNoarch>,
    kernel_id: &KernelIdentifier,
    file_set: &mut FileSet,
) -> Result<()> {
    for backend in &Backend::all() {
        let writer = file_set.entry(format!("metadata-{backend}.json"));

        let python_depends = general
            .python_depends()
            .map(|deps| Ok(deps?.0.to_owned()))
            .chain(
                general
                    .backend_python_depends(*backend)
                    .map(|deps| Ok(deps?.0.to_owned())),
            )
            .collect::<Result<Vec<_>>>()?;

        let metadata = Metadata {
            id: kernel_id.to_string_for_backend(*backend),
            name: general.name.clone(),
            version: general.version,
            license: general.license.clone(),
            upstream: general.upstream.clone(),
            python_depends,
            backend: BackendInfo {
                archs: noarch
                    .and_then(|noarch| noarch.backend_archs(*backend))
                    .cloned(),
                backend_type: *backend,
            },
        };

        serde_json::to_writer_pretty(writer, &metadata)?;
    }

    Ok(())
}

pub fn prefix_and_join_includes<S>(includes: impl AsRef<[S]>) -> String
where
    S: AsRef<str>,
{
    includes
        .as_ref()
        .iter()
        .map(|include| format!("${{CMAKE_SOURCE_DIR}}/{}", include.as_ref()))
        .collect_vec()
        .join(";")
}

pub fn write_add_build_metadata_py(file_set: &mut FileSet) {
    write_cmake_file(
        file_set,
        "add_build_metadata.py",
        ADD_BUILD_METADATA_PY.as_bytes(),
    );
}

/// Helper function to write a file to the cmake subdirectory
pub fn write_cmake_file(file_set: &mut FileSet, filename: &str, content: &[u8]) {
    let mut path = PathBuf::new();
    path.push("cmake");
    path.push(filename);
    file_set.entry(path).extend_from_slice(content);
}

#[cfg(test)]
mod tests {
    use kernels_data::config::{Backend, General, KernelName, TorchNoarch};
    use kernels_data::metadata::Metadata;

    use super::write_noarch_metadata;
    use crate::pyproject::fileset::FileSet;
    use crate::pyproject::ops_identifier::KernelIdentifier;

    fn general() -> General {
        General {
            name: KernelName::new("my-kernel").unwrap(),
            version: 1,
            license: "Apache-2.0".to_string(),
            upstream: None,
            backends: vec![Backend::Cuda, Backend::Rocm, Backend::Cpu],
            hub: None,
            python_depends: None,
            cuda: None,
            neuron: None,
            xpu: None,
        }
    }

    fn metadata_for(noarch: &TorchNoarch, backend: &str) -> Metadata {
        let general = general();
        let kernel_id =
            KernelIdentifier::new(".", general.name.to_string(), Some("abc1234".into()));

        let mut file_set = FileSet::default();
        write_noarch_metadata(&general, noarch, &kernel_id, &mut file_set).unwrap();

        let dir = tempfile::tempdir().unwrap();
        file_set.write(dir.path(), false).unwrap();
        Metadata::from_reader(
            std::fs::File::open(dir.path().join(format!("metadata-{backend}.json"))).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn noarch_capabilities_are_exported_to_metadata() {
        let noarch = TorchNoarch {
            cuda_capabilities: Some(vec!["9.0".to_string(), "10.0".to_string()]),
            rocm_archs: Some(vec!["gfx942".to_string()]),
        };

        assert_eq!(
            metadata_for(&noarch, "cuda").backend.archs.as_deref(),
            Some(["9.0".to_string(), "10.0".to_string()].as_slice()),
        );
        assert_eq!(
            metadata_for(&noarch, "rocm").backend.archs.as_deref(),
            Some(["gfx942".to_string()].as_slice()),
        );
        // Backends without declared archs (and CPU, which has no arch concept)
        // leave `archs` unset.
        assert_eq!(metadata_for(&noarch, "cpu").backend.archs, None);
    }

    #[test]
    fn metadata_archs_unset_when_not_declared() {
        let noarch = TorchNoarch {
            cuda_capabilities: None,
            rocm_archs: None,
        };

        assert_eq!(metadata_for(&noarch, "cuda").backend.archs, None);
    }
}
