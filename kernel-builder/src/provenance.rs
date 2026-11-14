use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use eyre::{Context, Result};
use kernels_common::git::GitStatus;
use kernels_common::metadata::{KernelBuilderVersion, Metadata, Provenance};
use serde::Deserialize;

use crate::util::{check_or_infer_kernel_dir, discover_variants};

/// The git state of the sources that a build was made from.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GitState {
    /// The kernel (or SDK) source.
    #[serde(deserialize_with = "Option::deserialize")]
    pub kernel: Option<GitStatus>,

    /// The `kernel-builder` source.
    #[serde(deserialize_with = "Option::deserialize")]
    pub builder: Option<GitStatus>,
}

impl GitState {
    /// Read the git state from a JSON file, e.g.:
    ///
    /// ```json
    /// {"kernel": {"commit": "d0610aa5...", "dirty": false}, "builder": null}
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let f = File::open(path).context(format!(
            "Cannot open provenance file `{}`",
            path.to_string_lossy()
        ))?;
        serde_json::from_reader(BufReader::new(f)).context(format!(
            "Cannot parse provenance file `{}`",
            path.to_string_lossy()
        ))
    }
}

/// Provenance of a build: the `kernel-builder` that produced it and the kernel
/// source, with their git state (each when known).
///
/// The version of the `kernel-builder` is baked into the binary. The git state
/// of the `kernel-builder` source is not, since that would make the binary
/// different for every commit.
fn provenance(git_state: GitState) -> Provenance {
    Provenance {
        kernel_builder: KernelBuilderVersion {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            git: git_state.builder,
        },
        kernel: git_state.kernel,
    }
}

/// Set the provenance in the metadata of the build variants of a kernel or
/// SDK.
///
/// The git state of the kernel (or SDK) and of the `kernel-builder` needs to
/// be passed explicitly, since we cannot infer it from paths that are not a
/// Git repo (e.g. Nix builds). Any existing provenance information is
/// overwritten.
pub fn add_provenance(kernel_dir: Option<PathBuf>, git_state: GitState) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let (_, variants) = discover_variants(&kernel_dir)?;

    let provenance = provenance(git_state);

    for variant in variants {
        eprintln!(
            "Writing provenance to variant `{}`...",
            variant.file_name().unwrap().to_string_lossy()
        );
        set_provenance(&variant, &provenance)?;
    }

    Ok(())
}

fn set_provenance(variant: &Path, provenance: &Provenance) -> Result<()> {
    let metadata_path = variant.join("metadata.json");

    let f = File::open(&metadata_path).context(format!(
        "Cannot open `{}` for reading",
        metadata_path.to_string_lossy()
    ))?;
    let mut metadata: Metadata = serde_json::from_reader(BufReader::new(f)).context(format!(
        "Cannot parse `{}`",
        metadata_path.to_string_lossy()
    ))?;

    metadata.provenance = Some(provenance.clone());

    let f = File::create(&metadata_path).context(format!(
        "Cannot open `{}` for writing provenance",
        metadata_path.to_string_lossy()
    ))?;
    serde_json::to_writer_pretty(BufWriter::new(f), &metadata).context(format!(
        "Cannot write updated metadata to `{}`",
        metadata_path.to_string_lossy()
    ))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    const KERNEL_SHA: &str = "d0610aa58db33b142c86b59598a2a1c730f52996";
    const BUILDER_SHA: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

    const METADATA: &str = r#"{"name": "test-kernel", "id": "kernel_id", "version": 1, "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cuda"}}"#;

    // Provenance that is not written by the builder.
    const METADATA_WITH_PROVENANCE: &str = r#"{"name": "test-kernel", "id": "kernel_id", "version": 1, "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cuda"}, "provenance": {"kernel-builder": {"version": "0.0.0", "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "dirty": false}, "kernel": {"commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": false}}}"#;

    fn git_state(kernel: Option<(&str, bool)>, builder: Option<(&str, bool)>) -> GitState {
        let status = |state: Option<(&str, bool)>| {
            state.map(|(sha, dirty)| format!(r#"{{"commit": "{sha}", "dirty": {dirty}}}"#))
        };
        let json = format!(
            r#"{{"kernel": {}, "builder": {}}}"#,
            status(kernel).unwrap_or("null".to_owned()),
            status(builder).unwrap_or("null".to_owned()),
        );
        parse(&json).unwrap()
    }

    /// Read a git state through a file, like the command does.
    fn parse(json: &str) -> Result<GitState> {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("provenance.json");
        fs::write(&path, json).unwrap();
        GitState::from_file(&path)
    }

    fn write_variant(dir: &Path, name: &str, metadata: &str) -> PathBuf {
        let variant = dir.join(name);
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), metadata).unwrap();
        variant
    }

    fn read_metadata(variant: &Path) -> Metadata {
        let f = File::open(variant.join("metadata.json")).unwrap();
        serde_json::from_reader(BufReader::new(f)).unwrap()
    }

    #[test]
    fn parses_git_state() {
        let state = parse(&format!(
            r#"{{"kernel": {{"commit": "{KERNEL_SHA}", "dirty": true}}, "builder": {{"commit": "{BUILDER_SHA}", "dirty": false}}}}"#
        ))
        .unwrap();

        let kernel = state.kernel.unwrap();
        assert_eq!(kernel.commit.as_str(), KERNEL_SHA);
        assert!(kernel.dirty);
        let builder = state.builder.unwrap();
        assert_eq!(builder.commit.as_str(), BUILDER_SHA);
        assert!(!builder.dirty);
    }

    #[test]
    fn parses_unknown_git_state() {
        let state = parse(r#"{"kernel": null, "builder": null}"#).unwrap();
        assert!(state.kernel.is_none());
        assert!(state.builder.is_none());
    }

    #[test]
    fn parses_sha_as_commit() {
        // The shape of the git provenance that Nix computes.
        let state = parse(&format!(
            r#"{{"kernel": {{"sha": "{KERNEL_SHA}", "dirty": false}}, "builder": null}}"#
        ))
        .unwrap();
        assert_eq!(state.kernel.unwrap().commit.as_str(), KERNEL_SHA);
    }

    #[test]
    fn missing_git_state_is_an_error() {
        // Unknown has to be explicit: `null`.
        assert!(parse(r#"{"kernel": null}"#).is_err());
        assert!(parse(r#"{"builder": null}"#).is_err());
        assert!(parse(r#"{}"#).is_err());
    }

    #[test]
    fn invalid_git_state_is_an_error() {
        // Unknown field.
        assert!(parse(r#"{"kernel": null, "builder": null, "kernal": null}"#).is_err());
        // Abbreviated commit.
        assert!(
            parse(r#"{"kernel": {"commit": "d0610aa", "dirty": false}, "builder": null}"#).is_err()
        );
        // Missing dirty flag.
        assert!(parse(&format!(
            r#"{{"kernel": {{"commit": "{KERNEL_SHA}"}}, "builder": null}}"#
        ))
        .is_err());
        assert!(parse("null").is_err());
        assert!(parse("not json").is_err());
    }

    #[test]
    fn errors_mention_the_file() {
        let temp_dir = tempfile::tempdir().unwrap();

        let missing = temp_dir.path().join("missing.json");
        let error = GitState::from_file(&missing).unwrap_err();
        assert!(format!("{error:#}").contains("missing.json"), "{error:#}");

        let invalid = temp_dir.path().join("invalid.json");
        fs::write(&invalid, r#"{"kernel": null}"#).unwrap();
        let error = GitState::from_file(&invalid).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("invalid.json"), "{message}");
        // The reason is reported as well.
        assert!(message.contains("builder"), "{message}");
    }

    #[test]
    fn adds_kernel_and_builder_provenance() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cuda = write_variant(temp_dir.path(), "torch-cuda", METADATA);
        let cpu = write_variant(temp_dir.path(), "torch-cpu", METADATA);

        add_provenance(
            Some(temp_dir.path().to_path_buf()),
            git_state(Some((KERNEL_SHA, true)), Some((BUILDER_SHA, false))),
        )
        .unwrap();

        for variant in [cuda, cpu] {
            let provenance = read_metadata(&variant).provenance.unwrap();

            let kernel = provenance.kernel.unwrap();
            assert_eq!(kernel.commit.as_str(), KERNEL_SHA);
            assert!(kernel.dirty);

            // The version comes from this binary, the commit from the caller.
            assert_eq!(provenance.kernel_builder.version, env!("CARGO_PKG_VERSION"));
            let builder = provenance.kernel_builder.git.unwrap();
            assert_eq!(builder.commit.as_str(), BUILDER_SHA);
            assert!(!builder.dirty);
        }
    }

    #[test]
    fn dirty_builder_provenance() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA);

        add_provenance(
            Some(temp_dir.path().to_path_buf()),
            git_state(Some((KERNEL_SHA, false)), Some((BUILDER_SHA, true))),
        )
        .unwrap();

        let provenance = read_metadata(&variant).provenance.unwrap();
        assert!(!provenance.kernel.unwrap().dirty);
        assert!(provenance.kernel_builder.git.unwrap().dirty);
    }

    #[test]
    fn unknown_kernel_provenance_is_omitted() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA);

        add_provenance(
            Some(temp_dir.path().to_path_buf()),
            git_state(None, Some((BUILDER_SHA, false))),
        )
        .unwrap();

        let provenance = read_metadata(&variant).provenance.unwrap();
        assert!(provenance.kernel.is_none());
        assert!(provenance.kernel_builder.git.is_some());
    }

    #[test]
    fn unknown_builder_commit_is_omitted() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA);

        add_provenance(
            Some(temp_dir.path().to_path_buf()),
            git_state(Some((KERNEL_SHA, false)), None),
        )
        .unwrap();

        let provenance = read_metadata(&variant).provenance.unwrap();
        assert!(provenance.kernel.is_some());
        // Only the version is known.
        assert_eq!(provenance.kernel_builder.version, env!("CARGO_PKG_VERSION"));
        assert!(provenance.kernel_builder.git.is_none());
    }

    #[test]
    fn existing_provenance_is_replaced() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA_WITH_PROVENANCE);

        // Without commits, nothing of the provenance of the metadata must
        // survive.
        add_provenance(Some(temp_dir.path().to_path_buf()), git_state(None, None)).unwrap();

        let provenance = read_metadata(&variant).provenance.unwrap();
        assert!(provenance.kernel.is_none());
        assert!(provenance.kernel_builder.git.is_none());
        assert_eq!(provenance.kernel_builder.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn other_metadata_is_preserved() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA);

        add_provenance(Some(temp_dir.path().to_path_buf()), git_state(None, None)).unwrap();

        let metadata = read_metadata(&variant);
        assert_eq!(metadata.id, "kernel_id");
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.license, "Apache-2.0");
    }
}
