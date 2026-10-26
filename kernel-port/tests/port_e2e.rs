use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn read_tree(root: &Path) -> BTreeMap<PathBuf, String> {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let name = PathBuf::from(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            for (path, text) in read_tree(&entry.path()) {
                files.insert(name.join(path), text);
            }
        } else {
            files.insert(name, fs::read_to_string(entry.path()).unwrap());
        }
    }
    files
}

fn check_port(flavor: &str) {
    let example = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/examples")
        .join(flavor);
    let temp = tempfile::tempdir().unwrap();
    let upstream = temp.path().join("upstream");
    let out = temp.path().join("out");
    let source = read_tree(&example.join("upstream"));
    for (path, text) in &source {
        let dest = upstream.join(path);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(dest, text).unwrap();
    }

    let output = Command::new(env!("CARGO_BIN_EXE_kernel-port"))
        .arg(example.join("port.kdl"))
        .arg("--dir")
        .arg(&upstream)
        .arg("--out")
        .arg(&out)
        .current_dir(temp.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{flavor}: stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let expected = read_tree(&example.join("expected"));
    let mut actual = read_tree(&out);
    // Provenance depends on the recipe text and is not part of the snapshot.
    assert!(actual.remove(Path::new("port-provenance.json")).is_some());
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (path, text) in expected {
        assert_eq!(actual[&path], text, "{flavor}: {}", path.display());
    }
    assert_eq!(read_tree(&upstream), source, "{flavor}: upstream changed");
}

#[test]
fn aot_relu_port() {
    check_port("relu-aot");
}

#[test]
fn jit_relu_port() {
    check_port("relu-jit");
}
