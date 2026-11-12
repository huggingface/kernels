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

// A Nix store path has no .git, so its revisions are passed in.
#[test]
fn revs_stand_in_for_git() {
    let temp = tempfile::tempdir().unwrap();
    let (upstream, vendored) = (temp.path().join("up"), temp.path().join("dep"));
    fs::create_dir_all(&upstream).unwrap();
    fs::create_dir_all(&vendored).unwrap();
    fs::write(vendored.join("b.h"), "// b\n").unwrap();
    let pin = "0123456789abcdef0123456789abcdef01234567";
    let recipe = temp.path().join("port.kdl");
    fs::write(
        &recipe,
        format!(
            "recipe version=1\n\
             source repo=\"https://example.com/up.git\" commit=\"{pin}\"\n\
             vendor name=\"dep\" repo=\"https://example.com/dep.git\" commit=\"{pin}\" \
             path=\".\" to=\"third_party\"\n"
        ),
    )
    .unwrap();

    let other = "f".repeat(40);
    for (source_rev, vendor_rev, ok) in
        [(pin, pin, true), (&other, pin, false), (pin, &other, false)]
    {
        let out = temp.path().join("out");
        let output = Command::new(env!("CARGO_BIN_EXE_kernel-port"))
            .arg(&recipe)
            .arg("--dir")
            .arg(&upstream)
            .arg("--out")
            .arg(&out)
            .args(["--source-rev", source_rev])
            .arg("--vendor")
            .arg(format!("dep={}", vendored.display()))
            .arg("--vendor-rev")
            .arg(format!("dep={vendor_rev}"))
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.success(), ok, "{stderr}");
        if ok {
            assert!(out.join("third_party/b.h").is_file());
        } else {
            assert!(
                stderr.contains("but this port is written against"),
                "{stderr}"
            );
        }
    }
}
