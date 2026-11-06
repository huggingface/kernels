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

fn port(args: &[&std::ffi::OsStr]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kernel-port"))
        .args(args)
        .output()
        .unwrap()
}

fn provenance_commit(out: &Path) -> String {
    let text = fs::read_to_string(out.join("port-provenance.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["sources"]["upstream"]["commit"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn source_rev_stands_in_for_git() {
    let temp = tempfile::tempdir().unwrap();
    let upstream = temp.path().join("upstream");
    let out = temp.path().join("out");
    fs::create_dir_all(&upstream).unwrap();
    fs::write(upstream.join("a.py"), "a = 1\n").unwrap();

    let pin = "0123456789abcdef0123456789abcdef01234567";
    let recipe = temp.path().join("port.kdl");
    fs::write(
        &recipe,
        format!(
            "recipe version=1\n\
             source repo=\"https://example.com/up.git\" commit=\"{pin}\"\n"
        ),
    )
    .unwrap();

    let run = |rev: &str| {
        port(&[
            recipe.as_os_str(),
            "--dir".as_ref(),
            upstream.as_os_str(),
            "--out".as_ref(),
            out.as_os_str(),
            "--source-rev".as_ref(),
            rev.as_ref(),
        ])
    };

    let output = run(pin);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(provenance_commit(&out), pin);

    // A recipe still rejects a revision other than its pin.
    let output = run(&"f".repeat(40));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("but this port is written against"),
        "{stderr}"
    );
}

#[test]
fn vendor_rev_stands_in_for_git() {
    let temp = tempfile::tempdir().unwrap();
    let upstream = temp.path().join("upstream");
    let vendored = temp.path().join("vendored");
    let out = temp.path().join("out");
    fs::create_dir_all(&upstream).unwrap();
    fs::create_dir_all(vendored.join("include")).unwrap();
    fs::write(upstream.join("a.py"), "a = 1\n").unwrap();
    fs::write(vendored.join("include/b.h"), "// b\n").unwrap();

    let pin = "0123456789abcdef0123456789abcdef01234567";
    let recipe = temp.path().join("port.kdl");
    fs::write(
        &recipe,
        format!(
            "recipe version=1\n\
             source repo=\"https://example.com/up.git\" commit=\"{pin}\"\n\
             vendor name=\"dep\" repo=\"https://example.com/dep.git\" commit=\"{pin}\" \
             path=\"include\" to=\"third_party\"\n"
        ),
    )
    .unwrap();

    let run = |vendor_rev: &str| {
        port(&[
            recipe.as_os_str(),
            "--dir".as_ref(),
            upstream.as_os_str(),
            "--out".as_ref(),
            out.as_os_str(),
            "--source-rev".as_ref(),
            pin.as_ref(),
            "--vendor".as_ref(),
            format!("dep={}", vendored.display()).as_ref(),
            "--vendor-rev".as_ref(),
            format!("dep={vendor_rev}").as_ref(),
        ])
    };

    let output = run(pin);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.join("third_party/b.h").is_file());

    // A vendor still rejects a revision other than its pin.
    let output = run(&"f".repeat(40));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("but this port is written against"),
        "{stderr}"
    );
}
