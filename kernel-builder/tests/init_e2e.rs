use std::fs;
use std::process::Command;

fn run_init(args: &[&str]) -> (bool, String, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/debug/kernel-builder");

    let output = Command::new(&bin)
        .args(["init"])
        .args(args)
        .current_dir(temp.path())
        .output()
        .expect("failed to run kernel-builder");

    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).to_string(),
        temp,
    )
}

#[test]
fn test_init_creates_expected_files() {
    let (ok, err, temp) = run_init(&["--name", "user/my-kernel", "--backends", "cuda", "metal"]);
    assert!(ok, "init failed: {err}");

    let dir = temp.path().join("my-kernel");

    // Core files
    for f in ["build.toml", "flake.nix", "CARD.md", "example.py", ".git"] {
        assert!(dir.join(f).exists(), "{f} missing");
    }

    // Only requested backends
    assert!(dir.join("my_kernel_cuda").exists());
    assert!(dir.join("my_kernel_metal").exists());
    assert!(!dir.join("my_kernel_cpu").exists());
    assert!(!dir.join("my_kernel_xpu").exists());
}

#[test]
fn test_init_templates_rendered() {
    let (ok, err, temp) = run_init(&["--name", "user/foo-bar", "--backends", "cpu"]);
    assert!(ok, "init failed: {err}");

    let binding =
        fs::read_to_string(temp.path().join("foo-bar/torch-ext/torch_binding.cpp")).unwrap();
    assert!(binding.contains("foo_bar"), "kernel name not substituted");
    assert!(!binding.contains("{{"), "template not rendered");
}

#[test]
fn test_init_fails_on_nonempty_dir() {
    let temp = tempfile::tempdir().unwrap();
    fs::create_dir_all(temp.path().join("exists")).unwrap();
    fs::write(temp.path().join("exists/file.txt"), "x").unwrap();

    let bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/debug/kernel-builder");

    let out = Command::new(&bin)
        .args(["init", "--name", "user/exists", "--backends", "cpu"])
        .current_dir(temp.path())
        .output()
        .unwrap();

    assert!(!out.status.success());
}

#[test]
fn test_init_overwrite() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("k");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("old.txt"), "x").unwrap();

    let bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/debug/kernel-builder");

    let out = Command::new(&bin)
        .args([
            "init",
            "--name",
            "user/k",
            "--backends",
            "cpu",
            "--overwrite",
        ])
        .current_dir(temp.path())
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "overwrite failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!dir.join("old.txt").exists());
    assert!(dir.join("build.toml").exists());
}
