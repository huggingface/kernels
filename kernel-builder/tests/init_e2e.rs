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
fn test_init_fails_on_existing_scaffold_file() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("exists");
    fs::create_dir_all(&dir).unwrap();
    // Pre-create a scaffold file - should cause init to fail
    fs::write(dir.join("build.toml"), "existing content").unwrap();

    let bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/debug/kernel-builder");

    let out = Command::new(&bin)
        .args(["init", "--name", "user/exists", "--backends", "cpu"])
        .current_dir(temp.path())
        .output()
        .unwrap();

    assert!(!out.status.success());
    // Original file should be preserved (atomic - no partial writes)
    assert_eq!(
        fs::read_to_string(dir.join("build.toml")).unwrap(),
        "existing content"
    );
}

#[test]
fn test_init_overwrite() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("k");
    fs::create_dir_all(&dir).unwrap();
    // Pre-create a scaffold file and a user file
    fs::write(dir.join("build.toml"), "old scaffold").unwrap();
    fs::write(dir.join("custom.txt"), "user content").unwrap();

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
    // Scaffold file should be overwritten with new content
    let build_toml = fs::read_to_string(dir.join("build.toml")).unwrap();
    assert!(
        build_toml.contains("[general]"),
        "build.toml not overwritten"
    );
    // User's custom file should be preserved
    assert!(
        dir.join("custom.txt").exists(),
        "user file should be preserved"
    );
    assert_eq!(
        fs::read_to_string(dir.join("custom.txt")).unwrap(),
        "user content"
    );
}
