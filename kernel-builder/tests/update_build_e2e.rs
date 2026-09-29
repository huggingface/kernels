use std::fs;
use std::path::Path;
use std::process::Command;

fn update_build(dir: &Path) {
    let output = Command::new(env!("CARGO_BIN_EXE_kernel-builder"))
        .arg("update-build")
        .arg(dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn legacy_builds_migrate_to_edition_six() {
    for (edition, framework) in [
        ("", ""),
        ("", "[torch-noarch]"),
        ("edition = 5", "[torch-noarch]"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("build.toml");
        fs::write(
            &path,
            format!(
                r#"[general]
name = "test"
version = 1
license = "MIT"
backends = ["cpu"]
upstream = "https://github.com/example/kernel"
{edition}
{framework}
"#
            ),
        )
        .unwrap();

        update_build(dir.path());
        let migrated = fs::read_to_string(&path).unwrap();
        let value: toml::Value = toml::from_str(&migrated).unwrap();
        assert_eq!(value["general"]["edition"].as_integer(), Some(6));
        assert_eq!(
            value["general"]["upstream"].as_array().unwrap(),
            &[toml::Value::from("https://github.com/example/kernel")]
        );
        kernels_common::config::Build::open(dir.path()).unwrap();

        // Already-current configurations must not be rewritten.
        fs::write(&path, format!("# Keep this comment.\n{migrated}")).unwrap();
        update_build(dir.path());
        assert_eq!(
            fs::read_to_string(path).unwrap(),
            format!("# Keep this comment.\n{migrated}")
        );
    }
}
