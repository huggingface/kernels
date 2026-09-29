use kernels_common::config::{Backend, Build, BuildCompat, CurrentConfig};
use kernels_common::metadata::Metadata;
use kernels_common::version::Version;
use serde_json::json;

const FIRST: &str = "https://github.com/ronghanghu/torch_generic_nms";
const SECOND: &str = "git@github.com:ronghanghu/cc_torch.git";

fn config(edition: &str, upstream: &str) -> String {
    format!(
        r#"[general]
name = "cv-utils"
version = 1
license = "MIT"
backends = ["cpu"]
{edition}
{upstream}
[torch-noarch]
"#
    )
}

#[test]
fn build_upstreams_round_trip_through_metadata_and_current_config() {
    for (field, expected) in [
        (String::new(), vec![]),
        ("upstream = []".into(), vec![]),
        (format!("upstream = [{FIRST:?}]"), vec![FIRST]),
        (
            format!("upstream = [{FIRST:?}, {SECOND:?}]"),
            vec![FIRST, SECOND],
        ),
    ] {
        let compat: BuildCompat = toml::from_str(&config("edition = 6", &field)).unwrap();
        assert!(matches!(&compat, BuildCompat::V6(_)));
        let build: Build = compat.try_into().unwrap();
        let metadata = Metadata::for_backend(&build, "cv-utils".into(), Backend::Cpu).unwrap();
        assert_eq!(
            metadata
                .upstream
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            metadata.kernels_minver,
            Some(if expected.len() > 1 {
                Version::new([0, 18, 0])
            } else {
                Version::new([0, 14, 0])
            })
        );

        let serialized = serde_json::to_value(&metadata).unwrap();
        match expected.as_slice() {
            [] => assert!(serialized.get("upstream").is_none()),
            [url] => assert_eq!(serialized["upstream"], json!(url)),
            urls => assert_eq!(serialized["upstream"], json!(urls)),
        }
        let parsed: Metadata = serde_json::from_value(serialized).unwrap();
        assert_eq!(parsed.upstream, metadata.upstream);

        let current: CurrentConfig = build.into();
        let serialized = toml::to_string(&current).unwrap();
        let value: toml::Value = toml::from_str(&serialized).unwrap();
        assert_eq!(value["general"]["edition"].as_integer(), Some(6));
        if !expected.is_empty() {
            assert_eq!(
                value["general"]["upstream"].as_array().unwrap(),
                &expected
                    .iter()
                    .map(|url| toml::Value::from(*url))
                    .collect::<Vec<_>>()
            );
        }
        let parsed: CurrentConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed.general.upstream, metadata.upstream);
    }
}

#[test]
fn legacy_build_upstream_survives_migration() {
    for (edition, framework) in [
        ("", ""),
        ("", "[torch-noarch]"),
        ("edition = 5", "[torch-noarch]"),
    ] {
        for (field, expected) in [
            (String::new(), vec![]),
            (format!("upstream = {FIRST:?}"), vec![FIRST]),
        ] {
            let input = config(edition, &field).replace("[torch-noarch]", framework);
            let compat: BuildCompat = toml::from_str(&input).unwrap();
            if !edition.is_empty() {
                assert!(matches!(&compat, BuildCompat::V5(_)));
            } else if framework.is_empty() {
                assert!(matches!(&compat, BuildCompat::V3(_)));
            } else {
                assert!(matches!(&compat, BuildCompat::V4(_)));
            }
            let build: Build = compat.try_into().unwrap();
            let current: CurrentConfig = build.into();
            let serialized = toml::to_string(&current).unwrap();
            let parsed: BuildCompat = toml::from_str(&serialized).unwrap();
            assert!(matches!(parsed, BuildCompat::V6(_)));
            assert_eq!(
                current
                    .general
                    .upstream
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn legacy_build_rejects_upstream_lists() {
    for (edition, framework) in [
        ("", ""),
        ("", "[torch-noarch]"),
        ("edition = 5", "[torch-noarch]"),
    ] {
        for urls in [vec![], vec![FIRST], vec![FIRST, SECOND]] {
            let input = config(edition, &format!("upstream = {urls:?}"))
                .replace("[torch-noarch]", framework);
            assert!(toml::from_str::<BuildCompat>(&input).is_err());
        }
    }
}

#[test]
fn build_open_upgrades_supported_editions_in_memory() {
    for (edition, field) in [
        ("", format!("upstream = {FIRST:?}")),
        ("edition = 5", format!("upstream = {FIRST:?}")),
        ("edition = 6", format!("upstream = [{FIRST:?}]")),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let input = config(edition, &field);
        let path = dir.path().join("build.toml");
        std::fs::write(&path, &input).unwrap();
        let build = Build::open(dir.path()).unwrap();
        assert_eq!(build.general.upstream[0].to_string(), FIRST);
        assert_eq!(std::fs::read_to_string(path).unwrap(), input);
    }
}

#[test]
fn unsupported_build_editions_are_rejected() {
    let err = toml::from_str::<BuildCompat>(&config("edition = 7", "")).unwrap_err();
    assert!(err.to_string().contains("unsupported build edition 7"));

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("build.toml"),
        config("", "").replace("[torch-noarch]", ""),
    )
    .unwrap();
    assert!(
        Build::open(dir.path())
            .err()
            .unwrap()
            .to_string()
            .contains("update-build")
    );
}

#[test]
fn metadata_accepts_legacy_and_list_upstreams() {
    for (upstream, expected) in [
        (json!(null), vec![]),
        (json!([]), vec![]),
        (json!(FIRST), vec![FIRST]),
        (json!([FIRST]), vec![FIRST]),
        (json!([FIRST, SECOND]), vec![FIRST, SECOND]),
    ] {
        let metadata: Metadata = serde_json::from_value(json!({
            "name": "cv-utils", "id": "cv-utils", "version": 1, "license": "MIT",
            "python-depends": [], "backend": {"type": "cpu"}, "upstream": upstream,
        }))
        .unwrap();
        assert_eq!(
            metadata
                .upstream
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn invalid_upstreams_are_rejected() {
    for upstream in [
        json!(42),
        json!({}),
        json!("not a url"),
        json!([FIRST, "ftp://example.com/repo"]),
        json!([FIRST, null]),
    ] {
        let metadata = json!({
            "name": "cv-utils", "id": "cv-utils", "version": 1, "license": "MIT",
            "python-depends": [], "backend": {"type": "cpu"}, "upstream": upstream,
        });
        assert!(serde_json::from_value::<Metadata>(metadata).is_err());
    }
    for upstream in [
        "42",
        "\"https://example.com/repo\"",
        "{}",
        "\"not a url\"",
        "[\"https://example.com/repo\", 42]",
        "[\"ftp://example.com/repo\"]",
    ] {
        assert!(
            toml::from_str::<BuildCompat>(&config(
                "edition = 6",
                &format!("upstream = {upstream}")
            ))
            .is_err()
        );
    }
}
