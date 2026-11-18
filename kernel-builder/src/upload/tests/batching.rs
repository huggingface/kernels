use std::collections::BTreeMap;

use hf_hub::repository::{AddSource, CommitOperation};

use crate::upload::{operation_path, operation_variant, plan_batches, BUILD_COMMIT_BATCH_SIZE};

type Files = BTreeMap<String, String>;

const DEEP_GEMM_VARIANT_FILES: usize = 1447;

fn delete(path: &str) -> CommitOperation {
    CommitOperation::Delete {
        path_in_repo: path.to_owned(),
    }
}

fn add(path: &str) -> CommitOperation {
    add_content(path, "")
}

fn add_content(path: &str, content: &str) -> CommitOperation {
    CommitOperation::Add {
        path_in_repo: path.to_owned(),
        source: AddSource::Bytes(content.as_bytes().to_vec().into()),
    }
}

fn batch_paths(batches: &[Vec<CommitOperation>]) -> Vec<Vec<&str>> {
    batches
        .iter()
        .map(|batch| batch.iter().map(operation_path).collect())
        .collect()
}

fn variant(files: &mut Files, variant: &str, paths: &[String], content: &str) {
    for path in paths {
        files.insert(
            format!("build/{variant}/{path}"),
            format!("{content}:{path}"),
        );
    }
}

fn variant_files(files: &Files, variant: &str) -> Files {
    let prefix = format!("build/{variant}/");
    files
        .iter()
        .filter(|(path, _)| path.starts_with(&prefix))
        .map(|(path, content)| (path.clone(), content.clone()))
        .collect()
}

fn metadata(files: &Files, variant: &str) -> [Option<String>; 2] {
    ["metadata.json", "metadata.json.sigstore"]
        .map(|file| files.get(&format!("build/{variant}/{file}")).cloned())
}

/// In the order `collect_commit_ops` makes them: all deletes, then all adds.
fn upload_ops(before: &Files, after: &Files) -> Vec<CommitOperation> {
    let uploaded = |path: &str| {
        after
            .keys()
            .any(|p| operation_variant(&delete(p)) == operation_variant(&delete(path)))
    };
    let deletes = before
        .keys()
        .filter(|path| !after.contains_key(*path) && uploaded(path))
        .map(|path| delete(path));
    let adds = after
        .iter()
        .map(|(path, content)| add_content(path, content));
    deletes.chain(adds).collect()
}

fn apply(files: &mut Files, batch: &[CommitOperation]) {
    for op in batch {
        match op {
            CommitOperation::Add {
                path_in_repo,
                source: AddSource::Bytes(bytes),
            } => {
                files.insert(
                    path_in_repo.clone(),
                    String::from_utf8_lossy(bytes).into_owned(),
                );
            }
            CommitOperation::Delete { path_in_repo } => {
                files.remove(path_in_repo);
            }
            _ => unreachable!(),
        }
    }
}

fn check_every_failure(
    before: &Files,
    after: &Files,
    variants: &[&str],
    batch_size: usize,
    split: bool,
) {
    let batches = plan_batches(upload_ops(before, after), batch_size).unwrap();
    assert!(batches.len() > 1, "{} batches", batches.len());
    assert!(batches.iter().all(|batch| batch.len() <= batch_size));

    for fail_at in 0..=batches.len() {
        let mut files = before.clone();
        for batch in &batches[..fail_at] {
            apply(&mut files, batch);
        }
        for variant in variants {
            let on_hub = variant_files(&files, variant);
            let uploaded = variant_files(after, variant);
            if fail_at == batches.len() {
                assert_eq!(on_hub, uploaded, "`build/{variant}` was not uploaded");
                continue;
            }
            let unchanged = on_hub == variant_files(before, variant);
            let without_new_metadata = metadata(&on_hub, variant) == metadata(before, variant);
            assert!(
                on_hub == uploaded || unchanged || (split && without_new_metadata),
                "`build/{variant}` is half-uploaded after the commit of batch {fail_at} \
                 failed:\n{:#?}",
                on_hub.keys().collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn operation_variant_of_paths() {
    assert_eq!(
        operation_variant(&add("build/torch-cpu/metadata.json")),
        Some("build/torch-cpu")
    );
    assert_eq!(
        operation_variant(&delete("build/torch-cpu/sub/_ops.py")),
        Some("build/torch-cpu")
    );
    assert_eq!(operation_variant(&add("README.md")), None);
    assert_eq!(operation_variant(&add("benchmarks/benchmark.py")), None);
    assert_eq!(operation_variant(&add("build/stray-file")), None);
}

#[test]
fn plan_batches_keeps_variants_whole() {
    let operations = vec![
        add("README.md"),
        delete("build/a/stale.so"),
        delete("build/b/stale.so"),
        add("build/a/1"),
        add("build/a/2"),
        add("build/b/1"),
        add("build/b/2"),
        add("build/c/1"),
    ];

    let batches = plan_batches(operations, 4).unwrap();

    assert_eq!(
        batch_paths(&batches),
        vec![
            vec!["README.md", "build/a/stale.so", "build/a/1", "build/a/2"],
            vec!["build/b/stale.so", "build/b/1", "build/b/2", "build/c/1"],
        ]
    );
}

#[test]
fn plan_batches_single_batch() {
    let operations = vec![add("README.md"), add("build/a/1"), add("build/b/1")];
    let batches = plan_batches(operations, BUILD_COMMIT_BATCH_SIZE).unwrap();
    assert_eq!(
        batch_paths(&batches),
        vec![vec!["README.md", "build/a/1", "build/b/1"]]
    );
}

#[test]
fn plan_batches_no_operations() {
    assert!(plan_batches(Vec::new(), 4).unwrap().is_empty());
}

#[test]
fn plan_batches_splits_large_variant_with_metadata_last() {
    let operations = vec![
        add("README.md"),
        delete("build/a/stale.so"),
        add("build/a/1"),
        add("build/a/metadata.json"),
        add("build/a/metadata.json.sigstore"),
        add("build/a/utils/2"),
        add("build/a/utils/3"),
        add("build/b/1"),
    ];

    let batches = plan_batches(operations, 3).unwrap();

    assert_eq!(
        batch_paths(&batches),
        vec![
            vec!["README.md"],
            vec!["build/a/stale.so", "build/a/1", "build/a/utils/2"],
            vec![
                "build/a/utils/3",
                "build/a/metadata.json",
                "build/a/metadata.json.sigstore"
            ],
            vec!["build/b/1"],
        ]
    );
}

#[test]
fn plan_batches_too_many_files_outside_variants() {
    let operations = vec![add("README.md"), add("benchmarks/benchmark.py")];
    let err = plan_batches(operations, 1).unwrap_err().to_string();
    assert!(err.contains("outside of the build variants"), "{err}");
}

#[test]
fn upload_failing_between_batches_never_leaves_a_variant_half_uploaded() {
    let built = ["__init__.py", "_ops.py", "kernel.so"].map(String::from);
    let earlier = ["__init__.py", "_ops.py", "kernel.so", "stale.so"].map(String::from);

    let mut before = Files::from([("README.md".to_owned(), "readme".to_owned())]);
    variant(&mut before, "torch-a", &earlier, "earlier");
    variant(&mut before, "torch-c", &earlier, "earlier");
    let mut after = Files::from([("README.md".to_owned(), "card".to_owned())]);
    for name in ["torch-a", "torch-b", "torch-c"] {
        variant(&mut after, name, &built, "new");
    }

    check_every_failure(
        &before,
        &after,
        &["torch-a", "torch-b", "torch-c"],
        5,
        false,
    );
}

#[test]
fn upload_failing_between_batches_of_deep_gemm_sized_variants_keeps_earlier_metadata() {
    let files = |n: usize| -> Vec<String> {
        let mut files = vec![
            "metadata.json".to_owned(),
            "metadata.json.sigstore".to_owned(),
        ];
        files.extend((files.len()..n).map(|i| {
            let dir = if i % 2 == 0 { "include" } else { "utils" };
            format!("{dir}/{i}.h")
        }));
        files
    };

    let mut before = Files::new();
    variant(&mut before, "torch-a", &files(1200), "earlier");
    let mut after = Files::new();
    for name in ["torch-a", "torch-b"] {
        variant(&mut after, name, &files(DEEP_GEMM_VARIANT_FILES), "new");
    }

    check_every_failure(
        &before,
        &after,
        &["torch-a", "torch-b"],
        BUILD_COMMIT_BATCH_SIZE,
        true,
    );
}
