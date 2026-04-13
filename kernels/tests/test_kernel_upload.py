import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from kernels.cli import upload_kernels
from kernels.cli.upload import BUILD_COMMIT_BATCH_SIZE, upload_kernels_dir
from kernels.utils import _get_hf_api

REPO_ID = "__DUMMY_KERNELS_USER__/kernels-upload-test"


PY_CONTENT = """\
#!/usr/bin/env python3

def main():
    print("Hello from torch-universal!")

if __name__ == "__main__":
    main()
"""


@dataclass
class UploadArgs:
    kernel_dir: None
    repo_id: None
    private: False
    branch: None


def next_filename(path: Path) -> Path:
    """
    Given a path like foo_2050.py, return foo_2051.py.
    """
    m = re.match(r"^(.*?)(\d+)(\.py)$", path.name)
    if not m:
        raise ValueError(
            f"Filename {path.name!r} does not match pattern <prefix>_<number>.py"
        )

    prefix, number, suffix = m.groups()
    new_number = str(int(number) + 1).zfill(len(number))
    return path.with_name(f"{prefix}{new_number}{suffix}")


def get_filename_to_change(repo_filenames):
    for f in repo_filenames:
        if "foo" in f and f.endswith(".py"):
            filename_to_change = os.path.basename(f)
            break
    assert filename_to_change
    return filename_to_change


def get_filenames_from_a_repo(repo_id: str) -> list[str]:
    try:
        repo_info = _get_hf_api().model_info(repo_id=repo_id, files_metadata=True)
        repo_siblings = repo_info.siblings
        if repo_siblings is not None:
            return [f.rfilename for f in repo_siblings]
        else:
            raise ValueError("No repo siblings found.")
    except Exception as e:
        logging.error(f"Error connecting to the Hub: {e}.")


@pytest.mark.token
@pytest.mark.is_staging_test
@pytest.mark.parametrize("branch", (None, "foo"))
def test_kernel_upload_works_as_expected(branch):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/build/torch-universal/upload_test"
        build_dir = Path(path)
        build_dir.mkdir(parents=True, exist_ok=True)
        script_path = build_dir / "foo.py"
        script_path.write_text(PY_CONTENT)
        upload_kernels(UploadArgs(tmpdir, REPO_ID, False, branch))

    repo_filenames = get_filenames_from_a_repo(REPO_ID)
    assert any(str(script_path.name) for f in repo_filenames)

    api = _get_hf_api()
    if branch is not None:
        refs = api.list_repo_refs(repo_id=REPO_ID)
        assert any(ref_branch.name == branch for ref_branch in refs.branches)

    api.delete_repo(repo_id=REPO_ID)


def get_filenames_from_a_branch(repo_id: str, branch: str) -> list[str]:
    try:
        repo_info = _get_hf_api().model_info(
            repo_id=repo_id, revision=branch, files_metadata=True
        )
        repo_siblings = repo_info.siblings
        if repo_siblings is not None:
            return [f.rfilename for f in repo_siblings]
        else:
            raise ValueError("No repo siblings found.")
    except Exception as e:
        logging.error(f"Error connecting to the Hub: {e}.")


@pytest.mark.token
@pytest.mark.is_staging_test
def test_kernel_upload_new_branch_starts_fresh():
    api = _get_hf_api()

    # First upload to main to populate it.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/build/torch-universal/upload_test"
        build_dir = Path(path)
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "foo.py").write_text(PY_CONTENT)
        upload_kernels(UploadArgs(tmpdir, REPO_ID, False, None))

    main_files = get_filenames_from_a_repo(REPO_ID)
    assert any("foo.py" in f for f in main_files)

    # Now upload a different variant to a new branch.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/build/torch-universal/upload_test"
        build_dir = Path(path)
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "bar.py").write_text(PY_CONTENT)
        upload_kernels(UploadArgs(tmpdir, REPO_ID, False, "v2"))

    branch_files = get_filenames_from_a_branch(REPO_ID, "v2")

    assert any("bar.py" in f for f in branch_files), f"{branch_files=}"
    assert not any(
        "foo.py" in f for f in branch_files
    ), f"Branch v2 should not inherit foo.py from main: {branch_files=}"

    api.delete_repo(repo_id=REPO_ID)


@pytest.mark.token
@pytest.mark.is_staging_test
def test_kernel_upload_deletes_as_expected():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/build/torch-universal/upload_test"
        build_dir = Path(path)
        build_dir.mkdir(parents=True, exist_ok=True)
        script_path = build_dir / "foo_2025.py"
        script_path.write_text(PY_CONTENT)
        upload_kernels(UploadArgs(tmpdir, REPO_ID, False, None))

    repo_filenames = get_filenames_from_a_repo(REPO_ID)
    filename_to_change = get_filename_to_change(repo_filenames)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/build/torch-universal/upload_test"
        build_dir = Path(path)
        build_dir.mkdir(parents=True, exist_ok=True)
        changed_filename = next_filename(Path(filename_to_change))
        script_path = build_dir / changed_filename
        script_path.write_text(PY_CONTENT)
        upload_kernels(UploadArgs(tmpdir, REPO_ID, False, None))

    repo_filenames = get_filenames_from_a_repo(REPO_ID)
    assert any(str(changed_filename) in k for k in repo_filenames), f"{repo_filenames=}"
    assert not any(
        str(filename_to_change) in k for k in repo_filenames
    ), f"{repo_filenames=}"
    _get_hf_api().delete_repo(repo_id=REPO_ID)


def test_upload_includes_card_as_readme():
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_dir = Path(tmpdir).resolve()

        variant_dir = kernel_dir / "torch-cuda"
        variant_dir.mkdir(parents=True)
        (variant_dir / "metadata.json").write_text(
            '{"version": 1, "python-depends": []}'
        )

        build_dir = kernel_dir / "build"
        build_dir.mkdir()
        card_path = build_dir / "CARD.md"
        card_path.write_text("# Test Kernel\n")

        mock_api = MagicMock()
        mock_api.create_repo.return_value.repo_id = REPO_ID

        with patch("kernels.cli.upload._get_hf_api", return_value=mock_api):
            upload_kernels_dir(kernel_dir, repo_id=REPO_ID, branch=None, private=False)

        mock_api.upload_file.assert_called_once_with(
            repo_id=REPO_ID,
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            revision="main",
            commit_message="File uploaded using `kernels`.",
        )


def test_large_kernel_upload_uses_create_commit_batches(monkeypatch, tmp_path):
    kernel_root = tmp_path / "kernel"
    build_variant = kernel_root / "build" / "torch-cpu"
    build_variant.mkdir(parents=True, exist_ok=True)
    (build_variant / "metadata.json").write_text("{}")
    file_count = BUILD_COMMIT_BATCH_SIZE * 2
    for i in range(file_count):
        (build_variant / f"file_{i}.py").touch()

    api = Mock()
    api.create_repo.return_value = SimpleNamespace(repo_id=REPO_ID)
    api.list_repo_refs.return_value = SimpleNamespace(
        branches=[SimpleNamespace(name="main")]
    )
    api.list_repo_files.return_value = [
        "README.md",
        "build/torch-cpu/file_0.py",
        "build/torch-cpu/stale.py",
        "build/torch-cuda/keep.py",
    ]
    monkeypatch.setattr("kernels.cli.upload._get_hf_api", lambda: api)

    upload_kernels(UploadArgs(kernel_root, REPO_ID, False, "main"))

    # 2 full batches of adds, plus metadata and 1 stale-file delete.
    assert api.create_commit.call_count == 3
    batch_sizes = [
        len(call.kwargs["operations"]) for call in api.create_commit.call_args_list
    ]
    assert batch_sizes == [
        BUILD_COMMIT_BATCH_SIZE,
        BUILD_COMMIT_BATCH_SIZE,
        2,
    ]
    commit_messages = [
        call.kwargs["commit_message"] for call in api.create_commit.call_args_list
    ]
    assert commit_messages == [
        "Build uploaded using `kernels` (batch 1/3).",
        "Build uploaded using `kernels` (batch 2/3).",
        "Build uploaded using `kernels` (batch 3/3).",
    ]

    # Stale repo files should be deleted.
    operations = [
        operation
        for call in api.create_commit.call_args_list
        for operation in call.kwargs["operations"]
    ]
    delete_paths = {
        op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)
    }
    assert delete_paths == {"build/torch-cpu/stale.py"}

    add_paths = {
        op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)
    }
    assert len(add_paths) == file_count + 1
    assert "build/torch-cpu/metadata.json" in add_paths
    assert "build/torch-cpu/file_0.py" in add_paths
    assert "build/torch-cpu/file_399.py" in add_paths
    api.upload_folder.assert_not_called()


def test_new_branch_upload_replaces_inherited_build_tree(monkeypatch, tmp_path):
    kernel_root = tmp_path / "kernel"
    build_variant = kernel_root / "build" / "torch-cpu"
    build_variant.mkdir(parents=True, exist_ok=True)
    (build_variant / "metadata.json").write_text("{}")
    (build_variant / "current.py").write_text(PY_CONTENT)

    benchmarks_dir = kernel_root / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    (benchmarks_dir / "benchmark_current.py").write_text(PY_CONTENT)

    api = Mock()
    api.create_repo.return_value = SimpleNamespace(repo_id=REPO_ID)
    api.list_repo_refs.return_value = SimpleNamespace(branches=[])
    api.list_repo_files.return_value = [
        "README.md",
        "benchmarks/benchmark_old.py",
        "build/torch-cpu/stale.py",
        "build/torch-cuda/inherited.py",
    ]
    monkeypatch.setattr("kernels.cli.upload._get_hf_api", lambda: api)

    upload_kernels(UploadArgs(kernel_root, REPO_ID, False, "v2"))

    api.upload_folder.assert_called_once()
    assert api.upload_folder.call_args.kwargs["delete_patterns"] == ["**"]

    operations = [
        operation
        for call in api.create_commit.call_args_list
        for operation in call.kwargs["operations"]
    ]
    delete_paths = {
        op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)
    }
    assert delete_paths == {
        "build/torch-cpu/stale.py",
        "build/torch-cuda/inherited.py",
    }
