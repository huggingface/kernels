import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from kernels.cli import upload_kernels
from kernels.utils import _get_hf_api

REPO_ID = "valid_org/kernels-upload-test"


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
    create_card: bool = False
    card_path: str = None
    description: str = None
    create_pr: bool = False


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


@patch("kernels.cli.create_and_upload_card")
@patch("kernels.cli.upload_kernels_dir")
def test_kernel_upload_calls_create_and_upload_card(mock_upload_dir, mock_create_card):
    with tempfile.TemporaryDirectory() as tmpdir:
        args = UploadArgs(
            kernel_dir=tmpdir,
            repo_id=REPO_ID,
            private=False,
            branch=None,
            create_card=True,
        )
        upload_kernels(args)

        mock_create_card.assert_called_once_with(args)
        assert args.card_path == str(Path(tmpdir).resolve() / "README.md")
