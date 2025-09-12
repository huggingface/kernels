import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from huggingface_hub import model_info

from kernels.cli import upload_kernels
from kernels.utils import _get_filenames_from_a_repo

REPO_ID = "kernels-test/kernels-upload-test"

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


def _get_filenames_from_a_repo(repo_id: str) -> List[str]:
    try:
        repo_info = model_info(repo_id=repo_id, files_metadata=True)
        repo_siblings = repo_info.siblings
        if repo_siblings is not None:
            return [f.rfilename for f in repo_siblings]
        else:
            raise ValueError("No repo siblings found.")
    except Exception as e:
        logging.error(f"Error connecting to the Hub: {e}.")


def test_kernel_upload_deletes_as_expected():
    repo_filenames = _get_filenames_from_a_repo(REPO_ID)
    filename_to_change = get_filename_to_change(repo_filenames)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/build/torch-universal/upload_test"
        build_dir = Path(path)
        build_dir.mkdir(parents=True, exist_ok=True)
        changed_filename = next_filename(Path(filename_to_change))
        script_path = build_dir / changed_filename
        script_path.write_text(PY_CONTENT)
        upload_kernels(UploadArgs(tmpdir, REPO_ID, False))

    repo_filenames = _get_filenames_from_a_repo(REPO_ID)
    assert any(str(changed_filename) in k for k in repo_filenames), f"{repo_filenames=}"
    assert not any(
        str(filename_to_change) in k for k in repo_filenames
    ), f"{repo_filenames=}"
