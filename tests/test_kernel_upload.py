from kernels.cli import upload_kernels
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from kernels.utils import _get_filenames_from_a_repo
import re

# TODO: host this somewhere else.
REPO_ID = "sayakpaul/kernels-upload-test"

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
        upload_kernels(UploadArgs(f"{tmpdir}/build", REPO_ID, False))

    repo_filenames = _get_filenames_from_a_repo(REPO_ID)
    assert any(str(changed_filename) in k for k in repo_filenames), f"{repo_filenames=}"
