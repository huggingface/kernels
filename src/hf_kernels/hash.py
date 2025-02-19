from typing import List, Tuple
import hashlib
import os
from pathlib import Path


def content_hash(dir: Path) -> str:
    """Get a hash of the contents of a directory."""

    # Get the file paths. The first element is a byte-encoded relative path
    # used for sorting. The second element is the absolute path.
    paths: List[Tuple[bytes, Path]] = []
    # Ideally we'd use Path.walk, but it's only available in Python 3.12.
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            file_abs = Path(dirpath) / filename

            # Python likes to create files when importing modules from the
            # cache, only hash files that are symlinked blobs.
            if file_abs.is_symlink():
                paths.append(
                    (file_abs.relative_to(dir).as_posix().encode("utf-8"), file_abs)
                )

    m = hashlib.sha256()
    for filename, path in sorted(paths):
        m.update(filename)
        with open(path, "rb") as f:
            m.update(f.read())

    return f"sha256-{m.hexdigest()}"
