import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id, repo_type, revision, local_dir = sys.argv[1:]

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
    local_dir=local_dir,
)

# Downloading leaves per-file metadata (etags, timestamps) behind, which would
# make the hash of the snapshot unstable.
shutil.rmtree(Path(local_dir) / ".cache", ignore_errors=True)
