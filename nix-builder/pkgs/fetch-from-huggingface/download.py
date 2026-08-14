import sys

from huggingface_hub import snapshot_download

repo_id, repo_type, revision, local_dir = sys.argv[1:]

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
    local_dir=local_dir,
)
