from kernels._versions import _get_available_versions
from kernels.utils import _get_hf_api


def get_latest_version_readme(repo_id: str) -> str:
    versions = _get_available_versions(repo_id)

    if not versions:
        raise ValueError(
            f"No versions found for `{repo_id}`. "
            "Upload at least one versioned kernel before generating a main README."
        )

    latest_version = max(versions.keys())
    branch = f"v{latest_version}"

    api = _get_hf_api()
    readme_content = api.hf_hub_download(
        repo_id=repo_id,
        filename="README.md",
        revision=branch,
    )

    with open(readme_content) as f:
        return f.read()


def collate_readme_from_versions(
    repo_id: str,
    push_to_hub: bool = False,
):
    """Copy the most recent version's README to the main branch."""
    readme_content = get_latest_version_readme(repo_id)

    if push_to_hub:
        api = _get_hf_api()
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            revision="main",
            commit_message="Update main README from latest version.",
        )
        print(f"README pushed to https://huggingface.co/{repo_id}")
    else:
        print(readme_content)
