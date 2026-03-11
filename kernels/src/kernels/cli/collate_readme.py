from kernels._versions import _get_available_versions
from kernels.utils import _get_hf_api


def generate_main_readme(repo_id: str) -> str:
    versions = _get_available_versions(repo_id)

    if not versions:
        raise ValueError(
            f"No versions found for `{repo_id}`. "
            "Upload at least one versioned kernel before generating a main README."
        )

    kernel_name = repo_id.split("/")[-1]
    hub_url = f"https://huggingface.co/{repo_id}"

    lines = [
        "---",
        "tags:",
        "- kernels",
        "library_name: kernels",
        "---",
        "",
        f"# {kernel_name}",
        "",
        "This kernel is available in the following versions. "
        "Please refer to the version branches for details.",
        "",
        "## Available versions",
        "",
        "| Version | Branch |",
        "| ------- | ------ |",
    ]

    for version_num in sorted(versions.keys()):
        branch_name = f"v{version_num}"
        branch_url = f"{hub_url}/tree/{branch_name}"
        lines.append(f"| {version_num} | [{branch_name}]({branch_url}) |")

    lines.append("")
    lines.append("## Quick start")
    lines.append("")
    lines.append("```python")
    lines.append("from kernels import get_kernel")
    lines.append("")
    lines.append(f'kernel = get_kernel("{repo_id}", version=<version>)')
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def collate_readme_from_versions(
    repo_id: str,
    push_to_hub: bool = False,
):
    readme_content = generate_main_readme(repo_id)

    if push_to_hub:
        api = _get_hf_api()
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            revision="main",
            commit_message="Update main README with available versions.",
        )
        print(f"README pushed to https://huggingface.co/{repo_id}")
    else:
        print(readme_content)
