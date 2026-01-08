import re
from pathlib import Path
from typing import Optional

from huggingface_hub import create_branch, create_repo, upload_folder

from kernels.metadata import Metadata

BUILD_VARIANT_REGEX = re.compile(r"^(torch\d+\d+|torch-(cpu|cuda|metal|rocm|xpu))")


def upload_kernels_dir(
    kernel_dir: Path, *, repo_id: str, branch: Optional[str], private: bool
):
    kernel_dir = Path(kernel_dir).resolve()

    build_dir = None
    variants = None
    for candidate in [kernel_dir / "build", kernel_dir]:
        variants = [
            variant_path
            for variant_path in candidate.glob("torch*")
            if BUILD_VARIANT_REGEX.match(variant_path.name) is not None
            and (variant_path / "metadata.json").is_file()
        ]
        if variants:
            build_dir = candidate
            break
    if build_dir is None:
        raise ValueError(
            f"Couldn't find any build variants in: {kernel_dir.absolute()} or {(kernel_dir / 'build').absolute()}"
        )

    print(build_dir)

    if branch is None:
        assert variants is not None
        channels = set()
        for variant in variants:
            print(variant)
            metadata = Metadata.load_from_variant(variant)
            channels.add(metadata.channel)

        if len(channels) > 1:
            raise ValueError(
                f"Found multiple channels in build variants: {', '.join(channels)}"
            )

        print(channels)

        channel = channels.pop()
        if channel is not None:
            branch = f"channel-{channel}"

    repo_id = create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

    if branch is not None:
        create_branch(repo_id=repo_id, branch=branch, exist_ok=True)

    delete_patterns: set[str] = set()
    for build_variant in build_dir.iterdir():
        if build_variant.is_dir():
            delete_patterns.add(f"{build_variant.name}/**")

    upload_folder(
        repo_id=repo_id,
        folder_path=build_dir,
        revision=branch,
        path_in_repo="build",
        delete_patterns=list(delete_patterns),
        commit_message="Build uploaded using `kernels`.",
        allow_patterns=["torch*"],
    )
    print(f"✅ Kernel upload successful. Find the kernel in https://hf.co/{repo_id}.")
