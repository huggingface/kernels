from pathlib import Path

from huggingface_hub import create_branch, create_repo, upload_folder

from kernels.metadata import Metadata
from kernels.variants import BUILD_VARIANT_REGEX


def upload_kernels_dir(
    kernel_dir: Path,
    *,
    repo_id: str,
    branch: str | None,
    private: bool,
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

    if branch is None:
        assert variants is not None
        versions = set()
        for variant in variants:
            metadata = Metadata.load_from_variant(variant)
            versions.add(metadata.version)

        if len(versions) > 1:
            raise ValueError(
                f"Found multiple versions in build variants: {', '.join(str(version) for version in versions)}"
            )

        version = versions.pop()
        if version is not None:
            branch = f"v{version}"

    repo_id = create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

    if branch is not None:
        create_branch(repo_id=repo_id, branch=branch, exist_ok=True)

    delete_patterns: set[str] = set()
    for build_variant in build_dir.iterdir():
        if build_variant.is_dir():
            delete_patterns.add(f"{build_variant.name}/**")

    # in the case we have variants, upload to the same as the kernel_dir
    if (kernel_dir / "benchmarks").is_dir():
        upload_folder(
            repo_id=repo_id,
            folder_path=kernel_dir / "benchmarks",
            revision=branch,
            path_in_repo="benchmarks",
            delete_patterns=["benchmark*.py"],
            commit_message="Benchmarks uploaded using `kernels`.",
            allow_patterns=["benchmark*.py"],
        )

    upload_folder(
        repo_id=repo_id,
        folder_path=build_dir,
        revision=branch,
        path_in_repo="build",
        delete_patterns=list(delete_patterns),
        commit_message="Build uploaded using `kernels`.",
        allow_patterns=["torch*"],
    )
    print(f"✅ Kernel upload successful. Find the kernel in: https://hf.co/{repo_id}")
