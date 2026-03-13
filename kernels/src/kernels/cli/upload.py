from itertools import chain
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.utils import chunk_iterable

from kernels.metadata import Metadata
from kernels.utils import _get_hf_api
from kernels.variants import BUILD_VARIANT_REGEX

BUILD_COMMIT_BATCH_SIZE = 1_000


def _branch_exists(api, repo_id, branch):
    refs = api.list_repo_refs(repo_id=repo_id)
    return any(ref.name == branch for ref in refs.branches)


def _upload_build_dir(
    api,
    *,
    repo_id: str,
    revision: str | None,
    build_dir: Path,
    variants: list[Path],
    is_new_branch: bool,
):
    repo_paths = {}
    for variant in variants:
        for path in sorted(variant.rglob("*")):
            if path.is_file():
                repo_paths[f"build/{path.relative_to(build_dir).as_posix()}"] = path

    variant_prefixes = tuple(
        f"build/{variant.relative_to(build_dir).as_posix()}/" for variant in variants
    )
    delete_prefixes = ("build/",) if is_new_branch else variant_prefixes
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationDelete(path_in_repo=repo_file)
        for repo_file in sorted(
            api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
        )
        if repo_file.startswith(delete_prefixes) and repo_file not in repo_paths
    ]
    operations.extend(
        CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
        for repo_path, local_path in sorted(repo_paths.items())
    )

    if not operations:
        return

    batch_count = (
        len(operations) + BUILD_COMMIT_BATCH_SIZE - 1
    ) // BUILD_COMMIT_BATCH_SIZE
    if batch_count > 1:
        print(
            f"⚠️  Found {len(operations)} build operations, uploading in {batch_count} commits."
        )

    for batch_index, chunk in enumerate(
        chunk_iterable(operations, chunk_size=BUILD_COMMIT_BATCH_SIZE), start=1
    ):
        commit_message = "Build uploaded using `kernels`."
        if batch_count > 1:
            commit_message = (
                f"Build uploaded using `kernels` (batch {batch_index}/{batch_count})."
            )
        api.create_commit(
            repo_id=repo_id,
            operations=list(chunk),
            revision=revision,
            repo_type="model",
            commit_message=commit_message,
        )


def upload_kernels_dir(
    kernel_dir: Path,
    *,
    repo_id: str,
    branch: str | None,
    private: bool,
):
    api = _get_hf_api()
    kernel_dir = Path(kernel_dir).resolve()

    build_dir = None
    variants = None
    for candidate in [kernel_dir / "build", kernel_dir]:
        variants = [
            variant_path
            for variant_path in chain(
                candidate.glob("torch*"), candidate.glob("tvm-ffi*")
            )
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

    repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

    is_new_branch = False
    if branch is not None:
        is_new_branch = not _branch_exists(api, repo_id, branch)
        api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)

    # In the case we have benchmarks, upload to the same repo as the kernel_dir.
    if (kernel_dir / "benchmarks").is_dir():
        benchmark_delete = ["**"] if is_new_branch else ["benchmark*.py"]
        api.upload_folder(
            repo_id=repo_id,
            folder_path=kernel_dir / "benchmarks",
            revision=branch,
            path_in_repo="benchmarks",
            delete_patterns=benchmark_delete,
            commit_message="Benchmarks uploaded using `kernels`.",
            allow_patterns=["benchmark*.py"],
        )

    card_path = kernel_dir / "build" / "CARD.md"
    if (card_path).exists():
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            revision=branch,
            commit_message="File uploaded using `kernels`.",
        )

    assert variants is not None
    _upload_build_dir(
        api,
        repo_id=repo_id,
        revision=branch,
        build_dir=build_dir,
        variants=variants,
        is_new_branch=is_new_branch,
    )
    print(f"✅ Kernel upload successful. Find the kernel in: https://hf.co/{repo_id}")
