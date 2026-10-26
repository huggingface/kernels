import logging

from kernels.cli.variants import print_kernel_variants

logger = logging.getLogger(__name__)


def print_kernel_versions(repo_id: str):
    logger.warning(
        "`kernels versions` is deprecated and will be removed in kernels 0.20. "
        "Use `kernels variants --all-versions` instead.",
        stacklevel=2,
    )
    print_kernel_variants(repo_id, all_versions=True)
