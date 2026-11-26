"""Submit the benchmark CLI to Hugging Face Jobs."""

import argparse
import shlex
from pathlib import Path

from huggingface_hub.errors import HfHubHTTPError, HFValidationError, LocalTokenNotFoundError
from huggingface_hub.utils import validate_repo_id
from packaging.version import Version

from kernels import __version__
from kernels._internals.cli.benchmark_compare import DEFAULT_THRESHOLD
from kernels._internals.hf_hub import _get_hf_api


def submit_benchmark_jobs(args: argparse.Namespace) -> None:
    if any(not flavor.strip() for flavor in args.flavor):
        raise SystemExit("--flavor requires a hardware flavor from `hf jobs hardware`.")
    if Path(args.repo_id).exists() or args.repo_id.startswith(("./", "../", "/", "~")):
        raise SystemExit("--flavor requires a Hub kernel repository ID. Omit --flavor to benchmark a local checkout.")
    if args.output is not None:
        raise SystemExit("--output writes a local file. Use --json to include JSON Lines in the job logs.")
    if args.baseline is not None or args.fail_on_regression or args.threshold != DEFAULT_THRESHOLD:
        raise SystemExit("Baseline comparisons require local files and cannot be used with --flavor.")
    if args.iterations <= 0 or args.warmup < 0:
        raise SystemExit("Benchmarks require a positive --iterations count and a non-negative --warmup count.")

    repo_id = args.repo_id
    revision_args = []
    if args.version is not None:
        revision_args = ["--version", str(args.version)]
    elif args.branch:
        revision_args = [f"--branch={args.branch}"]
    else:
        repo_id, _, revision = repo_id.partition("@")
        if not revision or revision.isdigit():
            raise SystemExit("Specify a branch or version with --branch, --version, @branch, or @vN.")
    try:
        validate_repo_id(repo_id)
    except HFValidationError as error:
        raise SystemExit(str(error)) from None

    # Development checkouts can contain benchmark features that have yet to reach PyPI.
    requirement = (
        "kernels[benchmark] @ git+https://github.com/huggingface/kernels.git@main#subdirectory=kernels"
        if Version(__version__).is_devrelease
        else f"kernels[benchmark]=={__version__}"
    )
    command = [
        "sh",
        "-c",
        f'uv pip install --system --torch-backend=auto {shlex.quote(requirement)} && exec kernels benchmark "$@"',
        "benchmark",
        args.repo_id,
        *revision_args,
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
    ]
    if args.json:
        command.append("--json")
    if args.show_stats:
        command.append("--show-stats")

    api = _get_hf_api()
    for flavor in dict.fromkeys(args.flavor):
        try:
            job = api.run_job(image="ghcr.io/astral-sh/uv:python3.12-bookworm", command=command, flavor=flavor)
        except (HfHubHTTPError, LocalTokenNotFoundError) as error:
            raise SystemExit(f"Could not submit benchmark on {flavor}: {error}") from None
        print(f"{flavor}: {job.url}", flush=True)
