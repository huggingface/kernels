#!/usr/bin/env python3
"""Fetch an upstream kernel repository at a given tag.

This is the "get it" half of fetching upstream kernel sources; tag
resolution lives in `latest_tag.py` and is reused here, so `--tag latest`
resolves to the newest *stable* tag without a separate call.

The clone is shallow (`--depth 1`) and pinned to the resolved tag, and the
resolved commit SHA is reported so the exact upstream state can be recorded
in a sync commit message or PR description.

Examples:
    python tools/fetch_upstream.py https://github.com/Dao-AILab/flash-attention.git upstream/
    python tools/fetch_upstream.py https://github.com/Dao-AILab/flash-attention.git upstream/ \\
        --tag v2.8.3 --strip-git --json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from latest_tag import (
    EXIT_ERROR,
    EXIT_NO_MATCH,
    EXIT_OK,
    add_selection_args,
    latest_tag,
    list_remote_tags,
    parse_version,
)


def _run_git(args: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as err:
        raise RuntimeError("`git` was not found on PATH") from err
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(
            f"`git {' '.join(args)}` timed out after {timeout}s"
        ) from err


def _prepare_dest(dest: Path, force: bool) -> None:
    if not dest.exists():
        return
    if not dest.is_dir():
        raise RuntimeError(f"{dest} exists and is not a directory")
    if not any(dest.iterdir()):
        return
    if not force:
        raise RuntimeError(f"{dest} is not empty (pass --force to replace it)")
    shutil.rmtree(dest)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/fetch_upstream.py",
        description="Shallow-clone an upstream repository at a release tag.",
    )
    parser.add_argument("url", help="Git URL, as passed to `git clone`.")
    parser.add_argument("dest", type=Path, help="Directory to clone into.")
    parser.add_argument(
        "--tag",
        default="latest",
        help="Tag to check out, or 'latest' to resolve the newest tag (default: latest).",
    )
    add_selection_args(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination directory if it already has contents.",
    )
    parser.add_argument(
        "--strip-git",
        action="store_true",
        help="Remove the .git directory after cloning (for vendoring sources).",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON object.")
    args = parser.parse_args(argv)

    def bail(message: str, code: int) -> int:
        if args.json:
            json.dump(
                {"url": args.url, "ok": False, "error": message}, sys.stdout, indent=2
            )
            sys.stdout.write("\n")
        else:
            print(f"error: {message}", file=sys.stderr)
        return code

    prerelease = None
    try:
        if args.tag == "latest":
            resolved, candidates = latest_tag(
                args.url,
                include_prerelease=args.include_prerelease,
                pattern=args.tag_pattern,
                timeout=args.timeout,
            )
            if resolved is None:
                hint = "no tag with a parseable version matched"
                if not args.include_prerelease and candidates:
                    hint += " (try --include-prerelease)"
                return bail(hint, EXIT_NO_MATCH)
            tag = resolved.name
            prerelease = resolved.version.is_prerelease
        else:
            tag = args.tag
            known = {
                name for name, _ in list_remote_tags(args.url, timeout=args.timeout)
            }
            if tag not in known:
                return bail(f"tag {tag!r} does not exist in {args.url}", EXIT_NO_MATCH)
            version = parse_version(tag)
            prerelease = version.is_prerelease if version is not None else None

        _prepare_dest(args.dest, args.force)

        if not args.json:
            print(f"Cloning {args.url} at {tag} into {args.dest}", file=sys.stderr)
        clone = _run_git(
            [
                "clone",
                "--depth",
                "1",
                "--branch",
                tag,
                "--",
                args.url,
                str(args.dest),
            ],
            timeout=max(args.timeout, 600),
        )
        if clone.returncode != 0:
            return bail(f"clone failed: {clone.stderr.strip()}", EXIT_ERROR)

        rev = _run_git(
            ["-C", str(args.dest), "rev-parse", "HEAD"], timeout=args.timeout
        )
        sha = rev.stdout.strip() if rev.returncode == 0 else None

        if args.strip_git:
            shutil.rmtree(args.dest / ".git", ignore_errors=True)
    except RuntimeError as err:
        return bail(str(err), EXIT_ERROR)

    if args.json:
        json.dump(
            {
                "url": args.url,
                "ok": True,
                "tag": tag,
                "sha": sha,
                "prerelease": prerelease,
                "dest": str(args.dest),
                "git_stripped": args.strip_git,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(f"{tag}\t{sha}\t{args.dest}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
