#!/usr/bin/env python3
"""Resolve the latest release tag of an upstream Git repository.

This is the "which tag?" half of fetching upstream kernel sources; see
`fetch_upstream.py` for the "get it" half, which imports this module.

Tags are read with `git ls-remote`, so nothing is cloned and no working
copy is needed. Version numbers are extracted from the tag name, which
handles both plain tags (`v2.8.3`) and the prefixed tags upstream kernels
tend to use (`fa4-v4.0.0.beta8`, `release-1.2`).

Pre-releases (`rc`, `beta`, `dev`, ...) are excluded unless
`--include-prerelease` is passed, so the default answer is the latest
*stable* tag.

Examples:
    python tools/latest_tag.py https://github.com/Dao-AILab/flash-attention.git
    python tools/latest_tag.py https://github.com/Dao-AILab/flash-attention.git \\
        --tag-pattern '^fa4-' --include-prerelease --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

EXIT_OK = 0
EXIT_NO_MATCH = 1
EXIT_ERROR = 2

# A version anywhere inside a tag name, with an optional PEP 440-ish
# pre-release and post-release suffix.
_VERSION_RE = re.compile(
    r"(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-._]?(?P<pre_label>alpha|beta|preview|pre|dev|rc|a|b|c)[-._]?(?P<pre_num>\d+)?)?"
    r"(?:[-._]?(?P<post_label>post|rev|r)[-._]?(?P<post_num>\d+)?)?",
    re.IGNORECASE,
)

# Ordering of pre-release kinds, following PEP 440: dev < alpha < beta < rc.
_PRE_ORDER = {
    "dev": 0,
    "alpha": 1,
    "a": 1,
    "beta": 2,
    "b": 2,
    "c": 3,
    "rc": 3,
    "pre": 3,
    "preview": 3,
}

_RELEASE_PAD = 6


@dataclass(frozen=True)
class Version:
    release: Tuple[int, ...]
    pre: Optional[Tuple[str, int]]
    post: Optional[int]
    text: str

    @property
    def is_prerelease(self) -> bool:
        return self.pre is not None

    def sort_key(self) -> tuple:
        release = self.release + (0,) * (_RELEASE_PAD - len(self.release))
        if self.pre is None:
            # A final release outranks every pre-release of the same number.
            pre_key = (1, 0, 0)
        else:
            label, number = self.pre
            pre_key = (0, _PRE_ORDER.get(label, 3), number)
        return (release[:_RELEASE_PAD], pre_key, self.post or 0)


@dataclass(frozen=True)
class Tag:
    name: str
    sha: str
    version: Version

    def to_json(self) -> dict:
        return {
            "tag": self.name,
            "sha": self.sha,
            "version": self.version.text,
            "prerelease": self.version.is_prerelease,
        }


def parse_version(tag: str) -> Optional[Version]:
    """Extract a version from a tag name.

    When a tag contains several digit runs (`fa4-v4.0.0.beta8`), the
    candidate with the most dotted components wins, and ties are broken by
    taking the right-most one. That picks `4.0.0.beta8` rather than the `4`
    that is part of the `fa4` prefix.
    """
    best: Optional[Version] = None
    best_score: Optional[tuple] = None
    for match in _VERSION_RE.finditer(tag):
        release = tuple(int(part) for part in match.group("release").split("."))
        pre_label = match.group("pre_label")
        pre: Optional[Tuple[str, int]] = None
        if pre_label is not None:
            pre = (pre_label.lower(), int(match.group("pre_num") or 0))
        post: Optional[int] = None
        if match.group("post_label") is not None:
            post = int(match.group("post_num") or 0)

        candidate = Version(release=release, pre=pre, post=post, text=match.group(0))
        score = (len(release), match.start())
        if best_score is None or score > best_score:
            best, best_score = candidate, score
    return best


def list_remote_tags(url: str, *, timeout: int = 60) -> List[Tuple[str, str]]:
    """Return `(tag, sha)` pairs for `url` without cloning it.

    Raises:
        RuntimeError: if `git ls-remote` is unavailable or fails.
    """
    env = dict(os.environ)
    # Never block on a credential prompt: an unreachable or private repo
    # should fail fast rather than hang an agent's tool call.
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        completed = subprocess.run(
            ["git", "ls-remote", "--tags", "--refs", url],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as err:
        raise RuntimeError("`git` was not found on PATH") from err
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(f"`git ls-remote {url}` timed out after {timeout}s") from err

    if completed.returncode != 0:
        raise RuntimeError(f"`git ls-remote {url}` failed: {completed.stderr.strip()}")

    tags = []
    for line in completed.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if not ref.startswith("refs/tags/"):
            continue
        tags.append((ref[len("refs/tags/") :], sha))
    return tags


def select_tags(
    tags: Sequence[Tuple[str, str]],
    *,
    include_prerelease: bool = False,
    pattern: Optional[str] = None,
) -> List[Tag]:
    """Filter and version-sort `tags`, newest first."""
    compiled = re.compile(pattern) if pattern else None
    selected: List[Tag] = []
    for name, sha in tags:
        if compiled is not None and not compiled.search(name):
            continue
        version = parse_version(name)
        if version is None:
            continue
        if version.is_prerelease and not include_prerelease:
            continue
        selected.append(Tag(name=name, sha=sha, version=version))
    selected.sort(key=lambda tag: tag.version.sort_key(), reverse=True)
    return selected


def latest_tag(
    url: str,
    *,
    include_prerelease: bool = False,
    pattern: Optional[str] = None,
    timeout: int = 60,
) -> Tuple[Optional[Tag], List[Tag]]:
    """Return the newest tag of `url` plus the full sorted candidate list."""
    candidates = select_tags(
        list_remote_tags(url, timeout=timeout),
        include_prerelease=include_prerelease,
        pattern=pattern,
    )
    return (candidates[0] if candidates else None), candidates


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Register the tag-selection flags shared with `fetch_upstream.py`."""
    parser.add_argument(
        "--tag-pattern",
        metavar="REGEX",
        help="Only consider tags matching this regular expression (e.g. '^fa4-').",
    )
    parser.add_argument(
        "--include-prerelease",
        action="store_true",
        help="Also consider pre-release tags (rc, beta, dev, ...).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for git commands (default: 60).",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/latest_tag.py",
        description="Print the latest release tag of a Git repository.",
    )
    parser.add_argument("url", help="Git URL, as passed to `git clone`.")
    add_selection_args(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of candidate tags to report with --json (default: 10).",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON object.")
    args = parser.parse_args(argv)

    try:
        newest, candidates = latest_tag(
            args.url,
            include_prerelease=args.include_prerelease,
            pattern=args.tag_pattern,
            timeout=args.timeout,
        )
    except RuntimeError as err:
        if args.json:
            json.dump(
                {"url": args.url, "ok": False, "error": str(err)}, sys.stdout, indent=2
            )
            sys.stdout.write("\n")
        else:
            print(f"error: {err}", file=sys.stderr)
        return EXIT_ERROR

    if newest is None:
        message = "no tag with a parseable version matched"
        if args.json:
            json.dump(
                {"url": args.url, "ok": False, "tag": None, "error": message},
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
        else:
            print(f"error: {message}", file=sys.stderr)
        return EXIT_NO_MATCH

    if args.json:
        payload = {"url": args.url, "ok": True, **newest.to_json()}
        payload["candidates"] = [tag.to_json() for tag in candidates[: args.limit]]
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        # Bare tag name on stdout so it composes in a shell.
        print(newest.name)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
