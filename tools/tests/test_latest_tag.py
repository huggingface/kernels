import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latest_tag import parse_version, select_tags  # noqa: E402


def _sha(name: str) -> str:
    return f"{abs(hash(name)):040x}"[:40]


def tags(*names):
    return [(name, _sha(name)) for name in names]


@pytest.mark.parametrize(
    "tag,expected,prerelease",
    [
        ("v2.8.3", (2, 8, 3), False),
        ("2.8.3", (2, 8, 3), False),
        ("release-1.2", (1, 2), False),
        ("v0.1.5", (0, 1, 5), False),
        ("v1.0.0rc1", (1, 0, 0), True),
        ("v1.0.0-rc.2", (1, 0, 0), True),
        ("v4.0.0.beta8", (4, 0, 0), True),
        # The `fa4` prefix must not be mistaken for the version.
        ("fa4-v4.0.0.beta8", (4, 0, 0), True),
        ("v1.2.3.post1", (1, 2, 3), False),
    ],
)
def test_parse_version(tag, expected, prerelease):
    version = parse_version(tag)
    assert version is not None
    assert version.release == expected
    assert version.is_prerelease is prerelease


def test_parse_version_rejects_tags_without_numbers():
    assert parse_version("latest") is None


def test_stable_tags_sort_newest_first():
    selected = select_tags(tags("v0.9.0", "v0.10.0", "v0.10.1", "v0.2.0"))
    assert [tag.name for tag in selected] == ["v0.10.1", "v0.10.0", "v0.9.0", "v0.2.0"]


def test_prereleases_are_excluded_by_default():
    selected = select_tags(tags("v1.0.0", "v1.1.0rc1", "v1.1.0.beta2"))
    assert [tag.name for tag in selected] == ["v1.0.0"]


def test_prereleases_sort_below_their_release():
    selected = select_tags(
        tags("v1.1.0", "v1.1.0rc1", "v1.1.0b1", "v1.1.0a1", "v1.1.0.dev3"),
        include_prerelease=True,
    )
    assert [tag.name for tag in selected] == [
        "v1.1.0",
        "v1.1.0rc1",
        "v1.1.0b1",
        "v1.1.0a1",
        "v1.1.0.dev3",
    ]


def test_shorter_release_compares_as_zero_padded():
    selected = select_tags(tags("v1.2", "v1.2.1"))
    assert [tag.name for tag in selected] == ["v1.2.1", "v1.2"]


def test_tag_pattern_filters_candidates():
    selected = select_tags(
        tags("fa4-v4.0.0.beta8", "v2.8.3", "fa4-v4.0.0.beta7"),
        include_prerelease=True,
        pattern="^fa4-",
    )
    assert [tag.name for tag in selected] == ["fa4-v4.0.0.beta8", "fa4-v4.0.0.beta7"]


def test_post_release_outranks_release():
    selected = select_tags(tags("v1.2.3", "v1.2.3.post1"))
    assert selected[0].name == "v1.2.3.post1"
