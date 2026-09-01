import pytest

from kernels_data import Version


def test_version_parse():
    assert str(Version.from_str("12.8.0", 3)) == "12.8.0"
    assert str(Version.from_str("1", 1)) == "1"
    assert str(Version.from_str("1.2.3", 3)) == "1.2.3"


def test_version_ordering_and_hash():
    v1 = Version.from_str("1.2.0", 3)
    v2 = Version.from_str("1.2.0", 3)
    v3 = Version.from_str("1.3.0", 3)
    assert v1 == v2
    assert v1 < v3
    assert hash(v1) == hash(v2)
    assert {v1, v2, v3} == {v1, v3}


def test_version_invalid():
    with pytest.raises(ValueError):
        Version.from_str("abc", 1)
    with pytest.raises(ValueError):
        Version.from_str("", 1)


def test_version_must_have_expected_components():
    with pytest.raises(ValueError, match="has 2 components, expected 3"):
        Version.from_str("1.2", 3)
    with pytest.raises(ValueError, match="has 3 components, expected 2"):
        Version.from_str("1.2.0", 2)
