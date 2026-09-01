import pytest

from kernels_data import Version


def test_version_parse():
    assert str(Version.from_str("12.8.0")) == "12.8.0"
    assert str(Version.from_str("1")) == "1"
    assert str(Version.from_str("1.2.3")) == "1.2.3"


def test_version_ordering_and_hash():
    v1 = Version.from_str("1.2")
    v2 = Version.from_str("1.2.0")
    v3 = Version.from_str("1.3")
    assert v1 == v2
    assert v1 < v3
    assert hash(v1) == hash(v2)
    assert {v1, v2, v3} == {v1, v3}


def test_version_trailing_zeros_compare_equal():
    assert Version.from_str("0.17") == Version.from_str("0.17.0")
    assert Version.from_str("0.17") < Version.from_str("0.17.1")
    assert Version.from_str("0.17.0.1") > Version.from_str("0.17")


def test_version_invalid():
    with pytest.raises(ValueError):
        Version.from_str("abc")
    with pytest.raises(ValueError):
        Version.from_str("")
