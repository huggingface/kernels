import pytest

from kernels_data import Digest, DigestAlgorithm, DigestValidationError, DigestViolation


def test_hash_variant_hashes_file_with_sha256(tmp_path):
    (tmp_path / "_extension.so").write_bytes(b"hello world")

    digest = Digest.hash_variant(DigestAlgorithm.SHA256, tmp_path)

    assert len(digest.files) == 1
    assert (
        digest.files["_extension.so"] == "uU0nuZNNPgilLlLX2n2r+sSE7+N6U4DukIj3rOLvzek="
    )


def test_hash_variant_hashes_file_with_sha512(tmp_path):
    (tmp_path / "_extension.so").write_bytes(b"hello world")

    digest = Digest.hash_variant(DigestAlgorithm.SHA512, tmp_path)

    assert len(digest.files) == 1
    assert (
        digest.files["_extension.so"]
        == "MJ7MSJwS1utMxA9QyQLytNDtd+5RGnx6m808qG1M2G+YndNbxf9JlnDaNCVbRbDP2DDoH2Bdz33FVC6TrpzXbw=="
    )


def test_hash_variant_skips_excluded_files(tmp_path):
    (tmp_path / "_extension.so").write_bytes(b"content")
    (tmp_path / "metadata.json").write_bytes(b"{}")
    (tmp_path / "metadata.json.sigstore").write_bytes(b"sig")
    (tmp_path / "cache.pyc").write_bytes(b"bytecode")

    digest = Digest.hash_variant(DigestAlgorithm.SHA256, tmp_path)

    assert len(digest.files) == 1
    assert "_extension.so" in digest.files
    assert "metadata.json" not in digest.files
    assert "metadata.json.sigstore" not in digest.files
    assert "cache.pyc" not in digest.files


def test_hash_variant_uses_forward_slash_paths(tmp_path):
    subdir = tmp_path / "some" / "subdir"
    subdir.mkdir(parents=True)
    (subdir / "__init__.py").write_bytes(b"data")

    digest = Digest.hash_variant(DigestAlgorithm.SHA256, tmp_path)

    assert "some/subdir/__init__.py" in digest.files


def test_hash_variant_returns_empty_map_for_empty_dir(tmp_path):
    digest = Digest.hash_variant(DigestAlgorithm.SHA256, tmp_path)

    assert digest.files == {}


def test_hash_variant_produces_deterministic_results(tmp_path):
    (tmp_path / "_extension.so").write_bytes(b"hello world")

    digest1 = Digest.hash_variant(DigestAlgorithm.SHA256, tmp_path)
    digest2 = Digest.hash_variant(DigestAlgorithm.SHA256, tmp_path)

    assert digest1.files == digest2.files


def test_validate_returns_no_violations_for_identical_digests(tmp_path):
    (tmp_path / "_extension.so").write_bytes(b"hello world")

    expected = Digest.hash_variant(DigestAlgorithm.SHA256, tmp_path)
    actual = Digest.hash_variant(DigestAlgorithm.SHA256, tmp_path)

    expected.validate(actual)


@pytest.fixture
def mismatched_digests(tmp_path):
    expected_dir = tmp_path / "expected"
    actual_dir = tmp_path / "actual"
    expected_dir.mkdir()
    actual_dir.mkdir()

    # Present in both, but with different contents -> hash mismatch.
    (expected_dir / "shared.so").write_bytes(b"original")
    (actual_dir / "shared.so").write_bytes(b"tampered")

    # Present only in the expected digest -> missing file.
    (expected_dir / "gone.py").write_bytes(b"data")

    # Present only in the actual digest -> unknown file.
    (actual_dir / "extra.bin").write_bytes(b"data")

    expected = Digest.hash_variant(DigestAlgorithm.SHA256, expected_dir)
    actual = Digest.hash_variant(DigestAlgorithm.SHA256, actual_dir)
    return expected, actual


def test_validate_raises_with_structured_violations(mismatched_digests):
    expected, actual = mismatched_digests

    with pytest.raises(DigestValidationError) as exc_info:
        expected.validate(actual)

    violations = exc_info.value.violations
    assert len(violations) == 3

    by_path = {v.path: v for v in violations}

    assert isinstance(by_path["gone.py"], DigestViolation.MissingFile)
    assert isinstance(by_path["extra.bin"], DigestViolation.UnknownFile)

    mismatch = by_path["shared.so"]
    assert isinstance(mismatch, DigestViolation.HashMismatch)
    assert mismatch.expected != mismatch.got


def test_validate_violations_support_pattern_matching(mismatched_digests):
    expected, actual = mismatched_digests

    with pytest.raises(DigestValidationError) as exc_info:
        expected.validate(actual)

    violations = exc_info.value.violations
    assert len(violations) == 3

    for violation in violations:
        match violation:
            case DigestViolation.MissingFile(path=path):
                assert path == "gone.py"
            case DigestViolation.UnknownFile(path=path):
                assert path == "extra.bin"
            case DigestViolation.HashMismatch(path=path):
                assert path == "shared.so"
            case _:
                pytest.fail(f"unexpected violation: {violation!r}")


def test_validate_error_str_lists_every_violation(mismatched_digests):
    expected, actual = mismatched_digests

    with pytest.raises(DigestValidationError) as exc_info:
        expected.validate(actual)

    message = str(exc_info.value)
    assert "Missing file: gone.py" in message
    assert "Unknown file: extra.bin" in message
    assert "File hash mismatch for shared.so" in message

    # `str(violation)` reuses the same human-readable rendering.
    rendered = {str(v) for v in exc_info.value.violations}
    assert "Missing file: gone.py" in rendered
    assert "Unknown file: extra.bin" in rendered
