import platform

import pytest
from huggingface_hub import constants
from huggingface_hub.utils import build_hf_headers

from kernels.hf_hub import _get_hf_api, _platform


@pytest.fixture(autouse=True)
def telemetry_enabled(monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", False)


def test_user_agent_contains_core_fields():
    api = _get_hf_api()
    ua = api.user_agent

    assert "kernels/" in ua
    python_ver = ".".join(platform.python_version_tuple()[:2])
    assert f"python/{python_ver}" in ua
    assert "backend/" in ua
    assert f"platform/{_platform()}" in ua
    assert "file_type/kernel" in ua


def test_user_agent_with_string():
    api = _get_hf_api(user_agent="my-app/1.0")
    ua = api.user_agent

    assert "my-app/1.0" in ua
    assert "kernels/" in ua


def test_user_agent_with_dict():
    api = _get_hf_api(user_agent={"app": "test", "ver": "2.0"})
    ua = api.user_agent

    assert "app/test" in ua
    assert "ver/2.0" in ua
    assert "kernels/" in ua


def test_user_agent_telemetry_disabled(monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", True)
    api = _get_hf_api(user_agent="should-not-appear")
    assert api.user_agent is None


def test_user_agent_includes_torch():
    api = _get_hf_api()
    ua = api.user_agent

    # If torch is importable, it should appear in the UA.
    try:
        import torch

        assert f"torch/{torch.__version__}" in ua
    except ImportError:
        assert "torch/" not in ua


def test_user_agent_format_structure():
    api = _get_hf_api()
    ua = api.user_agent

    fields = [f.strip() for f in ua.split(";") if f.strip()]
    # Each field should be key/value
    for field in fields:
        assert "/" in field, f"Field missing key/value separator: {field}"


def test_platform_format():
    plat = _platform()
    assert "-" in plat
    parts = plat.split("-")
    assert len(parts) == 2
    assert parts[1] in ("linux", "darwin", "windows")


@pytest.mark.parametrize("disable_telemetry", [False, True])
def test_http_user_agent_header_is_well_formed(monkeypatch, disable_telemetry):
    # Ensure that the user agent string is well-formed, both with and wthout
    # telemetry enabled.
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", disable_telemetry)

    api = _get_hf_api()
    header = build_hf_headers(
        library_name=api.library_name,
        library_version=api.library_version,
        user_agent=api.user_agent,
    )["user-agent"]

    assert header == header.strip(), f"header has surrounding whitespace: {header!r}"
    for segment in header.split(";"):
        assert segment.strip() != "", f"empty segment found in header: {header!r}"


def test_user_agent_no_leading_or_empty_segment():
    # Regression: when no caller-supplied user_agent is passed, the resulting
    # string must not start with a separator and must not contain empty
    # segments. Empty segments downstream produce malformed User-Agent headers
    # (e.g. trailing "; ") which strict HTTP clients reject.
    for ua_input in (None, "", {}):
        api = _get_hf_api(user_agent=ua_input)
        ua = api.user_agent
        assert not ua.startswith(";"), f"user_agent must not start with ';': {ua!r}"
        for segment in ua.split(";"):
            assert segment.strip() != "", f"empty segment found in user_agent: {ua!r}"
