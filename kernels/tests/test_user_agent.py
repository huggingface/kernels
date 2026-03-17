import platform

from kernels.utils import _get_hf_api, _platform


def test_user_agent_contains_core_fields():
    api = _get_hf_api()
    ua = api.user_agent

    assert "kernels/" in ua
    python_ver = ".".join(platform.python_version_tuple()[:2])
    assert f"python/{python_ver}" in ua
    assert "backend/" in ua
    assert f"flatform/{_platform()}" in ua
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
    assert api.user_agent == ""


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
