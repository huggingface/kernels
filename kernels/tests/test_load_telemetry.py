from types import SimpleNamespace

import huggingface_hub.utils
from kernels_data import KernelName

from kernels.utils import RepoInfo, _send_load_telemetry


def _stub_metadata(name="relu", version=1):
    # `name` is a real `KernelName` so `str(...)` mirrors production: the normal
    # name is reported, not the underscore-separated Python name.
    return SimpleNamespace(name=KernelName(name), version=version)


def _capture(monkeypatch):
    calls = []
    monkeypatch.setattr(
        huggingface_hub.utils,
        "send_telemetry",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def test_load_telemetry_uses_dedicated_topic(monkeypatch):
    calls = _capture(monkeypatch)

    _send_load_telemetry(_stub_metadata(), RepoInfo(repo_id="org/relu", revision="abc123"))

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("kernels/loaded",)
    assert kwargs["library_name"] == "kernels"


def test_load_telemetry_includes_repo_id_when_available(monkeypatch):
    calls = _capture(monkeypatch)

    _send_load_telemetry(_stub_metadata(version=3), RepoInfo(repo_id="org/relu", revision="abc123"))

    ua = calls[0][1]["user_agent"]
    assert ua["repo_id"] == "org/relu"
    assert ua["revision"] == "abc123"
    # Defaults to a downloaded kernel.
    assert ua["local"] == "false"
    assert ua["kernel_name"] == "relu"
    assert ua["kernel_version"] == "3"
    # System info is merged in so loads can be sliced by backend/platform/etc.
    assert "backend" in ua
    assert ua["kernels"]


def test_load_telemetry_flags_locally_loaded_kernels(monkeypatch):
    calls = _capture(monkeypatch)

    # A pre-downloaded/locked kernel is flagged so downloads can be separated
    # from local loads.
    _send_load_telemetry(_stub_metadata(), RepoInfo(repo_id="org/relu", revision="abc123", local=True))

    assert calls[0][1]["user_agent"]["local"] == "true"


def test_load_telemetry_without_repo_info_omits_repo_id(monkeypatch):
    calls = _capture(monkeypatch)

    _send_load_telemetry(_stub_metadata(), None)

    ua = calls[0][1]["user_agent"]
    assert "repo_id" not in ua
    assert "revision" not in ua
    assert "local" not in ua
    assert ua["kernel_name"] == "relu"


def test_load_telemetry_reports_normal_kernel_name(monkeypatch):
    calls = _capture(monkeypatch)

    # The normal (hyphenated) name is reported, not the Python name where
    # dashes are replaced by underscores.
    _send_load_telemetry(_stub_metadata(name="flash-attn"), None)

    assert calls[0][1]["user_agent"]["kernel_name"] == "flash-attn"


def test_load_telemetry_always_sets_load_marker(monkeypatch):
    calls = _capture(monkeypatch)

    # The marker must be present whether or not a repo id is available, so loads
    # are separable from other Hub requests by the user agent alone.
    _send_load_telemetry(_stub_metadata(), RepoInfo(repo_id="org/relu", revision="abc123"))
    _send_load_telemetry(_stub_metadata(), None)

    assert calls[0][1]["user_agent"]["kernels_event"] == "load"
    assert calls[1][1]["user_agent"]["kernels_event"] == "load"


def test_load_telemetry_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(huggingface_hub.utils, "send_telemetry", boom)

    # Must be swallowed: telemetry should never break loading a kernel.
    _send_load_telemetry(_stub_metadata(), None)
