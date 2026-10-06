import sys
from unittest.mock import Mock, call

import httpx
import pytest
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.hf_api import GitRefInfo

from kernels.cli import describe as describe_cli
from kernels.cli import main
from kernels.variants import parse_variant

REPO_ID = "kernels-community/example"
VARIANT = "torch210-cxx11-cu128-x86_64-linux"
INIT = """
raise RuntimeError("Describing must not execute kernel code")
from missing_dependency import dependency
from . import layers

def hidden(): pass
def public(): pass
async def async_public(): pass
class NotAFunction: pass

__all__ = ["public", "layers", "NotAFunction"]
__all__ += ["async_public"]
"""
LAYERS = """
raise RuntimeError("Describing must not execute layer code")
class Hidden: pass
class Trainable:
    has_backward: bool = True
    can_torch_compile = False
class Inference:
    has_backward = False
    can_torch_compile: bool = True
class Unknown:
    can_torch_compile = compute_flag()
__all__ = ("Trainable", "Inference", "Unknown")
"""


def write_file(root, filename, content):
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["kernels", "describe", *map(str, args)])
    main()


@pytest.fixture
def hub(monkeypatch, tmp_path):
    versions = {v: GitRefInfo(name=f"v{v}", ref=f"refs/heads/v{v}", target_commit=str(v) * 40) for v in (1, 2)}
    available = Mock(return_value=versions)
    variants = Mock(return_value=[parse_variant(VARIANT)])
    write_file(tmp_path, f"build/{VARIANT}/__init__.py", INIT)
    write_file(tmp_path, f"build/{VARIANT}/layers.py", LAYERS)

    def download(repo_id, *, filename, **kwargs):
        path = tmp_path / filename
        if not path.is_file():
            raise RemoteEntryNotFoundError(
                filename, response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))
            )
        return str(path)

    # A strict API mock makes any snapshot/binary download fail the test.
    api = Mock(spec=["hf_hub_download"])
    api.hf_hub_download.side_effect = download
    monkeypatch.setattr(describe_cli, "_get_hf_api", lambda: api)
    monkeypatch.setattr(describe_cli, "_get_available_versions", available)
    monkeypatch.setattr(describe_cli, "get_variants", variants)
    return api, available, variants, tmp_path


def assert_api_output(output):
    assert "Functions:\n  public\n  async_public\n" in output
    assert "Trainable  has_backward=True  can_torch_compile=False" in output
    assert "Inference  has_backward=False  can_torch_compile=True" in output
    assert "Unknown  has_backward=unknown  can_torch_compile=unknown" in output
    for name in ("hidden", "Hidden", "NotAFunction", "dependency"):
        assert name not in output


@pytest.mark.parametrize("selection", [[], ["--version", "latest"], ["--version", "2"], ["--version", "1"]])
def test_hub_versions(monkeypatch, hub, capsys, selection):
    api, available, variants, _ = hub
    revision = "v1" if selection == ["--version", "1"] else "v2"
    run_cli(monkeypatch, REPO_ID, *selection)
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.startswith(f"Repository: {REPO_ID}\nRevision: {revision}\n")
    assert_api_output(output.out)
    available.assert_called_once_with(REPO_ID, local_files_only=describe_cli.constants.HF_HUB_OFFLINE)
    variants.assert_called_once_with(api, repo_id=REPO_ID, revision=revision)
    assert api.hf_hub_download.call_args_list == [
        call(
            REPO_ID,
            repo_type="kernel",
            filename=f"build/{VARIANT}/{filename}",
            revision=revision,
            cache_dir=describe_cli._get_cache_dir(),
            local_files_only=describe_cli.constants.HF_HUB_OFFLINE,
        )
        for filename in ("__init__.py", "layers/__init__.py", "layers.py")
    ]


@pytest.mark.parametrize("revision", ["main", "my-tag", "a" * 40])
def test_hub_revision(monkeypatch, hub, capsys, revision):
    run_cli(monkeypatch, REPO_ID, "--revision", revision)
    assert f"Revision: {revision}" in capsys.readouterr().out
    hub[1].assert_not_called()
    hub[2].assert_called_once_with(hub[0], repo_id=REPO_ID, revision=revision)


@pytest.mark.parametrize("selection", [["--version", "3"], []])
def test_missing_version_or_variants(monkeypatch, hub, capsys, selection):
    if not selection:
        hub[2].return_value = []
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, REPO_ID, *selection)
    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert ("Version 3 not found" if selection else "No build variants found") in output.err
    hub[0].hf_hub_download.assert_not_called()


@pytest.mark.parametrize("selection", [["--version", "1", "--revision", "main"], ["--version", "invalid"]])
def test_invalid_selection(monkeypatch, hub, capsys, selection):
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, REPO_ID, *selection)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    hub[1].assert_not_called()
    hub[2].assert_not_called()


@pytest.mark.parametrize("layout", ["flat", "nested", "package", "torch-ext", "tvm-ffi-ext"])
@pytest.mark.parametrize("layers_file", ["layers.py", "layers/__init__.py"])
def test_local(monkeypatch, tmp_path, capsys, layout, layers_file):
    if layout in ("torch-ext", "tvm-ffi-ext"):
        write_file(
            tmp_path,
            "build.toml",
            '[general]\nname = "test-kernel"\nversion = 1\nlicense = "MIT"\nbackends = ["cpu"]\n[torch]\n',
        )
        package = tmp_path / layout / "test_kernel"
    elif layout == "package":
        package = tmp_path
    else:
        package = tmp_path / "build" / VARIANT
        if layout == "nested":
            write_file(
                package,
                "metadata.json",
                '{"name": "test-kernel", "id": "test", "version": 1, '
                '"license": "MIT", "python-depends": [], "backend": {"type": "cuda"}}',
            )
            package /= "test_kernel"
    write_file(package, "__init__.py", INIT)
    write_file(package, layers_file, LAYERS)
    api = Mock(side_effect=AssertionError("Local inspection must not contact the Hub"))
    monkeypatch.setattr(describe_cli, "_get_hf_api", api)
    run_cli(monkeypatch, tmp_path)
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.startswith(f"Path: {tmp_path}\n")
    assert_api_output(output.out)
    api.assert_not_called()


@pytest.mark.parametrize("selection", [["--version", "1"], ["--version", "latest"], ["--revision", "main"]])
def test_local_rejects_selection(monkeypatch, tmp_path, capsys, selection):
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, tmp_path, *selection)
    assert error.value.code == 1
    assert "cannot be used with a local path" in capsys.readouterr().err


def test_no_exports(monkeypatch, tmp_path, capsys):
    write_file(tmp_path, "__init__.py", "def hidden(): pass")
    write_file(tmp_path, "layers.py", "class Hidden: pass")
    run_cli(monkeypatch, tmp_path)
    output = capsys.readouterr()
    assert output.err == ""
    assert "No functions declared in __all__." in output.out
    assert "No layers declared in __all__." in output.out
    assert "Hidden" not in output.out


def test_missing_layers(monkeypatch, tmp_path, capsys):
    write_file(tmp_path, "__init__.py", INIT)
    run_cli(monkeypatch, tmp_path)
    output = capsys.readouterr()
    assert "  public\n" in output.out
    assert "No layers declared in __all__." in output.out
    assert output.err == ""


def test_relative_reexports_and_private_symbols(monkeypatch, tmp_path, capsys):
    write_file(
        tmp_path,
        "__init__.py",
        "from .functions import impl as public\nfrom ._private_for_testing import secret\n"
        'alias = public\n__all__ = ["public", "alias", "secret"]',
    )
    write_file(tmp_path, "functions.py", "def impl(): pass\ndef hidden(): pass")
    write_file(tmp_path, "_private_for_testing.py", "def secret(): pass")
    write_file(tmp_path, "layers/__init__.py", 'from .impl import Layer as PublicLayer\n__all__ = ["PublicLayer"]')
    write_file(tmp_path, "layers/impl.py", "from ..implementation import Layer")
    write_file(tmp_path, "implementation.py", "class Layer:\n    has_backward = False\n    can_torch_compile = True")
    run_cli(monkeypatch, tmp_path)
    output = capsys.readouterr()
    assert "Functions:\n  public\n  alias\n" in output.out
    assert "PublicLayer  has_backward=False  can_torch_compile=True" in output.out
    assert "secret" not in output.out


def test_download_errors_are_not_hidden(monkeypatch, hub, capsys):
    hub[0].hf_hub_download.side_effect = OSError("download failed")
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, REPO_ID)
    assert error.value.code == 1
    assert "download failed" in capsys.readouterr().err


def test_hub_fetches_only_reexport_source(monkeypatch, hub, capsys):
    api, _, _, root = hub
    variant = root / "build" / VARIANT
    write_file(
        variant,
        "__init__.py",
        "from .functions import impl as public\nfrom .functions import impl as alias\n"
        'from .unused import hidden\n__all__ = ["public", "alias"]',
    )
    write_file(variant, "functions.py", "def impl(): pass")
    run_cli(monkeypatch, REPO_ID)
    output = capsys.readouterr()
    assert "Functions:\n  public\n  alias\n" in output.out
    assert output.err == ""
    filenames = [entry.kwargs["filename"] for entry in api.hf_hub_download.call_args_list]
    assert filenames.count(f"build/{VARIANT}/functions.py") == 1
    assert all("unused" not in filename for filename in filenames)


def test_direct_call_rejects_conflicting_selection():
    with pytest.raises(SystemExit) as error:
        describe_cli.print_kernel_description(REPO_ID, version=1, revision="main")
    assert error.value.code == 1
