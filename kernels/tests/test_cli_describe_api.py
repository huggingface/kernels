import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.hf_api import GitRefInfo

from kernels.cli import describe_api, main
from kernels.variants import parse_variant

REPO_ID = "kernels-community/example"


@pytest.fixture
def symbols():
    return {
        "schema_version": 1,
        "module": "_example_build",
        "functions": [
            {
                "name": "relu",
                "qualname": "relu",
                "module": "_example_build",
                "kind": "function",
                "doc": "Apply ReLU.",
                "signature": {
                    "parameters": [
                        {"name": "x", "kind": "POSITIONAL_ONLY", "default": None, "annotation": "torch.Tensor"}
                    ],
                    "return_annotation": "torch.Tensor",
                },
            }
        ],
        "layers": [
            {
                "name": "ReLU",
                "qualname": "ReLU",
                "module": "_example_build.layers",
                "kind": "class",
                "doc": None,
                "signature": None,
                "attributes": {"has_backward": False, "can_torch_compile": True},
            },
            {
                "name": "Unknown",
                "attributes": {"has_backward": None, "can_torch_compile": None},
            },
        ],
    }


@pytest.fixture
def local_kernel(tmp_path, symbols):
    variant = tmp_path / "build" / "torch-cuda"
    variant.mkdir(parents=True)
    (variant / "symbols.json").write_text(json.dumps(symbols))
    (variant / "__init__.py").write_text("raise RuntimeError('Kernel code must not be imported')")
    return tmp_path


@pytest.fixture
def hub(monkeypatch, local_kernel):
    api = Mock()
    api.list_repo_refs.return_value = SimpleNamespace(
        branches=[
            GitRefInfo(name=name, ref=f"refs/heads/{name}", target_commit="a" * 40)
            for name in ["main", "v10", "v0", "v2", "v-next"]
        ]
    )
    api.hf_hub_download.return_value = str(local_kernel / "build" / "torch-cuda" / "symbols.json")
    monkeypatch.setattr("kernels.hf_hub._get_hf_api", lambda: api)
    monkeypatch.setattr(describe_api, "_get_hf_api", lambda: api)
    variants = Mock(return_value=[parse_variant("torch-cuda")])
    monkeypatch.setattr(describe_api, "get_variants", variants)
    monkeypatch.setattr(describe_api.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv("KERNELS_CACHE", str(local_kernel / "cache"))
    return api, variants


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["kernels", "describe-api", *args])
    main()


@pytest.fixture
def missing_entry():
    response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/missing"))
    return RemoteEntryNotFoundError("Missing entry", response=response)


@pytest.mark.parametrize("suffix", ["", "build", "build/torch-cuda"])
def test_local_directories(monkeypatch, local_kernel, capsys, suffix):
    path = local_kernel / suffix
    run_cli(monkeypatch, str(path))
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == (
        f"Path: {path}\nVariant: torch-cuda\n\n"
        "Functions:\n  relu\n\n"
        "Layers:\n  ReLU  has_backward=False  can_torch_compile=True\n"
        "  Unknown  has_backward=unknown  can_torch_compile=unknown\n"
    )


def test_json_preserves_symbol_data(monkeypatch, local_kernel, symbols, capsys):
    run_cli(monkeypatch, str(local_kernel), "--json")
    assert json.loads(capsys.readouterr().out) == {**symbols, "path": str(local_kernel), "variant": "torch-cuda"}


@pytest.mark.parametrize("selection, revision", [([], "v10"), (["--version", "0"], "v0"), (["--version", "2"], "v2")])
def test_hub_version_selection(monkeypatch, hub, symbols, local_kernel, capsys, selection, revision):
    api, variants = hub
    run_cli(monkeypatch, REPO_ID, *selection, "--json")
    assert json.loads(capsys.readouterr().out) == {
        **symbols,
        "repo_id": REPO_ID,
        "revision": revision,
        "variant": "torch-cuda",
    }
    api.list_repo_refs.assert_called_once_with(repo_id=REPO_ID, repo_type="kernel")
    variants.assert_called_once_with(api, repo_id=REPO_ID, revision=revision)
    api.hf_hub_download.assert_called_once_with(
        REPO_ID,
        repo_type="kernel",
        filename="build/torch-cuda/symbols.json",
        revision=revision,
        cache_dir=str(local_kernel / "cache"),
        local_files_only=False,
    )


@pytest.mark.parametrize("revision", ["main", "my-tag", "a" * 40])
def test_hub_revision(monkeypatch, hub, capsys, revision):
    api, variants = hub
    run_cli(monkeypatch, REPO_ID, "--revision", revision)
    output = capsys.readouterr().out
    assert f"Repository: {REPO_ID}\nRevision: {revision}\n" in output
    api.list_repo_refs.assert_not_called()
    variants.assert_called_once_with(api, repo_id=REPO_ID, revision=revision)


def test_unversioned_repo(monkeypatch, hub, capsys):
    hub[0].list_repo_refs.return_value.branches = []
    run_cli(monkeypatch, REPO_ID)
    assert "Revision: main\n" in capsys.readouterr().out


def test_missing_version(monkeypatch, hub, capsys):
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, REPO_ID, "--version", "3")
    output = capsys.readouterr()
    assert output.out == ""
    assert "Version 3 not found" in output.err
    hub[1].assert_not_called()


@pytest.mark.parametrize(
    "selection, revision, context",
    [
        ([], "v10", "version: 10, revision: v10"),
        (["--version", "0"], "v0", "version: 0, revision: v0"),
        (["--version", "2"], "v2", "version: 2, revision: v2"),
        (["--revision", "my-tag"], "my-tag", "revision: my-tag"),
    ],
)
def test_hub_missing_symbols_does_not_try_another_variant(
    monkeypatch, hub, capsys, missing_entry, selection, revision, context
):
    api, variants = hub
    variants.return_value = [parse_variant("torch-xpu"), parse_variant("torch-cuda")]
    api.hf_hub_download.side_effect = [missing_entry, api.hf_hub_download.return_value]
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, REPO_ID, *selection)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith(f"Error: No symbols.json found in {REPO_ID} ({context}) (variant: torch-cuda).")
    api.hf_hub_download.assert_called_once()
    assert api.hf_hub_download.call_args.kwargs["filename"] == "build/torch-cuda/symbols.json"
    assert api.hf_hub_download.call_args.kwargs["revision"] == revision


@pytest.mark.parametrize("suffix", ["", "build"])
def test_local_missing_symbols_does_not_try_another_variant(monkeypatch, local_kernel, capsys, suffix):
    (local_kernel / "build" / "torch-cpu").mkdir()
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(local_kernel / suffix))
    output = capsys.readouterr()
    assert output.out == ""
    assert "No symbols.json found" in output.err
    assert "variant: torch-cpu" in output.err


def test_hub_errors_are_not_treated_as_missing_symbols(monkeypatch, hub, capsys):
    hub[0].hf_hub_download.side_effect = OSError("Authentication failed")
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, REPO_ID)
    assert capsys.readouterr().err == "Error: Authentication failed\n"


@pytest.mark.parametrize("selection", [["--revision", "main"], ["--version", "2"]])
def test_local_rejects_hub_options(monkeypatch, local_kernel, capsys, selection):
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(local_kernel), *selection)
    assert "cannot be used with a local path" in capsys.readouterr().err


def test_mutually_exclusive_options(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="2"):
        run_cli(monkeypatch, REPO_ID, "--revision", "main", "--version", "2")
    assert "not allowed with argument" in capsys.readouterr().err


def test_missing_local_symbols(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(tmp_path))
    assert "No symbols.json found" in capsys.readouterr().err


def test_missing_local_directory(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(tmp_path / "missing"))
    assert "Not a local kernel directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content, message",
    [
        ("{invalid", "Invalid symbols.json"),
        ("[]", "Unsupported symbols.json schema"),
        ('{"schema_version": 2}', "Unsupported symbols.json schema"),
        ('{"schema_version": 1, "functions": null, "layers": []}', "functions must be a list"),
        ('{"schema_version": 1, "functions": [{}], "layers": []}', "functions must be a list"),
        (
            '{"schema_version": 1, "functions": [], "layers": [{"name": "Layer", "attributes": []}]}',
            "layer attributes must be booleans or null",
        ),
    ],
)
def test_invalid_symbols(monkeypatch, tmp_path, capsys, content, message):
    (tmp_path / "symbols.json").write_text(content)
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(tmp_path))
    output = capsys.readouterr()
    assert output.out == ""
    assert message in output.err


def test_empty_api(monkeypatch, tmp_path, capsys):
    (tmp_path / "symbols.json").write_text(json.dumps({"schema_version": 1, "functions": [], "layers": []}))
    run_cli(monkeypatch, str(tmp_path))
    assert "Functions:\n  (none)\n\nLayers:\n  (none)\n" in capsys.readouterr().out
