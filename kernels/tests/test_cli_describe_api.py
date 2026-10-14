import json
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.hf_api import GitRefInfo
from packaging.version import Version

from kernels.backends import CUDA
from kernels.cli import describe_api, main
from kernels.variants import _resolve_variant_for_system, parse_variant

REPO_ID = "kernels-community/example"
PREFERRED = "torch210-cu128-x86_64-linux"
COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def cuda_system(monkeypatch):
    resolve = partial(
        _resolve_variant_for_system,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    monkeypatch.setattr("kernels.variants.resolve_variants", lambda variants, backend: resolve(variants=variants))


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


def write_metadata(variant):
    variant.mkdir(parents=True, exist_ok=True)
    (variant / "metadata.json").write_text(
        json.dumps(
            {
                "name": "example",
                "id": "example_build",
                "version": 1,
                "license": "Apache-2.0",
                "python-depends": [],
                "backend": {"type": "cuda"},
            }
        )
    )


@pytest.fixture
def local_kernel(tmp_path, symbols):
    variant = tmp_path / "build" / "torch-cuda"
    write_metadata(variant)
    (variant / "symbols.json").write_text(json.dumps(symbols))
    (variant / "__init__.py").write_text("raise RuntimeError('Kernel code must not be imported')")
    return tmp_path


@pytest.fixture
def hub(monkeypatch, local_kernel):
    api = Mock()
    api.get_organization_overview.return_value = SimpleNamespace(trustedKernelPublisher=True)
    api.list_repo_refs.return_value = SimpleNamespace(
        branches=[
            GitRefInfo(name=name, ref=f"refs/heads/{name}", target_commit=COMMIT)
            for name in ["main", "v10", "v0", "v2", "v-next"]
        ]
    )
    api.repo_info.return_value = SimpleNamespace(sha=COMMIT)
    api.hf_hub_download.side_effect = lambda repo_id, filename, **kwargs: str(local_kernel / filename)
    api.snapshot_download.return_value = str(local_kernel)
    monkeypatch.setattr("kernels.hf_hub._get_hf_api", lambda: api)
    monkeypatch.setattr(describe_api, "_get_hf_api", lambda: api)
    variants = Mock(return_value=[parse_variant("torch-cuda")])
    monkeypatch.setattr("kernels.resolver.get_variants", variants)
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


@pytest.mark.parametrize("source", ["local", "hub"])
@pytest.mark.parametrize(
    "variant_names, expected",
    [
        (["torch-cpu", "torch-cuda"], "torch-cuda"),
        (
            ["torch-cuda", "torch210-cu128-aarch64-linux", "torch29-cu128-x86_64-linux", PREFERRED],
            PREFERRED,
        ),
        (["torch-cuda", "torch210-cu126-x86_64-linux", PREFERRED], PREFERRED),
    ],
)
def test_selects_system_variant(monkeypatch, local_kernel, hub, symbols, capsys, source, variant_names, expected):
    for name in variant_names:
        variant_path = local_kernel / "build" / name
        write_metadata(variant_path)
        (variant_path / "symbols.json").write_text(json.dumps({**symbols, "module": name}))
        (variant_path / "__init__.py").write_text("raise RuntimeError('Kernel code must not be imported')")

    api, variants = hub
    variants.return_value = [parse_variant(name) for name in variant_names]
    run_cli(monkeypatch, str(local_kernel) if source == "local" else REPO_ID, "--json")
    info = json.loads(capsys.readouterr().out)
    assert info["variant"] == expected
    assert info["module"] == expected
    if source == "hub":
        assert [download.kwargs["filename"] for download in api.hf_hub_download.call_args_list] == [
            f"build/{expected}/metadata.json",
            f"build/{expected}/symbols.json",
        ]
    else:
        api.hf_hub_download.assert_not_called()
    api.snapshot_download.assert_not_called()


def test_explicit_local_variant(monkeypatch, tmp_path, symbols, capsys):
    variant_path = tmp_path / "torch-cpu"
    write_metadata(variant_path)
    (variant_path / "symbols.json").write_text(json.dumps(symbols))
    run_cli(monkeypatch, str(variant_path), "--json")
    assert json.loads(capsys.readouterr().out)["variant"] == "torch-cpu"


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
    api.list_repo_refs.assert_called_with(repo_id=REPO_ID, repo_type="kernel")
    api.get_organization_overview.assert_called_once_with("kernels-community")
    variants.assert_called_once_with(api, repo_id=REPO_ID, revision=COMMIT)
    assert api.hf_hub_download.call_args_list == [
        call(
            REPO_ID,
            repo_type="kernel",
            filename=f"build/torch-cuda/{filename}",
            revision=COMMIT,
            cache_dir=str(local_kernel / "cache"),
            local_files_only=False,
        )
        for filename in ["metadata.json", "symbols.json"]
    ]
    api.snapshot_download.assert_not_called()


@pytest.mark.parametrize("revision", ["main", "my-tag", "a" * 40])
def test_hub_revision(monkeypatch, hub, capsys, revision):
    api, variants = hub
    run_cli(monkeypatch, REPO_ID, "--revision", revision)
    output = capsys.readouterr().out
    assert f"Repository: {REPO_ID}\nRevision: {revision}\n" in output
    api.list_repo_refs.assert_not_called()
    variants.assert_called_once_with(api, repo_id=REPO_ID, revision=COMMIT)


@pytest.mark.parametrize("trust_status", ["untrusted", "missing", "unverified"])
def test_hub_enforces_publisher_trust(monkeypatch, hub, capsys, trust_status):
    api, variants = hub
    if trust_status == "unverified":
        api.get_organization_overview.side_effect = OSError("Cannot check publisher")
        expected_error = "could not verify publisher trust status"
    else:
        api.get_organization_overview.return_value = (
            SimpleNamespace(trustedKernelPublisher=False) if trust_status == "untrusted" else SimpleNamespace()
        )
        expected_error = "not from a trusted publisher"

    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, REPO_ID, "--version", "2")
    output = capsys.readouterr()
    assert output.out == ""
    assert f"Cannot resolve {REPO_ID} (version: 2, revision: v2)" in output.err
    assert expected_error in output.err
    api.get_organization_overview.assert_called_once_with("kernels-community")
    variants.assert_not_called()
    api.hf_hub_download.assert_not_called()
    api.snapshot_download.assert_not_called()


def test_hub_user_can_trust_remote_code(monkeypatch, hub, capsys):
    api, _ = hub
    api.get_organization_overview.return_value = SimpleNamespace(trustedKernelPublisher=False)
    run_cli(monkeypatch, REPO_ID, "--version", "2", "--trust-remote-code", "--json")
    info = json.loads(capsys.readouterr().out)
    assert info["repo_id"] == REPO_ID
    assert info["revision"] == "v2"
    assert info["functions"][0]["name"] == "relu"
    api.get_organization_overview.assert_not_called()
    assert api.hf_hub_download.call_count == 2
    api.snapshot_download.assert_not_called()


@pytest.mark.parametrize("trust_remote_code", [False, True])
@pytest.mark.parametrize("selection", [[], ["--version", "2"], ["--revision", "my-tag"]])
def test_hub_uses_cached_resolver_offline(monkeypatch, hub, local_kernel, capsys, selection, trust_remote_code):
    api, variants = hub
    if trust_remote_code:
        selection = [*selection, "--trust-remote-code"]
    run_cli(monkeypatch, REPO_ID, *selection, "--json")
    expected = json.loads(capsys.readouterr().out)

    api.reset_mock()
    variants.reset_mock()
    monkeypatch.setattr(describe_api.constants, "HF_HUB_OFFLINE", True)
    run_cli(monkeypatch, REPO_ID, *selection, "--json")
    assert json.loads(capsys.readouterr().out) == expected
    api.list_repo_refs.assert_not_called()
    api.repo_info.assert_not_called()
    api.get_organization_overview.assert_not_called()
    api.hf_hub_download.assert_not_called()
    variants.assert_not_called()
    api.snapshot_download.assert_called_once_with(
        REPO_ID,
        repo_type="kernel",
        ignore_patterns=["*.pyc", "**/__pycache__/**"],
        cache_dir=str(local_kernel / "cache"),
        revision=COMMIT,
        local_files_only=True,
    )


def test_unversioned_repo_requires_revision(monkeypatch, hub, capsys):
    api, variants = hub
    api.list_repo_refs.return_value.branches = [GitRefInfo(name="main", ref="refs/heads/main", target_commit=COMMIT)]
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, REPO_ID)
    output = capsys.readouterr()
    assert output.out == ""
    assert f"No numbered versions found for {REPO_ID}. Specify --revision explicitly." in output.err
    api.repo_info.assert_not_called()
    variants.assert_not_called()
    api.hf_hub_download.assert_not_called()
    api.snapshot_download.assert_not_called()


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
    monkeypatch, hub, local_kernel, missing_entry, capsys, selection, revision, context
):
    api, variants = hub
    variants.return_value = [parse_variant("torch-cuda"), parse_variant(PREFERRED)]
    write_metadata(local_kernel / "build" / PREFERRED)

    def download(repo_id, filename, **kwargs):
        path = local_kernel / filename
        if not path.is_file():
            raise missing_entry
        return str(path)

    api.hf_hub_download.side_effect = download
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, REPO_ID, *selection)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith(f"Error: No symbols.json found in {REPO_ID} ({context}) (variant: {PREFERRED}).")
    assert [
        (download.kwargs["filename"], download.kwargs["revision"]) for download in api.hf_hub_download.call_args_list
    ] == [(f"build/{PREFERRED}/{filename}", COMMIT) for filename in ["metadata.json", "symbols.json"]]
    api.snapshot_download.assert_not_called()


@pytest.mark.parametrize("suffix", ["", "build"])
def test_local_missing_symbols_does_not_try_another_variant(monkeypatch, local_kernel, capsys, suffix):
    write_metadata(local_kernel / "build" / PREFERRED)
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(local_kernel / suffix))
    output = capsys.readouterr()
    assert output.out == ""
    assert "No symbols.json found" in output.err
    assert f"variant: {PREFERRED}" in output.err


def test_hub_errors_are_not_treated_as_missing_symbols(monkeypatch, hub, capsys):
    hub[0].hf_hub_download.side_effect = OSError("Authentication failed")
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, REPO_ID)
    error = capsys.readouterr().err
    assert f"Cannot resolve {REPO_ID} (version: 10, revision: v10)" in error
    assert "Authentication failed" in error
    assert "No symbols.json found" not in error


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
    write_metadata(tmp_path / "build" / "torch-cuda")
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(tmp_path))
    assert "No symbols.json found" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["local", "hub", "hub-missing-build"])
def test_no_build_variants(monkeypatch, tmp_path, hub, missing_entry, capsys, source):
    api, variants = hub
    if source == "hub-missing-build":
        variants.side_effect = missing_entry
    else:
        variants.return_value = []
    path = tmp_path / "empty"
    path.mkdir()
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(path) if source == "local" else REPO_ID)
    output = capsys.readouterr()
    assert output.out == ""
    expected_error = (
        "Missing entry" if source == "hub-missing-build" else "Cannot find a build variant for this system"
    )
    assert expected_error in output.err
    api.hf_hub_download.assert_not_called()
    api.snapshot_download.assert_not_called()
    if source != "local":
        assert f"{REPO_ID} (version: 10, revision: v10)" in output.err


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
    write_metadata(tmp_path)
    (tmp_path / "symbols.json").write_text(content)
    with pytest.raises(SystemExit, match="1"):
        run_cli(monkeypatch, str(tmp_path))
    output = capsys.readouterr()
    assert output.out == ""
    assert message in output.err


def test_empty_api(monkeypatch, tmp_path, capsys):
    write_metadata(tmp_path)
    (tmp_path / "symbols.json").write_text(json.dumps({"schema_version": 1, "functions": [], "layers": []}))
    run_cli(monkeypatch, str(tmp_path))
    assert "Functions:\n  (none)\n\nLayers:\n  (none)\n" in capsys.readouterr().out
