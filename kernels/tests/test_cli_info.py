import json
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub.hf_api import GitRefInfo
from packaging.version import Version

from kernels.backends import CUDA
from kernels.cli import info as info_module
from kernels.cli import main
from kernels.cli.info import print_kernel_info
from kernels.variants import _resolve_variant_for_system, get_variants_local

REPO_ID = "kernels-community/activation"
COMMIT = "a" * 40
PREFERRED = "torch28-cxx11-cu128-x86_64-linux"


@pytest.fixture(autouse=True)
def cuda_system(monkeypatch):
    resolve = partial(
        _resolve_variant_for_system,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.8"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    monkeypatch.setattr("kernels.variants.resolve_variants", lambda variants, backend: resolve(variants=variants))


def write_variant(path, *, variant="torch-cuda", deps=(), python_deps=(), **metadata):
    variant_path = path / "build" / variant
    variant_path.mkdir(parents=True, exist_ok=True)
    (variant_path / "metadata.json").write_text(
        json.dumps(
            {
                "id": "activation_1_cuda",
                "name": "activation",
                "version": 1,
                "license": "Apache-2.0",
                "python-depends": list(python_deps),
                "kernel-depends": list(deps),
                "backend": {"type": "cuda"},
                **metadata,
            }
        )
    )
    (variant_path / "__init__.py").write_text("raise RuntimeError('Kernel code must not be imported')")
    return variant_path


@pytest.fixture
def hub(monkeypatch, tmp_path):
    def add_kernel(repo_id, **kwargs):
        path = tmp_path / repo_id
        write_variant(path, **kwargs)
        return path

    def download(repo_id, *, filename, **kwargs):
        assert filename.endswith("/metadata.json")
        return str(tmp_path / repo_id / filename)

    api = Mock()
    api.list_repo_refs.return_value = SimpleNamespace(
        branches=[GitRefInfo(name="v1", ref="refs/heads/v1", target_commit=COMMIT)]
    )
    api.repo_info.return_value = SimpleNamespace(sha=COMMIT)
    api.hf_hub_download.side_effect = download
    api.snapshot_download.side_effect = AssertionError("Kernel artifacts must not be downloaded")
    variants = Mock(side_effect=lambda api, *, repo_id, revision: get_variants_local(tmp_path / repo_id / "build"))
    monkeypatch.setattr(info_module, "_get_hf_api", lambda: api)
    monkeypatch.setattr("kernels.hf_hub._get_hf_api", lambda: api)
    monkeypatch.setattr(info_module, "get_variants", variants)
    monkeypatch.setattr("kernels.resolver.get_variants", variants)
    monkeypatch.setattr(info_module.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path / "cache"))
    add_kernel(REPO_ID)
    return SimpleNamespace(api=api, add_kernel=add_kernel, variants=variants)


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["kernels", "info", *args])
    main()


def test_info_hub(hub, capsys):
    print_kernel_info("kernels-community/activation", version=1)
    out = capsys.readouterr().out
    assert "Repository: kernels-community/activation" in out
    assert "Revision: v1" in out
    assert "Name: activation" in out
    assert "Version: 1" in out
    assert "License: Apache-2.0" in out
    assert "Backends:" in out


def test_info_hub_json(hub, capsys):
    print_kernel_info("kernels-community/activation", version=1, json_output=True)
    info = json.loads(capsys.readouterr().out)
    assert info["repo_id"] == "kernels-community/activation"
    assert info["revision"] == "v1"
    assert info["name"] == "activation"
    assert info["version"] == 1
    assert info["license"] == "Apache-2.0"
    assert "cuda" in info["backends"]
    assert info["variant"] == "torch-cuda"
    assert info["kernel_depends"] == []
    assert info["python_depends"] == []
    assert "dependency_tree" not in info


def test_info_hub_rejects_revision_and_version():
    with pytest.raises(SystemExit):
        print_kernel_info("kernels-community/activation", revision="main", version=1)


def test_info_local(tmp_path, capsys):
    variant_dir = tmp_path / "build" / "torch28-cxx11-cu128-x86_64-linux"
    variant_dir.mkdir(parents=True)
    (variant_dir / "metadata.json").write_text(
        json.dumps(
            {
                "id": "activation_1_cuda",
                "name": "activation",
                "version": 1,
                "license": "Apache-2.0",
                "upstream": "https://github.com/example/activation",
                "python-depends": ["torch"],
                "backend": {"type": "cuda"},
            }
        )
    )

    print_kernel_info(str(tmp_path), json_output=True)
    info = json.loads(capsys.readouterr().out)
    assert info["path"] == str(tmp_path)
    assert info["name"] == "activation"
    assert info["version"] == 1
    assert info["license"] == "Apache-2.0"
    assert info["upstream"] == "https://github.com/example/activation"
    assert info["source"] is None
    assert info["python_depends"] == ["torch"]
    assert info["backends"] == ["cuda"]


def test_info_local_rejects_revision(tmp_path):
    with pytest.raises(SystemExit):
        print_kernel_info(str(tmp_path), revision="main")


def test_info_local_no_variants(tmp_path):
    with pytest.raises(SystemExit):
        print_kernel_info(str(tmp_path))


def test_info_local_sdk_variants(tmp_path, capsys):
    for variant in ["sdk", "sdk-cuda", "sdk-cpu-cpython-3.9-abi3-x86_64-linux", "sdk-cpython-3.9-abi3-x86_64-linux"]:
        variant_dir = tmp_path / "build" / variant
        variant_dir.mkdir(parents=True)
        (variant_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": "sdk_1",
                    "name": "sdk",
                    "version": 1,
                    "license": "Apache-2.0",
                    "python-depends": [],
                    "backend": {"type": "cuda"},
                }
            )
        )

    print_kernel_info(str(tmp_path), json_output=True)
    info = json.loads(capsys.readouterr().out)
    assert sorted(info["backends"]) == ["cpu", "cuda"]


@pytest.fixture
def dependency_graph(hub):
    # A diamond with duplicate Python packages exercises recursive resolution
    # and deduplication without losing edges in the tree output.
    root = hub.add_kernel(
        REPO_ID,
        deps=[{"repo-id": "test/c", "version": 1}, {"repo-id": "test/b", "version": 1}],
        python_deps=["torch", "numpy"],
    )
    hub.add_kernel("test/b", deps=[{"repo-id": "test/d", "revision": "main"}], python_deps=["numpy", "einops"])
    hub.add_kernel("test/c", deps=[{"repo-id": "test/d", "revision": "main"}], python_deps=["torch"])
    hub.add_kernel("test/d", python_deps=["packaging", "numpy"])
    return root


@pytest.mark.parametrize("source", ["hub", "local"])
@pytest.mark.parametrize("dep_tree", [False, True])
def test_info_recursive_dependencies(monkeypatch, hub, dependency_graph, capsys, source, dep_tree):
    kernel = REPO_ID if source == "hub" else str(dependency_graph)
    run_cli(monkeypatch, kernel, "--json", *(["--dep-tree"] if dep_tree else []))
    info = json.loads(capsys.readouterr().out)
    assert info["kernel_depends"] == ["test/b", "test/c", "test/d"]
    assert info["python_depends"] == ["einops", "numpy", "packaging", "torch"]
    if dep_tree:
        assert info["dependency_tree"] == {"test/b": {"test/d": {}}, "test/c": {"test/d": {}}}
    else:
        assert "dependency_tree" not in info
    hub.api.snapshot_download.assert_not_called()
    hub.api.get_organization_overview.assert_not_called()
    for download in hub.api.hf_hub_download.call_args_list:
        assert download.kwargs["filename"] == "build/torch-cuda/metadata.json"
        if download.args[0] != REPO_ID:
            assert download.kwargs["revision"] == COMMIT
    hub.api.repo_info.assert_any_call(repo_id="test/d", repo_type="kernel", revision="main")


@pytest.mark.parametrize("source", ["hub", "local"])
def test_info_dependency_tree(monkeypatch, hub, dependency_graph, capsys, source):
    kernel = REPO_ID if source == "hub" else str(dependency_graph)
    run_cli(monkeypatch, kernel, "--dep-tree")
    out = capsys.readouterr().out
    assert (f"Kernel dependencies:\n{kernel}\n|-- test/b\n|   `-- test/d\n`-- test/c\n    `-- test/d\n") in out
    assert "Python dependencies: einops, numpy, packaging, torch\n" in out


def test_info_flat_dependencies(hub, dependency_graph, capsys):
    print_kernel_info(REPO_ID)
    out = capsys.readouterr().out
    assert "Kernel dependencies: test/b, test/c, test/d\n" in out
    assert "Python dependencies: einops, numpy, packaging, torch\n" in out


def test_info_empty_dependencies(monkeypatch, hub, capsys):
    run_cli(monkeypatch, REPO_ID)
    out = capsys.readouterr().out
    assert "Kernel dependencies: -\n" in out
    assert "Python dependencies: -\n" in out

    run_cli(monkeypatch, REPO_ID, "--dep-tree", "--json")
    assert json.loads(capsys.readouterr().out)["dependency_tree"] == {}


@pytest.mark.parametrize("source", ["hub", "local"])
def test_info_selects_matching_variant(hub, capsys, source):
    root = hub.add_kernel(REPO_ID, variant="torch-cpu", python_deps=["wrong-backend"])
    hub.add_kernel(REPO_ID, variant="torch-cuda", python_deps=["less-preferred"])
    hub.add_kernel(
        REPO_ID,
        variant=PREFERRED,
        deps=[{"repo-id": "test/b", "version": 1}],
        python_deps=["torch"],
    )
    hub.add_kernel("test/b", variant="torch-cpu", python_deps=["wrong-dependency-backend"])
    hub.add_kernel("test/b", variant="torch-cuda", python_deps=["numpy"])

    print_kernel_info(REPO_ID if source == "hub" else str(root), json_output=True)
    info = json.loads(capsys.readouterr().out)
    assert info["variant"] == PREFERRED
    assert info["backends"] == ["cpu", "cuda"]
    assert info["kernel_depends"] == ["test/b"]
    assert info["python_depends"] == ["numpy", "torch"]
    for download in hub.api.hf_hub_download.call_args_list:
        expected = PREFERRED if download.args[0] == REPO_ID else "torch-cuda"
        assert download.kwargs["filename"] == f"build/{expected}/metadata.json"


def test_info_cycle_is_an_error(hub, capsys):
    hub.add_kernel(REPO_ID, deps=[{"repo-id": "test/b", "version": 1}])
    hub.add_kernel("test/b", deps=[{"repo-id": REPO_ID, "version": 1}])
    with pytest.raises(SystemExit, match="1"):
        print_kernel_info(REPO_ID)
    output = capsys.readouterr()
    assert "Cyclic kernel dependency detected" in output.err
    assert output.out == ""


@pytest.mark.parametrize("source", ["hub", "local"])
def test_info_no_matching_variant(hub, capsys, source):
    root = hub.add_kernel("test/cpu-only", variant="torch-cpu")
    with pytest.raises(SystemExit, match="1"):
        print_kernel_info("test/cpu-only" if source == "hub" else str(root))
    output = capsys.readouterr()
    assert "Cannot find a build variant for this system" in output.err
    assert "torch-cpu" in output.err
    assert output.out == ""
    hub.api.hf_hub_download.assert_not_called()


def test_info_missing_dependency_is_an_error(hub, capsys):
    hub.add_kernel(REPO_ID, deps=[{"repo-id": "test/missing", "version": 1}])
    with pytest.raises(SystemExit, match="1"):
        print_kernel_info(REPO_ID, json_output=True)
    output = capsys.readouterr()
    assert "Cannot find a build variant for this system in test/missing" in output.err
    assert output.out == ""


def test_info_skips_metadata_validation(hub, capsys):
    hub.add_kernel(REPO_ID, deps=[{"repo-id": "test/b", "version": 1}], **{"kernels-minver": "999.0.0"})
    hub.add_kernel("test/b", python_deps=["uninstalled-package"], **{"kernels-minver": "999.0.0"})
    print_kernel_info(REPO_ID, json_output=True)
    info = json.loads(capsys.readouterr().out)
    assert info["python_depends"] == ["uninstalled-package"]
    assert info["kernels_minver"] == "999.0.0"


def test_info_offline_dependencies(monkeypatch, hub, dependency_graph, capsys):
    print_kernel_info(str(dependency_graph), json_output=True)
    online = json.loads(capsys.readouterr().out)
    hub.api.reset_mock()
    hub.variants.reset_mock()
    monkeypatch.setattr(info_module.constants, "HF_HUB_OFFLINE", True)

    def cached_snapshot(repo_id, *, local_files_only, **kwargs):
        assert local_files_only
        return str(dependency_graph.parent.parent / repo_id)

    hub.api.snapshot_download.side_effect = cached_snapshot
    print_kernel_info(str(dependency_graph), json_output=True)
    assert json.loads(capsys.readouterr().out) == online
    hub.api.hf_hub_download.assert_not_called()
    hub.api.repo_info.assert_not_called()
    hub.api.list_repo_refs.assert_not_called()
    hub.variants.assert_not_called()


@pytest.mark.parametrize("revision", [None, "custom"])
def test_info_unversioned_hub(hub, capsys, revision):
    hub.api.list_repo_refs.return_value = SimpleNamespace(branches=[])
    print_kernel_info(REPO_ID, revision=revision, json_output=True)
    info = json.loads(capsys.readouterr().out)
    assert info["revision"] == (revision or "main")
    assert hub.api.hf_hub_download.call_args.kwargs["revision"] == (revision or "main")
