import json
import logging
import sys
from functools import partial
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import tomlkit
from huggingface_hub.hf_api import GitRefInfo
from packaging.version import Version

from kernels.backends import CPU
from kernels.cli import main
from kernels.cli import variants as variants_cli
from kernels.variants import _resolve_variant_for_system, parse_variant

REPO_ID = "kernels-community/example"
PREFERRED = "torch210-cpu-x86_64-linux"
COMPATIBLE = "torch-cpu"
INCOMPATIBLE = "torch210-metal-aarch64-darwin"


@pytest.fixture
def hub(monkeypatch):
    versions = {v: GitRefInfo(name=f"v{v}", ref=f"refs/heads/v{v}", target_commit=str(v) * 40) for v in [0, 1, 2]}
    available_versions = Mock(return_value=versions)
    get_variants = Mock(return_value=[parse_variant(v) for v in [PREFERRED, COMPATIBLE, INCOMPATIBLE]])
    api = Mock()
    monkeypatch.setattr(variants_cli, "_get_hf_api", lambda: api)
    monkeypatch.setattr(variants_cli, "_get_available_versions", available_versions)
    monkeypatch.setattr(variants_cli, "get_variants", get_variants)
    resolve = partial(
        _resolve_variant_for_system,
        selected_backend=CPU(),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    monkeypatch.setattr(variants_cli, "resolve_variants", lambda variants, backend: resolve(variants=variants))
    return api, available_versions, get_variants


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["kernels", *args])
    main()


@pytest.mark.parametrize("selection", [[], ["--version", "latest"]])
def test_latest_version(monkeypatch, hub, capsys, selection):
    api, available_versions, get_variants = hub
    run_cli(monkeypatch, "variants", REPO_ID, *selection)
    output = capsys.readouterr()
    assert output.err == ""
    assert "Version 2:" in output.out
    assert "Version 1:" not in output.out
    assert f"{PREFERRED} compatible, preferred ✅" in output.out
    assert f"{COMPATIBLE} compatible ✅" in output.out
    assert f"{INCOMPATIBLE}:" in output.out
    available_versions.assert_called_once_with(REPO_ID, local_files_only=variants_cli.constants.HF_HUB_OFFLINE)
    get_variants.assert_called_once_with(api, repo_id=REPO_ID, revision="refs/heads/v2")


def test_all_versions(monkeypatch, hub, capsys):
    api, _, get_variants = hub
    run_cli(monkeypatch, "variants", REPO_ID, "--all-versions")
    output = capsys.readouterr().out
    assert output.index("Version 0:") < output.index("Version 1:") < output.index("Version 2:")
    assert get_variants.call_args_list == [call(api, repo_id=REPO_ID, revision=f"refs/heads/v{v}") for v in [0, 1, 2]]


@pytest.mark.parametrize("version", [0, 1, 2])
def test_specific_version(monkeypatch, hub, capsys, version):
    api, _, get_variants = hub
    run_cli(monkeypatch, "variants", REPO_ID, "--version", str(version))
    assert capsys.readouterr().out.startswith(f"Version {version}:\n")
    get_variants.assert_called_once_with(api, repo_id=REPO_ID, revision=f"refs/heads/v{version}")


@pytest.mark.parametrize("revision", ["main", "my-tag", "abc123" * 6 + "abcd"])
def test_specific_revision(monkeypatch, hub, capsys, revision):
    api, available_versions, get_variants = hub
    run_cli(monkeypatch, "variants", REPO_ID, "--revision", revision)
    assert capsys.readouterr().out.startswith(f"Revision {revision}:\n")
    available_versions.assert_not_called()
    get_variants.assert_called_once_with(api, repo_id=REPO_ID, revision=revision)


@pytest.mark.parametrize(
    "selection", [[], ["--all-versions"], ["--version", "2"], ["--version", "latest"], ["--revision", "main"]]
)
def test_only_compatible(monkeypatch, hub, capsys, selection):
    run_cli(monkeypatch, "variants", REPO_ID, "--only-compatible", *selection)
    output = capsys.readouterr().out
    assert f"{PREFERRED} compatible, preferred ✅" in output
    assert f"{COMPATIBLE} compatible ✅" in output
    assert INCOMPATIBLE not in output


def test_no_compatible_variants(monkeypatch, hub, capsys):
    _, _, get_variants = hub
    get_variants.return_value = [parse_variant(INCOMPATIBLE)]
    run_cli(monkeypatch, "variants", REPO_ID, "--only-compatible")
    assert capsys.readouterr().out == "Version 2:\n\nNo compatible variants found.\n"
    assert get_variants.call_count == 1  # Do not fall back to an older version.


def test_no_build_variants(monkeypatch, hub, capsys):
    hub[2].return_value = []
    run_cli(monkeypatch, "variants", REPO_ID)
    assert capsys.readouterr().out == "Version 2:\n\nNo build variants found.\n"


@pytest.mark.parametrize(
    "selection, expected_versions",
    [
        ([], [2]),
        (["--version", "latest"], [2]),
        (["--version", "0"], [0]),
        (["--all-versions"], [0, 1, 2]),
        (["--revision", "main"], [None]),
    ],
)
@pytest.mark.parametrize("only_compatible", [False, True])
def test_json(monkeypatch, hub, capsys, selection, expected_versions, only_compatible):
    api, available_versions, get_variants = hub
    filtering = ["--only-compatible"] if only_compatible else []
    run_cli(monkeypatch, "variants", REPO_ID, "--json", *selection, *filtering)
    output = capsys.readouterr()
    assert output.err == ""
    expected_variants = [
        {"variant": PREFERRED, "compatible": True, "preferred": True, "reason": None},
        {"variant": COMPATIBLE, "compatible": True, "preferred": False, "reason": None},
    ]
    if not only_compatible:
        expected_variants.append(
            {
                "variant": INCOMPATIBLE,
                "compatible": False,
                "preferred": False,
                "reason": "CPU (aarch64) does not match system CPU (x86_64)",
            }
        )
    expected_refs = [f"refs/heads/v{v}" if v is not None else "main" for v in expected_versions]
    assert json.loads(output.out) == {
        "repo_id": REPO_ID,
        "revisions": [
            {"version": v, "revision": ref, "variants": expected_variants}
            for v, ref in zip(expected_versions, expected_refs)
        ],
    }
    assert get_variants.call_args_list == [call(api, repo_id=REPO_ID, revision=ref) for ref in expected_refs]
    if expected_versions == [None]:
        available_versions.assert_not_called()


@pytest.mark.parametrize("variants", [[], [INCOMPATIBLE]])
@pytest.mark.parametrize("only_compatible", [False, True])
def test_json_no_compatible_variants(monkeypatch, hub, capsys, variants, only_compatible):
    hub[2].return_value = [parse_variant(v) for v in variants]
    filtering = ["--only-compatible"] if only_compatible else []
    run_cli(monkeypatch, "variants", REPO_ID, "--json", *filtering)
    result = json.loads(capsys.readouterr().out)
    assert len(result["revisions"]) == 1
    revision = result["revisions"][0]
    assert revision["version"] == 2
    if only_compatible or not variants:
        assert revision["variants"] == []
    else:
        assert revision["variants"] == [
            {
                "variant": INCOMPATIBLE,
                "compatible": False,
                "preferred": False,
                "reason": "CPU (aarch64) does not match system CPU (x86_64)",
            }
        ]
    assert hub[2].call_count == 1


@pytest.mark.parametrize("json_args", [[], ["--json"]])
def test_no_versions(monkeypatch, hub, capsys, json_args):
    hub[1].return_value = {}
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "variants", REPO_ID, *json_args)
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == f"Repository does not support kernel versions: {REPO_ID}\n"


@pytest.mark.parametrize("json_args", [[], ["--json"]])
def test_missing_version(monkeypatch, hub, capsys, json_args):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "variants", REPO_ID, "--version", "3", *json_args)
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Version 3 not found, available versions: 0, 1, 2" in output.err
    hub[2].assert_not_called()


@pytest.mark.parametrize(
    "selection",
    [
        ["--version", "2", "--revision", "main"],
        ["--version", "latest", "--revision", "main"],
        ["--all-versions", "--version", "2"],
        ["--all-versions", "--version", "latest"],
        ["--all-versions", "--revision", "main"],
        ["--version", "invalid"],
    ],
)
def test_invalid_selection(monkeypatch, hub, capsys, selection):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "variants", REPO_ID, *selection)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    hub[1].assert_not_called()
    hub[2].assert_not_called()


def test_versions_deprecated(monkeypatch, hub, capsys, caplog):
    project = tomlkit.parse((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert Version(str(project["project"]["version"])) < Version("0.20.0.dev0"), (
        "The deprecation cycle has ended: remove the `kernels versions` command, its implementation, and this test."
    )
    with caplog.at_level(logging.WARNING, logger="kernels.cli.versions"):
        run_cli(monkeypatch, "versions", REPO_ID)
    legacy = capsys.readouterr()
    assert caplog.record_tuples == [
        (
            "kernels.cli.versions",
            logging.WARNING,
            "`kernels versions` is deprecated and will be removed in kernels 0.20. "
            "Use `kernels variants --all-versions` instead.",
        )
    ]
    caplog.clear()
    run_cli(monkeypatch, "variants", REPO_ID, "--all-versions")
    replacement = capsys.readouterr()
    assert legacy.out == replacement.out
    assert replacement.err == ""
    assert caplog.records == []
