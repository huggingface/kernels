import hashlib
import json
import os
import socket
import subprocess
import sys

import pytest
from huggingface_hub import scan_cache_dir

from kernels.cli import cache as cache_cli
from kernels.cli import main

REPO_ID = "kernels-community/example"
COMMIT = "a" * 40
CUDA_128 = "torch28-cxx11-cu128-x86_64-linux"
CUDA_126 = "torch28-cxx11-cu126-x86_64-linux"
CUDA_129 = "torch28-cxx11-cu129-x86_64-linux"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Hub's CLI formatter initializes agent detection, which can refresh a registry.
    monkeypatch.setattr(cache_cli.constants, "HF_HUB_OFFLINE", True)

    def fail(*args, **kwargs):
        pytest.fail("Cache listing must not make network requests")

    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket.socket, "connect_ex", fail)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    path = tmp_path / "hub"
    path.mkdir()
    monkeypatch.delenv("KERNELS_CACHE", raising=False)
    monkeypatch.setattr(cache_cli.constants, "HF_HUB_CACHE", str(path))
    monkeypatch.setenv("COLUMNS", "240")
    return path


def make_snapshot(
    cache_dir, *, repo_id=REPO_ID, commit=COMMIT, refs=("v1",), variants=(), repo_type="kernel", symlinks=True
):
    repo = cache_dir / f"{repo_type}s--{repo_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    files = {"README.md": b"Kernel description"}
    for variant in variants:
        files[f"build/{variant}/metadata.json"] = b"{}"
        files[f"build/{variant}/example/__init__.py"] = b"raise AssertionError('Do not import cached kernel code')"

    for filename, content in files.items():
        path = snapshot / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if symlinks:
            blob = repo / "blobs" / hashlib.sha256(content).hexdigest()
            blob.parent.mkdir(exist_ok=True)
            blob.write_bytes(content)
            path.symlink_to(os.path.relpath(blob, path.parent))
        else:
            path.write_bytes(content)
    for ref in refs:
        path = repo / "refs" / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(commit)
    return snapshot


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["kernels", "cache", "ls", *args])
    main()


@pytest.mark.parametrize("symlinks", [False, True], ids=["copied-files", "symlinked-files"])
def test_cache_ls_json_keeps_variants_separate_for_each_cached_version(cache_dir, monkeypatch, capsys, symlinks):
    # Multiple refs can point to the same snapshot, including slash-containing refs.
    make_snapshot(cache_dir, refs=["v1", "main", "releases/v1"], variants=[CUDA_126, CUDA_128], symlinks=symlinks)
    make_snapshot(cache_dir, commit="b" * 40, refs=["v0"], variants=[CUDA_129], symlinks=symlinks)
    # Detached snapshots are not listed as kernel versions.
    make_snapshot(cache_dir, commit="c" * 40, refs=(), variants=[CUDA_128, "torch-cuda"], symlinks=symlinks)
    make_snapshot(cache_dir, repo_id="other/kernel", variants=[CUDA_128], symlinks=symlinks)
    make_snapshot(cache_dir, repo_id="other/model", repo_type="model", variants=[CUDA_128], symlinks=symlinks)
    make_snapshot(cache_dir, repo_id="other/dataset", repo_type="dataset", symlinks=symlinks)

    run_cli(monkeypatch, "--json")
    output = capsys.readouterr()
    assert output.err == ""
    result = json.loads(output.out)
    assert result["cache_dir"] == str(cache_dir)
    assert [repo["repo_id"] for repo in result["repos"]] == [REPO_ID, "other/kernel"]
    repo = result["repos"][0]
    assert set(repo) == {"repo_id", "size_on_disk", "last_accessed", "last_modified", "versions"}
    versions = repo["versions"]
    assert [entry["version"] for entry in versions] == ["v0", "v1"]
    assert [entry["variants"] for entry in versions] == [[CUDA_129], [CUDA_126, CUDA_128]]
    assert all("ref" not in entry for entry in versions)
    assert "torch-cuda" not in output.out
    assert COMMIT not in output.out
    # Shared blobs are only counted once, including across variants and revisions.
    unique_size = len(b"Kernel description{}raise AssertionError('Do not import cached kernel code')")
    assert repo["size_on_disk"] == (
        unique_size if symlinks else 3 * len(b"Kernel description") + 5 * (unique_size - len(b"Kernel description"))
    )


def test_cache_ls_shows_each_version_with_its_variants_on_separate_lines(cache_dir, monkeypatch, capsys):
    make_snapshot(cache_dir, refs=["v1"], variants=[CUDA_126, CUDA_128, CUDA_129], symlinks=False)
    make_snapshot(cache_dir, commit="b" * 40, refs=["v0"], variants=[CUDA_128, "torch-cpu"], symlinks=False)
    hub_repo = next(iter(scan_cache_dir(cache_dir).repos))
    run_cli(monkeypatch)
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.splitlines()[0].split() == [
        "ID",
        "SIZE",
        "LAST_ACCESSED",
        "LAST_MODIFIED",
        "VERSION",
        "VARIANTS",
    ]
    assert "REVISION" not in output.out
    assert "SELECTED" not in output.out
    assert COMMIT not in output.out
    lines = output.out.splitlines()
    assert len(lines) == 7  # Header, separator, two variants for v0, and three for v1.
    version_start = lines[0].index("VERSION")
    variants_start = lines[0].index("VARIANTS")
    for version, block, expected in [
        (0, lines[2:4], ["torch-cpu", CUDA_128]),
        (1, lines[4:], [CUDA_126, CUDA_128, CUDA_129]),
    ]:
        assert f"kernel/{REPO_ID}" in block[0]
        assert hub_repo.refs[f"v{version}"].size_on_disk_str in block[0]
        assert hub_repo.refs[f"v{version}"].last_modified_str in block[0]
        assert block[0][version_start:variants_start].strip() == f"v{version}"
        assert [line[variants_start:].strip() for line in block] == expected
        assert all(not line[:variants_start].strip() for line in block[1:])
    # A build cached for both versions must be shown under both versions.
    assert output.out.count(CUDA_128) == 2


def test_cache_ls_sorts_version_numbers_and_excludes_other_refs(cache_dir, monkeypatch, capsys):
    for version in [10, 2, 0]:
        make_snapshot(cache_dir, commit=f"{version:040x}", refs=[f"v{version}"], variants=[CUDA_128])
    make_snapshot(cache_dir, refs=["main", "release/preview", "v1.0"], variants=["torch-cuda"])
    make_snapshot(cache_dir, commit="f" * 40, refs=(), variants=["torch-cpu"])

    run_cli(monkeypatch)
    output = capsys.readouterr()
    assert output.err == ""
    lines = output.out.splitlines()
    start, end = lines[0].index("VERSION"), lines[0].index("VARIANTS")
    assert [line[start:end].strip() for line in lines[2:]] == ["v0", "v2", "v10"]
    assert "torch-cuda" not in output.out
    assert "torch-cpu" not in output.out


@pytest.mark.parametrize("columns", [80, 120])
def test_cache_ls_fits_long_variant_lists_in_the_terminal(cache_dir, monkeypatch, capsys, columns):
    make_snapshot(
        cache_dir,
        repo_id="kernels-community/activation-with-a-long-kernel-name",
        refs=["v1", "v2"],
        variants=[CUDA_126, CUDA_128, CUDA_129, "torch-cuda"],
    )
    make_snapshot(cache_dir, repo_id="other/kernel", variants=["torch-cpu"])
    monkeypatch.setenv("COLUMNS", str(columns))
    run_cli(monkeypatch)
    output = capsys.readouterr()
    lines = output.out.splitlines()
    assert len(lines) == 11  # Header, separator, four variants for each of two versions, and another kernel.
    assert all(len(line) <= columns for line in lines)
    assert "..." in output.out
    assert "--no-truncate" in output.err
    assert "--json" in output.err
    # Every column starts at the same position in the header and data rows.
    separators = lines[1].split()
    start = 0
    for column_index, column in enumerate(separators):
        end = start + len(column)
        # Continuation lines leave all metadata columns empty.
        if column_index == 5:
            assert all(line[start:end].strip() for line in lines)
        else:
            assert all(line[start:end].strip() for line in lines if line[: len(separators[0])].strip())
        if end < len(lines[1]):
            assert all(line[end] == " " for line in lines)
        start = end + 1


@pytest.mark.parametrize("output_args", [["--no-truncate"], ["--json"]], ids=["full-table", "json"])
def test_cache_ls_can_show_full_values_in_a_narrow_terminal(cache_dir, monkeypatch, capsys, output_args):
    variants = [CUDA_126, CUDA_128, CUDA_129]
    make_snapshot(cache_dir, variants=variants)
    monkeypatch.setenv("COLUMNS", "80")
    run_cli(monkeypatch, *output_args)
    output = capsys.readouterr()
    assert output.err == ""
    assert REPO_ID in output.out
    assert all(variant in output.out for variant in variants)


@pytest.mark.parametrize(
    "variants",
    [["torch-cuda", "torch-rocm", "torch-xpu"], ["torch-universal"]],
    ids=["triton-backends", "legacy-universal"],
)
def test_cache_ls_lists_jit_kernels_without_loading_them(cache_dir, monkeypatch, capsys, variants):
    repo_id = "kernels-test/relu-triton"
    snapshot = make_snapshot(cache_dir, repo_id=repo_id, refs=["v1"], variants=variants)
    # JIT builds contain Python source and metadata, without native binaries.
    # make_snapshot's __init__.py raises if the listing imports the kernel.
    for variant in variants:
        (snapshot / "build" / variant / "example" / "op.py").write_text(
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def relu_kernel(x):\n"
            "    value = tl.load(x)\n"
            "    tl.store(x, tl.maximum(value, 0))\n"
        )

    run_cli(monkeypatch, "--json")
    output = capsys.readouterr()
    assert output.err == ""
    repos = json.loads(output.out)["repos"]
    assert len(repos) == 1
    assert repos[0]["repo_id"] == repo_id
    assert len(repos[0]["versions"]) == 1
    assert repos[0]["versions"][0]["version"] == "v1"
    assert repos[0]["versions"][0]["variants"] == variants

    run_cli(monkeypatch)
    output = capsys.readouterr()
    assert output.err == ""
    lines = [line for line in output.out.splitlines() if repo_id in line]
    assert len(lines) == 1
    table = output.out.splitlines()
    variants_start = table[0].index("VARIANTS")
    assert [line[variants_start:].strip() for line in table[2:]] == variants


def test_cache_ls_reports_size_and_timestamps_for_each_version(cache_dir, monkeypatch, capsys):
    first = make_snapshot(cache_dir, refs=["v1"], variants=[CUDA_126], symlinks=False)
    second = make_snapshot(cache_dir, commit="b" * 40, refs=["v2"], variants=[CUDA_128, CUDA_129], symlinks=False)
    for snapshot, atime, mtime in [(first, 1_700_000_200, 1_700_000_000), (second, 1_700_000_100, 1_700_000_300)]:
        for file in snapshot.rglob("*"):
            if file.is_file():
                os.utime(file, (atime, mtime))
    run_cli(monkeypatch, "--json")
    repo = json.loads(capsys.readouterr().out)["repos"][0]
    assert repo["last_accessed"] == 1_700_000_200
    assert repo["last_modified"] == 1_700_000_300
    assert [entry["last_accessed"] for entry in repo["versions"]] == [1_700_000_200, 1_700_000_100]
    assert [entry["last_modified"] for entry in repo["versions"]] == [1_700_000_000, 1_700_000_300]
    assert [entry["size_on_disk"] for entry in repo["versions"]] == [
        sum(file.stat().st_size for file in snapshot.rglob("*") if file.is_file()) for snapshot in [first, second]
    ]


def test_cache_ls_ignores_build_directories_without_downloaded_files(cache_dir, monkeypatch, capsys):
    snapshot = make_snapshot(cache_dir, variants=[CUDA_126])
    # An empty build directory does not contain a cached variant.
    (snapshot / "build" / CUDA_128).mkdir()
    (snapshot / "build" / "not-a-variant.txt").write_text("build notes")
    run_cli(monkeypatch, "--json")
    repo = json.loads(capsys.readouterr().out)["repos"][0]
    assert repo["versions"][0]["variants"] == [CUDA_126]


@pytest.mark.parametrize(
    "variants", [[], ["unknown-framework-cpu-x86_64-linux"]], ids=["no-builds", "unknown-build-format"]
)
def test_cache_ls_shows_a_dash_when_no_recognized_variants_are_cached(cache_dir, monkeypatch, capsys, variants):
    make_snapshot(cache_dir, variants=variants)
    run_cli(monkeypatch, "--json")
    repo = json.loads(capsys.readouterr().out)["repos"][0]
    assert repo["versions"][0]["variants"] == []
    run_cli(monkeypatch)
    output = capsys.readouterr().out
    line = next(line for line in output.splitlines() if REPO_ID in line)
    assert line.split()[-1] == "-"
    assert "SELECTED" not in output
    assert all(variant not in output for variant in variants)


@pytest.mark.parametrize(
    "override", ["default", "kernels", "explicit"], ids=["hub-cache-default", "kernels-cache-env", "cache-dir-option"]
)
def test_cache_ls_uses_the_configured_cache_directory(cache_dir, tmp_path, monkeypatch, capsys, override):
    kernels_cache = tmp_path / "kernels-cache"
    explicit_cache = tmp_path / "explicit-cache"
    make_snapshot(cache_dir, repo_id="test/default", variants=[CUDA_126])
    make_snapshot(kernels_cache, repo_id="test/kernels", variants=[CUDA_128])
    make_snapshot(explicit_cache, repo_id="test/explicit", variants=[CUDA_129])
    args = []
    if override != "default":
        monkeypatch.setenv("KERNELS_CACHE", str(kernels_cache))
    if override == "explicit":
        args = ["--cache-dir", str(explicit_cache)]
    run_cli(monkeypatch, "--json", *args)
    result = json.loads(capsys.readouterr().out)
    assert [repo["repo_id"] for repo in result["repos"]] == [f"test/{override}"]


@pytest.mark.parametrize("env_var", ["HF_HUB_CACHE", "HF_HOME"])
def test_cache_ls_uses_hugging_face_cache_environment_variables(tmp_path, env_var):
    cache_dir = tmp_path if env_var == "HF_HUB_CACHE" else tmp_path / "hub"
    make_snapshot(cache_dir)
    env = {key: value for key, value in os.environ.items() if key not in {"KERNELS_CACHE", "HF_HUB_CACHE", "HF_HOME"}}
    env[env_var] = str(tmp_path)
    env["HF_HUB_OFFLINE"] = "1"
    result = subprocess.run(
        ["kernels", "cache", "ls", "--json"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    assert info["cache_dir"] == str(cache_dir)
    assert [repo["repo_id"] for repo in info["repos"]] == [REPO_ID]


@pytest.mark.parametrize("state", ["empty", "missing", "models-only", "unversioned-kernel", "detached-kernel"])
def test_cache_ls_reports_when_no_kernel_versions_are_cached(cache_dir, monkeypatch, capsys, state):
    if state == "missing":
        cache_dir.rmdir()
    elif state == "models-only":
        make_snapshot(cache_dir, repo_type="model")
    elif state == "unversioned-kernel":
        make_snapshot(cache_dir, refs=["main"])
    elif state == "detached-kernel":
        make_snapshot(cache_dir, refs=())
    run_cli(monkeypatch)
    assert "No cached kernel versions found" in capsys.readouterr().out
    run_cli(monkeypatch, "--json")
    assert json.loads(capsys.readouterr().out) == {"cache_dir": str(cache_dir), "repos": []}
    if state == "missing":
        assert not cache_dir.exists()


@pytest.mark.parametrize("output_args", [[], ["--json"]], ids=["table", "json"])
@pytest.mark.parametrize("show_warnings", [False, True], ids=["warning-summary", "warning-details"])
def test_cache_ls_keeps_working_when_one_cached_kernel_is_corrupt(
    cache_dir, monkeypatch, capsys, output_args, show_warnings
):
    make_snapshot(cache_dir, variants=[CUDA_128])
    broken = make_snapshot(cache_dir, repo_id="test/broken", variants=[CUDA_128])
    (broken / "README.md").resolve().unlink()
    run_cli(monkeypatch, *output_args, *(["--show-warnings"] if show_warnings else []))
    output = capsys.readouterr()
    assert "Warning:" in output.err
    if show_warnings:
        assert "broken" in output.err
        assert "Repo ignored." in output.err
    else:
        assert len(output.err.splitlines()) == 1
        assert "Found 1 cache inconsistencies" in output.err
        assert "--show-warnings" in output.err
        assert "broken" not in output.err
    if output_args:
        repos = json.loads(output.out)["repos"]
        assert [repo["repo_id"] for repo in repos] == [REPO_ID]
        assert repos[0]["versions"][0]["variants"] == [CUDA_128]
    else:
        assert REPO_ID in output.out
        assert CUDA_128 in output.out
        assert "test/broken" not in output.out


def test_cache_ls_rejects_a_file_passed_as_cache_dir(tmp_path, monkeypatch, capsys):
    path = tmp_path / "file"
    path.write_text("not a directory")
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "--cache-dir", str(path))
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Cannot scan kernel cache" in output.err


def test_cache_without_ls_reports_a_missing_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["kernels", "cache"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err
