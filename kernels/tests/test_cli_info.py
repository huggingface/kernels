import json

import pytest

from kernels.cli.info import print_kernel_info


def test_info_hub(capsys):
    print_kernel_info("kernels-community/activation", version=1)
    out = capsys.readouterr().out
    assert "Repository: kernels-community/activation" in out
    assert "Revision: v1" in out
    assert "Name: activation" in out
    assert "Version: 1" in out
    assert "License: Apache-2.0" in out
    assert "Backends:" in out


def test_info_hub_json(capsys):
    print_kernel_info("kernels-community/activation", version=1, json_output=True)
    info = json.loads(capsys.readouterr().out)
    assert info["repo_id"] == "kernels-community/activation"
    assert info["revision"] == "v1"
    assert info["name"] == "activation"
    assert info["version"] == 1
    assert info["license"] == "Apache-2.0"
    assert "cuda" in info["backends"]


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
