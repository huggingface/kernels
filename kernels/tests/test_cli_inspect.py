import json
import sys
from types import SimpleNamespace

from kernels.cli import main
from kernels.cli import inspect as inspect_cli


def test_inspect_human(monkeypatch, capsys):
    def activate():
        pass

    class Relu:
        pass

    monkeypatch.setattr(
        inspect_cli,
        "get_kernel",
        lambda repo_id, *, revision, version: SimpleNamespace(activate=activate, layers=SimpleNamespace(Relu=Relu)),
    )
    monkeypatch.setattr(sys, "argv", ["kernels", "inspect", "--version", "1", "example/kernel"])

    main()

    output = capsys.readouterr().out
    assert "Repository: example/kernel" in output
    assert "Revision: v1" in output
    assert "  activate" in output
    assert "  Relu" in output


def test_inspect_json(monkeypatch, capsys):
    def activate():
        pass

    class Relu:
        pass

    monkeypatch.setattr(
        inspect_cli,
        "get_kernel",
        lambda repo_id, *, revision, version: SimpleNamespace(activate=activate, layers=SimpleNamespace(Relu=Relu)),
    )
    monkeypatch.setattr(sys, "argv", ["kernels", "inspect", "--revision", "main", "--json", "example/kernel"])

    main()

    info = json.loads(capsys.readouterr().out)
    assert info == {
        "repo_id": "example/kernel",
        "revision": "main",
        "version": None,
        "functions": ["activate"],
        "layers": ["Relu"],
    }
