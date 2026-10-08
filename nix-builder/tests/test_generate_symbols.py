import functools
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_symbols.py"
SPEC = importlib.util.spec_from_file_location("generate_symbols", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def make_module(name="example", **exports):
    module = ModuleType(name)
    module.__dict__.update(exports)
    module.__all__ = list(exports)
    return module


def test_runtime_exports_aliases_and_layers():
    def implementation(x: int, /, *, scale: float = 1.0) -> int:
        """Scale the input."""
        raise AssertionError("Must not run the kernel")

    class Base:
        has_backward = True

    class Layer(Base):
        can_torch_compile = False

        def __init__(self, size: int = 4):
            raise AssertionError("Must not instantiate layers")

    layers = make_module("example.layers", Layer=Layer)
    module = make_module(layers=layers, alias=implementation)
    module.unexported = implementation
    module._private_for_testing = make_module(hidden=implementation)
    # __all__ need not be statically resolvable, and duplicates are harmless.
    module.__all__ = ["alias"] + [name for name in ("layers", "alias")]

    result = generator.generate_symbols(module)

    assert result["schema_version"] == 1
    (alias,) = result["functions"]
    assert alias["name"] == "alias"
    assert alias["qualname"].endswith("implementation")
    assert alias["kind"] == "function"
    assert alias["doc"] == "Scale the input."
    layer = result["layers"][0]
    assert layer["name"] == "Layer"
    assert layer["kind"] == "class"
    assert layer["attributes"] == {"has_backward": True, "can_torch_compile": False}
    assert layer["signature"]["parameters"] == [
        {
            "name": "size",
            "kind": "POSITIONAL_OR_KEYWORD",
            "default": "4",
            "annotation": "int",
        }
    ]
    assert json.loads(json.dumps(result)) == result


def test_signature_preserves_parameter_kinds_and_empty_values():
    def function(a, /, b: int = None, *args: float, c="é", **kwargs) -> None:
        pass

    assert generator.describe_signature(function) == {
        "parameters": [
            {
                "name": "a",
                "kind": "POSITIONAL_ONLY",
                "default": None,
                "annotation": None,
            },
            {
                "name": "b",
                "kind": "POSITIONAL_OR_KEYWORD",
                "default": "None",
                "annotation": "int",
            },
            {
                "name": "args",
                "kind": "VAR_POSITIONAL",
                "default": None,
                "annotation": "float",
            },
            {"name": "c", "kind": "KEYWORD_ONLY", "default": "'é'", "annotation": None},
            {
                "name": "kwargs",
                "kind": "VAR_KEYWORD",
                "default": None,
                "annotation": None,
            },
        ],
        "return_annotation": "None",
    }


def test_annotations_are_not_evaluated():
    def function(x: "UndefinedTensor") -> "UndefinedTensor":  # noqa: F821
        pass

    signature = generator.describe_signature(function)
    assert signature["parameters"][0]["annotation"] == "'UndefinedTensor'"
    assert signature["return_annotation"] == "'UndefinedTensor'"


def test_only_functions_and_layer_classes_are_included():
    def original(x: int, y: int = 2):
        pass

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        raise AssertionError("Must not call wrappers")

    async def async_function():
        raise AssertionError("Must not call async functions")

    class Layer:
        def __call__(self):
            raise AssertionError("Must not call exported objects")

        def method(self):
            pass

    layers = make_module("example.layers", Layer=Layer, function=original)
    layers.unexported = type("HiddenLayer", (), {})
    other_module = make_module("example.other", function=original, Layer=Layer)
    module = make_module(
        wrapped=wrapper,
        async_function=async_function,
        partial=functools.partial(original, y=3),
        callable=Layer(),
        method=Layer().method,
        builtin=len,
        data=42,
        TopLevelClass=Layer,
        other=other_module,
        layers=layers,
    )
    result = generator.generate_symbols(module)
    assert [item["name"] for item in result["functions"]] == [
        "wrapped",
        "async_function",
    ]
    assert result["functions"][0]["signature"] == generator.describe_signature(original)
    assert [item["name"] for item in result["layers"]] == ["Layer"]


def test_unavailable_signature_is_null():
    class Layer:
        __signature__ = "not an inspect.Signature"

    result = generator.generate_symbols(make_module(layers=make_module(Layer=Layer)))
    assert result["layers"][0]["signature"] is None


def test_non_boolean_layer_attributes_are_unknown():
    class Layer:
        has_backward = "True"
        can_torch_compile = 1

    symbol = generator.describe_symbol("Layer", Layer)
    assert symbol["attributes"] == {"has_backward": None, "can_torch_compile": None}


@pytest.mark.parametrize("exports", [None, "function", [1], {"function"}])
def test_invalid_all_fails(exports):
    module = make_module()
    module.__all__ = exports
    with pytest.raises(ValueError, match=r"example\.__all__ must be"):
        generator.generate_symbols(module)


def test_missing_export_fails_with_context():
    module = make_module()
    module.__all__ = ["missing"]
    with pytest.raises(
        ValueError, match=r"example\.__all__ names a missing symbol: 'missing'"
    ):
        generator.generate_symbols(module)


def test_no_all_exports_nothing():
    module = ModuleType("example")
    module.public = lambda: None
    module.layers = ModuleType("example.layers")
    module.layers.Layer = type("Layer", (), {})
    result = generator.generate_symbols(module)
    assert result["functions"] == []
    assert result["layers"] == []


def test_private_exports_and_testing_reexports_are_excluded():
    def public():
        pass

    def secret():
        pass

    class PrivateLayer:
        pass

    secret.__module__ = "example._private_for_testing.helpers"
    PrivateLayer.__module__ = "example._private_for_testing"
    module = make_module(
        _hidden=public,
        secret=secret,
        public=public,
        layers=make_module(_Hidden=type("Hidden", (), {}), Alias=PrivateLayer),
    )
    result = generator.generate_symbols(module)
    assert [item["name"] for item in result["functions"]] == ["public"]
    assert result["layers"] == []


def test_lazy_exports_not_in_dir():
    def public():
        pass

    module = make_module()
    module.__all__ = ["lazy"]
    module.__getattr__ = lambda name: public if name == "lazy" else None
    assert generator.generate_symbols(module)["functions"][0]["name"] == "lazy"


def test_layers_need_not_be_exported_by_package():
    module = make_module()
    module.layers = make_module(Layer=type("Layer", (), {}))
    result = generator.generate_symbols(module)
    assert result["functions"] == []
    assert [item["name"] for item in result["layers"]] == ["Layer"]


def test_missing_layers_is_supported():
    module = make_module(public=lambda: None)
    result = generator.generate_symbols(module)
    assert [item["name"] for item in result["functions"]] == ["public"]
    assert result["layers"] == []


def test_export_order_and_aliases_are_preserved():
    def implementation():
        pass

    module = make_module(z=implementation, a=implementation)
    module.__all__ = ("z", "a", "z")
    assert [
        item["name"] for item in generator.generate_symbols(module)["functions"]
    ] == ["z", "a"]


@pytest.fixture
def built_package(tmp_path):
    package = tmp_path / "variant"
    package.mkdir()
    (package / "__init__.py").write_text(
        textwrap.dedent("""\
            from . import layers
            from .implementation import function as alias
            print("Initialization output")
            __all__ = ["alias", "layers"]
        """),
        encoding="utf-8",
    )
    (package / "implementation.py").write_text(
        'def function(x: "UndefinedTensor", *, scale=2):\n    """Scale é."""\n',
        encoding="utf-8",
    )
    (package / "layers.py").write_text(
        textwrap.dedent("""\
            from dataclasses import dataclass
            @dataclass
            class Layer:
                size: int = 4
                has_backward = True
                can_torch_compile = True
            __all__ = ["Layer"]
        """),
        encoding="utf-8",
    )
    return package


def test_cli_loads_built_package_and_writes_json(built_package):
    output = built_package / "symbols.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "example_build",
        "--package-dir",
        str(built_package),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    content = output.read_text(encoding="utf-8")
    document = json.loads(content)
    assert "Initialization output" in result.stdout
    assert document["module"] == "example_build"
    (alias,) = document["functions"]
    assert alias["module"] == "example_build.implementation"
    assert alias["doc"] == "Scale é."
    assert document["layers"][0]["attributes"]["has_backward"] is True
    subprocess.run(command, capture_output=True, text=True, check=True)
    assert output.read_text(encoding="utf-8") == content


def test_cli_imports_from_python_path(built_package, tmp_path):
    output = tmp_path / "symbols.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "variant", "--output", str(output)],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(output.read_text())["module"] == "variant"


@pytest.mark.parametrize("has_layers", [False, True])
def test_cli_discovers_optional_layers(built_package, has_layers):
    # Match the PR: layers need not be imported or exported by __init__.py.
    (built_package / "__init__.py").write_text(
        "from .implementation import function\n__all__ = ['function']\n",
        encoding="utf-8",
    )
    if not has_layers:
        (built_package / "layers.py").unlink()
    output = built_package / "symbols.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "example_build",
            "--package-dir",
            str(built_package),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(output.read_text())
    assert [item["name"] for item in result["functions"]] == ["function"]
    assert [item["name"] for item in result["layers"]] == (
        ["Layer"] if has_layers else []
    )


def test_optional_layers_import_errors_are_not_hidden(built_package):
    (built_package / "__init__.py").write_text("", encoding="utf-8")
    (built_package / "layers.py").write_text(
        "import missing_layer_dependency_for_test\n", encoding="utf-8"
    )
    output = built_package / "symbols.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "example_build",
            "--package-dir",
            str(built_package),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing_layer_dependency_for_test" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "source", ["raise RuntimeError('broken import')", "__all__ = ['missing']"]
)
def test_cli_failure_preserves_existing_output(built_package, source):
    output = built_package / "symbols.json"
    output.write_text("previous output", encoding="utf-8")
    (built_package / "__init__.py").write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "example_build",
            "--package-dir",
            str(built_package),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "previous output"


def test_missing_package_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No kernel package found"):
        generator.load_package("missing_package", tmp_path)


def test_failed_import_cleans_up_submodules(built_package):
    (built_package / "__init__.py").write_text(
        "from . import layers\nraise RuntimeError('broken import')\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="broken import"):
        generator.load_package("failed_build", built_package)
    assert not any(
        name == "failed_build" or name.startswith("failed_build.")
        for name in sys.modules
    )
