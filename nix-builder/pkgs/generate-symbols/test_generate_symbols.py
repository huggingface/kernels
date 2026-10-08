import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

# The generator is a standalone script, so load it by its file path.
SCRIPT = Path(__file__).with_name("generate_symbols.py")
SPEC = importlib.util.spec_from_file_location("generate_symbols", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.fixture
def kernel():
    """A small kernel exporting one function and one layer."""

    def relu(x: int) -> int:
        """Apply ReLU."""
        raise AssertionError("Inspection must not call the function")

    class ReLU:
        has_backward = True
        can_torch_compile = False

        def __init__(self, size: int = 4):
            raise AssertionError("Inspection must not instantiate the layer")

    layers = ModuleType("example.layers")
    layers.ReLU = ReLU
    layers.__all__ = ["ReLU"]

    module = ModuleType("example")
    module.relu = relu
    module.layers = layers
    module.__all__ = ["relu", "layers"]
    return module


def test_public_function_is_included(kernel):
    result = generator.generate_symbols(kernel)

    assert len(result["functions"]) == 1
    function = result["functions"][0]
    assert function["name"] == "relu"
    assert function["kind"] == "function"
    assert function["doc"] == "Apply ReLU."


def test_public_layer_is_included(kernel):
    result = generator.generate_symbols(kernel)

    assert len(result["layers"]) == 1
    layer = result["layers"][0]
    assert layer["name"] == "ReLU"
    assert layer["kind"] == "class"
    assert layer["attributes"] == {
        "has_backward": True,
        "can_torch_compile": False,
    }


def test_symbols_not_in_all_are_excluded(kernel):
    kernel.helper = kernel.relu
    kernel.layers.OtherLayer = kernel.layers.ReLU

    result = generator.generate_symbols(kernel)

    assert [function["name"] for function in result["functions"]] == ["relu"]
    assert [layer["name"] for layer in result["layers"]] == ["ReLU"]


def test_underscore_names_are_excluded_even_when_exported(kernel):
    kernel._helper = kernel.relu
    kernel.__all__.append("_helper")
    kernel.layers._HiddenLayer = kernel.layers.ReLU
    kernel.layers.__all__.append("_HiddenLayer")

    result = generator.generate_symbols(kernel)

    assert [function["name"] for function in result["functions"]] == ["relu"]
    assert [layer["name"] for layer in result["layers"]] == ["ReLU"]


def test_function_from_private_testing_module_is_excluded(kernel):
    # A public export name must not expose a private testing implementation.
    kernel.relu.__module__ = "example._private_for_testing.helpers"

    result = generator.generate_symbols(kernel)

    assert result["functions"] == []
    assert [layer["name"] for layer in result["layers"]] == ["ReLU"]


def test_layer_from_private_testing_module_is_excluded(kernel):
    kernel.layers.ReLU.__module__ = "example._private_for_testing"

    result = generator.generate_symbols(kernel)

    assert result["layers"] == []
    assert [function["name"] for function in result["functions"]] == ["relu"]


def test_functions_in_layers_are_not_layers(kernel):
    kernel.layers.helper = kernel.relu
    kernel.layers.__all__.append("helper")

    result = generator.generate_symbols(kernel)

    assert [layer["name"] for layer in result["layers"]] == ["ReLU"]


def test_layers_are_optional(kernel):
    del kernel.layers
    kernel.__all__ = ["relu"]

    result = generator.generate_symbols(kernel)

    assert [function["name"] for function in result["functions"]] == ["relu"]
    assert result["layers"] == []


def test_layers_need_not_appear_in_package_all(kernel):
    kernel.__all__ = ["relu"]

    result = generator.generate_symbols(kernel)

    assert [layer["name"] for layer in result["layers"]] == ["ReLU"]


def test_parameter_kinds_are_preserved():
    def function(x, /, scale, *args, out=None, **kwargs):
        pass

    signature = generator.describe_signature(function)
    parameters = signature["parameters"]

    assert [parameter["name"] for parameter in parameters] == [
        "x",
        "scale",
        "args",
        "out",
        "kwargs",
    ]
    assert [parameter["kind"] for parameter in parameters] == [
        "POSITIONAL_ONLY",
        "POSITIONAL_OR_KEYWORD",
        "VAR_POSITIONAL",
        "KEYWORD_ONLY",
        "VAR_KEYWORD",
    ]


def test_missing_default_differs_from_explicit_none():
    def function(x, out=None):
        pass

    signature = generator.describe_signature(function)
    x, out = signature["parameters"]

    assert x["default"] is None
    assert out["default"] == "None"
    assert x["annotation"] is None
    assert signature["return_annotation"] is None


def test_string_annotations_are_not_evaluated():
    def function(x: "UndefinedTensor") -> "UndefinedTensor":  # noqa: F821
        pass

    signature = generator.describe_signature(function)

    assert signature["parameters"][0]["annotation"] == "'UndefinedTensor'"
    assert signature["return_annotation"] == "'UndefinedTensor'"


def test_layer_constructor_signature_is_included(kernel):
    signature = generator.describe_signature(kernel.layers.ReLU)

    assert signature["parameters"] == [
        {
            "name": "size",
            "kind": "POSITIONAL_OR_KEYWORD",
            "default": "4",
            "annotation": "int",
        }
    ]


def test_unavailable_signature_is_null(kernel):
    kernel.layers.ReLU.__signature__ = "invalid signature"

    result = generator.generate_symbols(kernel)

    assert result["layers"][0]["signature"] is None


def test_inherited_layer_flags_are_included(kernel):
    class ChildLayer(kernel.layers.ReLU):
        pass

    kernel.layers.ChildLayer = ChildLayer
    kernel.layers.__all__ = ["ChildLayer"]

    result = generator.generate_symbols(kernel)

    assert result["layers"][0]["attributes"] == {
        "has_backward": True,
        "can_torch_compile": False,
    }


def test_missing_layer_flags_are_unknown(kernel):
    del kernel.layers.ReLU.has_backward
    del kernel.layers.ReLU.can_torch_compile

    result = generator.generate_symbols(kernel)

    assert result["layers"][0]["attributes"] == {
        "has_backward": None,
        "can_torch_compile": None,
    }


def test_result_can_be_written_as_json(kernel):
    result = generator.generate_symbols(kernel)

    assert result["schema_version"] == 1
    assert result["module"] == "example"
    assert json.loads(json.dumps(result)) == result
