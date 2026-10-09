import base64
import hashlib
import json
import sys
from pathlib import Path

variant = Path(sys.argv[1])
symbols_path = variant / "symbols.json"
symbols = json.loads(symbols_path.read_text())
assert symbols["schema_version"] == 1
assert symbols["module"] == "_symbols_test"
assert [function["name"] for function in symbols["functions"]] == ["relu"]
assert symbols["functions"][0]["signature"]["return_annotation"] == "int"
assert [layer["name"] for layer in symbols["layers"]] == ["ReLU"]
assert symbols["layers"][0]["attributes"] == {
    "has_backward": True,
    "can_torch_compile": False,
}

metadata = json.loads((variant / "metadata.json").read_text())
assert (
    metadata["digest"]["files"]["symbols.json"]
    == base64.b64encode(hashlib.sha256(symbols_path.read_bytes()).digest()).decode()
)
assert not list(variant.rglob("__pycache__"))
