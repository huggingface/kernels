import ast
from pathlib import Path
from typing import Any

from huggingface_hub import ModelCard

from ..compat import tomllib

KERNEL_CARD_TEMPLATE_PATH = Path(__file__).parent / "card_template.md"
DESCRIPTION = """
This is the repository card of {repo_id} that has been pushed on the Hub. It was built to be used with the [`kernels` library](https://github.com/huggingface/kernels). This card was automatically generated.
"""
EXAMPLE_CODE = """```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("{repo_id}") # <- change the ID if needed
{func_name} = kernel_module.{func_name}

{func_name}(...)
```"""
LIBRARY_NAME = "kernels"


def _parse_build_toml(local_path: str | Path) -> dict | None:
    local_path = Path(local_path)
    build_toml_path = local_path / "build.toml"

    if not build_toml_path.exists():
        return None

    try:
        with open(build_toml_path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return None


def _find_torch_ext_init(local_path: str | Path) -> Path | None:
    local_path = Path(local_path)

    config = _parse_build_toml(local_path)
    if not config:
        return None

    try:
        kernel_name = config.get("general", {}).get("name")
        if not kernel_name:
            return None

        module_name = kernel_name.replace("-", "_")
        init_file = local_path / "torch-ext" / module_name / "__init__.py"

        if init_file.exists():
            return init_file

        return None
    except Exception:
        return None


def _extract_functions_from_all(init_file_path: Path) -> list[str] | None:
    try:
        content = init_file_path.read_text()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            functions = []
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    func_name = str(elt.value)
                                    functions.append(func_name)
                            return functions if functions else None
        return None
    except Exception:
        return None


def _parse_repo_id(local_path: str | Path) -> str:
    local_path = Path(local_path)

    config = _parse_build_toml(local_path)
    if not config:
        return None

    repo_id = config.get("general", {}).get("hub", {}).get("repo-id", None)
    return repo_id


def _build_kernel_card_vars(
    local_path: str | Path,
    repo_id: str = "REPO_ID",
) -> dict:
    local_path = Path(local_path)
    vars: dict[str, Any] = {}

    # --- usage example + available interfaces ---
    init_file = _find_torch_ext_init(local_path)
    func_names = _extract_functions_from_all(init_file) if init_file else None
    if func_names:
        vars["usage_example"] = EXAMPLE_CODE.format(
            repo_id=repo_id, func_name=func_names[0]
        )
        vars["available_functions"] = "\n".join(f"- `{fn}`" for fn in func_names)

    # --- backends, CUDA capabilities, upstream ---
    config = _parse_build_toml(local_path)
    if config:
        general_config = config.get("general", {})
        backends = general_config.get("backends")
        if backends:
            vars["supported_backends"] = "\n".join(f"- {b}" for b in backends)

        kernel_configs = config.get("kernel", {})
        cuda_capabilities: set[Any] = set()
        for k in kernel_configs:
            caps = kernel_configs[k].get("cuda-capabilities")
            if caps:
                cuda_capabilities.update(caps)
        if cuda_capabilities:
            vars["cuda_capabilities"] = "\n".join(
                f"- {cap}" for cap in cuda_capabilities
            )

        upstream_repo = config.get("upstream", None)
        if upstream_repo:
            vars["source_code"] = (
                f"Source code of this kernel originally comes from {upstream_repo}"
                " and it was repurposed for compatibility with `kernels`."
            )

    # --- benchmark ---
    benchmark_file = local_path / "benchmarks" / "benchmark.py"
    if benchmark_file.exists():
        vars["benchmark_content"] = (
            "Benchmarking script is available for this kernel. Make sure to run"
            " `kernels benchmark org-id/repo-id`"
            ' (replace "org-id" and "repo-id" with actual values).'
        )

    return vars


def _update_kernel_card_license(
    kernel_card: ModelCard, local_path: str | Path
) -> ModelCard:
    config = _parse_build_toml(local_path)
    if not config:
        return kernel_card

    existing_license = kernel_card.data.get("license", None)
    license_from_config = config.get("general", {}).get("license", None)
    final_license = license_from_config or existing_license
    kernel_card.data["license"] = final_license
    return kernel_card
