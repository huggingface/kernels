import ast
import re
from pathlib import Path

from ..compat import tomllib
from typing import Any
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

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

is_jinja_available = False
try:
    import jinja2  # noqa

    is_jinja_available = True
except ImportError:
    pass


def _load_or_create_kernel_card(
    repo_id_or_path: str = "REPO_ID",
    token: str | None = None,
    kernel_description: str | None = None,
    license: str | None = None,
    force_update_content: bool = False,
) -> ModelCard:
    if not is_jinja_available:
        raise ValueError(
            "Modelcard rendering is based on Jinja templates."
            " Please make sure to have `jinja` installed before using `load_or_create_model_card`."
            " To install it, please run `pip install Jinja2`."
        )

    kernel_card = None

    if not force_update_content:
        try:
            kernel_card = ModelCard.load(repo_id_or_path, token=token)
        except (EntryNotFoundError, RepositoryNotFoundError, TypeError):
            pass  # Will create from template below

    if kernel_card is None:
        kernel_description = kernel_description or DESCRIPTION
        kernel_card = ModelCard.from_template(
            card_data=ModelCardData(license=license, library_name=LIBRARY_NAME),
            template_path=str(KERNEL_CARD_TEMPLATE_PATH),
            model_description=kernel_description,
        )

    return kernel_card


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


def _extract_card_sections(card_content: str) -> dict:
    """Extract named sections from a rendered kernel card markdown string."""
    body = card_content
    front_matter_match = re.match(r"^---\n.*?\n---\n", body, re.DOTALL)
    if front_matter_match:
        body = body[front_matter_match.end() :]

    parts = re.split(r"\n## ", body)

    result: dict[str, str] = {}

    description = parts[0].strip()
    description = re.sub(r"<!--.*?-->", "", description, flags=re.DOTALL).strip()
    if description:
        result["description"] = description

    for part in parts[1:]:
        newline = part.find("\n")
        if newline == -1:
            continue
        section_name = part[:newline].strip().lower()
        section_body = part[newline:].strip()
        result[section_name] = section_body

    return result


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
