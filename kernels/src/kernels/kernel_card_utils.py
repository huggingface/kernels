import ast
import re
from pathlib import Path

from .compat import tomllib
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
        except (EntryNotFoundError, RepositoryNotFoundError):
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


def _update_kernel_card_usage(
    kernel_card: ModelCard,
    local_path: str | Path,
    repo_id: str = "REPO_ID",
) -> ModelCard:
    init_file = _find_torch_ext_init(local_path)

    if not init_file:
        return kernel_card

    func_names = _extract_functions_from_all(init_file)

    if not func_names:
        return kernel_card

    func_name = func_names[0]
    example_code = EXAMPLE_CODE.format(repo_id=repo_id, func_name=func_name)

    card_content = str(kernel_card.content)
    pattern = r"(## How to use\s*\n\n)```python\n# TODO: add an example code snippet for running this kernel\n```"

    if re.search(pattern, card_content):
        updated_content = re.sub(pattern, r"\1" + example_code, card_content)
        kernel_card.content = updated_content

    return kernel_card


def _update_kernel_card_available_funcs(
    kernel_card: ModelCard, local_path: str | Path
) -> ModelCard:
    init_file = _find_torch_ext_init(local_path)

    if not init_file:
        return kernel_card

    func_names = _extract_functions_from_all(init_file)

    if not func_names:
        return kernel_card

    functions_list = "\n".join(f"- `{func}`" for func in func_names)

    card_content = str(kernel_card.content)
    pattern = r"(## Available functions\s*\n\n)\[TODO: add the functions available through this kernel\]"

    if re.search(pattern, card_content):
        updated_content = re.sub(pattern, r"\1" + functions_list, card_content)
        kernel_card.content = updated_content

    return kernel_card


def _update_kernel_card_backends(
    kernel_card: ModelCard, local_path: str | Path
) -> ModelCard:
    config = _parse_build_toml(local_path)
    if not config:
        return kernel_card

    general_config = config.get("general", {})

    card_content = str(kernel_card.content)

    backends = general_config.get("backends")
    if backends:
        backends_list = "\n".join(f"- {backend}" for backend in backends)
        pattern = r"(## Supported backends\s*\n\n)\[TODO: add the backends this kernel supports\]"
        if re.search(pattern, card_content):
            card_content = re.sub(pattern, r"\1" + backends_list, card_content)

    # TODO: should we consider making it a separate utility?
    kernel_configs = config.get("kernel", {})
    cuda_capabilities = []
    if kernel_configs:
        for k in kernel_configs:
            cuda_cap_for_config = kernel_configs[k].get("cuda-capabilities")
            if cuda_cap_for_config:
                cuda_capabilities.extend(cuda_cap_for_config)
    cuda_capabilities: set[Any] = set(cuda_capabilities)  # type: ignore[no-redef]
    if cuda_capabilities:
        cuda_list = "\n".join(f"- {cap}" for cap in cuda_capabilities)
        cuda_section = f"## CUDA Capabilities\n\n{cuda_list}\n\n"
        pattern = r"(## Benchmarks)"
        if re.search(pattern, card_content):
            card_content = re.sub(pattern, cuda_section + r"\1", card_content)

    kernel_card.content = card_content
    return kernel_card


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


def _update_benchmark(kernel_card: ModelCard, local_path: str | Path):
    local_path = Path(local_path)

    benchmark_file = local_path / "benchmarks" / "benchmark.py"
    if not benchmark_file.exists():
        return kernel_card

    card_content = str(kernel_card.content)
    benchmark_text = '\n\nBenchmarking script is available for this kernel. Make sure to run `kernels benchmark org-id/repo-id` (replace "org-id" and "repo-id" with actual values).'

    pattern = r"(## Benchmarks)"
    if re.search(pattern, card_content):
        updated_content = re.sub(pattern, r"\1" + benchmark_text, card_content)
        kernel_card.content = updated_content

    return kernel_card
