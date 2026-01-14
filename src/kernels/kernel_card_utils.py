import ast
import re
from pathlib import Path

from .compat import tomllib
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

MODEL_CARD_TEMPLATE_PATH = Path(__file__).parent / "card_template.md"
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

is_jinja_available = False
try:
    import jinja2  # noqa

    is_jinja_available = True
except ImportError:
    pass


def _load_or_create_model_card(
    repo_id_or_path: str = "REPO_ID",
    token: str | None = None,
    kernel_description: str | None = None,
    license: str | None = None,
) -> ModelCard:
    if not is_jinja_available:
        raise ValueError(
            "Modelcard rendering is based on Jinja templates."
            " Please make sure to have `jinja` installed before using `load_or_create_model_card`."
            " To install it, please run `pip install Jinja2`."
        )

    try:
        # Check if the model card is present on the remote repo
        model_card = ModelCard.load(repo_id_or_path, token=token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        # Otherwise create a model card from template
        kernel_description = kernel_description or DESCRIPTION
        model_card = ModelCard.from_template(
            # Card metadata object that will be converted to YAML block
            card_data=ModelCardData(license=license, library_name="kernels"),
            template_path=str(MODEL_CARD_TEMPLATE_PATH),
            model_description=kernel_description,
        )

    return model_card


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
        # Get kernel name from general.name
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


def _extract_function_from_all(init_file_path: Path) -> str | None:
    try:
        content = init_file_path.read_text()

        # Parse the file as an AST
        tree = ast.parse(content)

        # Find the __all__ assignment
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        # Extract the list values
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    func_name = str(elt.value)
                                    # Skip module names, return the first function-like name
                                    if not func_name.endswith("s") or "_" in func_name:
                                        return func_name
                            # Fallback: return the first item if no function found
                            if node.value.elts:
                                first_elt = node.value.elts[0]
                                if isinstance(first_elt, ast.Constant):
                                    return str(first_elt.value)
        return None
    except Exception:
        return None


def _update_model_card_usage(
    model_card: ModelCard,
    local_path: str | Path,
    repo_id: str = "REPO_ID",
) -> ModelCard:
    init_file = _find_torch_ext_init(local_path)

    if not init_file:
        return model_card

    func_name = _extract_function_from_all(init_file)

    if not func_name:
        return model_card

    example_code = EXAMPLE_CODE.format(repo_id=repo_id, func_name=func_name)

    # Update the model card content
    card_content = str(model_card.content)
    pattern = r"(## How to use\s*\n\n)```python\n# TODO: add an example code snippet for running this kernel\n```"

    if re.search(pattern, card_content):
        updated_content = re.sub(pattern, r"\1" + example_code, card_content)
        model_card.content = updated_content

    return model_card


def _update_model_card_backends(
    model_card: ModelCard, local_path: str | Path
) -> ModelCard:
    config = _parse_build_toml(local_path)
    if not config:
        return model_card

    config = config.get("general", {})

    card_content = str(model_card.content)

    backends = config.get("backends")
    if backends:
        backends_list = "\n".join(f"- {backend}" for backend in backends)
        pattern = r"(## Supported backends\s*\n\n)\[TODO: add the backends this kernel supports\]"
        if re.search(pattern, card_content):
            card_content = re.sub(pattern, r"\1" + backends_list, card_content)

    # TODO: should we consider making it a separate utility?
    cuda_capabilities = config.get("cuda-capabilities")
    if cuda_capabilities:
        cuda_list = "\n".join(f"- {cap}" for cap in cuda_capabilities)
        cuda_section = f"## CUDA Capabilities\n\n{cuda_list}\n\n"
        pattern = r"(## Benchmarks)"
        if re.search(pattern, card_content):
            card_content = re.sub(pattern, cuda_section + r"\1", card_content)

    model_card.content = card_content
    return model_card
