from typing import Any, List


def _get_nested_attr(d, attr: List[str]) -> Any:
    for a in attr:
        d = d.get(a)
        if d is None:
            break
    return d


def write_egg_lockfile(cmd, basename, filename):
    import json
    import os
    import tomllib

    cwd = os.getcwd()
    pyproject = os.path.join(cwd, "pyproject.toml")
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    kernel_versions = _get_nested_attr(data, ["tool", "kernels", "dependencies"])
    if kernel_versions is None:
        return

    kernel_versions_json = json.dumps(kernel_versions, indent=2)

    cmd.write_or_delete_file(basename, filename, kernel_versions_json)
