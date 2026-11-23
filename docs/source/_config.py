from functools import wraps
from inspect import unwrap

from doc_builder import autodoc

from kernels import hf_hub, load, resolve, validate

# The pinned doc-builder uses the defining module to name re-exports. Use the
# public paths for names and anchors while keeping the defining source links.
_public_paths = {}
for _module in (hf_hub, load, resolve, validate):
    for _name in _module.__all__:
        _obj = getattr(_module, _name)
        if _obj.__module__.startswith("kernels._internals."):
            _public_paths[f"{_obj.__module__}.{_obj.__qualname__}"] = (
                f"{_module.__name__}.{_name}"
            )

# Doc-builder may load this config twice in the same build.
_get_shortest_path = unwrap(autodoc.get_shortest_path)


@wraps(_get_shortest_path)
def _get_public_path(obj, package):
    path = _get_shortest_path(obj, package)
    if package.__name__ != "kernels" or path is None:
        return path
    if path in _public_paths:
        return _public_paths[path]

    owner, _, member = path.rpartition(".")
    if owner in _public_paths:
        return f"{_public_paths[owner]}.{member}"
    return path


autodoc.get_shortest_path = _get_public_path
