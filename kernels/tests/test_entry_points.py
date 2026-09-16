from importlib.metadata import distribution

import pytest


def test_egg_info_writer_entry_point_is_importable():
    """The `egg_info.writers` entry point is loaded by setuptools during *any*
    `egg_info` run in an environment where `kernels` is installed, including
    builds of unrelated packages. If it points at a module that is not shipped,
    every such build fails with `ModuleNotFoundError`."""
    entry_points = [ep for ep in distribution("kernels").entry_points if ep.group == "egg_info.writers"]
    assert entry_points, "no egg_info.writers entry point registered"

    for ep in entry_points:
        try:
            writer = ep.load()
        except ModuleNotFoundError as e:
            pytest.fail(f"entry point {ep.name} = {ep.value!r} is not importable: {e}")
        assert callable(writer)
