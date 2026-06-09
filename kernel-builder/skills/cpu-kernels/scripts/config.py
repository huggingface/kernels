"""Shared config loader — reads config.yaml from project root."""

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULTS = {
    "max_trials": 8,
    "early_stop_speedup": 3.0,
    "perf_stat_enabled": True,
    "vtune_enabled": False,
    "build_command": "kernel-builder build --release",
}


def load_config() -> dict:
    """Load config.yaml, falling back to defaults for missing keys."""
    config_path = _CONFIG_DIR / "config.yaml"
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    return {**_DEFAULTS, **cfg}
