"""Shared config loader — reads config.yaml from project root."""

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULTS = {
    "max_trials": 10,
    "vtune_enabled": True,
    "vtune_bin": "/bin64/vtune",
}


def load_config() -> dict:
    """Load config.yaml, falling back to defaults for missing keys."""
    config_path = _CONFIG_DIR / "config.yaml"
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    return {**_DEFAULTS, **cfg}
