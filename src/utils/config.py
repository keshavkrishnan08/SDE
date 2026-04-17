"""YAML configuration loading and merging."""

import yaml
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override into base config."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_with_overrides(
    base_path: str | Path = "configs/default.yaml",
    override_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load base config and optionally merge an override config on top."""
    config = load_config(base_path)
    if override_path is not None:
        override = load_config(override_path)
        config = merge_configs(config, override)
    return config
