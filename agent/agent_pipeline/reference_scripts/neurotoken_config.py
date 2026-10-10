#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small TOML config helper for the NeuroToken demo pipeline.

The processing scripts keep their original defaults. A config file only replaces
per-dataset/runtime values such as paths, event column names, chunk length, and
release metadata. Relative paths in config files are resolved relative to the
config file's directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


_CONFIG_DIR_KEY = "__config_dir__"
_CONFIG_PATH_KEY = "__config_path__"


def load_config(path: Optional[str | Path]) -> dict:
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    config[_CONFIG_PATH_KEY] = str(config_path)
    config[_CONFIG_DIR_KEY] = str(config_path.parent)
    validate_config(config)
    return config


def cfg_get(config: dict, dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def none_if_empty(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def cfg_path(config: dict, dotted_key: str, default: Any = None) -> Optional[Path]:
    value = none_if_empty(cfg_get(config, dotted_key, default))
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and config.get(_CONFIG_DIR_KEY):
        path = Path(config[_CONFIG_DIR_KEY]) / path
    return path


def cfg_bool(config: dict, dotted_key: str, default: bool = False) -> bool:
    value = cfg_get(config, dotted_key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def cfg_chunking_strategy(config: dict) -> str:
    """Return the required canonical ``rest``/``event`` chunking strategy."""
    strategy_value = none_if_empty(cfg_get(config, "chunking.strategy"))
    if strategy_value is None:
        raise ValueError("chunking.strategy is required")

    strategy = str(strategy_value).strip().lower()
    if strategy not in {"rest", "event"}:
        raise ValueError("chunking.strategy must be 'rest' or 'event'")
    return strategy


def validate_config(config: dict) -> None:
    cfg_chunking_strategy(config)
    split_points = cfg_get(config, "chunking.split_points")
    downsample_factor = cfg_get(config, "chunking.downsample_factor")
    if split_points is None:
        raise ValueError("chunking.split_points is required")
    if downsample_factor is None:
        raise ValueError("chunking.downsample_factor is required")

    split_points = int(split_points)
    downsample_factor = int(downsample_factor)
    if split_points <= 0:
        raise ValueError("chunking.split_points must be positive")
    if downsample_factor <= 0:
        raise ValueError("chunking.downsample_factor must be positive")
    if split_points % downsample_factor != 0:
        raise ValueError(
            "chunking.split_points must be divisible by chunking.downsample_factor: "
            f"{split_points} % {downsample_factor} != 0"
        )
