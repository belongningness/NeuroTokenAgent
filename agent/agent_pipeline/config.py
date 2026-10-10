"""Small TOML config helpers for the agent pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


CONFIG_DIR_KEY = "__config_dir__"
CONFIG_PATH_KEY = "__config_path__"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    config[CONFIG_PATH_KEY] = str(config_path)
    config[CONFIG_DIR_KEY] = str(config_path.parent)
    return config


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
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


def cfg_path(config: dict[str, Any], dotted_key: str, default: Any = None) -> Path | None:
    value = none_if_empty(cfg_get(config, dotted_key, default))
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and config.get(CONFIG_DIR_KEY):
        path = Path(config[CONFIG_DIR_KEY]) / path
    return path
