"""Source adapter registry."""

from __future__ import annotations

from importlib import import_module, invalidate_caches
import re
from typing import Any

from agent_pipeline.adapters.eeg_h5 import EegH5Adapter
from agent_pipeline.adapters.meg_h5 import MEGH5Adapter


ADAPTER_MODULE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _generated_adapter(config: dict[str, Any], adapter_name: str):
    if not ADAPTER_MODULE_PATTERN.fullmatch(adapter_name):
        return None
    invalidate_caches()
    try:
        module = import_module(f"agent_pipeline.adapters.{adapter_name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"agent_pipeline.adapters.{adapter_name}":
            return None
        raise
    class_name = "".join(part.capitalize() for part in adapter_name.split("_")) + "Adapter"
    adapter_class = getattr(module, class_name, None)
    return adapter_class(config) if isinstance(adapter_class, type) else None


def build_adapter(config: dict[str, Any]):
    adapter_name = str(config.get("source", {}).get("adapter", "")).strip().lower()
    if adapter_name in {"meg_h5", "meg"}:
        return MEGH5Adapter(config)
    if adapter_name in {"eeg_h5", "eeg"}:
        return EegH5Adapter(config)
    adapter = _generated_adapter(config, adapter_name)
    if adapter is not None:
        return adapter
    raise ValueError(f"Unsupported source.adapter={adapter_name!r}")
