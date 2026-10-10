"""Dataset source adapters for Stage 1 packaging."""

from __future__ import annotations

from typing import Any

from .eeg_h5 import EegH5Adapter


def build_adapter(config: dict[str, Any]):
    adapter_name = config.get("source", {}).get("adapter")
    if not adapter_name:
        raise ValueError("Missing required config field: source.adapter")
    adapter_name = str(adapter_name).strip().lower()
    if adapter_name in {"eeg_h5", "nod_eeg_h5", "h5_eeg"}:
        return EegH5Adapter(config)
    raise ValueError(f"Unsupported source.adapter={adapter_name!r}")
