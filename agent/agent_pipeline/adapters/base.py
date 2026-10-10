"""Shared source adapter structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np


def safe_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return name.strip("._") or "unknown"


@dataclass
class SourceSession:
    source_path: Path
    source_stem: str
    safe_stem: str
    modality: str
    signal: np.ndarray
    sfreq: float
    subject: dict[str, Any]
    channels: list[dict[str, Any]]
    events: list[dict[str, Any]]
    session_metadata: dict[str, Any]
