#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small JSON report helpers for the NeuroToken demo pipeline."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Optional


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_report(stage: str, status: str = "ok", **fields: Any) -> dict:
    report = {
        "format": "neurotoken_stage_report",
        "version": "1.0",
        "stage": stage,
        "status": status,
        "created_at": utc_now_iso(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    report.update(fields)
    return report


def write_report(path: Optional[str | Path], report: dict) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_report_path(config: dict, stage: str, fallback_root: Path) -> Path:
    from neurotoken_config import cfg_path

    root = cfg_path(config, "paths.report_root", fallback_root)
    return Path(root) / f"{stage}_report.json"
