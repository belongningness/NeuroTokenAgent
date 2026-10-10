#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 0: preflight checks before packaging/tokenization.

This stage intentionally does not import the UniTok project. It checks paths,
source HDF5 structure, release placeholders, and tokenizer filesystem inputs so
the pipeline can fail early with a machine-readable report.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Optional

from adapters import build_adapter
from neurotoken_config import cfg_chunking_strategy, cfg_get, cfg_path, load_config
from neurotoken_report import default_report_path, make_report, write_report


PLACEHOLDERS = ("TODO", "Please fill in", "TODO_DATASET_NAME")


def has_unitok_weight_files(path: Path) -> bool:
    expected = ["generator.pt", "ema_generator.pt", "checkpoint.pt", "model.pt", "ema_model.pt"]
    return path.exists() and any((path / name).exists() for name in expected)


def find_checkpoint_candidate(checkpoint_root: Path) -> Optional[Path]:
    checkpoint_root = Path(checkpoint_root)
    if checkpoint_root.is_file() or has_unitok_weight_files(checkpoint_root):
        return checkpoint_root
    if not checkpoint_root.exists():
        return None
    candidates = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir() or not has_unitok_weight_files(child):
            continue
        base = child.name.replace("_final", "")
        if base.isdigit():
            candidates.append((int(base), child.name.endswith("_final"), child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def add_finding(findings: list[dict], severity: str, category: str, message: str, **fields: Any) -> None:
    item = {"severity": severity, "category": category, "message": message}
    item.update(fields)
    findings.append(item)


def require_cfg(config: dict, dotted_key: str) -> Any:
    value = cfg_get(config, dotted_key, None)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def require_cfg_path(config: dict, dotted_key: str) -> Path:
    value = cfg_path(config, dotted_key, None)
    if value is None:
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def check_module(findings: list[dict], module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        add_finding(findings, "error", "environment", f"Missing Python module: {module_name}")


def check_release_fields(config: dict, findings: list[dict]) -> None:
    for key in ["dataset_name", "description", "tokenizer_version"]:
        value = str(cfg_get(config, f"release.{key}", ""))
        if not value.strip() or any(marker in value for marker in PLACEHOLDERS):
            add_finding(findings, "warning", "release_metadata", f"release.{key} is empty or still a placeholder")


def check_h5_inputs(config: dict, findings: list[dict], max_files: Optional[int] = None) -> list[dict]:
    try:
        h5_root = require_cfg_path(config, "paths.h5_root")
    except Exception as exc:
        add_finding(findings, "error", "config", str(exc))
        return []
    summaries: list[dict] = []
    if h5_root is None or not h5_root.exists():
        add_finding(findings, "error", "input_data", f"HDF5 root does not exist: {h5_root}")
        return summaries

    adapter = build_adapter(config)
    files = adapter.discover(h5_root)
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        add_finding(findings, "error", "input_data", f"No .h5 files found under {h5_root}")
        return summaries

    try:
        split_points = int(require_cfg(config, "chunking.split_points"))
        downsample_factor = int(require_cfg(config, "chunking.downsample_factor"))
    except Exception as exc:
        add_finding(findings, "error", "config", str(exc))
        return summaries
    if downsample_factor <= 0 or split_points % downsample_factor != 0:
        add_finding(findings, "error", "config", "chunking.split_points must be divisible by chunking.downsample_factor")
    try:
        cfg_chunking_strategy(config)
    except ValueError as exc:
        add_finding(findings, "error", "config", str(exc))

    for path in files:
        summary, adapter_findings = adapter.inspect_file(Path(path), split_points=split_points)
        summaries.append(summary)
        findings.extend(adapter_findings)
    return summaries


def check_tokenizer_inputs(config: dict, findings: list[dict], skip_tokenizer: bool) -> dict:
    if skip_tokenizer:
        return {"skipped": True}
    try:
        project_root = require_cfg_path(config, "tokenizer.project_root")
        checkpoint_root = require_cfg_path(config, "tokenizer.checkpoint_root")
        checkpoint_path = cfg_path(config, "tokenizer.checkpoint_path", None)
        train_config = require_cfg_path(config, "tokenizer.train_config")
    except Exception as exc:
        add_finding(findings, "error", "config", str(exc))
        return {}

    summary = {
        "project_root": str(project_root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "train_config": str(train_config),
    }
    if project_root is None or not (project_root / "modeling").exists():
        add_finding(findings, "error", "tokenizer_environment", f"UniTok project root is missing or invalid: {project_root}")
    if train_config is None or not train_config.exists():
        add_finding(findings, "error", "tokenizer_environment", f"Tokenizer train_config is missing: {train_config}")
    resolved = checkpoint_path if checkpoint_path else find_checkpoint_candidate(Path(checkpoint_root))
    summary["resolved_checkpoint"] = str(resolved) if resolved else ""
    if resolved is None or not Path(resolved).exists():
        add_finding(findings, "error", "tokenizer_environment", f"No tokenizer checkpoint found under: {checkpoint_root}")
    for module_name in ["torch", "webdataset", "pandas"]:
        check_module(findings, module_name)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0: preflight checks for the NeuroToken demo pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="TOML pipeline config file.")
    parser.add_argument("--report-path", type=Path, default=None, help="Write machine-readable JSON report.")
    parser.add_argument("--max-files", type=int, default=None, help="Inspect only the first N h5 files.")
    parser.add_argument("--skip-tokenizer-check", action="store_true", help="Skip UniTok filesystem checks.")
    parser.add_argument("--strict-release", action="store_true", help="Treat release metadata placeholders as errors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    findings: list[dict] = []

    h5_summaries = check_h5_inputs(config, findings, max_files=args.max_files)
    tokenizer_summary = check_tokenizer_inputs(config, findings, skip_tokenizer=args.skip_tokenizer_check)
    check_release_fields(config, findings)
    if args.strict_release:
        for item in findings:
            if item["category"] == "release_metadata" and item["severity"] == "warning":
                item["severity"] = "error"

    status = "ok" if not any(item["severity"] == "error" for item in findings) else "failed"
    report = make_report(
        "preflight",
        status=status,
        config=str(Path(args.config).resolve()),
        findings=findings,
        h5_files=h5_summaries,
        tokenizer=tokenizer_summary,
    )
    report_path = args.report_path or default_report_path(config, "preflight", Path("reports"))
    write_report(report_path, report)

    print(f"Preflight status: {status}")
    print(f"HDF5 files inspected: {len(h5_summaries)}")
    print(f"Findings: errors={sum(1 for x in findings if x['severity'] == 'error')} warnings={sum(1 for x in findings if x['severity'] == 'warning')}")
    print(f"Report: {report_path}")
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
