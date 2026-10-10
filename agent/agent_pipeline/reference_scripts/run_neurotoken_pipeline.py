#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin runner for the NeuroToken demo pipeline.

This script only orchestrates the existing stage scripts. It does not implement
packaging, tokenization, documentation, or validation logic itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from neurotoken_config import cfg_path, load_config
from neurotoken_report import default_report_path, make_report, write_report

STAGE_ORDER = ["preflight", "package", "tokenize", "readme", "docs", "validate", "reconstruct"]
STAGE_SCRIPTS = {
    "preflight": "0_preflight_check.py",
    "package": "1_make_tar_packages4BrainToken_final.py",
    "tokenize": "2_neural2token_human_new.py",
    "readme": "3_make_readme.py",
    "docs": "4_generate_release_docs_and_schemas.py",
    "validate": "5_validate_neurotoken_dataset.py",
    "reconstruct": "6_verify_token_reconstruction.py",
}


def load_report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc)}


def count_findings(report: dict, severity: str | None = None, category: str | None = None) -> int:
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        return 0
    count = 0
    for item in findings:
        if not isinstance(item, dict):
            continue
        if severity is not None and item.get("severity") != severity:
            continue
        if category is not None and item.get("category") != category:
            continue
        count += 1
    return count


def top_findings(report: dict, limit: int = 8) -> list[dict]:
    findings = report.get("findings", [])
    if not isinstance(findings, list):
        return []
    severity_rank = {"error": 0, "warning": 1, "info": 2}
    rows = [item for item in findings if isinstance(item, dict) and item.get("severity") in {"error", "warning"}]
    rows.sort(key=lambda item: (severity_rank.get(item.get("severity"), 9), str(item.get("category", "")), str(item.get("message", ""))))
    return rows[:limit]


def markdown_status(status: str) -> str:
    if status == "ok":
        return "OK"
    if status == "dry_run":
        return "DRY RUN"
    if status == "failed":
        return "FAILED"
    if not status:
        return "NOT RUN"
    return status.upper()


def executed_stage_status(pipeline_report: dict, stage_name: str) -> str:
    for stage in pipeline_report.get("stages", []):
        if isinstance(stage, dict) and stage.get("stage") == stage_name:
            return str(stage.get("status", ""))
    return ""


def load_stage_report_if_current(pipeline_report: dict, report_root: Path, stage_name: str, filename: str) -> dict:
    status = executed_stage_status(pipeline_report, stage_name)
    if not status or status == "dry_run":
        return {}
    return load_report(report_root / filename)


def build_human_summary_markdown(pipeline_report: dict, report_root: Path) -> str:
    preflight = load_stage_report_if_current(pipeline_report, report_root, "preflight", "preflight_report.json")
    validate = load_stage_report_if_current(pipeline_report, report_root, "validate", "validate_report.json")
    reconstruct = load_stage_report_if_current(pipeline_report, report_root, "reconstruct", "reconstruct_report.json")

    lines = [
        "# NeuroToken Pipeline Summary",
        "",
        f"- Overall status: **{markdown_status(str(pipeline_report.get('status', '')))}**",
        f"- Config: `{pipeline_report.get('config', '')}`",
        f"- Report root: `{report_root}`",
        f"- Created at: `{pipeline_report.get('created_at', '')}`",
        f"- Smoke test: `{pipeline_report.get('smoke_test', False)}`",
        "",
        "## Stage Results",
        "",
        "| Stage | Status | Duration (s) |",
        "| --- | --- | ---: |",
    ]
    for stage in pipeline_report.get("stages", []):
        if not isinstance(stage, dict):
            continue
        lines.append(
            f"| {stage.get('stage', '')} | {markdown_status(str(stage.get('status', '')))} | {stage.get('duration_seconds', '')} |"
        )

    lines += [
        "",
        "## Validation",
        "",
        "| Check | Status | Errors | Warnings | Notes |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    lines.append(
        "| Preflight | "
        f"{markdown_status(str(preflight.get('status', '')))} | "
        f"{count_findings(preflight, 'error')} | "
        f"{count_findings(preflight, 'warning')} | "
        f"HDF5 files inspected: {len(preflight.get('h5_files', []) if isinstance(preflight.get('h5_files'), list) else [])} |"
    )
    validate_summary = validate.get("summary", {}) if isinstance(validate.get("summary"), dict) else {}
    lines.append(
        "| Release structure | "
        f"{markdown_status(str(validate.get('status', '')))} | "
        f"{validate_summary.get('errors', count_findings(validate, 'error'))} | "
        f"{validate_summary.get('warnings', count_findings(validate, 'warning'))} | "
        f"Info: {validate_summary.get('infos', count_findings(validate, 'info'))} |"
    )
    reconstruct_sessions = reconstruct.get("sessions", []) if isinstance(reconstruct.get("sessions"), list) else []
    lines.append(
        "| Token reconstruction | "
        f"{markdown_status(str(reconstruct.get('status', '')))} | "
        f"{count_findings(reconstruct, 'error')} | "
        f"{count_findings(reconstruct, 'warning')} | "
        f"Sessions checked: {len(reconstruct_sessions)} |"
    )

    if reconstruct_sessions:
        lines += [
            "",
            "## Reconstruction Metrics",
            "",
            "| Session | Windows | Pearson | MSE | MAE | Relative L2 | SSIM |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for session in reconstruct_sessions:
            if not isinstance(session, dict):
                continue
            metrics = session.get("metrics_mean", {}) if isinstance(session.get("metrics_mean"), dict) else {}
            lines.append(
                f"| {session.get('session_id', '')} | "
                f"{session.get('evaluated_windows', '')} | "
                f"{float(metrics.get('pearson', float('nan'))):.4f} | "
                f"{float(metrics.get('mse', float('nan'))):.6f} | "
                f"{float(metrics.get('mae', float('nan'))):.6f} | "
                f"{float(metrics.get('relative_l2', float('nan'))):.4f} | "
                f"{float(metrics.get('ssim', float('nan'))):.4f} |"
            )

    notable = []
    for label, report in [("Preflight", preflight), ("Release structure", validate), ("Token reconstruction", reconstruct)]:
        for item in top_findings(report):
            notable.append((label, item))
    if notable:
        lines += ["", "## Notable Findings", ""]
        for label, item in notable[:12]:
            lines.append(
                f"- **{label} / {item.get('severity', '').upper()} / {item.get('category', '')}**: {item.get('message', '')}"
            )
    else:
        lines += ["", "## Notable Findings", "", "No errors or warnings were reported by the validation stages."]

    lines += [
        "",
        "## Report Files",
        "",
        "- `pipeline_report.json`: machine-readable stage execution report",
        "- `preflight_report.json`: input/config/tokenizer preflight checks",
        "- `validate_report.json`: release structure and metadata validation",
        "- `reconstruct_report.json`: token reconstruction quality metrics",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_human_summary(report_root: Path, pipeline_report: dict) -> Path:
    path = report_root / "pipeline_summary.md"
    path.write_text(build_human_summary_markdown(pipeline_report, report_root), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    demo_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Run NeuroToken demo stages in order.")
    parser.add_argument(
        "--config",
        type=Path,
        default=demo_root / "configs" / "nod_eeg_demo.toml",
        help="TOML pipeline config file.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=STAGE_ORDER,
        help="Stages to run, in the order provided.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run each stage.")
    parser.add_argument("--max-samples", type=int, default=None, help="Forwarded to tokenize stage for smoke tests.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Forwarded to tokenize stage for smoke tests.")
    parser.add_argument("--max-reconstruction-samples", type=int, default=None, help="Forwarded to reconstruct stage.")
    parser.add_argument("--max-reconstruction-sessions", type=int, default=None, help="Forwarded to reconstruct stage.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end check. Defaults to --max-sessions 1 --max-samples 2 and allows README placeholders.",
    )
    parser.add_argument("--report-root", type=Path, default=None, help="Directory for machine-readable stage reports.")
    parser.add_argument("--overwrite-token-output", action="store_true", help="Forward --overwrite to the tokenize stage.")
    parser.add_argument("--allow-placeholder-readme", action="store_true", help="Forward --allow-placeholder to README generation.")
    parser.add_argument("--strict-validation", action="store_true", help="Forward --strict to validation.")
    parser.add_argument(
        "--check-event-duration-semantics",
        action="store_true",
        help="Forward optional event duration semantics warning to validation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def build_command(args: argparse.Namespace, stage: str, script_dir: Path, config_path: Path, report_root: Path) -> list[str]:
    cmd = [args.python, "-B", str(script_dir / STAGE_SCRIPTS[stage]), "--config", str(config_path)]
    if stage == "preflight":
        cmd += ["--report-path", str(report_root / "preflight_report.json")]
        if args.smoke_test:
            cmd += ["--max-files", "1"]
    elif stage == "tokenize":
        max_samples = args.max_samples
        max_sessions = args.max_sessions
        if args.smoke_test:
            max_samples = 2 if max_samples is None else max_samples
            max_sessions = 1 if max_sessions is None else max_sessions
        if max_samples is not None:
            cmd += ["--max-samples", str(max_samples)]
        if max_sessions is not None:
            cmd += ["--max-sessions", str(max_sessions)]
        if args.overwrite_token_output:
            cmd.append("--overwrite")
    elif stage == "readme":
        if args.allow_placeholder_readme or args.smoke_test:
            cmd.append("--allow-placeholder")
    elif stage == "validate":
        cmd += ["--json-report", str(report_root / "validate_report.json")]
        if args.strict_validation:
            cmd.append("--strict")
        if args.check_event_duration_semantics:
            cmd.append("--check-event-duration-semantics")
    elif stage == "reconstruct":
        cmd += ["--report-path", str(report_root / "reconstruct_report.json")]
        max_samples = args.max_reconstruction_samples
        max_sessions = args.max_reconstruction_sessions
        if args.smoke_test:
            max_samples = 2 if max_samples is None else max_samples
            max_sessions = 1 if max_sessions is None else max_sessions
        if max_samples is not None:
            cmd += ["--max-samples", str(max_samples)]
        if max_sessions is not None:
            cmd += ["--max-sessions", str(max_sessions)]
    return cmd


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    pipeline_config = load_config(config_path)
    report_root = Path(args.report_root or cfg_path(pipeline_config, "paths.report_root", script_dir / "reports")).resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    stage_reports = []
    for stage in args.stages:
        cmd = build_command(args, stage, script_dir, config_path, report_root)
        print("\n===", stage, "===")
        print(" ".join(cmd))
        start = time.time()
        returncode = 0
        try:
            if not args.dry_run:
                completed = subprocess.run(cmd, cwd=script_dir, check=True)
                returncode = int(completed.returncode)
        except subprocess.CalledProcessError as exc:
            returncode = int(exc.returncode)
            stage_reports.append({
                "stage": stage,
                "command": cmd,
                "status": "failed",
                "returncode": returncode,
                "duration_seconds": round(time.time() - start, 3),
            })
            write_report(
                report_root / "pipeline_report.json",
                make_report(
                    "pipeline",
                    status="failed",
                    config=str(config_path),
                    report_root=str(report_root),
                    smoke_test=bool(args.smoke_test),
                    stages=stage_reports,
                ),
            )
            failed_report = load_report(report_root / "pipeline_report.json")
            summary_path = write_human_summary(report_root, failed_report)
            print(f"\nHuman summary report: {summary_path}")
            raise
        stage_reports.append({
            "stage": stage,
            "command": cmd,
            "status": "dry_run" if args.dry_run else "ok",
            "returncode": returncode,
            "duration_seconds": round(time.time() - start, 3),
        })

    pipeline_report = make_report(
        "pipeline",
        status="dry_run" if args.dry_run else "ok",
        config=str(config_path),
        report_root=str(report_root),
        smoke_test=bool(args.smoke_test),
        stages=stage_reports,
    )
    write_report(report_root / "pipeline_report.json", pipeline_report)
    summary_path = write_human_summary(report_root, pipeline_report)
    print(f"\nPipeline report: {report_root / 'pipeline_report.json'}")
    print(f"Human summary report: {summary_path}")


if __name__ == "__main__":
    main()
