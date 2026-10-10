"""Tool functions used by the LangGraph pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys
import tempfile

from agent_pipeline.adapters import build_adapter
from agent_pipeline.config import cfg_get, cfg_path, load_config
from agent_pipeline.package import event_sample_bounds, package_all, require_cfg
from agent_pipeline.pipeline_stages import REFERENCE_STAGE_SCRIPTS, STAGE_ORDER


def _bounded_text_tail(path: Path, max_bytes: int = 32_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        size = path.stat().st_size
        if size > max_bytes:
            stream.seek(-max_bytes, os.SEEK_END)
        return stream.read().decode("utf-8", errors="replace").strip()


def load_config_tool(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "config": config,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "dataset": cfg_get(config, "dataset.name"),
        "modality": cfg_get(config, "dataset.modality"),
        "adapter": cfg_get(config, "source.adapter"),
    }


def preflight_tool(config: dict[str, Any], max_files: int | None = None, target_h5: str | None = None) -> dict[str, Any]:
    h5_root = cfg_path(config, "paths.h5_root", None)
    strategy = str(cfg_get(config, "chunking.strategy", "rest")).strip().lower()
    if strategy not in {"rest", "event"}:
        return {"status": "failed", "findings": [{"severity": "error", "category": "config", "message": "chunking.strategy must be 'rest' or 'event'"}]}
    split_points = int(require_cfg(config, "chunking.split_points")) if strategy == "rest" else 1
    adapter = build_adapter(config)
    files = [Path(target_h5).expanduser().resolve()] if target_h5 else adapter.discover(h5_root)
    if max_files is not None and not target_h5:
        files = files[:max_files]

    findings = []
    summaries = []
    if not files:
        findings.append({"severity": "error", "category": "input_data", "message": f"No .h5 files found under {h5_root}"})
    for path in files:
        summary, file_findings = adapter.inspect_file(path, split_points)
        if strategy == "event" and not any(item.get("severity") == "error" for item in file_findings):
            session = adapter.load_session(path)
            summary["n_events"] = len(session.events)
            summary["estimated_chunks"] = 0
            if not session.events:
                file_findings.append({"severity": "error", "category": "events", "message": f"{path.name} has no events for event chunking", "file": path.name})
            for event_index, event in enumerate(session.events):
                try:
                    event_sample_bounds(event, float(session.sfreq), int(session.signal.shape[1]))
                    summary["estimated_chunks"] += 1
                except (KeyError, TypeError, ValueError) as exc:
                    file_findings.append({"severity": "error", "category": "events", "message": f"{path.name} event {event_index}: {exc}", "file": path.name})
        summaries.append(summary)
        findings.extend(file_findings)

    status = "ok" if not any(item["severity"] == "error" for item in findings) else "failed"
    return {
        "status": status,
        "adapter": adapter.__class__.__name__,
        "chunking_strategy": strategy,
        "h5_root": str(h5_root),
        "h5_files": summaries,
        "findings": findings,
    }


def package_tool(config: dict[str, Any], max_files: int | None = None, target_h5: str | None = None) -> dict[str, Any]:
    return package_all(config, max_files=max_files, target_h5=target_h5)


def stage_command(
    stage: str,
    config_path: str,
    report_root: Path,
    reference_scripts_dir: Path,
    python: str,
    smoke_test: bool,
    overwrite_token_output: bool,
    allow_placeholder_readme: bool,
    strict_validation: bool,
) -> list[str]:
    cmd = [python, "-B", str(reference_scripts_dir / REFERENCE_STAGE_SCRIPTS[stage]), "--config", config_path]
    if stage == "tokenize":
        if smoke_test:
            cmd += ["--max-sessions", "1", "--max-samples", "2"]
        if overwrite_token_output:
            cmd.append("--overwrite")
    elif stage == "readme":
        if allow_placeholder_readme or smoke_test:
            cmd.append("--allow-placeholder")
    elif stage == "validate":
        cmd += ["--json-report", str(report_root / "validate_report.json")]
        if strict_validation:
            cmd.append("--strict")
    elif stage == "reconstruct":
        cmd += ["--report-path", str(report_root / "reconstruct_report.json")]
        if smoke_test:
            cmd += ["--max-sessions", "1", "--max-samples", "2"]
    return cmd


def tokenize_postcondition_failure(config_path: str) -> dict[str, Any] | None:
    config = load_config(config_path)
    output_base = cfg_path(config, "paths.token_root", None)
    if output_base is None:
        return {
            "severity": "error",
            "category": "tokenization_output",
            "message": "Tokenization exited successfully but paths.token_root is not configured.",
        }
    manifest_path = output_base / "manifest.json"
    if not manifest_path.is_file():
        return {
            "severity": "error",
            "category": "tokenization_output",
            "message": f"Tokenization exited successfully but did not produce {manifest_path}.",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "severity": "error",
            "category": "tokenization_output",
            "message": f"Tokenization produced an unreadable manifest at {manifest_path}: {exc}",
        }
    sessions = manifest.get("sessions")
    num_sessions = manifest.get("num_sessions")
    if not isinstance(sessions, list) or not isinstance(num_sessions, int):
        return {
            "severity": "error",
            "category": "tokenization_output",
            "message": f"Tokenization manifest at {manifest_path} is missing valid sessions/num_sessions fields.",
        }
    if num_sessions != len(sessions):
        return {
            "severity": "error",
            "category": "tokenization_output",
            "message": (
                f"Tokenization manifest is inconsistent: num_sessions={num_sessions}, "
                f"len(sessions)={len(sessions)}."
            ),
        }
    if num_sessions == 0:
        failed_sessions = manifest.get("failed_sessions", [])
        return {
            "severity": "error",
            "category": "tokenization_no_sessions",
            "message": (
                "Tokenization produced zero successful sessions"
                f"; {len(failed_sessions) if isinstance(failed_sessions, list) else 0} session(s) failed. "
                f"See {manifest_path} and the worker stderr for the underlying errors."
            ),
        }
    return None


def run_reference_stage_tool(
    stage: str,
    config_path: str,
    report_root: str,
    reference_scripts_dir: str,
    python: str = sys.executable,
    smoke_test: bool = False,
    overwrite_token_output: bool = False,
    allow_placeholder_readme: bool = False,
    strict_validation: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    cmd = stage_command(
        stage=stage,
        config_path=config_path,
        report_root=Path(report_root),
        reference_scripts_dir=Path(reference_scripts_dir),
        python=python,
        smoke_test=smoke_test,
        overwrite_token_output=overwrite_token_output,
        allow_placeholder_readme=allow_placeholder_readme,
        strict_validation=strict_validation,
    )
    if dry_run:
        return {"stage": stage, "status": "dry_run", "command": cmd}
    stdout_path = os.environ.get("NEUROTOKEN_STAGE_STDOUT_LOG")
    stderr_path = os.environ.get("NEUROTOKEN_STAGE_STDERR_LOG")
    if stdout_path and stderr_path:
        with Path(stdout_path).open("a", encoding="utf-8") as stdout_file, Path(stderr_path).open("a", encoding="utf-8") as stderr_file:
            completed = subprocess.run(
                cmd,
                cwd=str(reference_scripts_dir),
                check=False,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        stdout = _bounded_text_tail(Path(stdout_path))
        stderr = _bounded_text_tail(Path(stderr_path))
    else:
        completed = subprocess.run(
            cmd,
            cwd=str(reference_scripts_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
    result = {
        "stage": stage,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "command": cmd,
        "stdout": stdout,
        "stderr": stderr,
    }
    report_path = Path(report_root) / f"{stage}_report.json"
    if report_path.exists():
        result["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        result["report_path"] = str(report_path)
    if completed.returncode != 0:
        return result
    if stage == "tokenize":
        postcondition_failure = tokenize_postcondition_failure(config_path)
        if postcondition_failure is not None:
            result["status"] = "failed"
            result["findings"] = [postcondition_failure]
            result["report"] = {
                "status": "failed",
                "summary": {"errors": 1},
                "findings": [postcondition_failure],
            }
            return result
    if stage == "reconstruct":
        report_path = Path(report_root) / "reconstruct_report.json"
        if report_path.exists():
            result.update(json.loads(report_path.read_text(encoding="utf-8")))
            result["report_path"] = str(report_path)
    return result


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
