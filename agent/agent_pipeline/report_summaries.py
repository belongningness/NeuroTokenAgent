"""Compact report projections used in LLM prompts."""

from __future__ import annotations

from typing import Any


def compact_h5_file(row: dict[str, Any]) -> dict[str, Any]:
    attrs = row.get("attrs") or {}
    return {
        "file": row.get("file"),
        "path": row.get("path"),
        "shape": row.get("shape"),
        "sfreq": row.get("sfreq"),
        "estimated_chunks": row.get("estimated_chunks"),
        "n_events": row.get("n_events"),
        "n_channels": row.get("n_channels"),
        "subject": attrs.get("sub") or attrs.get("subject_id"),
        "task": attrs.get("task"),
        "split": attrs.get("split"),
    }


def compact_stage_report(stage: dict[str, Any]) -> dict[str, Any]:
    row = {
        "stage": stage.get("stage"),
        "status": stage.get("status"),
        "duration_seconds": stage.get("duration_seconds"),
    }
    for key in ["adapter", "h5_root", "input_base", "output_base", "report_path", "returncode"]:
        if stage.get(key) is not None:
            row[key] = stage.get(key)
    if stage.get("findings") is not None:
        row["findings"] = stage.get("findings", [])[:5]
        row["n_findings"] = len(stage.get("findings", []))
    if stage.get("h5_files") is not None:
        h5_files = stage.get("h5_files", [])
        row["n_h5_files"] = len(h5_files)
        row["first_h5_files"] = [compact_h5_file(item) for item in h5_files[:3]]
        row["total_estimated_chunks"] = sum(
            int(item.get("estimated_chunks", 0)) for item in h5_files
        )
    if stage.get("sessions") is not None:
        sessions = stage.get("sessions", [])
        row["n_sessions"] = len(sessions)
        row["samples"] = stage.get("samples")
        row["files"] = stage.get("files")
        if stage.get("stage") == "reconstruct":
            row["metrics"] = stage.get("metrics")
            row["evaluated_sessions"] = stage.get("evaluated_sessions", len(sessions))
            row["evaluated_windows"] = stage.get("evaluated_windows", sum(
                int(session.get("evaluated_windows", 0))
                for session in sessions
                if isinstance(session, dict)
            ))
            row["sessions"] = [
                {
                    "session_id": session.get("session_id"),
                    "modality": session.get("modality"),
                    "evaluated_windows": session.get("evaluated_windows"),
                    "metrics_mean": session.get("metrics_mean"),
                }
                for session in sessions[:3]
            ]
        else:
            row["sessions"] = sessions[:3]
    if stage.get("command") is not None:
        row["command"] = stage.get("command")
    return row


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "status": report.get("status"),
        "config": report.get("config"),
        "stages": [],
    }
    compact["stages"] = [compact_stage_report(stage) for stage in report.get("stages", [])]
    return compact
