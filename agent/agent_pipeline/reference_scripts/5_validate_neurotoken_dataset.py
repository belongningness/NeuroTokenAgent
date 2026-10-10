#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate a NeuroToken dataset before public release.

Checks include:
- Required top-level files and directories.
- Optional JSON Schema validation when jsonschema is installed.
- manifest/session path consistency.
- <modality>_token_ids.npy shape/dtype/id range consistency with metadata/codebook.
- compact token_metadata summaries and token array consistency.
- session_metadata/events/channels sidecar structure.
- event timing range against session duration.
- common public-release warnings, such as placeholder docs and absolute paths.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neurotoken_config import cfg_bool, cfg_path, load_config
from neurotoken_report import default_report_path, make_report, write_report

import numpy as np

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    jsonschema = None

ABSOLUTE_PATH_RE = re.compile(r"(?:/share/project|/home/|/root/|/Users/|[A-Za-z]:\\\\)")
PLACEHOLDER_RE = re.compile(r"TODO|Please fill in|empty_placeholder", re.IGNORECASE)


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def print(self) -> None:
        print("\n=== NeuroToken validation report ===")
        print(f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)} | Info: {len(self.infos)}")
        if self.errors:
            print("\nErrors:")
            for idx, message in enumerate(self.errors, start=1):
                print(f"  {idx}. {message}")
        if self.warnings:
            print("\nWarnings:")
            for idx, message in enumerate(self.warnings, start=1):
                print(f"  {idx}. {message}")
        if self.infos:
            print("\nInfo:")
            for idx, message in enumerate(self.infos, start=1):
                print(f"  {idx}. {message}")

    @staticmethod
    def classify(severity: str, message: str) -> str:
        text = message.lower()
        if severity == "error":
            if any(word in text for word in ["schema", "missing", "path", "json", "shape", "dtype", "manifest"]):
                return "format_error"
            return "release_blocker"
        if any(word in text for word in ["privacy", "consent", "subject.", "behavioral"]):
            return "privacy_warning"
        if any(word in text for word in ["placeholder", "absolute/local path", "todo"]):
            return "release_blocker"
        if any(word in text for word in ["event", "duration", "token", "channel", "short"]):
            return "quality_warning"
        return "info" if severity == "info" else "quality_warning"

    def findings(self) -> List[dict]:
        rows: List[dict] = []
        for severity, messages in [
            ("error", self.errors),
            ("warning", self.warnings),
            ("info", self.infos),
        ]:
            for message in messages:
                rows.append({
                    "severity": severity,
                    "category": self.classify(severity, message),
                    "message": message,
                })
        return rows

    def to_json_report(self, root: Path, strict: bool) -> dict:
        status = "failed" if self.errors or (strict and self.warnings) else "ok"
        return make_report(
            "validate",
            status=status,
            output_base=str(root),
            strict=bool(strict),
            summary={
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "infos": len(self.infos),
            },
            findings=self.findings(),
        )


def load_json(path: Path, report: ValidationReport) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        report.error(f"Failed to parse JSON {path}: {exc}")
        return None


def rel_path(path_value: Any) -> Optional[Path]:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return None
    return path


def file_exists(root: Path, rel: Any) -> bool:
    relp = rel_path(rel)
    return bool(relp is not None and (root / relp).exists())


def scan_json_text_for_release_warnings(path: Path, report: ValidationReport) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        report.warning(f"Cannot scan {path}: {exc}")
        return
    if ABSOLUTE_PATH_RE.search(text):
        report.warning(f"Possible absolute/local path found in {path}")
    if PLACEHOLDER_RE.search(text):
        report.warning(f"Placeholder text found in {path}; fill before public release")


def validate_with_schema(instance: Any, schema_path: Path, report: ValidationReport, label: str) -> None:
    if not schema_path.exists():
        report.warning(f"Schema missing for {label}: {schema_path}")
        return
    schema = load_json(schema_path, report)
    if schema is None:
        return
    if jsonschema is None:
        report.info(f"jsonschema not installed; skipped schema validation for {label}")
        return
    try:
        jsonschema.Draft202012Validator(schema).validate(instance)
    except Exception as exc:
        report.error(f"Schema validation failed for {label}: {exc}")


def validate_required_top_level(root: Path, report: ValidationReport) -> None:
    required_files = [
        "manifest.json",
        "subjects.json",
        "README.md",
        "DATA_DICTIONARY.md",
        "tokenizer/codebook.npz",
        "tokenizer/codebook_metadata.json",
    ]
    required_dirs = ["sessions", "tokenizer", "schemas"]
    for rel in required_files:
        path = root / rel
        if not path.exists():
            report.error(f"Missing required file: {rel}")
    for rel in required_dirs:
        path = root / rel
        if not path.is_dir():
            report.error(f"Missing required directory: {rel}")


def validate_codebook(root: Path, manifest: dict, report: ValidationReport) -> dict:
    tokenizer_info = manifest.get("tokenizer", {}) if isinstance(manifest, dict) else {}
    codebook_rel = tokenizer_info.get("codebook", "tokenizer/codebook.npz")
    metadata_rel = tokenizer_info.get("codebook_metadata", "tokenizer/codebook_metadata.json")
    codebook_path = root / codebook_rel if rel_path(codebook_rel) is not None else root / "tokenizer/codebook.npz"
    metadata_path = root / metadata_rel if rel_path(metadata_rel) is not None else root / "tokenizer/codebook_metadata.json"
    metadata = load_json(metadata_path, report) if metadata_path.exists() else {}
    if metadata_path.exists():
        validate_with_schema(metadata, root / "schemas" / "codebook_metadata.schema.json", report, "codebook_metadata.json")
        scan_json_text_for_release_warnings(metadata_path, report)
    else:
        report.error(f"Missing codebook metadata: {metadata_path}")

    if not codebook_path.exists():
        report.error(f"Missing codebook file: {codebook_path}")
        return metadata or {}

    try:
        z = np.load(codebook_path)
        expected_keys = {"codebook_raw", "codebook_normalized"}
        missing = expected_keys - set(z.files)
        if missing:
            report.error(f"codebook.npz missing arrays: {sorted(missing)}")
        if "codebook_raw" in z.files and "codebook_normalized" in z.files:
            raw_shape = tuple(int(v) for v in z["codebook_raw"].shape)
            normalized_shape = tuple(int(v) for v in z["codebook_normalized"].shape)
            if raw_shape != normalized_shape:
                report.error(f"codebook_raw shape {raw_shape} does not match codebook_normalized shape {normalized_shape}")
        for array_name in expected_keys.intersection(z.files):
            if not np.isfinite(z[array_name]).all():
                report.error(f"codebook.npz array {array_name} contains NaN/Inf")
        if "codebook_normalized" in z.files:
            shape = tuple(int(v) for v in z["codebook_normalized"].shape)
            num_codebooks = int(metadata.get("num_codebooks", shape[0] if shape else -1)) if metadata else shape[0]
            vocab_per = int(metadata.get("vocab_size_per_codebook", shape[1] if len(shape) > 1 else -1)) if metadata else shape[1]
            width_per = int(metadata.get("width_per_codebook", shape[2] if len(shape) > 2 else -1)) if metadata else shape[2]
            if len(shape) != 3:
                report.error(f"codebook_normalized should be 3D, got shape {shape}")
            else:
                if shape[0] != num_codebooks or shape[1] != vocab_per or shape[2] != width_per:
                    report.error(
                        "codebook_normalized shape does not match codebook_metadata: "
                        f"shape={shape}, metadata=({num_codebooks}, {vocab_per}, {width_per})"
                    )
                report.info(f"Codebook shape: {shape}")
    except Exception as exc:
        report.error(f"Failed to load codebook.npz: {exc}")
    return metadata or {}


def validate_subject(subject: Any, report: ValidationReport, label: str) -> None:
    if not isinstance(subject, dict):
        report.error(f"{label}.Subject is not an object")
        return
    for key in ["ID", "Age", "Sex", "Health_group"]:
        if key not in subject:
            report.error(f"{label}.Subject missing required key: {key}")
    if subject.get("Age") and not isinstance(subject.get("Age"), str):
        report.warning(f"{label}.Subject.Age should be stored as a string or age bin")


def compact_subject(subject: Any) -> dict:
    subject = subject if isinstance(subject, dict) else {}
    return {
        "ID": str(subject.get("ID", "") or ""),
        "Age": str(subject.get("Age", "") or ""),
        "Sex": str(subject.get("Sex", "") or ""),
        "Health_group": str(subject.get("Health_group", "") or ""),
    }


def validate_subjects_index(root: Path, manifest: dict, report: ValidationReport) -> None:
    path = root / "subjects.json"
    if not path.exists():
        report.error("Missing required file: subjects.json")
        return
    subjects = load_json(path, report)
    if subjects is None:
        return
    validate_with_schema(subjects, root / "schemas" / "subjects.schema.json", report, "subjects.json")
    scan_json_text_for_release_warnings(path, report)
    if not isinstance(subjects, list):
        report.error("subjects.json must be a JSON array")
        return

    sessions = manifest.get("sessions", []) if isinstance(manifest, dict) else []
    manifest_sessions = {str(session.get("session_id", "")): session for session in sessions if isinstance(session, dict)}
    covered_session_ids: List[str] = []
    seen_subject_ids = set()

    for idx, subject_entry in enumerate(subjects):
        label = f"subjects.json[{idx}]"
        if not isinstance(subject_entry, dict):
            report.error(f"{label} is not an object")
            continue
        extra_keys = set(subject_entry) - {"ID", "Age", "Sex", "Health_group", "session_ids"}
        if extra_keys:
            report.error(f"{label} contains unexpected keys: {sorted(extra_keys)}")
        for key in ["ID", "Age", "Sex", "Health_group"]:
            if key not in subject_entry:
                report.error(f"{label} missing required key: {key}")
            elif not isinstance(subject_entry.get(key), str):
                report.error(f"{label}.{key} must be a string")
        session_ids = subject_entry.get("session_ids")
        if not isinstance(session_ids, list) or not all(isinstance(v, str) for v in session_ids):
            report.error(f"{label}.session_ids must be a string array")
            continue
        if not session_ids:
            report.warning(f"{label}.session_ids is empty")
        covered_session_ids.extend(session_ids)

        subject_id = subject_entry.get("ID", "")
        if subject_id:
            if subject_id in seen_subject_ids:
                report.error(f"subjects.json contains duplicate subject ID: {subject_id!r}")
            seen_subject_ids.add(subject_id)

        compact_entry = compact_subject(subject_entry)
        for session_id in session_ids:
            session = manifest_sessions.get(session_id)
            if session is None:
                report.error(f"{label}.session_ids contains unknown session_id: {session_id!r}")
                continue
            compact_session_subject = compact_subject(session.get("subject"))
            if compact_entry != compact_session_subject:
                report.error(
                    f"{label} subject fields do not match manifest session {session_id!r}: "
                    f"subjects.json={compact_entry}, manifest={compact_session_subject}"
                )

    duplicate_session_ids = sorted({sid for sid in covered_session_ids if covered_session_ids.count(sid) > 1})
    if duplicate_session_ids:
        report.error(f"subjects.json lists session_id(s) more than once: {duplicate_session_ids}")
    missing_session_ids = sorted(set(manifest_sessions) - set(covered_session_ids))
    if missing_session_ids:
        report.error(f"subjects.json is missing manifest session_id(s): {missing_session_ids}")


def validate_events(
    events: Any,
    session_metadata: dict,
    report: ValidationReport,
    label: str,
    check_duration_semantics: bool = False,
) -> None:
    if not isinstance(events, list):
        report.error(f"{label} should be a JSON array")
        return
    sfreq = session_metadata.get("sfreq")
    n_samples_total = session_metadata.get("n_samples_total")
    duration = None
    if isinstance(sfreq, (int, float)) and isinstance(n_samples_total, int) and sfreq > 0:
        duration = float(n_samples_total) / float(sfreq)
    prev_start = -math.inf
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            report.error(f"{label}[{idx}] is not an object")
            continue
        for key in ["start_time", "end_time", "type", "value", "other_info"]:
            if key not in event:
                report.error(f"{label}[{idx}] missing key: {key}")
        if "type" in event and not isinstance(event.get("type"), str):
            report.error(f"{label}[{idx}].type must be a string")
        if "value" in event and not isinstance(event.get("value"), str):
            report.error(f"{label}[{idx}].value must be a string")
        start = event.get("start_time")
        end = event.get("end_time")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            report.error(f"{label}[{idx}] start_time/end_time must be numbers")
            continue
        if end < start:
            report.error(f"{label}[{idx}] end_time < start_time")
        if start < prev_start:
            report.warning(f"{label}[{idx}] start_time is not sorted ascending")
        prev_start = start
        if duration is not None and (start < -1e-6 or end > duration + 1e-6):
            report.warning(f"{label}[{idx}] time range [{start}, {end}] outside recording duration {duration:.6f}s")
        other = event.get("other_info")
        if not isinstance(other, dict):
            report.error(f"{label}[{idx}].other_info must be an object")
        elif check_duration_semantics and "duration" in other and isinstance(other["duration"], (int, float)):
            event_duration = float(end) - float(start)
            source_duration = float(other["duration"])
            if abs(event_duration - source_duration) > 0.1 and idx < 5:
                report.warning(
                    f"{label}[{idx}] end-start ({event_duration:.6f}s) differs from other_info.duration "
                    f"({source_duration:.6f}s); document timing semantics"
                )


def validate_channels(channels: Any, expected_channels: Optional[int], report: ValidationReport, label: str) -> None:
    if not isinstance(channels, list):
        report.error(f"{label} should be a JSON array")
        return
    if expected_channels is not None and len(channels) != expected_channels:
        report.error(f"{label} length {len(channels)} does not match expected tokenized channel count {expected_channels}")
    for idx, channel in enumerate(channels):
        if not isinstance(channel, dict):
            report.error(f"{label}[{idx}] is not an object")
            continue
        for key in ["channel_name", "channel_type", "mean", "std"]:
            if key not in channel:
                report.error(f"{label}[{idx}] missing key: {key}")
        for dropped_key in ["noise_index_before", "noise_index_after"]:
            if dropped_key in channel:
                report.error(f"{label}[{idx}] contains removed field: {dropped_key}")
        std = channel.get("std")
        mean = channel.get("mean")
        if not isinstance(std, (int, float)) or not np.isfinite(std) or std <= 0:
            report.error(f"{label}[{idx}].std must be a positive finite number")
        if not isinstance(mean, (int, float)) or not np.isfinite(mean):
            report.error(f"{label}[{idx}].mean must be a finite number")


def expected_channel_count_from_token_metadata(token_metadata: dict) -> Optional[int]:
    layout = token_metadata.get("spatial_layout") if isinstance(token_metadata, dict) else None
    if not isinstance(layout, dict) or layout.get("type") != "channels":
        return None
    value = layout.get("channel_count")
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return int(value)
    return None


def rel_path_str(path_value: Any) -> Optional[str]:
    path = rel_path(path_value)
    return str(path) if path is not None else None


def validate_temporal_index(token_metadata: dict, total_tokens: Optional[int], report: ValidationReport, session_id: str) -> None:
    if total_tokens is None:
        return
    temporal_index = token_metadata.get("temporal_index") if isinstance(token_metadata, dict) else None
    if not isinstance(temporal_index, dict):
        report.error(f"{session_id}: token_metadata.temporal_index must be an object")
        return
    segments = temporal_index.get("segments")
    if not isinstance(segments, list):
        report.error(f"{session_id}: token_metadata.temporal_index.segments must be a list")
        return
    num_segments = temporal_index.get("num_segments")
    if num_segments != len(segments):
        report.error(f"{session_id}: temporal_index.num_segments={num_segments} but len(segments)={len(segments)}")

    expected_start = 0
    prev_source_start: Optional[float] = None
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            report.error(f"{session_id}: temporal_index.segments[{idx}] is not an object")
            continue
        token = segment.get("token")
        if not isinstance(token, dict):
            report.error(f"{session_id}: temporal_index.segments[{idx}].token must be an object")
            continue

        start = token.get("start_index")
        end = token.get("end_index")
        count = token.get("count")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in [start, end, count]):
            report.error(f"{session_id}: temporal_index.segments[{idx}].token start/end/count must be integers")
            continue
        if start != expected_start:
            report.error(f"{session_id}: temporal_index token ranges have a gap/overlap at segment {idx}: expected start {expected_start}, got {start}")
        if end < start:
            report.error(f"{session_id}: temporal_index.segments[{idx}].token.end_index < start_index")
        if count != end - start:
            report.error(f"{session_id}: temporal_index.segments[{idx}].token.count={count} but end-start={end - start}")
        if end > total_tokens:
            report.error(f"{session_id}: temporal_index.segments[{idx}].token.end_index {end} exceeds token rows {total_tokens}")

        grid_shape = token.get("grid_shape")
        if isinstance(grid_shape, list) and grid_shape and all(isinstance(v, int) and v >= 0 for v in grid_shape):
            grid_count = int(np.prod(grid_shape, dtype=np.int64))
            if grid_count != count:
                report.error(f"{session_id}: temporal_index.segments[{idx}].token.grid_shape product {grid_count} != count {count}")

        source_time = segment.get("source_time")
        if isinstance(source_time, dict):
            source_start = source_time.get("start")
            source_end = source_time.get("end")
            if isinstance(source_start, (int, float)) and isinstance(source_end, (int, float)):
                if source_end < source_start:
                    report.error(f"{session_id}: temporal_index.segments[{idx}].source_time.end < start")
                if prev_source_start is not None and source_start < prev_source_start:
                    report.warning(f"{session_id}: temporal_index source_time.start is not sorted at segment {idx}")
                prev_source_start = float(source_start)

        expected_start = end

    if expected_start != total_tokens:
        report.error(f"{session_id}: temporal_index covers {expected_start} token rows but token array has {total_tokens}")


def validate_session_summary_matches_token_metadata(manifest: dict, session_summary: dict, token_metadata: dict, report: ValidationReport, session_id: str) -> None:
    session_info = token_metadata.get("session") if isinstance(token_metadata, dict) else None
    if isinstance(session_info, dict):
        token_session_id = session_info.get("session_id")
        if token_session_id is not None and token_session_id != session_id:
            report.error(f"{session_id}: token_metadata.session.session_id={token_session_id!r} does not match manifest session_id")
        token_modality = session_info.get("modality")
        manifest_modality = session_summary.get("modality") or manifest.get("modality")
        if token_modality is not None and manifest_modality is not None and token_modality != manifest_modality:
            report.error(f"{session_id}: token_metadata.session.modality={token_modality!r} does not match manifest modality {manifest_modality!r}")

    tokens_info = token_metadata.get("tokens") if isinstance(token_metadata.get("tokens"), dict) else {}
    token_path = rel_path_str(tokens_info.get("path"))
    manifest_token_path = rel_path_str(session_summary.get("token_file") or session_summary.get("output_token_ids"))
    if token_path is not None and manifest_token_path is not None and token_path != manifest_token_path:
        report.error(f"{session_id}: manifest token_file {manifest_token_path!r} != token_metadata.tokens.path {token_path!r}")

    for manifest_key, token_key in [("token_shape", "shape"), ("token_dtype", "dtype"), ("num_codebooks", "num_codebooks")]:
        manifest_value = session_summary.get(manifest_key)
        token_value = tokens_info.get(token_key)
        if manifest_value is not None and token_value is not None and manifest_value != token_value:
            report.error(f"{session_id}: manifest {manifest_key}={manifest_value!r} != token_metadata.tokens.{token_key}={token_value!r}")

    manifest_tokenizer = manifest.get("tokenizer") if isinstance(manifest.get("tokenizer"), dict) else {}
    token_tokenizer = token_metadata.get("tokenizer") if isinstance(token_metadata.get("tokenizer"), dict) else {}
    for key in ["codebook", "codebook_metadata"]:
        manifest_path = rel_path_str(manifest_tokenizer.get(key))
        token_path_value = rel_path_str(token_tokenizer.get(key))
        if manifest_path is not None and token_path_value is not None and manifest_path != token_path_value:
            report.error(f"{session_id}: manifest.tokenizer.{key}={manifest_path!r} != token_metadata.tokenizer.{key}={token_path_value!r}")


def validate_token_metadata_and_array(
    root: Path,
    manifest: dict,
    session_summary: dict,
    token_metadata: dict,
    codebook_metadata: dict,
    report: ValidationReport,
) -> Tuple[Optional[int], Optional[int]]:
    session_id = session_summary.get("session_id", "<unknown>")
    validate_session_summary_matches_token_metadata(manifest, session_summary, token_metadata, report, str(session_id))
    tokens_info = token_metadata.get("tokens", {}) if isinstance(token_metadata.get("tokens"), dict) else {}
    token_rel = tokens_info.get("path") or session_summary.get("token_file") or token_metadata.get("output_token_ids") or session_summary.get("output_token_ids")
    token_path_rel = rel_path(token_rel)
    if token_path_rel is None:
        report.error(f"{session_id}: output_token_ids is missing or absolute: {token_rel!r}")
        return None, None
    token_path = root / token_path_rel
    if not token_path.exists():
        report.error(f"{session_id}: token file missing: {token_path_rel}")
        return None, None

    for private_key in ["samples", "source_tar", "source_session_dir", "num_samples"]:
        if private_key in token_metadata:
            report.error(f"{session_id}: token_metadata.{private_key} should not be included in public release metadata")

    tokenizer_info = token_metadata.get("tokenizer")
    if tokenizer_info is not None:
        if not isinstance(tokenizer_info, dict):
            report.error(f"{session_id}: token_metadata.tokenizer must be an object")
        else:
            checkpoint = tokenizer_info.get("checkpoint")
            if isinstance(checkpoint, str) and ABSOLUTE_PATH_RE.search(checkpoint):
                report.error(f"{session_id}: token_metadata.tokenizer.checkpoint must be a short identifier, not an absolute/local path: {checkpoint!r}")
            for key in ["codebook", "codebook_metadata"]:
                value = tokenizer_info.get(key)
                if value is None:
                    continue
                value_rel = rel_path(value)
                if value_rel is None:
                    report.error(f"{session_id}: token_metadata.tokenizer.{key} must be a relative dataset path: {value!r}")
                elif not (root / value_rel).exists():
                    report.error(f"{session_id}: token_metadata.tokenizer.{key} path missing: {value}")

    try:
        token_ids = np.load(token_path, mmap_mode="r")
    except Exception as exc:
        report.error(f"{session_id}: failed to load token file: {exc}")
        return None, None

    actual_shape = [int(v) for v in token_ids.shape]
    if len(actual_shape) != 2:
        report.error(f"{session_id}: token_ids should be 2D [total_tokens, num_codebooks], got shape {actual_shape}")
    elif actual_shape[0] <= 0 or actual_shape[1] <= 0:
        report.error(f"{session_id}: token_ids shape must be non-empty, got {actual_shape}")
    actual_num_codebooks = actual_shape[1] if len(actual_shape) == 2 else None
    expected_shape = tokens_info.get("shape") or token_metadata.get("token_ids_shape") or session_summary.get("token_shape") or session_summary.get("token_ids_shape")
    if expected_shape != actual_shape:
        report.error(f"{session_id}: token_ids shape mismatch: metadata={expected_shape}, actual={actual_shape}")
    manifest_shape = session_summary.get("token_shape") or session_summary.get("token_ids_shape")
    if manifest_shape != actual_shape:
        report.error(f"{session_id}: manifest token shape mismatch: manifest={manifest_shape}, actual={actual_shape}")
    dtype = str(token_ids.dtype)
    expected_dtype = tokens_info.get("dtype") or token_metadata.get("token_ids_dtype") or session_summary.get("token_dtype") or session_summary.get("token_ids_dtype")
    if expected_dtype != dtype:
        report.error(f"{session_id}: token_ids dtype mismatch: metadata={expected_dtype}, actual={dtype}")

    metadata_num_codebooks = tokens_info.get("num_codebooks") or token_metadata.get("num_codebooks") or session_summary.get("num_codebooks")
    if metadata_num_codebooks is not None and actual_num_codebooks is not None and actual_num_codebooks != int(metadata_num_codebooks):
        report.error(f"{session_id}: token_ids second dimension {actual_num_codebooks} != metadata num_codebooks {metadata_num_codebooks}")
    expected_num_codebooks = codebook_metadata.get("num_codebooks")
    if expected_num_codebooks is not None and actual_num_codebooks is not None and actual_num_codebooks != int(expected_num_codebooks):
        report.error(f"{session_id}: token_ids second dimension {actual_num_codebooks} != num_codebooks {expected_num_codebooks}")

    vocab_per = codebook_metadata.get("vocab_size_per_codebook")
    if vocab_per is not None and token_ids.size:
        min_id = int(token_ids.min())
        max_id = int(token_ids.max())
        if min_id < 0 or max_id >= int(vocab_per):
            report.error(f"{session_id}: token ids out of range [0, {int(vocab_per)-1}]: min={min_id}, max={max_id}")
        report.info(f"{session_id}: token_ids shape={tuple(actual_shape)}, dtype={dtype}, id range={min_id}..{max_id}")

    total_tokens = actual_shape[0] if len(actual_shape) == 2 else None
    validate_temporal_index(token_metadata, total_tokens, report, str(session_id))
    return expected_channel_count_from_token_metadata(token_metadata), total_tokens


def validate_session(
    root: Path,
    manifest: dict,
    session_summary: dict,
    codebook_metadata: dict,
    report: ValidationReport,
    check_event_duration_semantics: bool = False,
) -> None:
    session_id = str(session_summary.get("session_id", "<unknown>"))
    output_session_dir = rel_path(session_summary.get("output_session_dir"))
    if output_session_dir is None:
        token_meta_rel_for_dir = rel_path(session_summary.get("token_metadata"))
        output_session_dir = token_meta_rel_for_dir.parent if token_meta_rel_for_dir is not None else None
    if output_session_dir is None:
        report.error(f"{session_id}: session directory missing or absolute")
        return
    session_dir = root / output_session_dir
    if not session_dir.is_dir():
        report.error(f"{session_id}: session directory missing: {output_session_dir}")
        return

    token_meta_rel = rel_path(session_summary.get("token_metadata"))
    token_meta_path = root / token_meta_rel if token_meta_rel is not None else session_dir / "token_metadata.json"
    session_meta_path = session_dir / "session_metadata.json"
    events_path = session_dir / "events_info.json"
    modality = str(session_summary.get("modality") or manifest.get("modality") or "eeg")
    channel_path = session_dir / f"{modality}_channels.json"
    if not channel_path.exists() and modality == "eeg":
        channel_path = session_dir / "eeg_channels.json"

    for path in [token_meta_path, session_meta_path, events_path, channel_path]:
        if not path.exists():
            report.error(f"{session_id}: missing sidecar {path.relative_to(root) if path.is_relative_to(root) else path}")

    token_metadata = load_json(token_meta_path, report) if token_meta_path.exists() else {}
    session_metadata = load_json(session_meta_path, report) if session_meta_path.exists() else {}
    events = load_json(events_path, report) if events_path.exists() else []
    channels = load_json(channel_path, report) if channel_path.exists() else []

    if token_metadata:
        validate_with_schema(token_metadata, root / "schemas" / "token_metadata.schema.json", report, f"{session_id}/token_metadata.json")
        scan_json_text_for_release_warnings(token_meta_path, report)
    if session_metadata:
        validate_with_schema(session_metadata, root / "schemas" / "session_metadata.schema.json", report, f"{session_id}/session_metadata.json")
        validate_subject(session_metadata.get("Subject"), report, f"{session_id}/session_metadata.json")
        scan_json_text_for_release_warnings(session_meta_path, report)
    if events_path.exists():
        validate_with_schema(events, root / "schemas" / "events_info.schema.json", report, f"{session_id}/events_info.json")
        validate_events(
            events,
            session_metadata if isinstance(session_metadata, dict) else {},
            report,
            f"{session_id}/events_info.json",
            check_duration_semantics=check_event_duration_semantics,
        )
        scan_json_text_for_release_warnings(events_path, report)

    expected_channels, _ = validate_token_metadata_and_array(root, manifest, session_summary, token_metadata or {}, codebook_metadata, report)
    if channel_path.exists():
        channel_schema = root / "schemas" / f"{modality}_channels.schema.json"
        if not channel_schema.exists() and modality == "eeg":
            channel_schema = root / "schemas" / "eeg_channels.schema.json"
        validate_with_schema(channels, channel_schema, report, f"{session_id}/{channel_path.name}")
        validate_channels(channels, expected_channels, report, f"{session_id}/{channel_path.name}")
        scan_json_text_for_release_warnings(channel_path, report)

def validate_manifest(root: Path, report: ValidationReport) -> dict:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path, report) if manifest_path.exists() else None
    if not isinstance(manifest, dict):
        report.error("manifest.json is missing or not an object")
        return {}
    validate_with_schema(manifest, root / "schemas" / "manifest.schema.json", report, "manifest.json")
    scan_json_text_for_release_warnings(manifest_path, report)

    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        report.error("manifest.sessions must be a list")
        return manifest
    if manifest.get("num_sessions") != len(sessions):
        report.error(f"manifest.num_sessions={manifest.get('num_sessions')} but len(sessions)={len(sessions)}")

    for rel in [manifest.get("readme"), manifest.get("documentation", {}).get("data_dictionary")]:
        if rel and not file_exists(root, rel):
            report.error(f"manifest documentation path does not exist: {rel}")

    for session in sessions:
        if not isinstance(session, dict):
            report.error("manifest.sessions contains non-object entry")
            continue
        validate_subject(session.get("subject") if "subject" in session else session.get("Subject"), report, f"manifest session {session.get('session_id', '<unknown>')}")
        for private_key in ["source_tar", "source_session_dir", "samples", "num_samples"]:
            if private_key in session:
                report.error(f"manifest session {session.get('session_id', '<unknown>')}: {private_key} should not be included in public release manifest")
        path_fields = {
            'token_file': session.get('token_file') or session.get('output_token_ids'),
            'token_metadata': session.get('token_metadata'),
        }
        for key, value in path_fields.items():
            if rel_path(value) is None:
                report.error(f"manifest session {session.get('session_id', '<unknown>')}: {key} missing or absolute: {value!r}")
            elif not (root / rel_path(value)).exists():
                report.error(f"manifest session {session.get('session_id', '<unknown>')}: {key} path missing: {value}")
        output_session_value = session.get('output_session_dir')
        if output_session_value is not None and rel_path(output_session_value) is None:
            report.error(f"manifest session {session.get('session_id', '<unknown>')}: output_session_dir is absolute: {output_session_value!r}")
    return manifest


def validate_docs_text(root: Path, report: ValidationReport) -> None:
    for rel in ["README.md", "DATA_DICTIONARY.md"]:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 200:
            report.warning(f"{rel} is very short; add release details before public upload")
        if PLACEHOLDER_RE.search(text):
            report.warning(f"{rel} contains placeholder text; fill before public release")
        if ABSOLUTE_PATH_RE.search(text):
            report.warning(f"{rel} contains possible absolute/local path")


def validate_json_files_for_standard_json(root: Path, report: ValidationReport) -> None:
    for path in root.rglob("*.json"):
        data = load_json(path, report)
        if data is None:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bNaN\b|\bInfinity\b|-Infinity", text):
            report.warning(f"{path.relative_to(root)} contains non-standard JSON numeric value NaN/Infinity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a NeuroToken dataset release directory.")
    parser.add_argument("--config", type=Path, default=None, help="TOML pipeline config file.")
    parser.add_argument("--output-base", type=Path, default=None, help="NeuroToken dataset directory. Overrides config/default.")
    parser.add_argument("--json-report", type=Path, default=None, help="Write machine-readable validation report JSON.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failure.")
    parser.add_argument(
        "--check-event-duration-semantics",
        action="store_true",
        help="Warn when event end-start differs from other_info.duration. Off by default because source datasets often use that field differently.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_config = load_config(args.config)
    root = args.output_base or cfg_path(pipeline_config, "paths.token_root")
    if root is None:
        raise ValueError("paths.token_root is not configured and --output-base was not provided")
    root = Path(root)
    check_event_duration_semantics = bool(
        args.check_event_duration_semantics
        or cfg_bool(pipeline_config, "validation.check_event_duration_semantics", False)
    )
    report = ValidationReport()
    if not root.exists():
        report.error(f"Dataset directory does not exist: {root}")
        report.print()
        report_path = args.json_report or default_report_path(pipeline_config, "validate", Path("reports"))
        write_report(report_path, report.to_json_report(root, strict=args.strict))
        sys.exit(1)

    validate_required_top_level(root, report)
    validate_docs_text(root, report)
    validate_json_files_for_standard_json(root, report)
    manifest = validate_manifest(root, report)
    validate_subjects_index(root, manifest, report)
    codebook_metadata = validate_codebook(root, manifest, report)

    sessions = manifest.get("sessions", []) if isinstance(manifest, dict) else []
    if isinstance(sessions, list):
        for session in sessions:
            if isinstance(session, dict):
                validate_session(
                    root,
                    manifest,
                    session,
                    codebook_metadata,
                    report,
                    check_event_duration_semantics=check_event_duration_semantics,
                )

    report.print()
    report_path = args.json_report or default_report_path(pipeline_config, "validate", Path("reports"))
    write_report(report_path, report.to_json_report(root, strict=args.strict))
    print(f"\nValidation JSON report: {report_path}")
    if report.errors or (args.strict and report.warnings):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
