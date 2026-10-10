"""Independent semantic review for adapter candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from agent_pipeline.adapter_authoring.models import AdapterIntentReview, AdapterSpec
from agent_pipeline.adapter_authoring.prompts import build_adapter_review_prompt, load_review_context


PREFLIGHT_SCRIPT = r'''
import json
from pathlib import Path
import sys

from agent_pipeline.adapters import build_adapter
from agent_pipeline.package import event_ranges, event_sample_bounds, segment_record

config = json.loads(sys.stdin.read())
target = Path(sys.argv[1]).expanduser().resolve()
adapter = build_adapter(config)
discovered = [Path(path).expanduser().resolve() for path in adapter.discover(target)]
if target not in discovered:
    raise ValueError(f"Target was not discovered by the candidate adapter: {target}")
session = adapter.load_session(target)
n_points = int(session.signal.shape[1])
valid_intervals = 0
semantic_labels = 0
source_file_paths = 0
synthetic_source_gap_records = 0
event_keys = set()
for event in session.events:
    event_sample_bounds(event, float(session.sfreq), n_points)
    valid_intervals += 1
    event_keys.update(str(key) for key in event)
    if any(str(event.get(key, "")).strip() for key in ("type", "value", "label", "trial_type", "event_type")):
        semantic_labels += 1
    if any(str(event.get(key, "")).strip() for key in ("file", "file_path", "source_file", "source_path")):
        source_file_paths += 1
    if event.get("type") == "gap" or event.get("segment_kind") == "gap":
        synthetic_source_gap_records += 1

strategy = str(config.get("chunking", {}).get("strategy", "rest")).strip().lower()
if synthetic_source_gap_records:
    raise ValueError(
        f"SourceSession.events contains {synthetic_source_gap_records} synthetic gap record(s); "
        "adapters must normalize real source events only"
    )
package_ranges = []
package_records = []
event_segments = 0
ordinary_segments = 0
invented_gap_labels = 0
fabricated_ordinary_payloads = 0
if strategy == "event":
    package_ranges = event_ranges(session)
    for chunk_index, start, end, event in package_ranges:
        record = segment_record(chunk_index, start, end, event, float(session.sfreq), strategy)
        package_records.append(record)
        if event is None:
            ordinary_segments += 1
            if any(key.startswith("event_") or key == "event" for key in record):
                fabricated_ordinary_payloads += 1
        else:
            event_segments += 1
        if record.get("segment_kind") == "gap" or record.get("type") == "gap":
            invented_gap_labels += 1
    if event_segments != len(session.events):
        raise ValueError(
            f"Event package ranges contain {event_segments} event segments for {len(session.events)} source events"
        )
    coverage_cursor = 0
    for _, start, end, _ in sorted(package_ranges, key=lambda item: (item[1], item[2], item[0])):
        if start > coverage_cursor:
            break
        coverage_cursor = max(coverage_cursor, end)
    full_recording_coverage = bool(package_ranges) and coverage_cursor == n_points
    if not full_recording_coverage:
        raise ValueError(
            f"Event package ranges do not cover the complete recording: covered_until={coverage_cursor}, "
            f"n_points={n_points}"
        )
    if invented_gap_labels or fabricated_ordinary_payloads:
        raise ValueError(
            "Event package ranges violate the reference metadata contract: "
            f"invented_gap_labels={invented_gap_labels}, "
            f"fabricated_ordinary_payloads={fabricated_ordinary_payloads}"
        )
else:
    full_recording_coverage = False

print(json.dumps({
    "status": "ok",
    "adapter_class": adapter.__class__.__name__,
    "target_h5": str(target),
    "source_session_loaded": True,
    "signal_shape": [int(value) for value in session.signal.shape],
    "sfreq": float(session.sfreq),
    "event_count": len(session.events),
    "valid_nonempty_clipped_event_intervals": valid_intervals,
    "events_with_semantic_labels": semantic_labels,
    "events_with_source_file_paths": source_file_paths,
    "normalized_event_keys": sorted(event_keys),
    "chunking_strategy": strategy,
    "package_segment_count": len(package_ranges),
    "package_event_segment_count": event_segments,
    "package_ordinary_segment_count": ordinary_segments,
    "package_full_recording_coverage": full_recording_coverage,
    "package_invented_gap_label_count": invented_gap_labels,
    "package_fabricated_ordinary_payload_count": fabricated_ordinary_payloads,
    "package_boundary_preview": [
        {
            "chunk_index": int(chunk_index),
            "start_sample": int(start),
            "end_sample": int(end),
            "has_source_event": event is not None,
        }
        for chunk_index, start, end, event in package_ranges[:12]
    ],
    "discovered_target": True,
}))
'''


def run_candidate_preflight(
    *,
    worktree: str | Path,
    effective_config: dict[str, Any],
    target_h5: str | Path,
    python: str = sys.executable,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    root = Path(worktree).expanduser().resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    try:
        completed = subprocess.run(
            [python, "-c", PREFLIGHT_SCRIPT, str(target_h5)],
            cwd=root,
            env=environment,
            input=json.dumps(effective_config, ensure_ascii=False, default=str),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "source_session_loaded": False,
            "target_h5": str(Path(target_h5).expanduser().resolve()),
            "error": f"Candidate preflight timed out after {timeout_seconds:g} seconds.",
            "stdout": (exc.stdout or "")[-4000:].strip(),
            "stderr": (exc.stderr or "")[-4000:].strip(),
        }
    if completed.returncode != 0:
        return {
            "status": "failed",
            "source_session_loaded": False,
            "target_h5": str(Path(target_h5).expanduser().resolve()),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:].strip(),
            "stderr": completed.stderr[-4000:].strip(),
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "source_session_loaded": False,
            "target_h5": str(Path(target_h5).expanduser().resolve()),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:].strip(),
            "stderr": completed.stderr[-4000:].strip(),
            "error": f"Candidate preflight returned invalid JSON: {exc}",
        }


def review_adapter_candidate(
    llm,
    *,
    spec: AdapterSpec,
    h5_summary: str,
    instruction: str,
    design_document: str,
    diff: str,
    worktree: str | Path,
    original_user_request: str = "",
    requirements_snapshot: dict[str, Any] | None = None,
    effective_config: dict[str, Any] | None = None,
    preflight_evidence: dict[str, Any] | None = None,
    allowed_paths: list[str] | None = None,
) -> AdapterIntentReview:
    reviewer = llm.with_structured_output(AdapterIntentReview, method="function_calling")
    prompt = build_adapter_review_prompt(
        spec,
        h5_summary,
        instruction,
        design_document,
        diff,
        load_review_context(worktree),
        original_user_request=original_user_request,
        requirements_snapshot=json.dumps(requirements_snapshot or {}, ensure_ascii=False, indent=2),
        effective_config=json.dumps(effective_config or {}, ensure_ascii=False, indent=2, default=str),
        preflight_evidence=json.dumps(preflight_evidence or {}, ensure_ascii=False, indent=2, default=str),
        allowed_paths=allowed_paths,
    )
    return AdapterIntentReview.model_validate(reviewer.invoke(prompt))


def review_feedback(review: AdapterIntentReview) -> str:
    lines = [review.summary.strip()]
    for finding in review.findings:
        lines.extend([
            f"[{finding.severity.upper()}] {finding.title}",
            f"Evidence: {finding.evidence}",
            f"Required change: {finding.required_change}",
        ])
    return "\n".join(line for line in lines if line)
