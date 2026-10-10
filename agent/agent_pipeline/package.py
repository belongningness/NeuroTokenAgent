"""Package SourceSession objects into NeuroToken tar directories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import tarfile

import numpy as np
import webdataset as wds

from agent_pipeline.adapters import build_adapter
from agent_pipeline.adapters.base import SourceSession, safe_filename
from agent_pipeline.config import cfg_get, cfg_path


TAR_FILENAME = "data.tar"
SESSION_METADATA_FILENAME = "session_metadata.json"
CHUNK_INDEX_PAD = 8
TAR_SAMPLE_MEMBER_LIMIT = 6
TAR_METADATA_PREVIEW_MAX_BYTES = 128_000
SEGMENTATION_PREVIEW_LIMIT = 24
SEGMENTATION_MANIFEST_FILENAME = "segmentation_manifest.json"


@dataclass(frozen=True)
class PackageConfig:
    input_root: Path
    output_root: Path
    dataset_name: str
    strategy: str
    split_points: int | None
    downsample_factor: int | None


def require_cfg(config: dict[str, Any], dotted_key: str) -> Any:
    value = cfg_get(config, dotted_key, None)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def require_cfg_path(config: dict[str, Any], dotted_key: str) -> Path:
    value = cfg_path(config, dotted_key, None)
    if value is None:
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def build_package_config(config: dict[str, Any]) -> PackageConfig:
    strategy = str(cfg_get(config, "chunking.strategy", "rest")).strip().lower()
    if strategy not in {"rest", "event"}:
        raise ValueError("chunking.strategy must be 'rest' or 'event'")
    split_points = None
    downsample_factor = None
    if strategy == "rest":
        split_points = int(require_cfg(config, "chunking.split_points"))
        downsample_factor = int(require_cfg(config, "chunking.downsample_factor"))
        if downsample_factor <= 0 or split_points % downsample_factor != 0:
            raise ValueError("chunking.split_points must be divisible by chunking.downsample_factor")
    return PackageConfig(
        input_root=require_cfg_path(config, "paths.h5_root"),
        output_root=require_cfg_path(config, "paths.tar_root"),
        dataset_name=str(require_cfg(config, "dataset.name")),
        strategy=strategy,
        split_points=split_points,
        downsample_factor=downsample_factor,
    )


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def event_sample_bounds(event: dict[str, Any], sfreq: float, n_points: int) -> tuple[int, int]:
    start_time = float(event["start_time"])
    end_time = float(event["end_time"])
    if not np.isfinite(start_time) or not np.isfinite(end_time) or end_time <= start_time:
        raise ValueError(f"Invalid event interval [{start_time}, {end_time})")
    if "start_sample" in event or "end_sample" in event:
        if "start_sample" not in event or "end_sample" not in event:
            raise ValueError("Event must provide both start_sample and end_sample")
        start = max(0, min(int(event["start_sample"]), n_points))
        end = max(0, min(int(event["end_sample"]), n_points))
    else:
        start = max(0, min(int(np.ceil(start_time * sfreq)), n_points))
        end = max(0, min(int(np.ceil(end_time * sfreq)), n_points))
    if end <= start:
        raise ValueError(f"Event [{start_time}, {end_time}) is empty after clipping to the signal")
    return start, end


def event_ranges(session: SourceSession) -> list[tuple[int, int, int, dict[str, Any] | None]]:
    n_points = int(session.signal.shape[1])
    sfreq = float(session.sfreq)
    normalized = []
    for source_index, source_event in enumerate(session.events):
        event = dict(source_event)
        start, end = event_sample_bounds(event, sfreq, n_points)
        event["event_index"] = int(source_index)
        normalized.append((start, end, source_index, event))
    normalized.sort(key=lambda item: (item[0], item[1], item[2]))

    ranges: list[tuple[int, int, int, dict[str, Any] | None]] = []
    covered_until = 0
    for start, end, _source_index, event in normalized:
        if covered_until < start:
            ranges.append((len(ranges), covered_until, start, None))
        ranges.append((len(ranges), start, end, event))
        covered_until = max(covered_until, end)
    if covered_until < n_points:
        ranges.append((len(ranges), covered_until, n_points, None))
    return ranges


def sample_ranges(session: SourceSession, package_config: PackageConfig):
    n_points = int(session.signal.shape[1])
    if package_config.strategy == "event":
        if not session.events:
            raise ValueError(f"{session.source_path} has no events for chunking.strategy='event'")
        yield from event_ranges(session)
        return

    points_per_split = int(package_config.split_points)
    for split_idx, start in enumerate(range(0, n_points, points_per_split)):
        yield split_idx, start, min(start + points_per_split, n_points), None


def segment_record(
    split_idx: int,
    start: int,
    end: int,
    event: dict[str, Any] | None,
    sfreq: float,
    strategy: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "chunk_index": int(split_idx),
        "start_sample": int(start),
        "end_sample": int(end),
        "n_samples": int(end - start),
        "start_time": float(start) / sfreq,
        "end_time": float(end) / sfreq,
        "duration_seconds": float(end - start) / sfreq,
        "chunking_strategy": strategy,
    }
    if event is not None:
        record.update({
            "event_index": int(event.get("event_index", split_idx)),
            "event_type": str(event.get("type", "")),
            "event_value": str(event.get("value", "")),
        })
    return record


def choose_segmentation_preview(segments: list[dict[str, Any]], limit: int = SEGMENTATION_PREVIEW_LIMIT) -> list[dict[str, Any]]:
    if len(segments) <= limit:
        return segments
    edge_count = max(1, min(4, limit // 3))
    selected = set(range(edge_count))
    selected.update(range(len(segments) - edge_count, len(segments)))
    while len(selected) < limit:
        available = [index for index in range(len(segments)) if index not in selected]
        if not available:
            break
        selected.add(max(available, key=lambda index: min(abs(index - chosen) for chosen in selected)))
    return [segments[index] for index in sorted(selected)]


def build_segmentation_review(
    *,
    total_samples: int,
    sfreq: float,
    strategy: str,
    segments: list[dict[str, Any]],
    split_points: int | None = None,
    downsample_factor: int | None = None,
) -> dict[str, Any]:
    sorted_segments = sorted(segments, key=lambda item: (item["start_sample"], item["end_sample"], item["chunk_index"]))
    gaps: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    cursor = 0
    for segment in sorted_segments:
        start = int(segment["start_sample"])
        end = int(segment["end_sample"])
        if start > cursor:
            gaps.append({
                "start_sample": cursor,
                "end_sample": start,
                "start_time": cursor / sfreq,
                "end_time": start / sfreq,
                "n_samples": start - cursor,
            })
        elif start < cursor:
            overlaps.append({
                "chunk_index": segment["chunk_index"],
                "start_sample": start,
                "overlap_until_sample": min(cursor, end),
                "n_samples": max(0, min(cursor, end) - start),
            })
        cursor = max(cursor, end)
    if cursor < total_samples:
        gaps.append({
            "start_sample": cursor,
            "end_sample": total_samples,
            "start_time": cursor / sfreq,
            "end_time": total_samples / sfreq,
            "n_samples": total_samples - cursor,
        })

    first_start = int(sorted_segments[0]["start_sample"]) if sorted_segments else total_samples
    last_end = int(sorted_segments[-1]["end_sample"]) if sorted_segments else 0
    for gap in gaps:
        if int(gap["end_sample"]) <= first_start:
            gap["position"] = "leading"
        elif int(gap["start_sample"]) >= last_end:
            gap["position"] = "trailing"
        else:
            gap["position"] = "internal"

    covered_samples = sum(int(segment["n_samples"]) for segment in sorted_segments) - sum(
        int(overlap["n_samples"]) for overlap in overlaps
    )
    lengths = [int(segment["n_samples"]) for segment in segments]
    event_segment_count = sum(segment.get("event_index") is not None for segment in segments)
    review = {
        "strategy": strategy,
        "sfreq": sfreq,
        "total_samples": total_samples,
        "total_duration_seconds": total_samples / sfreq,
        "segment_count": len(segments),
        "event_segment_count": event_segment_count,
        "ordinary_segment_count": len(segments) - event_segment_count if strategy == "event" else 0,
        "covered_samples": covered_samples,
        "coverage_percent": (covered_samples / total_samples * 100.0) if total_samples else 0.0,
        "min_segment_samples": min(lengths) if lengths else 0,
        "max_segment_samples": max(lengths) if lengths else 0,
        "min_segment_seconds": min(lengths) / sfreq if lengths else 0.0,
        "max_segment_seconds": max(lengths) / sfreq if lengths else 0.0,
        "uncovered_ranges": gaps,
        "overlaps": overlaps,
        "preview_limit": SEGMENTATION_PREVIEW_LIMIT,
        "preview_segments": choose_segmentation_preview(segments),
    }
    if strategy == "rest":
        review.update({
            "split_points": split_points,
            "downsample_factor": downsample_factor,
            "full_segment_count": sum(length == split_points for length in lengths),
            "tail_segment_samples": lengths[-1] if lengths and lengths[-1] != split_points else 0,
        })
    return review


def iter_samples(session: SourceSession, package_config: PackageConfig):
    signal = np.asarray(session.signal)
    if signal.ndim != 2:
        raise ValueError(f"{session.source_path} signal must be [C, T], got {signal.shape}")

    sfreq = float(session.sfreq)
    subject = session.session_metadata.get("Subject", {}) or {}
    sub_id = safe_filename(str(subject.get("ID", "")) or "unknown")
    dataset_name = str(session.session_metadata.get("Dataset", package_config.dataset_name))

    for split_idx, start, end, event in sample_ranges(session, package_config):
        signal_slice = signal[:, start:end]
        if not np.isfinite(signal_slice).all():
            raise ValueError(f"{session.source_path} chunk {split_idx} contains NaN or Inf")

        segment = segment_record(split_idx, start, end, event, sfreq, package_config.strategy)
        metadata = dict(session.session_metadata)
        metadata.update({
            "chunking_strategy": package_config.strategy,
            "chunk_index": int(split_idx),
            "chunk_start_sample": int(start),
            "chunk_end_sample": int(end),
            "chunk_start_time": float(start) / sfreq,
            "chunk_end_time": float(end) / sfreq,
            "n_samples": int(end - start),
        })
        if event is not None:
            metadata.update({
                "event_index": int(event.get("event_index", split_idx)),
                "event_start_time": float(event["start_time"]),
                "event_end_time": float(event["end_time"]),
                "event_type": str(event.get("type", "")),
                "event_value": str(event.get("value", "")),
                "event": event,
            })
        key = f"{dataset_name}-{sub_id}-{session.safe_stem}_{split_idx:0{CHUNK_INDEX_PAD}d}"
        timestamps = np.arange(0, end - start, dtype=np.float32) / np.float32(sfreq)
        yield {
            "__key__": key,
            "metadata.json": json.dumps(metadata, ensure_ascii=False, default=to_jsonable, allow_nan=True),
            f"{session.modality}.npy": np.ascontiguousarray(signal_slice.astype(np.float32)),
            f"{session.modality}_timestamps.npy": np.ascontiguousarray(timestamps.astype(np.float32)),
        }


def tar_sample_preview(
    tar_path: Path,
    *,
    member_limit: int = TAR_SAMPLE_MEMBER_LIMIT,
    metadata_max_bytes: int = TAR_METADATA_PREVIEW_MAX_BYTES,
) -> dict[str, Any]:
    """Inspect a bounded set of tar members and decode one JSON metadata sample."""
    entries: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    metadata_member: str | None = None
    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            if len(entries) < member_limit:
                entries.append({
                    "name": member.name,
                    "size": member.size,
                    "type": "file" if member.isfile() else "other",
                })
            if (
                metadata is None
                and member.isfile()
                and member.name.endswith(".metadata.json")
                and member.size <= metadata_max_bytes
            ):
                extracted = archive.extractfile(member)
                if extracted is not None:
                    try:
                        loaded = json.loads(extracted.read().decode("utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        loaded = None
                    if isinstance(loaded, dict):
                        metadata = loaded
                        metadata_member = member.name
            if len(entries) >= member_limit and metadata is not None:
                break
    return {
        "entries": entries,
        "metadata_member": metadata_member,
        "metadata": metadata,
    }


def tar_segmentation_review(tar_path: Path, strategy: str = "") -> dict[str, Any] | None:
    segments: list[dict[str, Any]] = []
    first_metadata: dict[str, Any] | None = None
    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".metadata.json") or member.size > TAR_METADATA_PREVIEW_MAX_BYTES:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                metadata = json.loads(extracted.read().decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            if first_metadata is None:
                first_metadata = metadata
            required = ("chunk_index", "chunk_start_sample", "chunk_end_sample")
            if not all(isinstance(metadata.get(key), (int, float)) for key in required):
                continue
            start = int(metadata["chunk_start_sample"])
            end = int(metadata["chunk_end_sample"])
            sfreq = float(metadata.get("sfreq", 0) or 0)
            if sfreq <= 0 or end <= start:
                continue
            event = metadata.get("event") if isinstance(metadata.get("event"), dict) else None
            segments.append(segment_record(int(metadata["chunk_index"]), start, end, event, sfreq, str(metadata.get("chunking_strategy") or strategy or "unknown")))
    if not segments or first_metadata is None:
        return None
    sfreq = float(first_metadata.get("sfreq", 0) or 0)
    total_samples = int(first_metadata.get("n_samples_total", max(segment["end_sample"] for segment in segments)))
    inferred_strategy = str(first_metadata.get("chunking_strategy") or strategy or segments[0]["chunking_strategy"])
    lengths = [int(segment["n_samples"]) for segment in segments]
    split_points = Counter(lengths).most_common(1)[0][0] if inferred_strategy == "rest" else None
    return build_segmentation_review(
        total_samples=total_samples,
        sfreq=sfreq,
        strategy=inferred_strategy,
        segments=segments,
        split_points=split_points,
    )


def ensure_package_report_previews(report: dict[str, Any]) -> dict[str, Any]:
    """Backfill bounded tar and segmentation previews in package reports."""
    if report.get("stage") != "package" or not isinstance(report.get("sessions"), list):
        return report

    sessions: list[Any] = []
    changed = False
    for value in report["sessions"]:
        if not isinstance(value, dict):
            sessions.append(value)
            continue
        tar_path = value.get("tar_path")
        if not isinstance(tar_path, str) or not tar_path:
            sessions.append(value)
            continue
        updates: dict[str, Any] = {}
        try:
            path = Path(tar_path)
            if "tar_sample_metadata" not in value or "tar_sample_metadata_member" not in value:
                preview = tar_sample_preview(path)
                updates.update({
                    "tar_sample": preview["entries"],
                    "tar_sample_metadata_member": preview["metadata_member"],
                    "tar_sample_metadata": preview["metadata"],
                })
            if not isinstance(value.get("segmentation_review"), dict):
                segmentation_review = tar_segmentation_review(path, str(report.get("chunking_strategy") or ""))
                if segmentation_review is not None:
                    updates["segmentation_review"] = segmentation_review
        except (OSError, tarfile.TarError):
            pass
        sessions.append({**value, **updates} if updates else value)
        changed = changed or bool(updates)

    return {**report, "sessions": sessions} if changed else report


def package_one_session(session: SourceSession, output_root: Path, package_config: PackageConfig) -> dict[str, Any]:
    output_dir = Path(output_root) / session.safe_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / TAR_FILENAME

    ranges = list(sample_ranges(session, package_config))
    segments = [
        segment_record(split_idx, start, end, event, float(session.sfreq), package_config.strategy)
        for split_idx, start, end, event in ranges
    ]
    segmentation_review = build_segmentation_review(
        total_samples=int(session.signal.shape[1]),
        sfreq=float(session.sfreq),
        strategy=package_config.strategy,
        segments=segments,
        split_points=package_config.split_points,
        downsample_factor=package_config.downsample_factor,
    )

    count = 0
    temporary_tar_path = output_dir / f".{TAR_FILENAME}.{os.getpid()}.tmp"
    try:
        with wds.TarWriter(str(temporary_tar_path)) as sink:
            for sample in iter_samples(session, package_config):
                sink.write(sample)
                count += 1
        tar_sample_preview(temporary_tar_path, member_limit=1)
        os.replace(temporary_tar_path, tar_path)
    finally:
        temporary_tar_path.unlink(missing_ok=True)

    (output_dir / SESSION_METADATA_FILENAME).write_text(
        json.dumps(session.session_metadata, ensure_ascii=False, default=to_jsonable, allow_nan=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{session.modality}_channels.json").write_text(
        json.dumps(session.channels, ensure_ascii=False, default=to_jsonable, allow_nan=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / "events_info.json").write_text(
        json.dumps(session.events, ensure_ascii=False, default=to_jsonable, allow_nan=True, indent=2),
        encoding="utf-8",
    )
    segmentation_manifest_path = output_dir / SEGMENTATION_MANIFEST_FILENAME
    segmentation_manifest_path.write_text(
        json.dumps({**segmentation_review, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    preview = tar_sample_preview(tar_path)
    return {
        "source": str(session.source_path),
        "output_dir": str(output_dir),
        "tar_path": str(tar_path),
        "tar_size_bytes": tar_path.stat().st_size,
        "samples": count,
        "segmentation_manifest_path": str(segmentation_manifest_path),
        "segmentation_review": segmentation_review,
        "tar_sample": preview["entries"],
        "tar_sample_metadata_member": preview["metadata_member"],
        "tar_sample_metadata": preview["metadata"],
    }


def package_all(config: dict[str, Any], max_files: int | None = None, target_h5: str | None = None) -> dict[str, Any]:
    package_config = build_package_config(config)
    package_config.output_root.mkdir(parents=True, exist_ok=True)
    adapter = build_adapter(config)
    files = [Path(target_h5).expanduser().resolve()] if target_h5 else adapter.discover(package_config.input_root)
    if max_files is not None and not target_h5:
        files = files[:max_files]
    results = []
    for path in files:
        session = adapter.load_session(path)
        results.append(package_one_session(session, package_config.output_root, package_config))
    return {
        "input_root": str(package_config.input_root),
        "output_root": str(package_config.output_root),
        "chunking_strategy": package_config.strategy,
        "files": len(files),
        "samples": sum(item["samples"] for item in results),
        "sessions": results,
    }
