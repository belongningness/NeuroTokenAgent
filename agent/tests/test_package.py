import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

from agent_pipeline.adapters.base import SourceSession
from agent_pipeline.package import (
    PackageConfig,
    choose_segmentation_preview,
    ensure_package_report_previews,
    package_one_session,
    sample_ranges,
    tar_sample_preview,
)


def test_tar_sample_preview_lists_bounded_members_and_decodes_metadata(tmp_path: Path):
    tar_path = tmp_path / "data.tar"
    payloads = {
        "sample.meg.npy": b"meg-bytes",
        "sample.meg_timestamps.npy": b"timestamp-bytes",
        "sample.metadata.json": json.dumps({"chunk_index": 0, "n_samples": 1024}).encode("utf-8"),
        "next.meg.npy": b"next",
    }
    with tarfile.open(tar_path, "w") as archive:
        for name, content in payloads.items():
            source = tmp_path / name
            source.write_bytes(content)
            archive.add(source, arcname=name)

    preview = tar_sample_preview(tar_path, member_limit=3)

    assert [entry["name"] for entry in preview["entries"]] == [
        "sample.meg.npy",
        "sample.meg_timestamps.npy",
        "sample.metadata.json",
    ]
    assert preview["metadata_member"] == "sample.metadata.json"
    assert preview["metadata"] == {"chunk_index": 0, "n_samples": 1024}


def test_tar_sample_preview_ignores_oversized_metadata(tmp_path: Path):
    tar_path = tmp_path / "data.tar"
    source = tmp_path / "sample.metadata.json"
    source.write_text(json.dumps({"description": "too large"}), encoding="utf-8")
    with tarfile.open(tar_path, "w") as archive:
        archive.add(source, arcname=source.name)

    preview = tar_sample_preview(tar_path, metadata_max_bytes=4)

    assert preview["entries"][0]["name"] == "sample.metadata.json"
    assert preview["metadata_member"] is None
    assert preview["metadata"] is None


def source_session(tmp_path: Path, *, n_samples: int = 314, events: list[dict] | None = None) -> SourceSession:
    return SourceSession(
        source_path=tmp_path / "source.h5",
        source_stem="source",
        safe_stem="source",
        modality="meg",
        signal=np.zeros((2, n_samples), dtype=np.float32),
        sfreq=10.0,
        subject={"ID": "01"},
        channels=[],
        events=events or [],
        session_metadata={"Dataset": "Example", "Subject": {"ID": "01"}, "sfreq": 10.0, "n_samples_total": n_samples},
    )


def test_event_segmentation_covers_recording_with_unlabeled_ordinary_ranges(tmp_path: Path):
    events = [
        {"start_time": 10.0, "end_time": 20.0, "type": "second", "value": "b"},
        {"start_time": 1.0, "end_time": 9.0, "type": "first", "value": "a"},
    ]
    session = source_session(tmp_path, n_samples=210, events=events)
    config = PackageConfig(tmp_path, tmp_path / "tar", "Example", "event", None, None)

    ranges = list(sample_ranges(session, config))

    assert [(start, end, event is not None) for _, start, end, event in ranges] == [
        (0, 10, False),
        (10, 90, True),
        (90, 100, False),
        (100, 200, True),
        (200, 210, False),
    ]
    assert ranges[1][3]["event_index"] == 1
    assert ranges[3][3]["event_index"] == 0


def test_event_segmentation_preserves_overlaps_without_negative_gap(tmp_path: Path):
    session = source_session(tmp_path, events=[
        {"start_time": 1.0, "end_time": 5.0},
        {"start_time": 3.0, "end_time": 6.0},
    ])
    config = PackageConfig(tmp_path, tmp_path / "tar", "Example", "event", None, None)

    ranges = list(sample_ranges(session, config))

    assert [(start, end, event is not None) for _, start, end, event in ranges] == [
        (0, 10, False),
        (10, 50, True),
        (30, 60, True),
        (60, 314, False),
    ]


def test_event_segmentation_starts_inter_event_range_after_nested_coverage(tmp_path: Path):
    session = source_session(tmp_path, n_samples=150, events=[
        {"start_time": 1.0, "end_time": 10.0},
        {"start_time": 2.0, "end_time": 3.0},
        {"start_time": 12.0, "end_time": 13.0},
    ])
    config = PackageConfig(tmp_path, tmp_path / "tar", "Example", "event", None, None)

    ranges = list(sample_ranges(session, config))

    assert [(start, end, event is not None) for _, start, end, event in ranges] == [
        (0, 10, False),
        (10, 100, True),
        (20, 30, True),
        (100, 120, False),
        (120, 130, True),
        (130, 150, False),
    ]


def test_event_segmentation_prefers_explicit_sample_bounds(tmp_path: Path):
    session = source_session(tmp_path, events=[{
        "start_time": 1.1,
        "end_time": 2.2,
        "start_sample": 12,
        "end_sample": 24,
    }])
    config = PackageConfig(tmp_path, tmp_path / "tar", "Example", "event", None, None)

    ranges = list(sample_ranges(session, config))

    assert [(start, end, event is not None) for _, start, end, event in ranges] == [
        (0, 12, False),
        (12, 24, True),
        (24, 314, False),
    ]


def test_event_package_keeps_inter_event_data_without_inventing_gap_metadata(tmp_path: Path):
    session = source_session(tmp_path, n_samples=100, events=[
        {"start_time": 1.0, "end_time": 3.0, "type": "first", "value": "a"},
        {"start_time": 5.0, "end_time": 7.0, "type": "second", "value": "b"},
    ])
    config = PackageConfig(tmp_path, tmp_path / "tar", "Example", "event", None, None)

    report = package_one_session(session, config.output_root, config)

    with tarfile.open(report["tar_path"], "r") as archive:
        metadata_members = sorted(
            (member for member in archive.getmembers() if member.name.endswith(".metadata.json")),
            key=lambda member: member.name,
        )
        metadata = [json.load(archive.extractfile(member)) for member in metadata_members]

    assert [(item["chunk_start_sample"], item["chunk_end_sample"]) for item in metadata] == [
        (0, 10),
        (10, 30),
        (30, 50),
        (50, 70),
        (70, 100),
    ]
    assert metadata[1]["event_type"] == "first"
    for item in (metadata[0], metadata[2], metadata[4]):
        assert not any(key.startswith("event_") or key in {"event", "segment_kind"} for key in item)
        assert "gap" not in json.dumps(item).lower()
    assert metadata[3]["event_type"] == "second"
    assert report["segmentation_review"]["event_segment_count"] == 2
    assert report["segmentation_review"]["ordinary_segment_count"] == 3
    assert report["segmentation_review"]["coverage_percent"] == 100.0
    assert report["segmentation_review"]["uncovered_ranges"] == []


def test_choose_segmentation_preview_keeps_exact_limit_and_edges():
    segments = [{"chunk_index": index} for index in range(100)]

    preview = choose_segmentation_preview(segments, limit=24)

    assert len(preview) == 24
    assert preview[0]["chunk_index"] == 0
    assert preview[-1]["chunk_index"] == 99
    assert [segment["chunk_index"] for segment in preview] == sorted(segment["chunk_index"] for segment in preview)


def test_package_report_exposes_rest_segmentation_boundaries_and_manifest(tmp_path: Path):
    config = PackageConfig(tmp_path, tmp_path / "tar", "Example", "rest", 100, 10)

    report = package_one_session(source_session(tmp_path), config.output_root, config)

    review = report["segmentation_review"]
    assert review["segment_count"] == 4
    assert review["coverage_percent"] == 100.0
    assert review["full_segment_count"] == 3
    assert review["tail_segment_samples"] == 14
    assert review["preview_segments"][0]["start_sample"] == 0
    assert review["preview_segments"][-1]["end_sample"] == 314
    manifest = json.loads(Path(report["segmentation_manifest_path"]).read_text(encoding="utf-8"))
    assert [segment["n_samples"] for segment in manifest["segments"]] == [100, 100, 100, 14]
    assert "segment_kind" not in report["tar_sample_metadata"]
    assert Path(report["tar_path"]).is_file()
    assert not list(Path(report["output_dir"]).glob(".data.tar.*.tmp"))


def test_package_failure_does_not_publish_partial_tar(tmp_path: Path):
    config = PackageConfig(tmp_path, tmp_path / "tar", "Example", "rest", 100, 10)
    session = source_session(tmp_path)

    with patch("agent_pipeline.package.iter_samples", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            package_one_session(session, config.output_root, config)

    output_dir = config.output_root / session.safe_stem
    assert not (output_dir / "data.tar").exists()
    assert not list(output_dir.glob(".data.tar.*.tmp"))


def test_ensure_package_report_previews_backfills_legacy_report(tmp_path: Path):
    tar_path = tmp_path / "data.tar"
    source = tmp_path / "sample.metadata.json"
    source.write_text(json.dumps({
        "chunking_strategy": "rest",
        "sfreq": 10.0,
        "n_samples_total": 1024,
        "chunk_index": 0,
        "chunk_start_sample": 0,
        "chunk_end_sample": 1024,
        "n_samples": 1024,
    }), encoding="utf-8")
    with tarfile.open(tar_path, "w") as archive:
        archive.add(source, arcname=source.name)
    report = {
        "stage": "package",
        "sessions": [{
            "tar_path": str(tar_path),
            "tar_sample": [{"name": source.name, "size": source.stat().st_size, "type": "file"}],
        }],
    }

    enriched = ensure_package_report_previews(report)

    session = enriched["sessions"][0]
    assert session["tar_sample_metadata_member"] == source.name
    assert session["tar_sample_metadata"]["chunk_index"] == 0
    assert session["tar_sample_metadata"]["n_samples"] == 1024
    assert session["segmentation_review"]["segment_count"] == 1
    assert session["segmentation_review"]["coverage_percent"] == 100.0
    assert session["segmentation_review"]["preview_segments"][0]["end_sample"] == 1024
    assert "tar_sample_metadata" not in report["sessions"][0]
