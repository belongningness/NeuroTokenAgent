from __future__ import annotations

import json
from pathlib import Path
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest

from agent_pipeline.adapters import build_adapter
from agent_pipeline.package import PackageConfig, sample_ranges


def write_recording(
    path: Path,
    *,
    dataset: str,
    schema_version: str,
    timestamps: np.ndarray,
    events: pd.DataFrame,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = pd.DataFrame({
        "channel_name": ["MEG001", "MEG002"],
        "channel_type": ["mag", "mag"],
    })
    metadata = {
        "Dataset": dataset,
        "Subject": {"ID": "sub-01", "Age": "", "Sex": "", "Health_group": ""},
    }
    with h5py.File(path, "w") as source:
        source.create_dataset("meg", data=np.zeros((2, len(timestamps)), dtype=np.float32))
        source.create_dataset("meg_timestamps", data=np.asarray(timestamps, dtype=np.float64))
        source.create_dataset(
            "metadata_json_u8",
            data=np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8),
        )
        source.create_dataset(
            "meg_channels_pkl_u8",
            data=np.frombuffer(pickle.dumps(channels), dtype=np.uint8),
        )
        source.create_dataset(
            "events_pkl_u8",
            data=np.frombuffer(pickle.dumps(events), dtype=np.uint8),
        )
        source.attrs.update(
            dataset=dataset,
            schema_version=schema_version,
            sfreq=10.0,
            subject="sub-01",
            task="example",
            n_events=len(events),
        )
    return path


def adapter_config(name: str, root: Path, strategy: str = "event") -> dict:
    return {
        "source": {"adapter": name},
        "dataset": {"name": "Example", "modality": "meg"},
        "paths": {"h5_root": str(root)},
        "chunking": {"strategy": strategy},
    }


def package_config(root: Path) -> PackageConfig:
    return PackageConfig(root, root / "tar", "Example", "event", None, None)


def test_smn_adapter_preserves_real_event_and_shared_package_covers_edges(tmp_path: Path):
    events = pd.DataFrame([{
        "start_time": 1.0,
        "end_time": 9.0,
        "type": "audio",
        "value": "audio/story.wav",
    }])
    path = write_recording(
        tmp_path / "smn.h5",
        dataset="SMN_MEG",
        schema_version="smn_meg_h5_v1",
        timestamps=np.arange(0.0, 10.0, 0.1),
        events=events,
    )
    adapter = build_adapter(adapter_config("smn_meg_h5", tmp_path, strategy="rest"))

    session = adapter.load_session(path)
    ranges = list(sample_ranges(session, package_config(tmp_path)))

    assert len(session.events) == 1
    assert session.events[0] == {
        "start_time": 1.0,
        "end_time": 9.0,
        "type": "audio",
        "value": "audio/story.wav",
        "other_info": {},
        "start_sample": 10,
        "end_sample": 90,
    }
    assert [(start, end, event is not None) for _, start, end, event in ranges] == [
        (0, 10, False),
        (10, 90, True),
        (90, 100, False),
    ]
    assert not any("gap" in json.dumps(event).lower() or "segment_kind" in event for event in session.events)


def test_nod_adapter_aligns_absolute_event_times_and_decodes_other_info(tmp_path: Path):
    events = pd.DataFrame([
        {
            "start_time": 2.0,
            "end_time": 3.0,
            "type": "image",
            "value": "stimuli/ImageNet/second.JPEG",
            "other_info": json.dumps({"image_id": "second", "RT": 0.5}),
        },
        {
            "start_time": 0.0,
            "end_time": 1.0,
            "type": "image",
            "value": "stimuli/ImageNet/first.JPEG",
            "other_info": json.dumps({"image_id": "first", "RT": 0.4}),
        },
    ])
    timestamps = np.arange(-1.0, 4.0, 0.1)
    path = write_recording(
        tmp_path / "nod.h5",
        dataset="NOD_MEG_2025",
        schema_version="nod-meg-h5-v2",
        timestamps=timestamps,
        events=events,
    )
    adapter = build_adapter(adapter_config("nod_meg_h5", tmp_path))

    session = adapter.load_session(path)
    ranges = list(sample_ranges(session, package_config(tmp_path)))
    second_start = int(np.searchsorted(timestamps, 2.0, side="left"))
    second_end = int(np.searchsorted(timestamps, 3.0, side="left"))
    first_start = int(np.searchsorted(timestamps, 0.0, side="left"))
    first_end = int(np.searchsorted(timestamps, 1.0, side="left"))

    assert session.events[0]["start_sample"] == second_start
    assert session.events[0]["end_sample"] == second_end
    assert session.events[1]["start_sample"] == first_start
    assert session.events[1]["end_sample"] == first_end
    assert session.events[0]["start_time"] == second_start / 10.0
    assert session.events[0]["end_time"] == second_end / 10.0
    assert session.events[1]["other_info"]["image_id"] == "first"
    assert session.events[1]["other_info"]["source_start_time"] == 0.0
    assert "other_info" not in session.events[1]["other_info"]
    assert "type" not in session.events[1]["other_info"]
    assert "value" not in session.events[1]["other_info"]
    assert [(start, end, event.get("event_index") if event else None) for _, start, end, event in ranges] == [
        (0, first_start, None),
        (first_start, first_end, 1),
        (first_end, second_start, None),
        (second_start, second_end, 0),
        (second_end, len(timestamps), None),
    ]
    assert not any("gap" in json.dumps(event).lower() or "segment_kind" in event for event in session.events)


def test_dataset_discovery_is_recursive_schema_aware_and_supports_hdf5(tmp_path: Path):
    events = pd.DataFrame([{
        "start_time": 0.0,
        "end_time": 0.5,
        "type": "audio",
        "value": "story.wav",
    }])
    valid = write_recording(
        tmp_path / "nested" / "valid.hdf5",
        dataset="SMN_MEG",
        schema_version="smn_meg_h5_v1",
        timestamps=np.arange(0.0, 1.0, 0.1),
        events=events,
    )
    write_recording(
        tmp_path / "wrong.h5",
        dataset="NOD_MEG_2025",
        schema_version="nod-meg-h5-v2",
        timestamps=np.arange(0.0, 1.0, 0.1),
        events=events,
    )
    with h5py.File(tmp_path / "malformed.h5", "w") as source:
        source.attrs["dataset"] = "SMN_MEG"
        source.attrs["schema_version"] = "smn_meg_h5_v1"

    adapter = build_adapter(adapter_config("smn_meg_h5", tmp_path))

    assert adapter.discover(tmp_path) == [valid]


@pytest.mark.parametrize("adapter_name", ["smn_meg_h5", "nod_meg_h5"])
def test_dataset_adapter_does_not_require_event_chunking_strategy(adapter_name: str, tmp_path: Path):
    adapter = build_adapter(adapter_config(adapter_name, tmp_path, strategy="rest"))

    assert adapter.config["chunking"]["strategy"] == "rest"


def test_smn_adapter_allows_empty_real_events_for_rest_chunking(tmp_path: Path):
    path = write_recording(
        tmp_path / "empty-events.h5",
        dataset="SMN_MEG",
        schema_version="smn_meg_h5_v1",
        timestamps=np.arange(0.0, 1.0, 0.1),
        events=pd.DataFrame(columns=["start_time", "end_time", "type", "value"]),
    )
    adapter = build_adapter(adapter_config("smn_meg_h5", tmp_path, strategy="rest"))

    session = adapter.load_session(path)

    assert session.events == []
