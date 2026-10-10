"""Shared timestamp-aware normalization for recording-level MEG HDF5 adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar
import json

import h5py
import numpy as np

from agent_pipeline.adapters.meg_h5 import MEGH5Adapter, json_value
from agent_pipeline.config import cfg_path


_EVENT_CORE_FIELDS = {
    "start_time",
    "end_time",
    "onset",
    "duration",
    "type",
    "trial_type",
    "event_type",
    "value",
    "stimulus",
    "condition",
    "other_info",
    "start_sample",
    "end_sample",
}


def _event_other_info(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("other_info")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {"source_other_info": raw}
    if raw in (None, ""):
        payload: dict[str, Any] = {}
    elif isinstance(raw, dict):
        payload = {str(key): json_value(value) for key, value in raw.items()}
    else:
        payload = {"source_other_info": json_value(raw)}
    for key, value in row.items():
        if key not in _EVENT_CORE_FIELDS:
            payload[str(key)] = json_value(value)
    return payload


class TimestampAlignedMEGH5Adapter(MEGH5Adapter):
    DEFAULT_H5_ROOT: ClassVar[Path]
    SCHEMA_VERSION: ClassVar[str]
    DATASET_ATTR: ClassVar[str]
    REQUIRED_DATASETS: ClassVar[frozenset[str]] = frozenset({
        "meg",
        "meg_timestamps",
        "metadata_json_u8",
        "meg_channels_pkl_u8",
        "events_pkl_u8",
    })
    LOCAL_PATH_METADATA_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def default_input_root(self) -> Path | None:
        return cfg_path(self.config, "paths.h5_root", self.DEFAULT_H5_ROOT)

    def discover(self, input_root: Path) -> list[Path]:
        root = Path(input_root)
        candidates = [root] if root.is_file() else sorted(
            path for pattern in ("*.h5", "*.hdf5") for path in root.rglob(pattern)
        )
        discovered: list[Path] = []
        for path in candidates:
            try:
                with h5py.File(path, "r") as source:
                    if str(source.attrs.get("schema_version", "")) != self.SCHEMA_VERSION:
                        continue
                    if str(source.attrs.get("dataset", "")) != self.DATASET_ATTR:
                        continue
                    if not self.REQUIRED_DATASETS.issubset(source.keys()):
                        continue
                    discovered.append(path)
            except OSError:
                continue
        return discovered

    def read_metadata(self, source: h5py.File) -> dict[str, Any]:
        metadata = super().read_metadata(source)
        for field in self.LOCAL_PATH_METADATA_FIELDS:
            metadata.pop(field, None)
        return metadata

    def build_events(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for source_index, row in enumerate(records):
            start = row.get("start_time", row.get("onset"))
            if start is None:
                raise ValueError(f"Source event {source_index} is missing start_time/onset")
            end = row.get("end_time")
            if end is None:
                duration = row.get("duration")
                if duration is None:
                    raise ValueError(f"Source event {source_index} is missing end_time/duration")
                end = float(start) + float(duration)
            start_time = float(start)
            end_time = float(end)
            if not np.isfinite(start_time) or not np.isfinite(end_time) or end_time <= start_time:
                raise ValueError(f"Source event {source_index} has invalid interval [{start_time}, {end_time})")
            event: dict[str, Any] = {
                "start_time": start_time,
                "end_time": end_time,
                "type": str(row.get("type", row.get("trial_type", row.get("event_type", "event")))),
                "value": str(row.get("value", row.get("stimulus", row.get("condition", "")))),
                "other_info": _event_other_info(row),
            }
            if "start_sample" in row or "end_sample" in row:
                if "start_sample" not in row or "end_sample" not in row:
                    raise ValueError(f"Source event {source_index} must provide both start_sample and end_sample")
                event["start_sample"] = int(row["start_sample"])
                event["end_sample"] = int(row["end_sample"])
            events.append(event)
        return events

    def load_session(self, path: Path):
        path = Path(path)
        session = super().load_session(path)
        with h5py.File(path, "r") as source:
            schema_version = str(source.attrs.get("schema_version", ""))
            dataset = str(source.attrs.get("dataset", ""))
            timestamps = np.asarray(source[self.timestamps_key][:], dtype=np.float64)
        if schema_version != self.SCHEMA_VERSION or dataset != self.DATASET_ATTR:
            raise ValueError(
                f"{path} is not {self.DATASET_ATTR} schema {self.SCHEMA_VERSION}: "
                f"dataset={dataset!r}, schema_version={schema_version!r}"
            )
        if timestamps.ndim != 1 or timestamps.size != session.signal.shape[1]:
            raise ValueError(f"{path} has invalid timestamps shape {timestamps.shape}")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{path} timestamps must be finite and strictly increasing")
        sfreq = float(session.sfreq)
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError(f"{path} has invalid sfreq={sfreq}")
        expected_step = 1.0 / sfreq
        if not np.allclose(np.diff(timestamps), expected_step, rtol=1e-6, atol=1e-9):
            raise ValueError(f"{path} timestamps are inconsistent with sfreq={sfreq}")

        time_origin = float(timestamps[0])
        aligned_events: list[dict[str, Any]] = []
        for event in session.events:
            source_start = float(event["start_time"])
            source_end = float(event["end_time"])
            if "start_sample" in event and "end_sample" in event:
                start_sample = int(event["start_sample"])
                end_sample = int(event["end_sample"])
            else:
                start_sample = int(np.searchsorted(timestamps, source_start, side="left"))
                end_sample = int(np.searchsorted(timestamps, source_end, side="left"))
            aligned = dict(event)
            aligned["start_time"] = start_sample / sfreq
            aligned["end_time"] = end_sample / sfreq
            aligned["start_sample"] = start_sample
            aligned["end_sample"] = end_sample
            if time_origin != 0.0:
                aligned["other_info"] = {
                    **dict(aligned.get("other_info") or {}),
                    "source_start_time": source_start,
                    "source_end_time": source_end,
                    "signal_time_origin": time_origin,
                }
            aligned_events.append(aligned)

        session.events = aligned_events
        session.session_metadata = {
            **session.session_metadata,
            "source_timestamp_start": time_origin,
            "source_timestamp_end": float(timestamps[-1]),
            "event_time_coordinate": "seconds relative to first MEG sample",
        }
        return session
