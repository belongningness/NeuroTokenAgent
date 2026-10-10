"""MEG HDF5 adapter for CamCAN-like session H5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import pickle

import h5py
import numpy as np
import pandas as pd

from agent_pipeline.adapters.base import SourceSession, safe_filename
from agent_pipeline.config import cfg_get, cfg_path


STRING_COLUMNS = {
    "ID", "id", "subject_id", "sub", "ses", "run", "task",
    "Age", "age", "Sex", "sex", "Gender", "gender",
    "Health_group", "health_group", "group", "diagnosis",
    "channel_name", "channel_type", "ch_name", "ch_type", "name", "label",
    "type", "status", "unit", "description", "trial_type", "event_type",
}


def decode_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def json_value(value: Any) -> Any:
    value = decode_bytes(value)
    if value is None:
        return ""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if pd.isna(value):
        return ""
    return value


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            value = json_value(row[col])
            rec[col] = str(value) if col in STRING_COLUMNS else value
        records.append(rec)
    return records


def object_records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, pd.DataFrame):
        return dataframe_records(obj)
    if isinstance(obj, list):
        return [json_value(item) for item in obj]
    if isinstance(obj, dict):
        return [json_value(obj)]
    raise TypeError(f"Cannot convert {type(obj).__name__} to JSON records")


def read_u8_bytes(f: h5py.File, key: str) -> bytes:
    return np.asarray(f[key][:], dtype=np.uint8).tobytes()


class MEGH5Adapter:
    """Adapter for MEG H5 files with signal, timestamps, JSON metadata and pickle sidecars."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.modality = str(cfg_get(config, "dataset.modality", "meg")).strip().lower()
        if self.modality != "meg":
            raise ValueError(f"MEGH5Adapter requires dataset.modality='meg', got {self.modality!r}")
        self.dataset_name = str(cfg_get(config, "dataset.name", ""))
        self.task_type = str(cfg_get(config, "dataset.task_type", "Unknown"))
        self.description = str(cfg_get(config, "dataset.description", ""))
        self.signal_key = str(cfg_get(config, "source_h5.signal_key", "meg"))
        self.timestamps_key = str(cfg_get(config, "source_h5.timestamps_key", "meg_timestamps"))
        self.metadata_key = str(cfg_get(config, "source_h5.metadata_key", "metadata_json_u8"))
        self.channels_key = str(cfg_get(config, "source_h5.channels_key", "meg_channels_pkl_u8"))
        self.events_key = str(cfg_get(config, "source_h5.events_key", "events_pkl_u8"))

    def default_input_root(self) -> Path | None:
        return cfg_path(self.config, "paths.h5_root", None)

    def discover(self, input_root: Path) -> list[Path]:
        root = Path(input_root)
        candidates = [root] if root.is_file() else sorted(
            path for pattern in ("*.h5", "*.hdf5") for path in root.rglob(pattern)
        )
        required = {self.signal_key, self.timestamps_key, self.metadata_key, self.channels_key}
        discovered: list[Path] = []
        for path in candidates:
            try:
                with h5py.File(path, "r") as source:
                    if required.issubset(source.keys()):
                        discovered.append(path)
            except OSError:
                continue
        return discovered

    def read_metadata(self, f: h5py.File) -> dict[str, Any]:
        metadata = json.loads(read_u8_bytes(f, self.metadata_key).decode("utf-8"))
        attrs = {str(k): json_value(v) for k, v in f.attrs.items()}
        metadata["h5_attrs"] = attrs
        return metadata

    def read_pickle_records(self, f: h5py.File, key: str) -> list[dict[str, Any]]:
        obj = pickle.loads(read_u8_bytes(f, key))
        return object_records(obj)

    def read_sfreq(self, f: h5py.File, metadata: dict[str, Any]) -> float:
        attrs = metadata.get("h5_attrs", {})
        if "sfreq" in attrs:
            return float(attrs["sfreq"])
        if "sfreq" in metadata:
            return float(metadata["sfreq"])
        timestamps = np.asarray(f[self.timestamps_key][:], dtype=np.float64)
        return float(1.0 / np.median(np.diff(timestamps)))

    def build_subject(self, metadata: dict[str, Any]) -> dict[str, Any]:
        subject = metadata.get("Subject", {})
        attrs = metadata.get("h5_attrs", {})
        if not subject:
            subject = {
                "ID": attrs.get("sub", attrs.get("subject_id", "")),
                "Age": metadata.get("age", ""),
                "Sex": metadata.get("sex", metadata.get("gender", "")),
                "Health_group": metadata.get("health_group", metadata.get("group", "")),
            }
        return json_value(subject)

    def build_events(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events = []
        for row in records:
            start = row["start_time"] if "start_time" in row else row["onset"]
            if "end_time" in row:
                end = row["end_time"]
            else:
                end = float(start) + float(row.get("duration", 0.0))
            event_type = row.get("type", row.get("trial_type", row.get("event_type", "event")))
            event_value = row.get("value", row.get("stimulus", row.get("condition", "")))
            events.append({
                "start_time": float(start),
                "end_time": float(end),
                "type": str(event_type),
                "value": str(event_value),
                "other_info": {k: v for k, v in row.items() if k not in {"start_time", "end_time", "onset", "duration"}},
            })
        return events

    def build_session_metadata(
        self,
        path: Path,
        metadata: dict[str, Any],
        subject: dict[str, Any],
        sfreq: float,
        n_samples_total: int,
    ) -> dict[str, Any]:
        attrs = metadata.get("h5_attrs", {})
        return {
            "Dataset": attrs.get("dataset", metadata.get("Dataset", self.dataset_name)),
            "Task_type": attrs.get("task", metadata.get("Task_type", self.task_type)),
            "Subject": subject,
            "Description": metadata.get("Description", self.description),
            "Source_file": path.name,
            "sfreq": float(sfreq),
            "n_samples_total": int(n_samples_total),
            "h5_metadata": metadata,
        }

    def load_session(self, path: Path) -> SourceSession:
        path = Path(path)
        with h5py.File(path, "r") as f:
            signal = np.asarray(f[self.signal_key][:], dtype=np.float32)
            if signal.ndim != 2:
                raise ValueError(f"{path.name} signal must be [C, T], got {signal.shape}")
            timestamps = np.asarray(f[self.timestamps_key][:], dtype=np.float64)
            if timestamps.shape[0] != signal.shape[1]:
                raise ValueError(f"{path.name} timestamps length {timestamps.shape[0]} != signal T {signal.shape[1]}")

            metadata = self.read_metadata(f)
            sfreq = self.read_sfreq(f, metadata)
            subject = self.build_subject(metadata)
            channels = self.read_pickle_records(f, self.channels_key)
            if len(channels) != signal.shape[0]:
                raise ValueError(f"{path.name} channels {len(channels)} != signal C {signal.shape[0]}")
            events = self.build_events(self.read_pickle_records(f, self.events_key)) if self.events_key in f else []

        session_metadata = self.build_session_metadata(path, metadata, subject, sfreq, signal.shape[1])
        return SourceSession(
            source_path=path,
            source_stem=path.stem,
            safe_stem=safe_filename(path.stem),
            modality=self.modality,
            signal=signal,
            sfreq=sfreq,
            subject=subject,
            channels=channels,
            events=events,
            session_metadata=session_metadata,
        )

    def inspect_file(self, path: Path, split_points: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        path = Path(path)
        findings: list[dict[str, Any]] = []
        with h5py.File(path, "r") as f:
            for key, label in [
                (self.signal_key, "signal"),
                (self.timestamps_key, "timestamps"),
                (self.metadata_key, "metadata"),
                (self.channels_key, "channels"),
            ]:
                if key not in f:
                    findings.append({"severity": "error", "category": "input_schema", "message": f"{path.name} missing {label} key: {key}", "file": path.name})
            if self.signal_key not in f:
                return {"file": path.name}, findings

            signal = f[self.signal_key]
            summary = {
                "file": path.name,
                "path": str(path),
                "signal_key": self.signal_key,
                "shape": [int(v) for v in signal.shape],
                "attrs": {str(k): json_value(v) for k, v in f.attrs.items()},
            }
            if signal.ndim != 2:
                findings.append({"severity": "error", "category": "input_schema", "message": f"{path.name} signal must be [C, T], got {signal.shape}", "file": path.name})
                return summary, findings
            if self.timestamps_key in f and f[self.timestamps_key].shape[0] != signal.shape[1]:
                findings.append({"severity": "error", "category": "input_schema", "message": f"{path.name} timestamps length mismatch", "file": path.name})

            metadata = self.read_metadata(f) if self.metadata_key in f else {"h5_attrs": dict(summary["attrs"])}
            sfreq = self.read_sfreq(f, metadata) if self.timestamps_key in f else math.nan
            summary["sfreq"] = float(sfreq) if math.isfinite(sfreq) else None
            summary["estimated_chunks"] = int(math.ceil(int(signal.shape[1]) / int(split_points)))
            summary["n_events"] = int(f.attrs.get("n_events", 0)) if self.events_key not in f else len(self.read_pickle_records(f, self.events_key))
            summary["n_channels"] = int(signal.shape[0])
        return summary, findings
