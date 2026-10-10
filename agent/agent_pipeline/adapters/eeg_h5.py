"""EEG HDF5 source adapter for Stage 1 packaging.

This adapter owns the dataset/source-specific part of Stage 1: how to read an
EEG-like HDF5 file, normalize subject/channel/event tables, and expose a
standard SourceSession to the generic tar writer.
"""

from __future__ import annotations

import math
from glob import glob
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd

from agent_pipeline.config import cfg_get, cfg_path, none_if_empty

from .base import SourceSession, safe_filename


FORCE_STRING_COLUMNS = {
    "ID", "id", "subject_id",
    "Age", "age", "Sex", "sex", "Gender", "gender",
    "Health_group", "health_group", "group", "diagnosis",
    "Outcome", "Engel_score", "Ilae_score", "TR", "TE",
    "channel_name", "channel_type", "ch_name", "ch_type", "name", "label",
    "type", "status", "unit", "reference", "description",
    "coord_source", "brain_region", "region", "soz",
    "stim", "stimulus", "stim_path", "stimulus_path", "filename",
    "file_name", "path", "trial_type", "event_type",
    "condition", "category", "image_path",
}

FORCE_NUMERIC_COLUMNS = {
    "start_time", "end_time", "original_start_time", "original_end_time",
    "onset", "offset", "duration", "sample", "sample_index",
    "trial", "trial_index",
    "sfreq", "sampling_frequency", "x", "y", "z", "ori", "ori_x", "ori_y", "ori_z",
    "mean", "std", "absmax", "is_bad",
}

DROP_CHANNEL_COLUMNS = {"noise_index_before", "noise_index_after"}
METADATA_CHANNEL_CONTEXTS = {"metadata", "subject", "channels", "channel"}


def decode_if_bytes(x: Any) -> Any:
    if isinstance(x, bytes):
        for enc in ("utf-8", "gbk", "latin1"):
            try:
                return x.decode(enc)
            except UnicodeDecodeError:
                continue
        return x.decode("utf-8", errors="replace")
    return x


def change_dtype(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(decode_if_bytes)
    return df


def is_missing_scalar(x: Any) -> bool:
    x = decode_if_bytes(x)
    if x is None:
        return True
    if isinstance(x, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def is_numeric_column(col_name: str, series: Optional[pd.Series] = None) -> bool:
    col_name = str(col_name)
    if col_name in FORCE_STRING_COLUMNS:
        return False
    if col_name in FORCE_NUMERIC_COLUMNS:
        return True
    if series is not None:
        return bool(pd.api.types.is_numeric_dtype(series))
    return False


def normalize_json_value(
    value: Any,
    col_name: Optional[str] = None,
    series: Optional[pd.Series] = None,
    missing_policy: str = "typed",
) -> Any:
    value = decode_if_bytes(value)
    numeric_col = col_name is not None and is_numeric_column(col_name, series=series)
    string_col = col_name is not None and str(col_name) in FORCE_STRING_COLUMNS

    if is_missing_scalar(value):
        if missing_policy == "empty_string":
            return ""
        return np.nan if numeric_col else ""
    if string_col:
        return str(value)
    if numeric_col:
        if isinstance(value, (np.bool_, bool)):
            return int(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (np.floating, float)):
            v = float(value)
            return v if math.isfinite(v) else ("" if missing_policy == "empty_string" else np.nan)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return "" if missing_policy == "empty_string" else np.nan
            try:
                fv = float(stripped)
                if not math.isfinite(fv):
                    return "" if missing_policy == "empty_string" else np.nan
                return int(fv) if fv.is_integer() else fv
            except ValueError:
                return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else ("" if missing_policy == "empty_string" else np.nan)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [normalize_json_value(v, col_name=col_name, missing_policy=missing_policy) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(v, col_name=col_name, missing_policy=missing_policy) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_json_value(v, col_name=str(k), missing_policy=missing_policy) for k, v in value.items()}
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def dataframe_to_json_records(df: pd.DataFrame, context: str = "event") -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    df = change_dtype(df.copy())
    df.columns = [str(c).strip() for c in df.columns]
    missing_policy = "empty_string" if context in METADATA_CHANNEL_CONTEXTS else "typed"
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            rec[col] = normalize_json_value(
                row[col],
                col_name=str(col),
                series=df[col],
                missing_policy=missing_policy,
            )
        records.append(rec)
    return records


def cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("", "nan") else s


class EegH5Adapter:
    """Adapter for EEG HDF5 files with table-like metadata groups."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.modality = str(cfg_get(config, "dataset.modality", "eeg")).strip().lower()
        if self.modality != "eeg":
            raise ValueError(f"EegH5Adapter only supports dataset.modality='eeg', got {self.modality!r}")
        self.dataset_name = str(cfg_get(config, "dataset.name", ""))
        self.task_type = str(cfg_get(config, "dataset.task_type", "Unknown"))
        self.description = str(cfg_get(config, "dataset.description", ""))
        self.signal_key = str(cfg_get(config, "source_h5.signal_key", "eeg"))
        self.sfreq_key = str(cfg_get(config, "source_h5.sfreq_key", "metadata/sfreq"))
        self.channel_key = str(cfg_get(config, "source_h5.channel_key", "metadata/channel"))
        self.subject_key = str(cfg_get(config, "source_h5.subject_key", "metadata/subject"))
        self.event_key = str(cfg_get(config, "source_h5.event_key", "event"))
        self.event_type = str(cfg_get(config, "event.type", "image"))
        self.event_type_source_column = none_if_empty(cfg_get(config, "event.type_source_column", None))
        self.value_source_column = none_if_empty(cfg_get(config, "event.value_source_column", None))

    def default_input_root(self) -> Optional[Path]:
        return cfg_path(self.config, "paths.h5_root", None)

    def discover(self, input_root: Path) -> list[Path]:
        return [Path(p) for p in sorted(glob(str(Path(input_root) / "*.h5")))]

    def read_scalar_sfreq(self, f: h5py.File) -> float:
        if self.sfreq_key not in f:
            raise ValueError(f"HDF5 file does not contain {self.sfreq_key!r}")
        raw = np.asarray(f[self.sfreq_key][()])
        if raw.shape == ():
            return float(raw)
        first = raw.reshape(-1)[0]
        if isinstance(first, np.void) and first.dtype.names:
            return float(first[first.dtype.names[0]])
        if hasattr(first, "dtype") and getattr(first.dtype, "names", None):
            return float(first[first.dtype.names[0]])
        return float(first)

    def read_table_as_dataframe(self, f: h5py.File, key: str) -> pd.DataFrame:
        if key not in f:
            return pd.DataFrame()
        df = change_dtype(pd.DataFrame(f[key][:]))
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def build_session_events(self, events_df_source: pd.DataFrame) -> list[dict[str, Any]]:
        if events_df_source is None or events_df_source.empty:
            return []
        if "start_time" not in events_df_source.columns:
            raise ValueError("event table does not contain required column 'start_time'")
        if "end_time" not in events_df_source.columns:
            raise ValueError("event table does not contain required column 'end_time'")
        if self.value_source_column is not None and self.value_source_column not in events_df_source.columns:
            raise KeyError(
                f"value_source_column={self.value_source_column!r} is not in event columns: {list(events_df_source.columns)}"
            )
        if self.event_type_source_column is not None and self.event_type_source_column not in events_df_source.columns:
            raise KeyError(
                f"type_source_column={self.event_type_source_column!r} is not in event columns: {list(events_df_source.columns)}"
            )

        records = dataframe_to_json_records(events_df_source, context="event")
        events = []
        for row_dict in records:
            event_type = (
                self.event_type
                if self.event_type_source_column is None
                else cell_to_str(row_dict.get(self.event_type_source_column)) or self.event_type
            )
            event_value = "" if self.value_source_column is None else cell_to_str(row_dict.get(self.value_source_column))
            other_info = {k: v for k, v in row_dict.items() if k not in {"start_time", "end_time"}}
            events.append({
                "start_time": float(row_dict["start_time"]),
                "end_time": float(row_dict["end_time"]),
                "type": event_type,
                "value": event_value,
                "other_info": other_info,
            })
        return events

    def build_session_metadata(self, subject: dict[str, Any], source_file: str, sfreq: float, n_samples_total: int) -> dict[str, Any]:
        return {
            "Dataset": self.dataset_name,
            "Task_type": self.task_type,
            "Subject": subject,
            "Description": self.description,
            "Source_file": source_file,
            "sfreq": float(sfreq),
            "n_samples_total": int(n_samples_total),
        }

    def load_session(self, path: Path) -> SourceSession:
        path = Path(path)
        source_stem = path.stem
        safe_stem = safe_filename(source_stem)
        with h5py.File(path, "r") as f:
            if self.signal_key not in f:
                raise KeyError(f"{path.name} does not contain signal key {self.signal_key!r}")
            signal = np.asarray(f[self.signal_key][:])
            if signal.ndim != 2:
                raise ValueError(f"{path.name} signal must be [C, T], got {signal.shape}")
            sfreq = self.read_scalar_sfreq(f)

            subject = {"ID": "", "Age": "", "Sex": "", "Health_group": ""}
            subject_df = self.read_table_as_dataframe(f, self.subject_key)
            if not subject_df.empty:
                subject_records = dataframe_to_json_records(subject_df, context="subject")
                if subject_records:
                    subject = subject_records[0]

            channels_df = self.read_table_as_dataframe(f, self.channel_key)
            if not channels_df.empty:
                channels_df = channels_df.drop(columns=[c for c in DROP_CHANNEL_COLUMNS if c in channels_df.columns])
            channels = dataframe_to_json_records(channels_df, context="channels")

            events_df = self.read_table_as_dataframe(f, self.event_key)
            events = self.build_session_events(events_df)

        n_samples_total = int(signal.shape[1])
        session_metadata = self.build_session_metadata(
            subject=subject,
            source_file=path.name,
            sfreq=sfreq,
            n_samples_total=n_samples_total,
        )
        return SourceSession(
            source_path=path,
            source_stem=source_stem,
            safe_stem=safe_stem,
            modality=self.modality,
            signal=signal,
            sfreq=sfreq,
            subject=subject,
            channels=channels,
            events=events,
            session_metadata=session_metadata,
        )

    def inspect_file(self, path: Path, split_points: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        findings: list[dict[str, Any]] = []
        path = Path(path)
        summary: dict[str, Any] = {"file": path.name}
        try:
            with h5py.File(path, "r") as f:
                for key, label, severity in [
                    (self.signal_key, "signal", "error"),
                    (self.sfreq_key, "sfreq", "error"),
                    (self.channel_key, "channels", "error"),
                    (self.subject_key, "subject", "warning"),
                    (self.event_key, "event", "warning"),
                ]:
                    if key not in f:
                        findings.append({
                            "severity": severity,
                            "category": "input_schema",
                            "message": f"{path.name} missing {label} key: {key}",
                            "file": path.name,
                        })
                if self.signal_key not in f:
                    return summary, findings
                signal = f[self.signal_key]
                summary["signal_key"] = self.signal_key
                summary["shape"] = [int(v) for v in signal.shape]
                if signal.ndim != 2:
                    findings.append({
                        "severity": "error",
                        "category": "input_schema",
                        "message": f"{path.name} signal must be 2D [C, T], got {signal.shape}",
                        "file": path.name,
                    })
                sfreq = self.read_scalar_sfreq(f) if self.sfreq_key in f else math.nan
                summary["sfreq"] = float(sfreq) if math.isfinite(sfreq) else None
                summary["estimated_chunks"] = (
                    int(math.ceil(int(signal.shape[-1]) / int(split_points)))
                    if signal.ndim == 2 and split_points > 0
                    else None
                )
                if self.event_key in f:
                    cols = list(getattr(f[self.event_key].dtype, "names", None) or [])
                    for col in [self.value_source_column, self.event_type_source_column]:
                        if col and col not in cols:
                            findings.append({
                                "severity": "error",
                                "category": "input_schema",
                                "message": f"{path.name} event table missing configured column: {col}",
                                "file": path.name,
                            })
        except Exception as exc:
            findings.append({
                "severity": "error",
                "category": "input_data",
                "message": f"Failed to inspect {path.name}: {exc}",
                "file": path.name,
            })
        return summary, findings
