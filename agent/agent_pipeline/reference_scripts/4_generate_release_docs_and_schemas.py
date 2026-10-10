#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate release documentation and JSON Schemas for a NeuroToken dataset.

This stage is intended to be run after Stage 2 tokenization. It creates the
minimal public-release documentation set plus machine-readable schemas:

neurotoken_dataset/
├── DATA_DICTIONARY.md
└── schemas/
    ├── manifest.schema.json
    ├── subjects.schema.json
    ├── session_metadata.schema.json
    ├── token_metadata.schema.json
    ├── events_info.schema.json
    ├── eeg_channels.schema.json
    └── codebook_metadata.schema.json

The generated files define field meanings and machine-readable validation rules
for the public NeuroToken dataset directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurotoken_config import cfg_path, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_VERSION = "https://json-schema.org/draft/2020-12/schema"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def session_dir_from_summary(output_base: Path, session: dict) -> Path:
    explicit = session.get("output_session_dir")
    if isinstance(explicit, str) and explicit:
        return output_base / explicit
    token_metadata = session.get("token_metadata")
    if isinstance(token_metadata, str) and token_metadata:
        return output_base / Path(token_metadata).parent
    session_id = str(session.get("session_id", ""))
    return output_base / "sessions" / session_id


def first_existing_events(output_base: Path, manifest: dict) -> list:
    for session in manifest.get("sessions", []):
        session_dir = session_dir_from_summary(output_base, session)
        events = load_json(session_dir / "events_info.json", default=None)
        if isinstance(events, list):
            return events
    return []


def first_existing_channels(output_base: Path, manifest: dict, modality: str) -> list:
    candidates = [f"{modality}_channels.json", "eeg_channels.json", "meg_channels.json", "ieeg_channels.json"]
    for session in manifest.get("sessions", []):
        session_dir = session_dir_from_summary(output_base, session)
        for name in candidates:
            channels = load_json(session_dir / name, default=None)
            if isinstance(channels, list):
                return channels
    return []


def schema_string(description: str, *, enum: Optional[list] = None) -> dict:
    result = {"type": "string", "description": description}
    if enum is not None:
        result["enum"] = enum
    return result


def schema_number(description: str) -> dict:
    return {"type": "number", "description": description}


def schema_integer(description: str) -> dict:
    return {"type": "integer", "description": description}


def schema_array(description: str, items: dict) -> dict:
    return {"type": "array", "description": description, "items": items}


def generate_schemas(modality: str) -> Dict[str, dict]:
    subject_schema = {
        "type": "object",
        "description": "Subject-level information. Required fields may be empty strings when unavailable.",
        "required": ["ID", "Age", "Sex", "Health_group"],
        "properties": {
            "ID": schema_string("Anonymized subject identifier, e.g. sub-01."),
            "Age": schema_string("Age or age bin. Stored as a string to allow values such as 20-29."),
            "Sex": schema_string("Biological sex or reported sex label, if released.", enum=["Female", "Male", "Other", ""]),
            "Health_group": schema_string("Health or disease group label, e.g. Healthy."),
        },
        "additionalProperties": True,
    }

    subject_index_entry_schema = {
        "type": "object",
        "description": "Subject-level index entry. Required metadata fields may be empty strings when unavailable.",
        "required": ["ID", "Age", "Sex", "Health_group", "session_ids"],
        "properties": {
            "ID": schema_string("Anonymized subject identifier, e.g. sub-01."),
            "Age": schema_string("Age or age bin. Stored as a string to allow values such as 20-29."),
            "Sex": schema_string("Biological sex or reported sex label, if released.", enum=["Female", "Male", "Other", ""]),
            "Health_group": schema_string("Health or disease group label, e.g. Healthy."),
            "session_ids": schema_array("Session IDs belonging to this subject.", {"type": "string"}),
        },
        "additionalProperties": False,
    }

    subjects_schema = {
        "$schema": SCHEMA_VERSION,
        "title": "NeuroToken subjects.json",
        "type": "array",
        "description": "Dataset-level subject index generated from manifest session metadata.",
        "items": subject_index_entry_schema,
    }

    session_metadata_schema = {
        "$schema": SCHEMA_VERSION,
        "title": "NeuroToken session_metadata.json",
        "type": "object",
        "required": ["Dataset", "Task_type", "Subject", "Description", "Source_file", "sfreq", "n_samples_total"],
        "properties": {
            "Dataset": schema_string("Dataset name."),
            "Task_type": schema_string("Recording/task type.", enum=["Resting", "Task", "Unknown", ""]),
            "Subject": subject_schema,
            "Description": schema_string("Short dataset or session description."),
            "Source_file": schema_string("Original source file name, not an absolute path."),
            "sfreq": schema_number("Sampling frequency in Hz."),
            "n_samples_total": schema_integer("Total number of original time samples in this recording."),
        },
        "additionalProperties": True,
    }

    event_schema = {
        "type": "object",
        "required": ["start_time", "end_time", "type", "value", "other_info"],
        "properties": {
            "start_time": schema_number("Event/window start time in absolute seconds from recording start."),
            "end_time": schema_number("Event/window end time in absolute seconds from recording start."),
            "type": schema_string("Stimulus/event modality or controlled event type."),
            "value": schema_string("Stimulus identifier, relative path, label, or empty string."),
            "other_info": {
                "type": "object",
                "description": "Dataset-specific event annotations copied from the source event table.",
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }

    events_info_schema = {
        "$schema": SCHEMA_VERSION,
        "title": "NeuroToken events_info.json",
        "type": "array",
        "description": "Session-level events with absolute timing in seconds.",
        "items": event_schema,
    }

    channel_schema = {
        "type": "object",
        "required": ["channel_name", "channel_type", "std", "mean"],
        "properties": {
            "channel_name": schema_string("Channel label, e.g. Fp1."),
            "channel_type": schema_string("Channel modality/type, e.g. EEG."),
            "std": schema_number("Per-channel standard deviation used by the tokenizer normalization."),
            "mean": schema_number("Per-channel mean used by the tokenizer normalization."),
            "absmax": schema_number("Maximum absolute amplitude observed for this channel, if available."),
            "is_bad": {"type": ["integer", "boolean", "number"], "description": "Bad-channel flag."},
            "x": schema_number("Sensor x coordinate, coordinate system described by coord_source."),
            "y": schema_number("Sensor y coordinate, coordinate system described by coord_source."),
            "z": schema_number("Sensor z coordinate, coordinate system described by coord_source."),
            "coord_source": schema_string("Source or coordinate system for x/y/z."),
        },
        "additionalProperties": True,
    }

    channels_schema = {
        "$schema": SCHEMA_VERSION,
        "title": f"NeuroToken {modality}_channels.json",
        "type": "array",
        "description": "Session-level channel metadata and normalization statistics.",
        "items": channel_schema,
    }

    token_metadata_schema = {
        "$schema": SCHEMA_VERSION,
        "title": "NeuroToken token_metadata.json",
        "type": "object",
        "required": ["format", "version", "session", "tokens", "tokenizer", "temporal_index", "spatial_layout", "provenance"],
        "properties": {
            "format": schema_string("Session metadata format identifier.", enum=["neurotoken_session"]),
            "version": schema_string("Session metadata schema version."),
            "session": {
                "type": "object",
                "required": ["session_id", "modality"],
                "properties": {"session_id": schema_string("Output session identifier."), "modality": schema_string("Neural modality.")},
                "additionalProperties": True,
            },
            "tokens": {
                "type": "object",
                "required": ["file", "path", "shape", "dtype", "num_codebooks", "id_space", "array_layout"],
                "properties": {
                    "file": schema_string("Token file name inside the session directory."),
                    "path": schema_string("<modality>_token_ids.npy path relative to dataset root."),
                    "shape": schema_array("Shape of <modality>_token_ids.npy: [total_tokens, num_codebooks].", {"type": "integer"}),
                    "dtype": schema_string("Expected dtype of <modality>_token_ids.npy, usually int32."),
                    "num_codebooks": schema_integer("Number of token id columns/codebooks."),
                    "id_space": schema_string("Token id namespace.", enum=["local_per_codebook"]),
                    "array_layout": {"type": "object", "additionalProperties": True},
                },
                "additionalProperties": True,
            },
            "tokenizer": {"type": "object", "additionalProperties": True},
            "temporal_index": {"type": "object", "required": ["num_segments", "segments"], "additionalProperties": True},
            "spatial_layout": {"type": "object", "required": ["type"], "additionalProperties": True},
            "provenance": {"type": "object", "additionalProperties": True},
        },
        "additionalProperties": True,
    }

    manifest_session_schema = {
        "type": "object",
        "required": ["session_id", "modality", "token_file", "token_metadata", "token_shape", "token_dtype", "num_codebooks", "subject"],
        "properties": {
            "session_id": schema_string("Output session identifier."),
            "modality": schema_string("Neural modality."),
            "token_file": schema_string("<modality>_token_ids.npy path relative to dataset root."),
            "token_metadata": schema_string("Path to token_metadata.json relative to dataset root."),
            "token_shape": schema_array("Shape of <modality>_token_ids.npy.", {"type": "integer"}),
            "token_dtype": schema_string("Dtype of <modality>_token_ids.npy."),
            "num_codebooks": schema_integer("Number of token id columns/codebooks."),
            "dataset": schema_string("Dataset name copied from session metadata."),
            "task_type": schema_string("Task type copied from session metadata."),
            "subject": subject_schema,
            "has_temporal_index": {"type": "boolean", "description": "Whether token_metadata contains temporal_index."},
            "has_events": {"type": "boolean", "description": "Whether events_info.json exists for this session."},
            "provenance": {"type": "object", "description": "Compact provenance fields.", "additionalProperties": True},
        },
        "additionalProperties": True,
    }

    manifest_schema = {
        "$schema": SCHEMA_VERSION,
        "title": "NeuroToken manifest.json",
        "type": "object",
        "required": ["format", "version", "modality", "tokenizer", "sessions_dir", "num_sessions", "sessions"],
        "properties": {
            "format": schema_string("Dataset format identifier.", enum=["neurotoken_dataset"]),
            "version": schema_string("Dataset schema version."),
            "modality": schema_string("Dataset neural modality."),
            "input_base": schema_string("Original tar-package root name used to generate this dataset."),
            "output_base": schema_string("Dataset output root name."),
            "readme": schema_string("README path relative to dataset root."),
            "documentation": {
                "type": "object",
                "description": "Documentation file index relative to dataset root.",
                "additionalProperties": True,
            },
            "tokenizer": {
                "type": "object",
                "required": ["codebook", "codebook_metadata"],
                "properties": {
                    "codebook": schema_string("codebook.npz path relative to dataset root."),
                    "codebook_metadata": schema_string("codebook_metadata.json path relative to dataset root."),
                },
                "additionalProperties": True,
            },
            "stimuli": {
                "type": "object",
                "properties": {
                    "path": schema_string("Stimuli folder path relative to dataset root."),
                    "status": schema_string("Stimuli inclusion/access note."),
                },
                "additionalProperties": True,
            },
            "sessions_dir": schema_string("Sessions directory relative to dataset root."),
            "num_sessions": schema_integer("Number of sessions listed in manifest.sessions."),
            "split_policy": schema_string("Split availability note."),
            "sessions": schema_array("Session summaries.", manifest_session_schema),
            "failed_sessions": schema_array("Optional source sessions that failed tokenization.", {"type": "string"}),
            "failed_session_errors": schema_array(
                "Structured error provenance for failed source sessions.",
                {
                    "type": "object",
                    "required": ["session", "error_type", "message"],
                    "properties": {
                        "session": schema_string("Source session identifier."),
                        "error_type": schema_string("Exception class name."),
                        "message": schema_string("Exception message."),
                    },
                    "additionalProperties": True,
                },
            ),
        },
        "additionalProperties": True,
    }

    codebook_metadata_schema = {
        "$schema": SCHEMA_VERSION,
        "title": "NeuroToken codebook_metadata.json",
        "type": "object",
        "required": ["format", "num_codebooks", "vocab_size", "vocab_size_per_codebook", "vocab_width", "width_per_codebook", "token_id_layout", "embedding_lookup", "modality"],
        "properties": {
            "format": schema_string("Codebook format name."),
            "num_codebooks": schema_integer("Number of codebooks / token id columns."),
            "vocab_size": schema_integer("Total vocabulary size."),
            "vocab_size_per_codebook": schema_integer("Valid local ids are 0..vocab_size_per_codebook-1."),
            "vocab_width": schema_integer("Total embedding width after codebook concatenation."),
            "width_per_codebook": schema_integer("Embedding width contributed by one codebook."),
            "token_id_layout": schema_string("Description of token id array layout."),
            "embedding_lookup": schema_string("Human-readable lookup expression."),
            "checkpoint": schema_string("Tokenizer checkpoint identifier. Prefer relative id plus hash/provenance."),
            "modality": schema_string("Neural modality."),
        },
        "additionalProperties": True,
    }

    return {
        "manifest.schema.json": manifest_schema,
        "subjects.schema.json": subjects_schema,
        "session_metadata.schema.json": session_metadata_schema,
        "token_metadata.schema.json": token_metadata_schema,
        "events_info.schema.json": events_info_schema,
        f"{modality}_channels.schema.json": channels_schema,
        "eeg_channels.schema.json": channels_schema if modality != "eeg" else channels_schema,
        "codebook_metadata.schema.json": codebook_metadata_schema,
    }


def field_list(fields: List[tuple]) -> str:
    return "\n".join(f"- `{name}` ({kind}): {desc}" for name, kind, desc in fields)


def build_data_dictionary(manifest: dict, events: list, channels: list, modality: str) -> str:
    channel_extra_keys = []
    for channel in channels[:20]:
        if isinstance(channel, dict):
            for key in channel.keys():
                if key in {"noise_index_before", "noise_index_after"}:
                    continue
                if key not in channel_extra_keys:
                    channel_extra_keys.append(key)

    manifest_fields = [
        ("format", "string", "Dataset format identifier. Expected value: `neurotoken_dataset`."),
        ("modality", "string", "Neural modality for this dataset, e.g. `eeg`, `meg`, or `fmri`."),
        ("input_base", "string", "Generation provenance only; not needed for loading the public dataset."),
        ("output_base", "string", "Dataset root name used during generation."),
        ("readme", "string", "README path relative to dataset root, if provided by Stage 3."),
        ("documentation", "object", "Index of documentation and schema files relative to dataset root."),
        ("tokenizer", "object", "Paths to tokenizer codebook and codebook metadata."),
        ("stimuli", "object", "Stimulus folder path and access/inclusion status."),
        ("sessions_dir", "string", "Directory containing released session folders."),
        ("num_sessions", "integer", "Number of released sessions listed in `sessions`."),
        ("sessions", "array", "Compact summaries of released sessions."),
        ("failed_sessions", "array", "Optional source session identifiers that failed tokenization."),
        ("failed_session_errors", "array", "Optional structured exception provenance for failed source sessions."),
    ]

    session_metadata_fields = [
        ("Dataset", "string", "Dataset name or dataset label."),
        ("Task_type", "string", "Task category: `Resting`, `Task`, or `Unknown`."),
        ("Subject.ID", "string", "Anonymized subject identifier."),
        ("Subject.Age", "string", "Age or age bin."),
        ("Subject.Sex", "string", "Sex label if released. Allowed examples: `Female`, `Male`, `Other`, or empty string."),
        ("Subject.Health_group", "string", "Health/disease group label."),
        ("Description", "string", "Short description of the recording or source dataset."),
        ("Source_file", "string", "Original source file name."),
        ("sfreq", "number", "Sampling frequency in Hz."),
        ("n_samples_total", "integer", "Number of original time samples in the recording."),
    ]

    subject_index_fields = [
        ("ID", "string", "Anonymized subject identifier. Empty string when unavailable."),
        ("Age", "string", "Age or age bin. Empty string when unavailable."),
        ("Sex", "string", "Sex label. Empty string when unavailable."),
        ("Health_group", "string", "Health/disease group label. Empty string when unavailable."),
        ("session_ids", "array", "Session IDs belonging to this subject."),
    ]

    event_fields = [
        ("start_time", "number", "Absolute start time in seconds from recording start."),
        ("end_time", "number", "Absolute end time in seconds from recording start."),
        ("type", "string", "Event/stimulus type such as `image`, `video`, `audio`, `text`, `label`, or `resting_event`."),
        ("value", "string", "Primary event or stimulus identifier, relative path, label, or empty string."),
        ("other_info", "object", "Optional dataset-specific fields copied from the original source data."),
    ]

    channel_fields = [
        ("channel_name", "string", "Channel label, e.g. `Fp1`."),
        ("channel_type", "string", "Channel type/modality, e.g. `EEG`."),
        ("std", "number", "Per-channel standard deviation used by tokenization-time z-score normalization."),
        ("mean", "number", "Per-channel mean used by tokenization-time z-score normalization."),
        ("absmax", "number", "Maximum absolute amplitude for the channel if available."),
        ("is_bad", "integer/boolean", "Bad-channel flag. Interpret according to the source preprocessing pipeline."),
        ("x", "number", "Sensor x coordinate."),
        ("y", "number", "Sensor y coordinate."),
        ("z", "number", "Sensor z coordinate."),
        ("coord_source", "string", "Source or coordinate system for channel coordinates."),
    ]

    token_metadata_fields = [
        ("format", "string", "Fixed value `neurotoken_session`."),
        ("version", "string", "Session metadata schema version."),
        ("session.session_id", "string", "Output session identifier."),
        ("session.modality", "string", "Neural modality for this session."),
        ("tokens.file", "string", "Token file name inside the session directory."),
        ("tokens.path", "string", "Path to `<modality>_token_ids.npy` relative to dataset root."),
        ("tokens.shape", "array", "Shape `[total_tokens, num_codebooks]` of `<modality>_token_ids.npy`."),
        ("tokens.dtype", "string", "Dtype of `<modality>_token_ids.npy`, usually `int32`."),
        ("tokens.num_codebooks", "integer", "Number of token id columns/codebooks."),
        ("tokens.id_space", "string", "Token id namespace, currently `local_per_codebook`."),
        ("tokenizer", "object", "Tokenizer and codebook provenance for tracking token version."),
        ("temporal_index", "object", "Optional-for-training token-range to source-time index; useful for time slicing and event alignment."),
        ("temporal_index.segments[]", "array", "Compact segment entries grouped into `token` (token_ids row range and token grid) and `source_time` (source signal time range)."),
        ("spatial_layout", "object", "How flattened tokens map to channel groups or fMRI patch grids."),
        ("provenance", "object", "Generation and preprocessing details for reproducibility/debugging."),
    ]

    codebook_fields = [
        ("format", "string", "Codebook format name."),
        ("num_codebooks", "integer", "Number of codebooks and token id columns."),
        ("vocab_size", "integer", "Total vocabulary size across codebooks."),
        ("vocab_size_per_codebook", "integer", "Valid local id range is `0..vocab_size_per_codebook-1`."),
        ("vocab_width", "integer", "Concatenated embedding width."),
        ("width_per_codebook", "integer", "Embedding width contributed by one codebook."),
        ("token_id_layout", "string", "Description of `<modality>_token_ids.npy` layout."),
        ("embedding_lookup", "string", "Human-readable codebook lookup expression."),
        ("checkpoint", "string", "Tokenizer checkpoint identifier."),
        ("modality", "string", "Neural modality."),
    ]

    channel_extra_section = "\n".join(f"- `{key}`" for key in channel_extra_keys) or "- No channel keys were inferred from example channels."

    return f"""# NeuroToken Data Dictionary

This document defines the fields visible in the released NeuroToken dataset. It is intended for public dataset users. Machine-readable JSON Schemas are provided in `schemas/` for validation; most users can rely on this data dictionary and README.

## Missing values

- Missing string/categorical metadata should be stored as an empty string `""`.

## `manifest.json`

{field_list(manifest_fields)}

### `manifest.sessions[]`

Each entry is a compact summary of one released session. Use `token_file` to locate the token array and `token_metadata` to locate the session-level token contract.

Important fields include `session_id`, `token_file`, `token_shape`, `token_metadata`, `dataset`, `task_type`, and `subject`. This release does not provide official train/validation/test splits; users should create task-appropriate splits from session and subject metadata.

## `subjects.json`

`subjects.json` is a dataset-level subject index generated from `manifest.sessions[].subject`. It is provided for convenience so users can quickly group sessions by subject without scanning every session folder.

{field_list(subject_index_fields)}

## `sessions/<session>/session_metadata.json`

{field_list(session_metadata_fields)}

## `sessions/<session>/events_info.json`

`events_info.json` is a session-level JSON array. The fixed public fields are `start_time`, `end_time`, `type`, and `value`. `other_info` may contain additional fields copied from the original source data; these fields are dataset-specific and are not part of the fixed NeuroToken schema.

{field_list(event_fields)}

### Timing notes

- `start_time` and `end_time` are absolute seconds from the beginning of the source recording.
- For task datasets, event timing follows the source event table. Depending on the source dataset, `end_time` may correspond to stimulus offset, trial/window end, or another source-defined endpoint.
- For resting-state datasets, events may be synthetic windows added to describe the released recording segments.

## `sessions/<session>/{modality}_channels.json`

Channel sidecars are session-level JSON arrays. For EEG/MEG-like modalities, channel order describes the neural data order used before tokenization. It is useful for interpreting source-channel metadata, but `<modality>_token_ids.npy` stores token ids rather than a direct `[channel, time]` signal matrix.

{field_list(channel_fields)}

### Channel keys inferred from this dataset

{channel_extra_section}

## `sessions/<session>/token_metadata.json`

{field_list(token_metadata_fields)}

This file is the session-level token contract. Plain sequence-modeling users can load `tokens.path` directly and ignore `temporal_index`; users who need time slicing or event alignment can use `temporal_index.segments[]` to map token ranges back to source time windows.

## `sessions/<session>/<modality>_token_ids.npy`

- Shape: `[total_tokens, num_codebooks]`
- Dtype: usually `int32`
- Each row is one neural token.
- Each column is the local token id for one codebook.
- Valid ids should be in `0..vocab_size_per_codebook-1` for each codebook.

## `tokenizer/codebook_metadata.json`

{field_list(codebook_fields)}

## `tokenizer/codebook.npz`

Expected arrays:

- `codebook_raw`: raw codebook vectors.
- `codebook_normalized`: normalized codebook vectors for embedding lookup.

Given `token_ids`, a typical embedding lookup is described in `codebook_metadata.json`.

## `stimuli/`

You may map `events_info.value` or `events_info.other_info` identifiers to external stimulus resources. Note that the stimuli directory may be empty due to license limitations.
"""


def patch_manifest_documentation(output_base: Path, manifest: dict) -> None:
    manifest["documentation"] = {
        "data_dictionary": "DATA_DICTIONARY.md",
        "schemas": "schemas/",
    }
    stimuli = manifest.setdefault("stimuli", {})
    stimuli.setdefault("path", "stimuli")
    old_status = str(stimuli.get("status", ""))
    if not old_status or "placeholder" in old_status.lower():
        stimuli["status"] = (
            "Stimulus files are not included by default. Use events_info.json "
            "value/other_info identifiers to map stimuli according to the source dataset license."
        )
    write_json(output_base / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NeuroToken data dictionary and JSON schemas.")
    parser.add_argument("--config", type=Path, default=None, help="TOML pipeline config file.")
    parser.add_argument("--output-base", type=Path, default=None, help="NeuroToken dataset directory. Overrides config/default.")
    parser.add_argument("--no-patch-manifest", action="store_true", help="Do not add documentation paths to manifest.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_config = load_config(args.config)
    output_base = args.output_base or cfg_path(pipeline_config, "paths.token_root")
    if output_base is None:
        raise ValueError("paths.token_root is not configured and --output-base was not provided")
    output_base = Path(output_base)
    manifest_path = output_base / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")

    manifest = load_json(manifest_path, default={})
    modality = str(manifest.get("modality", "eeg"))
    events = first_existing_events(output_base, manifest)
    channels = first_existing_channels(output_base, manifest, modality)

    schemas = generate_schemas(modality)

    for obsolete_path in [output_base / "dataset_description.json", output_base / "schemas" / "dataset_description.schema.json"]:
        if obsolete_path.exists():
            obsolete_path.unlink()
            logger.info("Removed obsolete %s", obsolete_path)

    write_text(output_base / "DATA_DICTIONARY.md", build_data_dictionary(manifest, events, channels, modality))

    schemas_dir = output_base / "schemas"
    for filename, schema in schemas.items():
        write_json(schemas_dir / filename, schema)

    if not args.no_patch_manifest:
        patch_manifest_documentation(output_base, manifest)

    logger.info("Data dictionary and JSON schemas generated.")


if __name__ == "__main__":
    main()
