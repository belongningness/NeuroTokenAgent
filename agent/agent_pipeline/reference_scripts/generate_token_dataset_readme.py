#!/usr/bin/env python3
"""NeuroToken README generator.

Produces a Hugging Face Dataset Card-style README.md for a tokenized
neurophysiological dataset folder. Dataset-specific metadata comes from a
TOML configuration file; structural metadata is auto-discovered from the
actual dataset files.

Usage:
    python tools/generate_token_dataset_readme.py \
        --config docs/datasets/nod_eeg_2025.toml \
        --dataset-root . \
        --output README.md

    python tools/generate_token_dataset_readme.py \
        --config docs/datasets/nod_eeg_2025.toml --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# TOML support  (Python 3.11+ has tomllib; 3.10 uses the tomli backport)
# ---------------------------------------------------------------------------
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        sys.exit(
            "This script requires tomli (pip install tomli) on Python < 3.11, "
            "or Python >= 3.11 for the built-in tomllib module."
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENERATOR_NAME = "generate_token_dataset_readme.py"
DATA_DICT = "DATA_DICTIONARY.md"
MANIFEST_FILE = "manifest.json"
SUBJECTS_FILE = "subjects.json"
CODEBOOK_META = "tokenizer/codebook_metadata.json"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a NeuroToken dataset README.md from a TOML config."
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to the dataset TOML configuration file.",
    )
    p.add_argument(
        "--dataset-root",
        default=".",
        help="Root directory of the tokenized dataset (default: .).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output README path (default: <dataset-root>/README.md).",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Validate config and data without writing output.",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum directory depth for the tree (default: 4).",
    )
    p.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directories to exclude from the tree (repeatable).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict[str, Any]:
    """Load and return the TOML config as a nested dict."""
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    errors: list[str] = []
    required_top = ["dataset_name", "short_description"]
    for key in required_top:
        if key not in cfg or not str(cfg[key]).strip():
            errors.append(f"Missing or empty required field: [{key}]")

    src = cfg.get("source_dataset", {})
    if "modality" not in src or not str(src["modality"]).strip():
        errors.append("Missing required field: [source_dataset.modality]")

    doc = cfg.get("documentation", {})
    if "version" not in doc or not str(doc["version"]).strip():
        errors.append("Missing required field: [documentation.version]")

    paths = cfg.get("paths", {})
    if "notebook_path" not in paths or not str(paths["notebook_path"]).strip():
        errors.append("Missing required field: [paths.notebook_path]")

    return errors


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------


def resolve(root: Path, *parts: str) -> Path:
    return (root / Path(*parts)).resolve()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_read_json(path: Path) -> Any:
    """Read JSON if file exists; return None otherwise."""
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def get_representative_session(
    manifest: dict[str, Any], cfg: dict[str, Any], root: Path
) -> tuple[dict[str, Any], Path]:
    """Return the manifest session entry and its directory path."""
    sessions: list[dict[str, Any]] = manifest.get("sessions", [])
    if not sessions:
        raise SystemExit("No sessions found in manifest.json")

    cfg_session: str = (cfg.get("paths", {}) or {}).get(
        "representative_session", ""
    )
    if cfg_session:
        for s in sessions:
            if s.get("session_id") == cfg_session:
                token_file = s.get("token_file", "")
                session_dir = resolve(root, Path(token_file).parent)
                return s, session_dir
        raise SystemExit(
            f"Configured representative_session '{cfg_session}' "
            f"not found in manifest."
        )

    # Default: first session
    s = sessions[0]
    token_file = s.get("token_file", "")
    session_dir = resolve(root, Path(token_file).parent)
    return s, session_dir


# ---------------------------------------------------------------------------
# Directory tree
# ---------------------------------------------------------------------------


def discover_tree(
    root: Path, max_depth: int, exclude: set[str]
) -> str:
    """Return a pretty-printed directory tree string.

    Collapses repeated session directory names into a representative
    placeholder like ``<session_id>``.
    """
    lines: list[str] = []
    real_root = root.resolve()

    def _walk(
        current: Path,
        prefix: str,
        depth: int,
        collapse_session: bool = False,
    ) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                current.iterdir(), key=lambda e: (not e.is_dir(), e.name)
            )
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return
        except OSError:
            return

        # Filter excluded entries
        entries = [e for e in entries if e.name not in exclude]

        # Collapse sessions/ directory: show only the first session,
        # then a placeholder for the rest.
        if current.name == "sessions" and current.parent == real_root:
            dirs = [e for e in entries if e.is_dir()]
            if len(dirs) == 0:
                return
            first = dirs[0]
            rest_count = len(dirs) - 1
            # First session — show real name
            lines.append(f"{prefix}├── {first.name}/")
            _walk(first, prefix + "│   ", depth + 1)
            if rest_count == 1:
                last = dirs[-1]
                lines.append(f"{prefix}└── {last.name}/")
                _walk(last, prefix + "    ", depth + 1)
            elif rest_count > 1:
                lines.append(
                    f"{prefix}└── ... ({rest_count} more session directories)"
                )
            return

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                child_prefix = prefix + ("    " if is_last else "│   ")
                _walk(entry, child_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    lines.append(f"{real_root.name}/")
    _walk(real_root, "", 0)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File reference table
# ---------------------------------------------------------------------------


def build_file_reference(
    root: Path, manifest: dict[str, Any], cfg: dict[str, Any]
) -> list[dict[str, str]]:
    """Return rows for the File and Folder Reference table."""
    modality = cfg.get("source_dataset", {}).get("modality", "<modality>")

    rows: list[dict[str, str]] = []

    # manifest.json
    rows.append(
        {
            "path": "`manifest.json`",
            "type": "JSON",
            "required": "Yes",
            "description": "Dataset-level index listing all sessions.",
            "fields": f"[manifest.json]({DATA_DICT}#manifestjson)",
        }
    )

    # subjects.json
    rows.append(
        {
            "path": "`subjects.json`",
            "type": "JSON",
            "required": "Yes",
            "description": "Subject-level demographic and grouping index.",
            "fields": f"[subjects.json]({DATA_DICT}#subjectsjson)",
        }
    )

    # schemas/
    rows.append(
        {
            "path": "`schemas/`",
            "type": "Directory",
            "required": "Yes",
            "description": "JSON Schema validation files (7 files).",
            "fields": "—",
        }
    )

    # stimuli/
    rows.append(
        {
            "path": "`stimuli/`",
            "type": "Directory",
            "required": "Yes",
            "description": (
                "Stimulus files. Empty by default; map event identifiers "
                "to external stimulus resources."
            ),
            "fields": f"[stimuli/]({DATA_DICT}#stimuli)",
        }
    )

    # tokenizer/
    rows.append(
        {
            "path": "`tokenizer/codebook.npz`",
            "type": "NumPy archive",
            "required": "Yes",
            "description": "Codebook vectors for embedding lookup.",
            "fields": f"[codebook.npz]({DATA_DICT}#tokenizercodebooknpz)",
        }
    )
    rows.append(
        {
            "path": "`tokenizer/codebook_metadata.json`",
            "type": "JSON",
            "required": "Yes",
            "description": "Codebook parameters (vocab size, embedding dims).",
            "fields": f"[codebook_metadata.json]({DATA_DICT}#tokenizercodebook_metadatajson)",
        }
    )

    # sessions/
    rows.append(
        {
            "path": "`sessions/<session_id>/`",
            "type": "Directory",
            "required": "Yes",
            "description": "One directory per recording session.",
            "fields": "—",
        }
    )

    # Token file
    rows.append(
        {
            "path": f"`sessions/<session_id>/{modality}_token_ids.npy`",
            "type": "NumPy array",
            "required": "Yes",
            "description": (
                f"Token ID matrix, shape `[total_tokens, num_codebooks]`, int32. "
                "Each row is one neural token; each column holds a local codebook id."
            ),
            "fields": (
                f"[{modality}_token_ids.npy]"
                f"({DATA_DICT}#sessionssessionmodality_token_idsnpy)"
            ),
        }
    )

    # token_metadata.json
    rows.append(
        {
            "path": "`sessions/<session_id>/token_metadata.json`",
            "type": "JSON",
            "required": "Yes",
            "description": (
                "Token shape, temporal index, spatial layout, and provenance."
            ),
            "fields": (
                f"[token_metadata.json]"
                f"({DATA_DICT}#sessionssessiontoken_metadatajson)"
            ),
        }
    )

    # session_metadata.json
    rows.append(
        {
            "path": "`sessions/<session_id>/session_metadata.json`",
            "type": "JSON",
            "required": "Yes",
            "description": "Recording parameters (sfreq, sample count, subject info).",
            "fields": (
                f"[session_metadata.json]"
                f"({DATA_DICT}#sessionssessionsession_metadatajson)"
            ),
        }
    )

    # events_info.json (conditional)
    has_events = any(
        s.get("has_events") for s in manifest.get("sessions", [])
    )
    rows.append(
        {
            "path": "`sessions/<session_id>/events_info.json`",
            "type": "JSON",
            "required": "Yes" if has_events else "Conditional",
            "description": (
                "Event annotations with absolute timing in seconds."
                if has_events
                else "Event annotations (not present in this dataset)."
            ),
            "fields": (
                f"[events_info.json]"
                f"({DATA_DICT}#sessionssessionevents_infojson)"
            ),
        }
    )

    # channels
    rows.append(
        {
            "path": f"`sessions/<session_id>/{modality}_channels.json`",
            "type": "JSON",
            "required": "Conditional",
            "description": (
                "Per-channel metadata: names, coordinates, normalization statistics."
            ),
            "fields": (
                f"[{modality}_channels.json]"
                f"({DATA_DICT}#sessionssessionmodality_channelsjson)"
            ),
        }
    )

    return rows


def render_file_reference_table(rows: list[dict[str, str]]) -> str:
    """Render the file reference as a Markdown table."""
    header = "| Path or pattern | Type | Required | Description | Detailed fields |"
    sep = "|---|---|---:|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['path']} | {r['type']} | {r['required']} "
            f"| {r['description']} | {r['fields']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anchor generation for DATA_DICTIONARY.md links
# ---------------------------------------------------------------------------


def dd_anchor(heading_text: str) -> str:
    """Convert a DATA_DICTIONARY.md heading to a GitHub-Flavored anchor.

    GFM rules: lowercase, remove punctuation (except hyphens/underscores),
    replace spaces with hyphens, collapse multiple hyphens.
    """
    anchor = heading_text.lower()
    # Keep alphanumeric, hyphens, underscores, spaces
    anchor = "".join(
        ch for ch in anchor if ch.isalnum() or ch in (" ", "-", "_")
    )
    anchor = anchor.strip().replace(" ", "-")
    # Collapse consecutive hyphens
    while "--" in anchor:
        anchor = anchor.replace("--", "-")
    return anchor


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Minimal Markdown escaping for inline text."""
    return text.replace("|", "\\|")


def render_dataset_description(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    codebook_meta: dict[str, Any],
    rep_session: dict[str, Any],
    rep_session_meta: dict[str, Any],
) -> str:
    name = cfg.get("dataset_name", "")
    short = cfg.get("short_description", "")
    src = cfg.get("source_dataset", {})
    doc = cfg.get("documentation", {})
    scope = cfg.get("scope", {})
    prep = (cfg.get("preprocessing_summary") or "").strip()
    trepr = (cfg.get("token_representation") or "").strip()
    intended = (doc.get("intended_use") or "").strip()
    lic = doc.get("license", "")

    modality = src.get("modality", manifest.get("modality", ""))
    task = src.get("task", "")
    source_name = src.get("name", "")
    source_url = src.get("url", "")
    source_citation = src.get("citation", "")
    source_desc = src.get("description", "")

    num_sessions = scope.get("num_sessions") or manifest.get("num_sessions", 0)
    num_subjects = scope.get("num_subjects", 0)
    if not num_subjects:
        subj_path = resolve(
            Path(cfg.get("_dataset_root", ".")), SUBJECTS_FILE
        )
        subj_data = safe_read_json(subj_path)
        if isinstance(subj_data, list):
            num_subjects = len(subj_data)

    tokenizer_ckpt = cfg.get("tokenizer", {}).get("checkpoint", "")
    if not tokenizer_ckpt:
        tokenizer_ckpt = codebook_meta.get("checkpoint", "")

    # Build property table
    rows = [
        ("Dataset name", name),
        ("Source dataset", source_name or "—"),
        ("Modality", modality),
        ("Task", task or "—"),
        ("Tokenizer checkpoint", tokenizer_ckpt or "—"),
        ("Token representation", trepr or "—"),
        ("Number of sessions", str(num_sessions)),
        ("Number of subjects", str(num_subjects)),
        ("Documentation version", doc.get("version", "")),
    ]
    if lic:
        rows.append(("License", lic))

    max_key = max(len(k) for k, _ in rows)
    table = "\n".join(
        f"| {k:<{max_key}} | {_esc(v)} |" for k, v in rows
    )
    header = f"| {'Property':<{max_key}} | Value |"
    sep = f"|{'-' * (max_key + 2)}|-------|"

    parts: list[str] = []
    parts.append(f"# {name}\n")
    parts.append(f"{short}\n")
    parts.append("## Dataset Description\n")
    parts.append(header)
    parts.append(sep)
    parts.append(table)

    if source_desc:
        parts.append(f"\n### Source Dataset\n\n{source_desc}")
    if source_citation:
        parts.append(f"\n**Citation:** {source_citation}")
    if source_url:
        parts.append(f"\n**URL:** {source_url}")
    if prep:
        parts.append(f"\n### Preprocessing\n\n{prep}")
    if intended:
        parts.append(f"\n### Intended Use\n\n{intended}")

    return "\n".join(parts) + "\n"


def render_quick_start(
    cfg: dict[str, Any], manifest: dict[str, Any]
) -> str:
    modality = cfg.get("source_dataset", {}).get(
        "modality", manifest.get("modality", "<modality>")
    )
    notebook_path = cfg.get("paths", {}).get("notebook_path", "")

    code = textwrap.dedent(f"""\
    ```python
    import json
    import numpy as np
    from pathlib import Path

    ROOT = Path("<PATH_TO_TOKENIZED_DATASET>")

    # Load the dataset manifest
    manifest = json.load(open(ROOT / "manifest.json"))
    print(f"Format: {{manifest['format']}}, Sessions: {{manifest['num_sessions']}}")

    # Load the first session's tokens
    s = manifest["sessions"][0]
    tokens = np.load(ROOT / s["token_file"])
    print(f"Token shape: {{tokens.shape}}, dtype: {{tokens.dtype}}")
    print(f"Token value range: [{{tokens.min()}}, {{tokens.max()}}]")
    ```
    """)

    parts = [
        "## Quick Start\n",
        code,
    ]
    if notebook_path:
        parts.append(
            f"For a complete walkthrough, see the "
            f"[Usage Notebook]({notebook_path}).\n"
        )
    return "\n".join(parts)


def render_directory_tree(
    root: Path, max_depth: int, exclude: set[str]
) -> str:
    tree = discover_tree(root, max_depth, exclude)
    return f"## Directory Tree\n\n```text\n{tree}\n```\n"


def render_file_reference_section(
    root: Path, manifest: dict[str, Any], cfg: dict[str, Any]
) -> str:
    rows = build_file_reference(root, manifest, cfg)
    table = render_file_reference_table(rows)
    return (
        "## File and Folder Reference\n\n"
        "For detailed field definitions, see "
        f"[{DATA_DICT}]({DATA_DICT}).\n\n"
        f"{table}\n"
    )


def render_token_data_overview(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    rep_session: dict[str, Any],
    token_meta: dict[str, Any],
    codebook_meta: dict[str, Any],
) -> str:
    modality = cfg.get("source_dataset", {}).get(
        "modality", manifest.get("modality", "<modality>")
    )
    tokens = token_meta.get("tokens", {})
    shape = tokens.get("shape", ["?", "?"])
    dtype = tokens.get("dtype", "int32")
    num_cb = token_meta.get("tokens", {}).get(
        "num_codebooks", codebook_meta.get("num_codebooks", "?")
    )
    vocab_per_cb = codebook_meta.get("vocab_size_per_codebook", "?")
    vocab_width = codebook_meta.get("vocab_width", "?")
    width_per_cb = codebook_meta.get("width_per_codebook", "?")
    ckpt = codebook_meta.get("checkpoint", "")

    # Token rate from temporal_index
    ti = token_meta.get("temporal_index", {})
    sfreq = ti.get("sampling_rate_hz", None)
    stride = ti.get("time_stride_samples", None)
    token_rate_line = ""
    if sfreq and stride:
        token_rate = sfreq / stride
        ms_per_token = (stride / sfreq) * 1000
        token_rate_line = (
            f"One neural token represents {stride} source samples "
            f"({ms_per_token:.0f} ms at {sfreq} Hz). "
            f"Token rate: {token_rate:.1f} tokens/s.\n"
        )

    parts = [
        "## Token Data\n",
        f"Each session contains a `{modality}_token_ids.npy` file.\n",
        f"| Property | Value |",
        f"|---|---|",
        f"| Shape | `{shape}` — `[total_tokens, num_codebooks]` |",
        f"| Dtype | `{dtype}` |",
        f"| Number of codebooks | {num_cb} |",
        f"| Valid id range per codebook | `0..{vocab_per_cb - 1}` |",
        f"| Embedding dimension | {vocab_width} ({num_cb} × {width_per_cb}) |",
        f"| Codebook shape | `({num_cb}, {vocab_per_cb}, {width_per_cb})` |",
    ]
    if ckpt:
        parts.append(f"| Tokenizer checkpoint | `{ckpt}` |")
    parts.append("")
    if token_rate_line:
        parts.append(token_rate_line)
    parts.append(
        f"\nSee [{DATA_DICT}]({DATA_DICT}#sessionssessionmodality_token_idsnpy) "
        f"for the complete token array specification.\n"
    )
    parts.append(f"\n### Embedding Lookup\n")
    parts.append(textwrap.dedent(f"""\
    ```python
    import numpy as np
    cb = np.load("tokenizer/codebook.npz")["codebook_normalized"]
    # cb.shape == ({num_cb}, {vocab_per_cb}, {width_per_cb})

    token_ids = np.load("<session>/{modality}_token_ids.npy")
    # token_ids.shape == (N, {num_cb})

    # Look up embeddings column-by-column
    pieces = [cb[k, token_ids[:, k]] for k in range({num_cb})]
    embedding = np.concatenate(pieces, axis=-1)
    # embedding.shape == (N, {vocab_width})
    ```
    """))
    return "\n".join(parts)


def render_time_events(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    token_meta: dict[str, Any],
    rep_session: dict[str, Any],
) -> str:
    ti = token_meta.get("temporal_index", {})
    sfreq = ti.get("sampling_rate_hz", "?")
    stride = ti.get("time_stride_samples", "?")
    num_segments = ti.get("num_segments", "?")
    segments = ti.get("segments", [])

    seg_size = "?"
    if segments:
        seg_size = segments[0].get("token", {}).get("count", "?")
        grid = segments[0].get("token", {}).get("grid_shape", "?")

    parts = [
        "## Time and Event Alignment\n",
        "### Temporal Index\n",
        f"The `temporal_index` in `token_metadata.json` maps token ranges "
        f"to source time windows.\n",
        f"| Property | Value |",
        f"|---|---|",
        f"| Source sampling rate | {sfreq} Hz |",
        f"| Samples per token stride | {stride} |",
        f"| Number of segments | {num_segments} |",
        f"| Tokens per segment | {seg_size} |",
    ]
    if segments:
        grid = segments[0].get("token", {}).get("grid_shape", [])
        if grid:
            parts.append(f"| Segment grid shape | `{grid}` — `[channel_group, time_token]` |")
        seg_dur = segments[0].get("source_time", {}).get("end_seconds", 0)
        if seg_size != "?":
            parts.append(f"| Segment duration | {seg_dur} s ({seg_size} tokens) |")

    has_events = rep_session.get("has_events", False)
    parts.append("")
    parts.append(
        f"Each segment covers a fixed time window. "
        f"Use `temporal_index.segments[]` to map between token indices "
        f"and source time in seconds.\n"
    )

    if has_events:
        parts.append(
            f"\n### Events\n\n"
            f"Event annotations are available in `events_info.json` "
            f"for this dataset. See "
            f"[events_info.json]({DATA_DICT}#sessionssessionevents_infojson) "
            f"for field definitions.\n"
        )
    else:
        parts.append(
            f"\n### Events\n\n"
            f"No event annotations are present in this dataset.\n"
        )

    parts.append(
        f"\nFor detailed code examples (time-to-token mapping, event "
        f"alignment), see the [Usage Notebook]"
        f"({cfg.get('paths', {}).get('notebook_path', '')}).\n"
    )

    return "\n".join(parts)


def render_channel_info(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    token_meta: dict[str, Any],
    channels_data: list[dict[str, Any]] | None,
) -> str:
    modality = cfg.get("source_dataset", {}).get(
        "modality", manifest.get("modality", "<modality>")
    )
    spatial = token_meta.get("spatial_layout", {})
    n_channels = spatial.get("channel_count", "?")
    channel_stride = spatial.get("channel_stride", "?")
    groups = "?"
    if isinstance(n_channels, int) and isinstance(channel_stride, int) and channel_stride > 0:
        groups = (n_channels + channel_stride - 1) // channel_stride

    parts = [
        "## Channel Information\n",
        f"| Property | Value |",
        f"|---|---|",
        f"| Total channels | {n_channels} |",
        f"| Channels per group | {channel_stride} |",
        f"| Channel groups | {groups} |",
    ]

    if channels_data:
        bad_count = sum(
            1 for ch in channels_data if ch.get("is_bad", 0)
        )
        parts.append(f"| Bad channels | {bad_count} |")
        # Sample channel names
        names = [ch.get("channel_name", "?") for ch in channels_data[:10]]
        parts.append(
            f"| Example channels | {', '.join(names)}, … |"
        )

    parts.append("")
    parts.append(
        f"The `{modality}_channels.json` file records per-channel "
        f"metadata (name, type, coordinates, normalization statistics). "
        f"Token channels are organized into groups of {channel_stride} "
        f"consecutive channels; the final group is zero-padded if the "
        f"channel count is not an exact multiple.\n"
    )
    parts.append(
        f"\nSee [{modality}_channels.json]"
        f"({DATA_DICT}#sessionssessionmodality_channelsjson) "
        f"for the complete field specification.\n"
    )

    return "\n".join(parts)


def render_usage(
    cfg: dict[str, Any], manifest: dict[str, Any]
) -> str:
    notebook = cfg.get("paths", {}).get("notebook_path", "")
    modality = cfg.get("source_dataset", {}).get(
        "modality", manifest.get("modality", "<modality>")
    )

    parts = [
        "## Usage\n",
        "### Basic Data Loading\n",
        textwrap.dedent(f"""\
        ```python
        import json
        import numpy as np
        from pathlib import Path

        ROOT = Path("<PATH_TO_TOKENIZED_DATASET>")
        manifest = json.load(open(ROOT / "manifest.json"))

        for s in manifest["sessions"]:
            tokens = np.load(ROOT / s["token_file"])
            print(f"{{s['session_id']}}: {{tokens.shape}}")
        ```
        """),
        "### Full Workflow\n",
        f"The [Usage Notebook]({notebook}) provides a complete walkthrough:\n",
        f"- Inspecting the dataset directory",
        f"- Loading token arrays and metadata",
        f"- Mapping token indices to time via `temporal_index`",
        f"- Reading channel metadata and spatial layout",
        f"- Loading and aligning event annotations",
        f"- Selecting time ranges and channel groups",
        f"- Computing token embeddings via the codebook",
        f"- Preparing data for downstream models\n",
        f"Open the notebook, replace the placeholders with your paths, "
        f"and run all cells in order.\n",
    ]
    return "\n".join(parts)


def render_tokenizer_info(
    codebook_meta: dict[str, Any], cfg: dict[str, Any]
) -> str:
    ckpt = codebook_meta.get("checkpoint", "?")
    num_cb = codebook_meta.get("num_codebooks", "?")
    vocab = codebook_meta.get("vocab_size", "?")
    vocab_per_cb = codebook_meta.get("vocab_size_per_codebook", "?")
    vw = codebook_meta.get("vocab_width", "?")
    wpc = codebook_meta.get("width_per_codebook", "?")
    tscheme = codebook_meta.get("tokenization_scheme", "")
    fmt = codebook_meta.get("format", "")

    parts = [
        "## Tokenizer\n",
        f"| Property | Value |",
        f"|---|---|",
        f"| Checkpoint | `{ckpt}` |",
        f"| Format | {fmt} |",
        f"| Number of codebooks | {num_cb} |",
        f"| Total vocabulary size | {vocab} |",
        f"| Vocabulary per codebook | {vocab_per_cb} |",
        f"| Embedding width (total) | {vw} |",
        f"| Embedding width per codebook | {wpc} |",
    ]
    if tscheme:
        parts.append(f"| Tokenization scheme | {tscheme} |")
    parts.append("")
    parts.append(
        f"See [codebook_metadata.json]"
        f"({DATA_DICT}#tokenizercodebook_metadatajson) "
        f"for details.\n"
    )
    return "\n".join(parts)


def render_limitations_and_notes(cfg: dict[str, Any]) -> str:
    parts: list[str] = []
    limitations = cfg.get("limitations", "").strip()
    notes = cfg.get("notes", "").strip()

    if limitations:
        parts.append("## Limitations\n")
        parts.append(limitations + "\n")
    if notes:
        parts.append("## Notes\n")
        parts.append(notes + "\n")
    return "\n".join(parts)


def render_footer(
    cfg_path: str, generated_at: str, warnings: list[str]
) -> str:
    parts = [
        "---\n",
        f"*README generated by `{GENERATOR_NAME}` on {generated_at}.*  \n",
        f"*Configuration: `{cfg_path}`.*  \n",
    ]
    if warnings:
        parts.append(
            "\n**Warnings during generation:**\n"
            + "\n".join(f"- {w}" for w in warnings)
            + "\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def generate(
    cfg: dict[str, Any],
    dataset_root: Path,
    max_depth: int,
    exclude_dirs: set[str],
    cfg_path: str,
    check_only: bool = False,
) -> tuple[str, list[str]]:
    """Return (markdown_string, warnings_list)."""
    warnings: list[str] = []

    # Inject dataset root for path resolution
    cfg["_dataset_root"] = str(dataset_root)

    # Load manifest
    manifest_path = resolve(dataset_root, MANIFEST_FILE)
    if not manifest_path.exists():
        raise SystemExit(f"Required file not found: {manifest_path}")
    manifest = read_json(manifest_path)

    # Load codebook metadata
    cb_meta_path = resolve(dataset_root, CODEBOOK_META)
    codebook_meta = {}
    if cb_meta_path.exists():
        codebook_meta = read_json(cb_meta_path)
    else:
        warnings.append(f"Codebook metadata not found: {cb_meta_path}")

    # Representative session
    rep_session, rep_dir = get_representative_session(
        manifest, cfg, dataset_root
    )

    # Load representative session metadata
    token_meta_path = resolve(
        dataset_root, rep_session.get("token_metadata", "")
    )
    token_meta = read_json(token_meta_path) if token_meta_path.exists() else {}
    if not token_meta:
        warnings.append(
            f"Could not load token_metadata for {rep_session.get('session_id')}"
        )

    session_meta_path = rep_dir / "session_metadata.json"
    session_meta = safe_read_json(session_meta_path) or {}

    channels_path = rep_dir / f"{cfg.get('source_dataset',{}).get('modality','eeg')}_channels.json"
    channels_data = safe_read_json(channels_path)

    # Build sections
    sections: list[str] = []

    sections.append(
        render_dataset_description(
            cfg, manifest, codebook_meta, rep_session, session_meta
        )
    )
    sections.append(render_quick_start(cfg, manifest))
    sections.append(
        render_directory_tree(dataset_root, max_depth, exclude_dirs)
    )
    sections.append(
        render_file_reference_section(dataset_root, manifest, cfg)
    )
    sections.append(
        render_token_data_overview(
            cfg, manifest, rep_session, token_meta, codebook_meta
        )
    )
    sections.append(
        render_time_events(cfg, manifest, token_meta, rep_session)
    )
    sections.append(
        render_channel_info(cfg, manifest, token_meta, channels_data)
    )
    sections.append(render_usage(cfg, manifest))
    sections.append(render_tokenizer_info(codebook_meta, cfg))
    sections.append(render_limitations_and_notes(cfg))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    # Use path relative to dataset root for the footer
    try:
        rel_cfg = str(Path(cfg_path).resolve().relative_to(dataset_root))
    except ValueError:
        rel_cfg = cfg_path
    sections.append(render_footer(rel_cfg, generated_at, warnings))

    return "\n".join(sections), warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Resolve paths
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        print(
            f"Error: dataset root is not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output).resolve()
        if args.output
        else dataset_root / "README.md"
    )

    exclude_dirs = set(args.exclude_dir) | {
        "__pycache__",
        ".git",
        ".ipynb_checkpoints",
        "tools",
        "docs",
    }

    # Load config
    try:
        cfg = load_config(str(config_path))
    except Exception as exc:
        print(f"Error reading config: {exc}", file=sys.stderr)
        return 1

    # Validate
    errors = validate_config(cfg)
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    # Generate
    try:
        markdown, warnings = generate(
            cfg,
            dataset_root,
            args.max_depth,
            exclude_dirs,
            str(config_path),
            check_only=args.check,
        )
    except SystemExit as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if warnings:
        print("Warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    if args.check:
        print("Check passed: all required files found, config valid.")
        print(f"Would write to: {output_path}")
        print(f"Output size: {len(markdown)} bytes")
        return 0

    # Write output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"README written to: {output_path}")
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
