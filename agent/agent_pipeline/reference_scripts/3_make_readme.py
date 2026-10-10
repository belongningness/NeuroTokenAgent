#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3: Generate README.md for neurotoken_dataset.

Produces a concise Dataset Card for a tokenized neurophysiological dataset
folder.  Dataset-specific metadata comes from the pipeline TOML config;
structural metadata (directory tree, session count, …) is auto-discovered.

Pipeline integration:
    python 3_make_readme.py --config <pipeline.toml>           (strict)
    python 3_make_readme.py --config <pipeline.toml> --allow-placeholder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurotoken_config import cfg_get, cfg_path, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GENERATOR_NAME = "3_make_readme.py"
_DATA_DICT = "DATA_DICTIONARY.md"
_MANIFEST_FILE = "manifest.json"
_SUBJECTS_FILE = "subjects.json"

# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------
_PLACEHOLDER_MARKERS = ("TODO", "TODO_DATASET_NAME")


def _looks_like_placeholder(text: str) -> bool:
    return any(marker in text for marker in _PLACEHOLDER_MARKERS)


# ===================================================================
# File-system helpers
# ===================================================================


def _resolve(root: Path, *parts: str) -> Path:
    return (root / Path(*parts)).resolve()


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_read_json(path: Path) -> Any:
    try:
        return _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ===================================================================
# Directory tree
# ===================================================================


def _discover_tree(root: Path, max_depth: int, exclude: set[str]) -> str:
    """Pretty-printed directory tree. Collapses session dirs to representative."""
    lines: list[str] = []
    real_root = root.resolve()

    def _walk(current: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return
        except OSError:
            return

        entries = [e for e in entries if e.name not in exclude]

        if current.name == "sessions" and current.parent == real_root:
            dirs = [e for e in entries if e.is_dir()]
            if len(dirs) == 0:
                return
            first = dirs[0]
            rest_count = len(dirs) - 1
            lines.append(f"{prefix}├── {first.name}/")
            _walk(first, prefix + "│   ", depth + 1)
            if rest_count == 1:
                last = dirs[-1]
                lines.append(f"{prefix}└── {last.name}/")
                _walk(last, prefix + "    ", depth + 1)
            elif rest_count > 1:
                lines.append(f"{prefix}└── ... ({rest_count} more session directories)")
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


# ===================================================================
# File reference table  (Path + Description only)
# ===================================================================


def _build_file_reference(
    manifest: dict[str, Any], modality: str
) -> list[dict[str, str]]:
    """Return (path, description) rows for the File and Folder Reference table."""
    has_events = any(s.get("has_events") for s in manifest.get("sessions", []))
    return [
        ("`manifest.json`",
         "Dataset-level index listing all sessions."),
        ("`subjects.json`",
         "Subject-level demographic and grouping index."),
        ("`schemas/`",
         "JSON Schema validation files (7 files)."),
        ("`stimuli/`",
         "Stimulus files."),
        ("`tokenizer/codebook.npz`",
         "Codebook vectors for embedding lookup."),
        ("`tokenizer/codebook_metadata.json`",
         "Codebook parameters (vocab size, embedding dims)."),
        ("`sessions/<session_id>/`",
         "One directory per recording session."),
        (f"`sessions/<session_id>/{modality}_token_ids.npy`",
         "Token ID matrix, shape `[total_tokens, num_codebooks]`, int32. "
         "Each row is one neural token; each column holds a local codebook id."),
        ("`sessions/<session_id>/token_metadata.json`",
         "Token shape, temporal index, spatial layout, and provenance."),
        ("`sessions/<session_id>/session_metadata.json`",
         "Recording parameters (sfreq, sample count, subject info)."),
        ("`sessions/<session_id>/events_info.json`",
         "Event annotations with absolute timing in seconds."
         if has_events else
         "Event annotations (not present in this dataset)."),
        (f"`sessions/<session_id>/{modality}_channels.json`",
         "Per-channel metadata: names, coordinates, normalization statistics."),
    ]


def _render_file_reference_table(rows: list[dict[str, str]]) -> str:
    header = "| Path or pattern | Description |"
    sep = "|---|---|"
    return "\n".join([header, sep] +
                     [f"| {path} | {desc} |" for path, desc in rows])


# ===================================================================
# Section renderers
# ===================================================================


def _esc(text: str) -> str:
    return text.replace("|", "\\|")


def _render_dataset_description(cfg: dict[str, Any], manifest: dict[str, Any]) -> str:
    name = cfg.get("dataset_name", "")
    short = cfg.get("short_description", "")
    src = cfg.get("source_dataset", {})
    doc = cfg.get("documentation", {})
    scope = cfg.get("scope", {})
    prep = (cfg.get("preprocessing_summary") or "").strip()

    modality = src.get("modality", manifest.get("modality", ""))
    task = src.get("task", "")
    source_name = src.get("name", "")
    source_url = src.get("url", "")
    source_citation = src.get("citation", "")
    source_desc = src.get("description", "")

    num_sessions = scope.get("num_sessions") or manifest.get("num_sessions", 0)
    num_subjects = scope.get("num_subjects", 0)
    if not num_subjects:
        subj_path = _resolve(Path(cfg.get("_dataset_root", ".")), _SUBJECTS_FILE)
        subj_data = _safe_read_json(subj_path)
        if isinstance(subj_data, list):
            num_subjects = len(subj_data)

    rows = [
        ("Source dataset", source_name or "—"),
        ("Modality", modality),
        ("Task", task or "—"),
        ("Number of sessions", str(num_sessions)),
        ("Number of subjects", str(num_subjects)),
        ("Documentation version", doc.get("version", "")),
    ]

    max_key = max(len(k) for k, _ in rows)
    header = f"| {'Property':<{max_key}} | Value |"
    sep = f"|{'-' * (max_key + 2)}|-------|"
    table = "\n".join(f"| {k:<{max_key}} | {_esc(v)} |" for k, v in rows)

    parts: list[str] = [f"# {name}\n", f"{short}\n", "## Dataset Description\n", header, sep, table]
    if source_desc:
        parts.append(f"\n### Source Dataset\n\n{source_desc}")
    if source_citation:
        parts.append(f"\n**Citation:** {source_citation}")
    if source_url:
        parts.append(f"\n**URL:** {source_url}")
    if prep:
        parts.append(f"\n### Preprocessing\n\n{prep}")
    return "\n".join(parts) + "\n"


def _release_safe_link(value: str) -> str:
    value = str(value).strip()
    if value.startswith(("https://", "http://")):
        return value
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        return ""
    return path.as_posix()


def _render_quick_start(cfg: dict[str, Any]) -> str:
    notebook_path = _release_safe_link(cfg.get("paths", {}).get("notebook_path", ""))
    lines = ["## Quick Start\n"]
    if notebook_path:
        lines.append(
            f"For a complete walkthrough, see the [Usage Notebook]({notebook_path}).\n"
        )
    else:
        lines.append("See the usage notebook for a complete walkthrough.\n")
    return "\n".join(lines)


def _render_directory_tree(root: Path, max_depth: int, exclude: set[str]) -> str:
    tree = _discover_tree(root, max_depth, exclude)
    return f"## Directory Tree\n\n```text\n{tree}\n```\n"


def _render_file_reference_section(manifest: dict[str, Any], modality: str) -> str:
    rows = _build_file_reference(manifest, modality)
    table = _render_file_reference_table(rows)
    return (
        f"## File and Folder Reference\n\n"
        f"For detailed field definitions, see [{_DATA_DICT}]({_DATA_DICT}).\n\n"
        f"{table}\n"
    )


def _render_notes(cfg: dict[str, Any]) -> str:
    notes = cfg.get("notes", "").strip()
    if notes:
        return "## Notes\n\n" + notes + "\n"
    return ""


def _render_footer(cfg_path: str, generated_at: str, warnings: list[str]) -> str:
    parts = [
        "---\n",
        f"*README generated by `{_GENERATOR_NAME}` on {generated_at}.*  \n",
        f"*Configuration: `{cfg_path}`.*  \n",
    ]
    if warnings:
        parts.append("\n**Warnings during generation:**\n"
                     + "\n".join(f"- {w}" for w in warnings) + "\n")
    return "\n".join(parts)


# ===================================================================
# Main generation logic
# ===================================================================


def _generate(
    cfg: dict[str, Any],
    dataset_root: Path,
    max_depth: int,
    exclude_dirs: set[str],
    cfg_path: str,
) -> tuple[str, list[str]]:
    """Return (markdown_string, warnings_list)."""
    warnings: list[str] = []

    cfg["_dataset_root"] = str(dataset_root)

    # Load manifest
    manifest_path = _resolve(dataset_root, _MANIFEST_FILE)
    if not manifest_path.exists():
        raise SystemExit(f"Required file not found: {manifest_path}")
    manifest = _read_json(manifest_path)

    modality = cfg.get("source_dataset", {}).get("modality", manifest.get("modality", "eeg"))

    # Build sections
    sections: list[str] = [
        _render_dataset_description(cfg, manifest),
        _render_quick_start(cfg),
        _render_directory_tree(dataset_root, max_depth, exclude_dirs),
        _render_file_reference_section(manifest, modality),
        _render_notes(cfg),
    ]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    config_name = Path(cfg_path).name if cfg_path else "pipeline_config.toml"
    sections.append(_render_footer(config_name, generated_at, warnings))

    return "\n".join(sections), warnings


# ===================================================================
# Config mapping: pipeline TOML → _generate() config dict
# ===================================================================


def build_readme_config(pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """Map the merged pipeline TOML into the nested dict expected by ``_generate()``.

    Key distinction:
    - ``[dataset]``  — the **original/source** dataset (raw neural recordings).
    - ``[release]``  — the **tokenized** derivative (this README's subject).
    - ``[documentation]`` — version and other card metadata.
    """
    short_desc = str(cfg_get(pipeline_config, "release.description", ""))
    prep = str(cfg_get(pipeline_config, "release.preprocessing_summary", ""))
    trepr = str(cfg_get(pipeline_config, "release.token_representation", ""))

    source_dataset = {
        "name": str(cfg_get(pipeline_config, "dataset.name", "")),
        "url": str(cfg_get(pipeline_config, "documentation.source_url", "")),
        "citation": str(cfg_get(pipeline_config, "documentation.source_citation", "")),
        "modality": str(cfg_get(pipeline_config, "dataset.modality", "")),
        "task": str(cfg_get(pipeline_config, "dataset.task_type", "")),
        "description": str(cfg_get(pipeline_config, "dataset.description", "")),
    }

    tokenizer = {"checkpoint": str(cfg_get(pipeline_config, "release.tokenizer_version", ""))}

    scope: dict[str, Any] = {"num_sessions": 0, "num_subjects": 0}
    if "scope" in pipeline_config and isinstance(pipeline_config["scope"], dict):
        scope.update(pipeline_config["scope"])

    paths = {
        "representative_session": str(cfg_get(pipeline_config, "paths.representative_session", "")),
        "notebook_path": str(cfg_get(pipeline_config, "paths.notebook_path", "")),
    }

    doc_section = pipeline_config.get("documentation", {})
    if not isinstance(doc_section, dict):
        doc_section = {}
    documentation = {
        "version": str(doc_section.get("version", "")),
    }

    # Notes from TOML (if present)
    notes = str(cfg_get(pipeline_config, "release.notes", ""))

    return {
        "dataset_name": str(cfg_get(pipeline_config, "release.dataset_name", "")),
        "short_description": short_desc,
        "preprocessing_summary": prep,
        "token_representation": trepr,
        "source_dataset": source_dataset,
        "tokenizer": tokenizer,
        "scope": scope,
        "paths": paths,
        "documentation": documentation,
        "notes": notes,
    }


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Generate README.md for neurotoken_dataset.")
    parser.add_argument("--config", type=Path, default=None, help="TOML pipeline config file (merged format).")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="Allow TODO placeholders instead of erroring (for smoke tests).")
    return parser.parse_args()


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    args = parse_args()
    pipeline_config = load_config(args.config)

    output_base = cfg_path(pipeline_config, "paths.token_root")
    if output_base is None:
        logger.error("paths.token_root is not configured – cannot determine dataset root.")
        sys.exit(1)
    if not output_base.exists():
        logger.error("Output directory does not exist: %s (run Stage 2 tokenize first).", output_base)
        sys.exit(1)

    readme_cfg = build_readme_config(pipeline_config)

    # Placeholder guard
    key_fields = [
        ("dataset_name", readme_cfg.get("dataset_name", "")),
        ("short_description", readme_cfg.get("short_description", "")),
        ("tokenizer checkpoint", readme_cfg.get("tokenizer", {}).get("checkpoint", "")),
    ]
    unfilled = [name for name, value in key_fields if _looks_like_placeholder(str(value))]

    if unfilled and not args.allow_placeholder:
        logger.error("The following fields are still placeholders – please fill them in the TOML config "
                     "before generating README: %s (or use --allow-placeholder to force generation).",
                     ", ".join(unfilled))
        sys.exit(1)
    if unfilled:
        logger.warning("The following fields are still placeholders (generating anyway "
                       "because --allow-placeholder is set): %s", ", ".join(unfilled))

    exclude_dirs: set[str] = {"__pycache__", ".git", ".ipynb_checkpoints"}
    cfg_path_str = str(Path(args.config).resolve()) if args.config else ""

    try:
        markdown, warnings = _generate(readme_cfg, output_base, max_depth=4,
                                       exclude_dirs=exclude_dirs, cfg_path=cfg_path_str)
    except SystemExit as exc:
        logger.error("README generation failed: %s", exc)
        sys.exit(1)

    for w in warnings:
        logger.warning("%s", w)

    readme_path = output_base / "README.md"
    readme_path.write_text(markdown, encoding="utf-8")
    logger.info("README written to: %s", readme_path)

    dataset_name = readme_cfg.get("dataset_name", "")
    modality = readme_cfg.get("source_dataset", {}).get("modality", "")
    logger.info("Dataset: %s | modality: %s", dataset_name, modality)


if __name__ == "__main__":
    main()
