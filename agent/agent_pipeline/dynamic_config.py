"""Helpers for creating per-run pipeline configs from a base TOML file."""

from __future__ import annotations

from copy import deepcopy
import os
import uuid
from pathlib import Path
from typing import Any
import re

from agent_pipeline.config import cfg_get, load_config

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # noqa: F401


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in data.items():
        if section.startswith("__"):
            continue
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def safe_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return name.strip("._") or "unknown"


def derive_dataset_name(target_h5: str | Path, base_name: str | None = None) -> str:
    path = Path(target_h5)
    stem = path.stem
    if stem.upper().startswith("NOD_") or "NOD" in stem.upper():
        return "NOD_MEG"
    return base_name or "MEG"


def make_run_slug(target_h5: str | Path) -> str:
    return safe_filename(Path(target_h5).stem)


def build_effective_config(
    base_config_path: str | Path,
    output_root: str | Path,
    target_h5: str | Path | None = None,
    input_root: str | Path | None = None,
    source_adapter: str | None = None,
    config_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base_path = Path(base_config_path).expanduser().resolve()
    config = load_config(base_path)
    config = deepcopy(config)

    target_path = Path(target_h5).expanduser().resolve() if target_h5 else None
    input_path = Path(input_root).expanduser().resolve() if input_root else None
    source_path = target_path or input_path
    if source_path is None:
        raise ValueError("Dynamic config requires target_h5 or input_root.")
    output_base = Path(output_root).expanduser().resolve()
    run_slug = make_run_slug(source_path)
    dataset_name = derive_dataset_name(source_path, cfg_get(config, "dataset.name", "MEG"))

    run_root = output_base / run_slug
    tar_root = run_root / "tar"
    token_root = run_root / "neurotoken"
    report_root = run_root / "reports"
    config.setdefault("dataset", {})["name"] = dataset_name
    config.setdefault("dataset", {})["modality"] = str(cfg_get(config, "dataset.modality", "meg"))
    config.setdefault("dataset", {})["task_type"] = str(cfg_get(config, "dataset.task_type", "MEG"))
    config.setdefault("dataset", {})["description"] = f"Per-run pipeline config for {source_path.name}."
    for section, values in (config_overrides or {}).items():
        if not isinstance(values, dict):
            raise ValueError(f"Config override section {section!r} must be a table.")
        config.setdefault(section, {}).update(values)
    if source_adapter:
        config.setdefault("source", {})["adapter"] = source_adapter

    config.setdefault("paths", {})["h5_root"] = str(target_path.parent if target_path else input_path)
    config.setdefault("paths", {})["tar_root"] = str(tar_root)
    config.setdefault("paths", {})["token_root"] = str(token_root)
    config.setdefault("paths", {})["report_root"] = str(report_root)
    config["selection"] = {
        "mode": "target_h5" if target_path else "input_root",
        "source": str(source_path),
        "target_h5": str(target_path) if target_path else "",
        "input_root": str(input_path) if input_path else "",
    }

    config.setdefault("release", {})["dataset_name"] = dataset_name
    config.setdefault("release", {})["description"] = f"NeuroToken dataset generated from {source_path.name}."

    config.pop("__config_dir__", None)
    config.pop("__config_path__", None)
    return config


def create_dynamic_config(
    base_config_path: str | Path,
    output_root: str | Path,
    target_h5: str | Path | None = None,
    input_root: str | Path | None = None,
    source_adapter: str | None = None,
    config_overrides: dict[str, dict[str, Any]] | None = None,
) -> Path:
    config = build_effective_config(
        base_config_path=base_config_path,
        output_root=output_root,
        target_h5=target_h5,
        input_root=input_root,
        source_adapter=source_adapter,
        config_overrides=config_overrides,
    )
    source_path = Path(target_h5 or input_root or "")
    config_root = Path(output_root).expanduser().resolve() / make_run_slug(source_path) / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    dynamic_path = config_root / "pipeline_config.toml"
    temporary = dynamic_path.with_name(f".{dynamic_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_text(_dump_toml(config), encoding="utf-8")
        os.replace(temporary, dynamic_path)
    finally:
        temporary.unlink(missing_ok=True)
    return dynamic_path