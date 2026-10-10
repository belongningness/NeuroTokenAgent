"""Filesystem and ownership policy for Agent Server requests."""

from __future__ import annotations

import os
from pathlib import Path

from agent_pipeline.adapters import build_adapter
from agent_pipeline.dynamic_config import build_effective_config
from agent_pipeline.pipeline_models import PipelineRequest


ROOT = Path(__file__).resolve().parents[1]


def configured_roots(name: str, defaults: list[Path]) -> list[Path]:
    raw = os.environ.get(name, "")
    roots = [Path(item).expanduser().resolve() for item in raw.split(os.pathsep) if item.strip()]
    return roots or [path.resolve() for path in defaults]


def is_within(path: str | Path, roots: list[Path]) -> bool:
    resolved = Path(path).expanduser().resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def select_linked_validation_target(request: PipelineRequest) -> PipelineRequest:
    if request.execution_strategy != "linked":
        return request
    full_root = Path(request.full_input_root or "").expanduser().resolve()
    if not full_root.is_dir():
        raise ValueError(f"full_input_root must be an existing directory: {full_root}")
    config = build_effective_config(
        base_config_path=request.config_path,
        output_root=request.report_root or Path.cwd(),
        input_root=full_root,
        source_adapter=request.source_adapter,
        config_overrides=request.config_overrides,
    )
    adapter = build_adapter(config)
    candidates = [Path(path).expanduser().resolve() for path in adapter.discover(full_root)]
    candidates = sorted(
        (path for path in candidates if path.is_file() and full_root in path.parents),
        key=lambda path: path.relative_to(full_root).as_posix(),
    )
    adapter_name = request.source_adapter or config.get("source", {}).get("adapter") or "configured adapter"
    if request.validation_target_h5:
        target = Path(request.validation_target_h5).expanduser().resolve()
        if target not in candidates:
            raise ValueError(f"validation_target_h5 was not discovered by {adapter_name}: {target}")
    else:
        if not candidates:
            raise ValueError(f"{adapter_name} did not discover a validation H5 under full_input_root: {full_root}")
        target = candidates[0]
    return request.with_updates(validation_target_h5=str(target))


def validate_pipeline_paths(request: PipelineRequest) -> None:
    input_roots = configured_roots(
        "NEUROTOKEN_ALLOWED_INPUT_ROOTS",
        [ROOT / "data"],
    )
    output_roots = configured_roots("NEUROTOKEN_ALLOWED_OUTPUT_ROOTS", [ROOT / "outputs"])
    sources = [request.target_h5 or request.input_root]
    if request.execution_strategy == "linked":
        sources = [request.validation_target_h5, request.full_input_root]
    for source in sources:
        if source is None or not is_within(source, input_roots):
            raise PermissionError(f"Input path is outside allowed roots: {source}")
    if request.execution_strategy == "linked":
        target = Path(request.validation_target_h5 or "").expanduser().resolve()
        full_root = Path(request.full_input_root or "").expanduser().resolve()
        if not target.is_file() or target.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"validation_target_h5 must be an existing H5 file: {target}")
        if not full_root.is_dir():
            raise ValueError(f"full_input_root must be an existing directory: {full_root}")
        if full_root not in target.parents:
            raise ValueError("validation_target_h5 must be within full_input_root")
    if request.report_root is None or not is_within(request.report_root, output_roots):
        raise PermissionError(f"Output path is outside allowed roots: {request.report_root}")


def validate_adapter_source(path: str | Path) -> None:
    roots = configured_roots(
        "NEUROTOKEN_ALLOWED_INPUT_ROOTS",
        [ROOT / "data"],
    )
    if not is_within(path, roots):
        raise PermissionError(f"Adapter source is outside allowed roots: {path}")
