"""Persistent Stage 2 model-worker functions used by the multi-GPU scheduler.

Each spawned process initializes one CUDA device, one UniTok model replica and
one modality transform. Dataset setup/finalization and whole-session conversion
reuse the canonical Stage 2 implementation; scheduling lives in
``tokenize_sessions_multi_gpu.py``.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

PARALLEL_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PARALLEL_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from neurotoken_config import cfg_bool, cfg_get, cfg_path, load_config


STAGE2_SCRIPTS = {
    "cuda": SCRIPT_DIR / "2_neural2token_human_new.py",
}
_WORKER_STATE: dict[str, Any] = {}


def require_cfg(config: dict, dotted_key: str) -> Any:
    value = cfg_get(config, dotted_key, None)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def require_cfg_path(config: dict, dotted_key: str) -> Path:
    value = cfg_path(config, dotted_key, None)
    if value is None:
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value.expanduser().resolve()


def load_stage2_module(accelerator: str):
    stage2_path = STAGE2_SCRIPTS[accelerator]
    module_name = f"parallel_neural2token_stage2_{accelerator}"
    spec = importlib.util.spec_from_file_location(module_name, str(stage2_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Stage 2 module from {stage2_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def shared_tokenizer_signature(config: dict) -> dict[str, Any]:
    checkpoint_path = cfg_path(config, "tokenizer.checkpoint_path", None)
    require_cfg(config, "tokenizer.use_ema")
    return {
        "modality": str(require_cfg(config, "dataset.modality")),
        "project_root": str(require_cfg_path(config, "tokenizer.project_root")),
        "checkpoint_root": str(require_cfg_path(config, "tokenizer.checkpoint_root")),
        "checkpoint_path": str(checkpoint_path.expanduser().resolve()) if checkpoint_path else None,
        "train_config": str(require_cfg_path(config, "tokenizer.train_config")),
        "use_ema": cfg_bool(config, "tokenizer.use_ema", False),
    }


def _resolve_device(mod, accelerator: str, config: dict, device_override: Optional[str]):
    if accelerator != "cuda":
        raise ValueError(f"Only the CUDA Stage 2 worker is supported, got {accelerator!r}")
    device_name = device_override or str(require_cfg(config, "runtime.device"))
    if device_name == "auto":
        device_name = "cuda" if mod.torch.cuda.is_available() else "cpu"
    device = mod.torch.device(device_name)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"CUDA Stage 2 does not support device {device}")
    return device


def _build_runtime(config: dict):
    mod = _WORKER_STATE["mod"]
    device = _WORKER_STATE["device"]
    require_cfg(config, "runtime.use_bfloat16")
    use_bfloat16 = cfg_bool(config, "runtime.use_bfloat16", False)
    if use_bfloat16 and device.type != "cuda":
        raise ValueError("runtime.use_bfloat16 requires a CUDA device")
    runtime_kwargs = {
        "modality": str(require_cfg(config, "dataset.modality")),
        "device": device,
        "use_bfloat16": use_bfloat16,
        "checkpoint_path": _WORKER_STATE["checkpoint"],
        "train_config_path": require_cfg_path(config, "tokenizer.train_config"),
        "use_ema": cfg_bool(config, "tokenizer.use_ema", False),
        "data_seed": int(require_cfg(config, "tokenizer.data_seed")),
        "storage_dtype": str(require_cfg(config, "tokenizer.storage_dtype")),
    }
    if _WORKER_STATE["accelerator"] == "cuda":
        runtime_kwargs["max_chunk_batch_size"] = int(
            cfg_get(config, "tokenizer.max_chunk_batch_size", 8)
        )
    runtime = mod.RuntimeConfig(**runtime_kwargs)
    runtime.quality_cfg = _WORKER_STATE.get("quality_cfg")
    runtime.zscore_mode = _WORKER_STATE.get("zscore_mode", "voxel")
    return runtime


def _seed_worker(runtime) -> None:
    random.seed(runtime.data_seed)
    np.random.seed(runtime.data_seed)
    _WORKER_STATE["mod"].torch.manual_seed(runtime.data_seed)


def initialize_gpu_session_worker(
    accelerator: str,
    config_path_strs: list[str],
    device_override: Optional[str],
) -> None:
    """Initialize one persistent model replica in a Stage 2 worker process."""
    mod = load_stage2_module(accelerator)
    configs = {str(Path(path).resolve()): load_config(path) for path in config_path_strs}
    first_config = next(iter(configs.values()))
    signature = shared_tokenizer_signature(first_config)
    mod.configure_unitok_project_root(Path(signature["project_root"]))
    device = _resolve_device(mod, accelerator, first_config, device_override)
    checkpoint_path = Path(signature["checkpoint_path"]) if signature["checkpoint_path"] else None
    resolved_checkpoint = mod.resolve_checkpoint_path(
        checkpoint_path,
        Path(signature["checkpoint_root"]),
    ).expanduser().resolve()

    _WORKER_STATE.update({
        "mod": mod,
        "configs": configs,
        "device": device,
        "checkpoint": resolved_checkpoint,
        "accelerator": accelerator,
    })
    runtime = _build_runtime(first_config)
    _seed_worker(runtime)
    model, dataset_meta, resolved_sizes, *_ = mod.build_unitok_for_inference(
        train_config_path=runtime.train_config_path,
        checkpoint_path=resolved_checkpoint,
        device=device,
        use_ema=runtime.use_ema,
    )
    neural_transform = mod.build_modality_neural_transform(dataset_meta, runtime.modality)
    quality_cfg, zscore_mode = mod.build_modality_quality_cfg(dataset_meta, runtime.modality)
    _WORKER_STATE.update({
        "model": model,
        "neural_transform": neural_transform,
        "quality_cfg": quality_cfg,
        "zscore_mode": zscore_mode,
        "resolved_sizes": resolved_sizes,
        "modality": runtime.modality,
    })


def gpu_worker_info() -> dict:
    mod = _WORKER_STATE["mod"]
    return {
        "pid": mp.current_process().pid,
        "device": str(_WORKER_STATE["device"]),
        "checkpoint": str(_WORKER_STATE["checkpoint"]),
        "modality": _WORKER_STATE["modality"],
        "resolved_sizes": _WORKER_STATE["resolved_sizes"],
        "cuda_memory_allocated": (
            int(mod.torch.cuda.memory_allocated(_WORKER_STATE["device"]))
            if _WORKER_STATE["device"].type == "cuda"
            else None
        ),
    }


def prepare_dataset(config_path_str: str, max_sessions: Optional[int], overwrite: bool) -> dict:
    """Initialize one dataset and assign stable session names before fan-out."""
    mod = _WORKER_STATE["mod"]
    config_key = str(Path(config_path_str).resolve())
    pipeline_config = _WORKER_STATE["configs"][config_key]
    runtime = _build_runtime(pipeline_config)
    _seed_worker(runtime)

    input_base = require_cfg_path(pipeline_config, "paths.tar_root")
    output_base = require_cfg_path(pipeline_config, "paths.token_root")
    session_dirs = mod.find_session_folders(input_base)
    if max_sessions is not None:
        session_dirs = session_dirs[: int(max_sessions)]

    tokenizer_dir, stimuli_dir, sessions_dir = mod.initialize_output_dataset(
        output_base,
        overwrite=overwrite,
    )
    codebook_path, codebook_metadata_path = mod.export_codebook(
        _WORKER_STATE["model"], tokenizer_dir, runtime,
    )

    used_names: set[str] = set()
    session_tasks = []
    for session_order, source_session_dir in enumerate(session_dirs):
        output_name = mod.ensure_unique_session_name(
            mod.session_output_name(source_session_dir, runtime.modality),
            used_names,
        )
        session_tasks.append({
            "config": config_path_str,
            "session_order": session_order,
            "source_session_dir": str(source_session_dir),
            "source_session_name": source_session_dir.name,
            "output_session_name": output_name,
            "output_session_dir": str(sessions_dir / output_name),
            "input_base": str(input_base),
            "output_base": str(output_base),
            "codebook_path": str(codebook_path),
            "codebook_metadata_path": str(codebook_metadata_path),
        })

    return {
        "config": config_path_str,
        "input_base": str(input_base),
        "output_base": str(output_base),
        "stimuli_dir": str(stimuli_dir),
        "sessions_dir": str(sessions_dir),
        "codebook_path": str(codebook_path),
        "codebook_metadata_path": str(codebook_metadata_path),
        "session_tasks": session_tasks,
    }


def convert_session(task: dict, max_samples: Optional[int]) -> dict:
    """Convert one whole session; chunk/window order remains sequential."""
    started = time.time()
    mod = _WORKER_STATE["mod"]
    config_key = str(Path(task["config"]).resolve())
    pipeline_config = _WORKER_STATE["configs"][config_key]
    runtime = _build_runtime(pipeline_config)
    _seed_worker(runtime)

    source_session_dir = Path(task["source_session_dir"])
    output_session_dir = Path(task["output_session_dir"])
    try:
        token_metadata = mod.convert_session_tar_to_token_ids(
            source_session_dir,
            output_session_dir,
            _WORKER_STATE["model"],
            runtime,
            _WORKER_STATE["neural_transform"],
            max_samples=max_samples,
            input_base=Path(task["input_base"]),
            output_base=Path(task["output_base"]),
            codebook_path=Path(task["codebook_path"]),
            codebook_metadata_path=Path(task["codebook_metadata_path"]),
        )
        token_metadata.setdefault("session", {})["session_id"] = task["output_session_name"]
        (output_session_dir / "token_metadata.json").write_text(
            json.dumps(token_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        session_metadata = mod.load_session_metadata(source_session_dir)
        summary = mod.summarize_session(
            token_metadata,
            session_metadata,
            task["output_session_name"],
            task["source_session_name"],
            output_session_dir,
        )
        return {
            "status": "ok",
            "session_order": int(task["session_order"]),
            "source_session_name": task["source_session_name"],
            "session_summary": summary,
            "duration_seconds": round(time.time() - started, 3),
            "worker_pid": mp.current_process().pid,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "session_order": int(task["session_order"]),
            "source_session_name": task["source_session_name"],
            "error": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "duration_seconds": round(time.time() - started, 3),
            "worker_pid": mp.current_process().pid,
        }


def finalize_dataset(prepared: dict, session_results: list[dict]) -> dict:
    """Write manifest and subjects once, ordered by the original session list."""
    mod = _WORKER_STATE["mod"]
    config_key = str(Path(prepared["config"]).resolve())
    pipeline_config = _WORKER_STATE["configs"][config_key]
    runtime = _build_runtime(pipeline_config)
    output_base = Path(prepared["output_base"])
    input_base = Path(prepared["input_base"])

    ordered = sorted(session_results, key=lambda item: int(item["session_order"]))
    session_summaries = [item["session_summary"] for item in ordered if item["status"] == "ok"]
    failed_items = [item for item in ordered if item["status"] != "ok"]
    failed_sessions = [item["source_session_name"] for item in failed_items]
    failed_session_errors = [
        {
            "session": item["source_session_name"],
            "error_type": str(item.get("error_type", "RuntimeError")),
            "message": str(item.get("message", item.get("error", "Unknown error"))),
        }
        for item in failed_items
    ]
    manifest = {
        "format": "neurotoken_dataset",
        "version": "1.0",
        "modality": runtime.modality,
        "input_base": input_base.name,
        "output_base": output_base.name,
        "readme": "README.md",
        "tokenizer": {
            "codebook": mod.to_relative_str(Path(prepared["codebook_path"]), output_base),
            "codebook_metadata": mod.to_relative_str(Path(prepared["codebook_metadata_path"]), output_base),
        },
        "stimuli": {
            "path": mod.to_relative_str(Path(prepared["stimuli_dir"]), output_base),
            "status": "empty_placeholder; user should move/copy real stimuli files here later",
        },
        "sessions_dir": mod.to_relative_str(Path(prepared["sessions_dir"]), output_base),
        "num_sessions": len(session_summaries),
        "split_policy": "not_provided; users should create task-appropriate splits from the session/subject metadata",
        "sessions": session_summaries,
    }
    if failed_sessions:
        manifest["failed_sessions"] = failed_sessions
        manifest["failed_session_errors"] = failed_session_errors
    (output_base / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    subjects = mod.build_subjects_index(session_summaries)
    (output_base / "subjects.json").write_text(
        json.dumps(subjects, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "failed_sessions": failed_sessions,
        "failed_session_errors": failed_session_errors,
        "session_results": ordered,
    }
