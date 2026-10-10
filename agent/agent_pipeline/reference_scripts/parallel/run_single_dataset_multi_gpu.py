#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one dataset through H5 parallelism, then one token worker per GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

PARALLEL_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PARALLEL_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapters import build_adapter
from neurotoken_config import cfg_get, cfg_path, load_config
from neurotoken_report import make_report, write_report
from parallel.package_h5_parallel import run as package_h5_files
from parallel.resource_planner import (
    DEFAULT_MAX_AUTO_GPU_DEVICES,
    DEFAULT_TOKEN_WORKER_MEMORY_GB,
    plan_h5_workers,
    plan_multi_gpu_devices,
)
from parallel.tokenize_sessions_multi_gpu import process_one_dataset_multi_gpu


def discover_source_h5(config: dict) -> list[Path]:
    input_root = cfg_path(config, "paths.h5_root", None)
    if input_root is None:
        raise KeyError("Missing required config field: paths.h5_root")
    return build_adapter(config).discover(input_root.expanduser().resolve())


def count_session_tars(config: dict, max_sessions: Optional[int]) -> int:
    tar_root = cfg_path(config, "paths.tar_root", None)
    if tar_root is None:
        raise KeyError("Missing required config field: paths.tar_root")
    tar_root = tar_root.expanduser().resolve()
    if (tar_root / "data.tar").is_file():
        sessions = [tar_root]
    elif tar_root.exists():
        sessions = sorted(
            path for path in tar_root.iterdir()
            if path.is_dir() and (path / "data.tar").is_file()
        )
    else:
        sessions = []
    if max_sessions is not None:
        sessions = sessions[:int(max_sessions)]
    return len(sessions)


def run_h5_to_tar(config_path: Path, workers: int) -> dict:
    started = time.time()
    try:
        sample_count = package_h5_files(
            config_path=config_path,
            workers=workers,
            strict=True,
        )
        if sample_count <= 0:
            raise RuntimeError("H5 → tar produced no samples")
        return {
            "stage": "h5_to_tar",
            "status": "ok",
            "workers": int(workers),
            "sample_count": int(sample_count),
            "duration_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "stage": "h5_to_tar",
            "status": "failed",
            "workers": int(workers),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "duration_seconds": round(time.time() - started, 3),
        }


def print_resource_plan(label: str, plan: dict) -> None:
    devices = plan.get("devices", [])
    print(
        f"{label} resource plan: workers={plan['workers']} "
        f"({'auto' if plan['automatic'] else 'explicit'}), "
        f"devices={','.join(devices) if devices else 'none'}"
    )
    for warning in plan.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)


def run_tar_to_token_multi_gpu(
    config_path: Path,
    devices: list[str],
    max_samples: Optional[int],
    max_sessions: Optional[int],
    overwrite: bool,
) -> dict:
    """Run Stage 2 with exactly one persistent model worker on each device."""
    started = time.time()
    try:
        dataset_result = process_one_dataset_multi_gpu(
            config_path=config_path,
            devices=devices,
            max_samples=max_samples,
            max_sessions=max_sessions,
            overwrite=overwrite,
        )
        status = "ok" if dataset_result.get("status") == "ok" else "failed"
        errors = [] if status == "ok" else [
            str(dataset_result.get("error", "tar → token failed"))
        ]
    except Exception as exc:
        dataset_result = None
        status = "failed"
        errors = [f"{type(exc).__name__}: {exc}"]
    initialized = int(dataset_result.get("model_replicas_initialized", 0)) if dataset_result else 0
    surviving = int(dataset_result.get("model_replicas_surviving", 0)) if dataset_result else 0
    return {
        "stage": "tar_to_token",
        "status": status,
        "workers": initialized,
        "workers_planned": len(devices),
        "model_replicas": initialized,
        "model_replicas_surviving": surviving,
        "devices": list(devices),
        "dataset_result": dataset_result,
        "errors": errors,
        "duration_seconds": round(time.time() - started, 3),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process one dataset with one H5 per CPU task, wait at a strict barrier, "
            "then dynamically schedule sessions to one persistent worker per CUDA GPU."
        ),
    )
    parser.add_argument("--config", type=Path, required=True, help="One dataset TOML config.")
    parser.add_argument(
        "-j", "--h5-workers", type=int, default=None,
        help=(
            "Concurrent H5 → tar workers. Omit or set 0 to choose from CPU affinity, "
            "available RAM, source size and H5 count."
        ),
    )
    parser.add_argument(
        "--devices", nargs="+", default=None, metavar="CUDA_DEVICE",
        help=(
            "CUDA devices, for example --devices cuda:0 cuda:1. Omit or pass auto "
            "to select every safely provisioned visible GPU."
        ),
    )
    parser.add_argument(
        "--max-gpus", type=int, default=DEFAULT_MAX_AUTO_GPU_DEVICES,
        help=(
            "Maximum devices selected in auto mode "
            f"(default: {DEFAULT_MAX_AUTO_GPU_DEVICES}; also bounded by session count)."
        ),
    )
    parser.add_argument(
        "--token-worker-memory-gb", type=float, default=None,
        help=(
            f"Conservative free-memory estimate required for each GPU model worker "
            f"(default: {DEFAULT_TOKEN_WORKER_MEMORY_GB:g} GiB)."
        ),
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum tar samples per session for a smoke test.",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=None,
        help="Maximum sessions to tokenize for a smoke test.",
    )
    parser.add_argument(
        "--overwrite-token-output", action="store_true",
        help="Replace token_root once before Stage 2 session fan-out.",
    )
    parser.add_argument(
        "--skip-h5-to-tar", action="store_true",
        help="Tokenize an existing tar_root without running Stage 1.",
    )
    parser.add_argument(
        "--package-only", action="store_true",
        help="Run parallel H5 → tar packaging only.",
    )
    parser.add_argument("--report-path", type=Path, default=None, help="Combined JSON report path.")
    args = parser.parse_args(argv)
    if args.h5_workers is not None and args.h5_workers < 0:
        parser.error("--h5-workers must be >= 0")
    if args.max_gpus <= 0:
        parser.error("--max-gpus must be positive")
    if args.token_worker_memory_gb is not None and args.token_worker_memory_gb <= 0:
        parser.error("--token-worker-memory-gb must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_sessions is not None and args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    if args.skip_h5_to_tar and args.package_only:
        parser.error("--skip-h5-to-tar and --package-only cannot be used together")
    return args


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    pipeline_config = load_config(config_path)

    stages = []
    resource_plans = {}
    if args.skip_h5_to_tar:
        stages.append({
            "stage": "h5_to_tar",
            "status": "skipped",
            "reason": "--skip-h5-to-tar",
        })
    else:
        source_paths = discover_source_h5(pipeline_config)
        h5_plan = plan_h5_workers(
            source_paths,
            args.h5_workers,
            signal_key=str(cfg_get(pipeline_config, "source_h5.signal_key", "eeg")),
        )
        resource_plans["h5_to_tar"] = h5_plan.to_dict()
        print_resource_plan("H5 → tar", resource_plans["h5_to_tar"] | {"devices": []})
        h5_result = run_h5_to_tar(config_path, h5_plan.workers)
        stages.append(h5_result)
        if h5_result["status"] != "ok":
            return make_report(
                "single_dataset_multi_gpu",
                status="failed",
                config=str(config_path),
                strict_stage_barrier=True,
                one_worker_per_gpu=True,
                resource_plans=resource_plans,
                stages=stages,
                duration_seconds=round(time.time() - started, 3),
            )

    if args.package_only:
        return make_report(
            "single_dataset_multi_gpu",
            status="ok",
            config=str(config_path),
            strict_stage_barrier=True,
            one_worker_per_gpu=True,
            resource_plans=resource_plans,
            stages=stages,
            duration_seconds=round(time.time() - started, 3),
        )

    session_count = count_session_tars(pipeline_config, args.max_sessions)
    estimated_worker_memory_gb = float(
        args.token_worker_memory_gb
        if args.token_worker_memory_gb is not None
        else cfg_get(
            pipeline_config,
            "parallel.token_worker_memory_gb",
            DEFAULT_TOKEN_WORKER_MEMORY_GB,
        )
    )
    gpu_plan = plan_multi_gpu_devices(
        requested_devices=args.devices,
        session_count=session_count,
        estimated_worker_memory_gb=estimated_worker_memory_gb,
        max_auto_devices=args.max_gpus,
    )
    resource_plans["tar_to_token"] = gpu_plan.to_dict()
    print_resource_plan("tar → token", resource_plans["tar_to_token"])
    token_result = run_tar_to_token_multi_gpu(
        config_path=config_path,
        devices=gpu_plan.devices,
        max_samples=args.max_samples,
        max_sessions=args.max_sessions,
        overwrite=args.overwrite_token_output,
    )
    stages.append(token_result)
    return make_report(
        "single_dataset_multi_gpu",
        status="ok" if token_result["status"] == "ok" else "failed",
        config=str(config_path),
        strict_stage_barrier=True,
        one_worker_per_gpu=True,
        resource_plans=resource_plans,
        stages=stages,
        duration_seconds=round(time.time() - started, 3),
    )


def main() -> None:
    args = parse_args()
    report = run(args)
    pipeline_config = load_config(args.config)
    report_path = args.report_path or (
        Path(cfg_path(pipeline_config, "paths.report_root", SCRIPT_DIR / "reports"))
        / "single_dataset_multi_gpu_report.json"
    )
    report_path = report_path.expanduser().resolve()
    write_report(report_path, report)
    print(f"Status: {report['status']}")
    for stage in report.get("stages", []):
        print(f"{stage.get('stage')}: {stage.get('status')}")
    print(f"Report: {report_path}")
    if report.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
