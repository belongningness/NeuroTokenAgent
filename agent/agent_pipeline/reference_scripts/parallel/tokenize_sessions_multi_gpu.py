"""Session-parallel Stage 2 with one persistent worker per CUDA device.

The coordinator initializes one independent single-process executor for each
selected device. Dataset setup and final manifest writes happen exactly once;
only isolated session conversion tasks are distributed across devices.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Optional

from parallel.stage2_session_worker import (
    convert_session,
    finalize_dataset,
    gpu_worker_info,
    initialize_gpu_session_worker,
    prepare_dataset,
)


def _failed_session_result(task: dict, error: str, error_type: str) -> dict:
    return {
        "status": "failed",
        "session_order": int(task["session_order"]),
        "source_session_name": task["source_session_name"],
        "error": error,
        "error_type": error_type,
        "message": error,
    }


def _shutdown_executor(executor: ProcessPoolExecutor, *, wait_for_workers: bool) -> None:
    try:
        executor.shutdown(wait=wait_for_workers, cancel_futures=True)
    except Exception:
        pass


def _worker_counts(workers: list[dict]) -> dict:
    initialized = sum(worker.get("info") is not None for worker in workers)
    surviving = sum(bool(worker.get("healthy")) for worker in workers)
    return {
        "model_replicas": initialized,
        "model_replicas_initialized": initialized,
        "model_replicas_surviving": surviving,
    }


def process_one_dataset_multi_gpu(
    config_path: Path | str,
    devices: list[str],
    max_samples: Optional[int] = None,
    max_sessions: Optional[int] = None,
    overwrite: bool = False,
) -> dict:
    """Dynamically schedule sessions across one model worker per CUDA device."""
    started = time.time()
    config_path_str = str(Path(config_path).expanduser().resolve())
    if not devices:
        return {
            "kind": "dataset_result",
            "config": config_path_str,
            "status": "failed",
            "error": "No CUDA devices were selected for multi-GPU tokenization",
            "devices": [],
            "worker_startup": [],
            "model_replicas": 0,
            "model_replicas_initialized": 0,
            "model_replicas_surviving": 0,
            "duration_seconds": round(time.time() - started, 3),
        }

    canonical_devices = []
    for device in devices:
        value = str(device).strip().lower()
        if not value.startswith("cuda:"):
            raise ValueError(f"Multi-GPU workers require explicit CUDA devices, got {device!r}")
        if value not in canonical_devices:
            canonical_devices.append(value)

    ctx = mp.get_context("spawn")
    workers: list[dict] = []
    startup: list[dict] = []
    worker_failures: list[dict] = []
    try:
        # Separate one-process executors guarantee one model replica on each device.
        for device in canonical_devices:
            try:
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=initialize_gpu_session_worker,
                    initargs=("cuda", [config_path_str], device),
                )
            except Exception as exc:
                startup.append({
                    "device": device,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                continue
            worker = {
                "device": device,
                "executor": executor,
                "healthy": False,
                "info": None,
            }
            workers.append(worker)

        probe_futures: dict[Future, dict] = {}
        for worker in workers:
            try:
                future = worker["executor"].submit(gpu_worker_info)
            except Exception as exc:
                startup.append({
                    "device": worker["device"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                _shutdown_executor(worker["executor"], wait_for_workers=False)
                continue
            probe_futures[future] = worker

        for future, worker in probe_futures.items():
            try:
                info = future.result()
                worker["healthy"] = True
                worker["info"] = info
                startup.append({"device": worker["device"], "status": "ok", **info})
            except Exception as exc:
                startup.append({
                    "device": worker["device"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                _shutdown_executor(worker["executor"], wait_for_workers=False)

        healthy = [worker for worker in workers if worker["healthy"]]
        if not healthy:
            return {
                "kind": "dataset_result",
                "config": config_path_str,
                "status": "failed",
                "error": "Every selected CUDA worker failed during model initialization",
                "devices": canonical_devices,
                "worker_startup": startup,
                "worker_failures": worker_failures,
                **_worker_counts(workers),
                "duration_seconds": round(time.time() - started, 3),
            }

        coordinator = healthy[0]
        try:
            prepared = coordinator["executor"].submit(
                prepare_dataset,
                config_path_str,
                max_sessions,
                overwrite,
            ).result()
        except Exception as exc:
            coordinator["healthy"] = False
            worker_failures.append({
                "device": coordinator["device"],
                "phase": "dataset_preparation",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            return {
                "kind": "dataset_result",
                "config": config_path_str,
                "status": "failed",
                "error": f"Dataset preparation failed: {type(exc).__name__}: {exc}",
                "devices": canonical_devices,
                "worker_startup": startup,
                "worker_failures": worker_failures,
                **_worker_counts(workers),
                "duration_seconds": round(time.time() - started, 3),
            }

        pending = deque(prepared["session_tasks"])
        active: dict[Future, tuple[dict, dict]] = {}
        session_results: list[dict] = []

        def deactivate_worker(worker: dict, phase: str, exc: Exception) -> None:
            worker["healthy"] = False
            worker_failures.append({
                "device": worker["device"],
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            _shutdown_executor(worker["executor"], wait_for_workers=False)

        def submit_next(worker: dict) -> bool:
            if not worker["healthy"] or not pending:
                return False
            task = pending[0]
            try:
                future = worker["executor"].submit(convert_session, task, max_samples)
            except Exception as exc:
                deactivate_worker(worker, "session_submission", exc)
                return False
            pending.popleft()
            active[future] = (worker, task)
            return True

        for worker in healthy:
            submit_next(worker)

        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                worker, task = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    error = f"Worker on {worker['device']} crashed: {type(exc).__name__}: {exc}"
                    failed = _failed_session_result(task, error, type(exc).__name__)
                    failed["device"] = worker["device"]
                    session_results.append(failed)
                    deactivate_worker(worker, "session_conversion", exc)
                else:
                    result["device"] = worker["device"]
                    session_results.append(result)
                    submit_next(worker)

        if pending:
            while pending:
                task = pending.popleft()
                session_results.append(_failed_session_result(
                    task,
                    "No healthy CUDA worker remained to process this session",
                    "WorkerUnavailableError",
                ))

        finalization_errors = []
        finalized = None
        for worker in workers:
            if not worker["healthy"]:
                continue
            try:
                finalized = worker["executor"].submit(
                    finalize_dataset,
                    prepared,
                    session_results,
                ).result()
                break
            except Exception as exc:
                finalization_errors.append(
                    f"{worker['device']}: {type(exc).__name__}: {exc}"
                )
                deactivate_worker(worker, "dataset_finalization", exc)

        if finalized is None:
            return {
                "kind": "dataset_result",
                "config": config_path_str,
                "status": "failed",
                "error": "Dataset finalization failed on every healthy CUDA worker",
                "finalization_errors": finalization_errors,
                "devices": canonical_devices,
                "worker_startup": startup,
                "worker_failures": worker_failures,
                **_worker_counts(workers),
                "session_results": sorted(
                    session_results, key=lambda item: int(item["session_order"])
                ),
                "duration_seconds": round(time.time() - started, 3),
            }

        failed_sessions = finalized["failed_sessions"]
        num_sessions = int(finalized["manifest"].get("num_sessions", 0))
        status = "failed" if failed_sessions or num_sessions == 0 else "ok"
        result = {
            "kind": "dataset_result",
            "config": config_path_str,
            "status": status,
            "num_sessions": num_sessions,
            "failed_sessions": failed_sessions,
            "failed_session_errors": finalized["failed_session_errors"],
            "session_results": finalized["session_results"],
            "devices": canonical_devices,
            "worker_startup": startup,
            "worker_failures": worker_failures,
            **_worker_counts(workers),
            "duration_seconds": round(time.time() - started, 3),
        }
        if failed_sessions:
            result["error"] = (
                f"Stage 2 failed for {len(failed_sessions)} session(s): {failed_sessions}"
            )
        elif num_sessions == 0:
            result["error"] = "Stage 2 produced zero successful sessions"
        return result
    finally:
        for worker in workers:
            _shutdown_executor(worker["executor"], wait_for_workers=worker["healthy"])
