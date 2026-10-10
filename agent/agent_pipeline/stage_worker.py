"""Fixed worker entrypoint for durable tmux-managed pipeline stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
from typing import Any

from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.run_runtime import (
    TMUX_STAGES,
    _execute_stage_unlocked,
    atomic_write_json,
    attempt_root,
    exclusive_lock,
    utc_now,
)


class StageCancelled(RuntimeError):
    pass


def _load_job(path: Path) -> dict[str, Any]:
    job_path = path.expanduser().resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    stage = str(job.get("stage") or "")
    execution_id = str(job.get("stage_execution_id") or "")
    run_root = Path(str(job.get("run_root") or "")).expanduser().resolve()
    expected_root = attempt_root(run_root, stage, execution_id)
    if stage not in TMUX_STAGES or job_path != expected_root / "job.json":
        raise ValueError("Job path or stage identity is invalid.")
    if Path(str(job.get("attempt_root") or "")).expanduser().resolve() != expected_root:
        raise ValueError("Job attempt root does not match its execution identity.")
    config_path = Path(str(job.get("config_path") or "")).expanduser().resolve()
    worker_python = Path(str(job.get("worker_python") or "")).expanduser().resolve()
    if worker_python != Path(os.path.realpath(os.sys.executable)).resolve() and worker_python != Path(os.sys.executable).resolve():
        raise ValueError("Stage worker interpreter does not match the submitted job identity.")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != job.get("config_sha256"):
        raise ValueError("Pipeline config checksum changed after stage submission.")
    return job


def run_job(job_path: Path) -> int:
    job = _load_job(job_path)
    root = Path(job["attempt_root"])
    status_path = root / "status.json"
    cancelled = False

    def handle_signal(signum, _frame):
        nonlocal cancelled
        cancelled = True
        raise StageCancelled(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    lock_path = Path(job["run_root"]) / "locks" / f"{job['stage']}.lock"
    try:
        with exclusive_lock(lock_path):
            os.environ["NEUROTOKEN_STAGE_STDOUT_LOG"] = str(root / "stdout.log")
            os.environ["NEUROTOKEN_STAGE_STDERR_LOG"] = str(root / "stderr.log")
            atomic_write_json(status_path, {
                "status": "running",
                "worker_pid": os.getpid(),
                "started_at": utc_now(),
                "updated_at": utc_now(),
            })
            report = _execute_stage_unlocked(
                request=PipelineRequest.model_validate(job["request"]),
                stage=job["stage"],
                run_root=job["run_root"],
                config_path=job["config_path"],
                report_root=job["report_root"],
                reference_scripts_dir=job["reference_scripts_dir"],
                python=job["python"],
                retry_failed=bool(job.get("retry_failed")),
                archive_failed=False,
                stage_execution_id=job["stage_execution_id"],
            )
            failed = report.get("status") == "failed"
            atomic_write_json(status_path, {
                "status": "failed" if failed else "completed",
                "completed_at": utc_now(),
                "updated_at": utc_now(),
                "report_status": report.get("status"),
            })
            return 1 if failed else 0
    except StageCancelled as exc:
        atomic_write_json(status_path, {
            "status": "cancelled",
            "cancelled_at": utc_now(),
            "updated_at": utc_now(),
            "error": str(exc),
        })
        return 143
    except BaseException as exc:
        atomic_write_json(status_path, {
            "status": "cancelled" if cancelled else "failed",
            "failed_at": utc_now(),
            "updated_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    arguments = parser.parse_args()
    job_path = arguments.job.expanduser().resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    root = Path(job["attempt_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stdout.log").open("a", encoding="utf-8") as stdout, (root / "stderr.log").open("a", encoding="utf-8") as stderr:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
            return run_job(job_path)
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


if __name__ == "__main__":
    raise SystemExit(main())
