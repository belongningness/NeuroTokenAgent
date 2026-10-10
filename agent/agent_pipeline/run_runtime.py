"""Run isolation, manifests, and stage execution for durable pipeline workflows."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agent_pipeline.config import cfg_path
from agent_pipeline.dynamic_config import create_dynamic_config
from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.tools import load_config_tool, package_tool, preflight_tool, run_reference_stage_tool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TMUX_STAGES = frozenset({"package", "tokenize", "reconstruct"})
TERMINAL_ATTEMPT_STATUSES = frozenset({"completed", "failed", "cancelled", "orphaned"})


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def validate_run_id(value: str, field: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {field}: {value!r}")
    return value


def safe_run_root(output_root: str | Path, workflow_id: str, run_id: str) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    workflow_id = validate_run_id(workflow_id, "workflow_id")
    run_id = validate_run_id(run_id, "run_id")
    run_root = (output_root / "runs" / workflow_id / run_id).resolve()
    if output_root not in run_root.parents:
        raise ValueError(f"Run root escapes output root: {run_root}")
    return run_root


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Run is locked: {path}") from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} created_at={utc_now()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def materialize_run(
    request: PipelineRequest,
    workflow_id: str | None = None,
    run_id: str | None = None,
    *,
    phase: str = "single",
    parent_run_id: str | None = None,
) -> dict[str, str]:
    if request.report_root is None:
        raise ValueError("report_root is required for Agent Server execution.")
    source = request.target_h5 or request.input_root
    if source is None:
        raise ValueError("target_h5 or input_root is required for Agent Server execution.")
    workflow_id = workflow_id or new_id("wf")
    run_id = run_id or new_id("run")
    run_root = safe_run_root(request.report_root, workflow_id, run_id)
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_request = request.model_dump(mode="json")
        if (
            existing.get("workflow_id") != workflow_id
            or existing.get("run_id") != run_id
            or existing.get("phase", "single") != phase
            or existing.get("parent_run_id") != parent_run_id
            or existing.get("request") != expected_request
        ):
            raise RuntimeError(f"Run identity already exists with a different specification: {run_root}")
        config_path = Path(existing["config_path"])
        if not config_path.is_file():
            raise RuntimeError(f"Existing run config is missing: {config_path}")
        config = load_config_tool(str(config_path))["config"]
        return {
            "workflow_id": workflow_id,
            "run_id": run_id,
            "run_root": str(run_root),
            "config_path": str(config_path),
            "report_root": str(cfg_path(config, "paths.report_root", run_root / "reports")),
        }
    config_path = create_dynamic_config(
        base_config_path=request.config_path,
        output_root=run_root,
        target_h5=request.target_h5,
        input_root=request.input_root,
        source_adapter=request.source_adapter,
        config_overrides=request.config_overrides,
    )
    config_bytes = config_path.read_bytes()
    manifest = {
        "schema_version": "1.0",
        "workflow_id": workflow_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "phase": phase,
        "parent_run_id": parent_run_id,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "selection": {
            "mode": "target_h5" if request.target_h5 else "input_root",
            "source": source,
            "target_h5": request.target_h5,
            "input_root": request.input_root,
            "max_files": request.max_files,
        },
        "request": request.model_dump(mode="json"),
    }
    atomic_write_json(run_root / "run_manifest.json", manifest)
    config = load_config_tool(str(config_path))["config"]
    return {
        "workflow_id": workflow_id,
        "run_id": run_id,
        "run_root": str(run_root),
        "config_path": str(config_path),
        "report_root": str(cfg_path(config, "paths.report_root", run_root / "reports")),
    }


def materialize_child_run(
    request: PipelineRequest,
    *,
    workflow_id: str,
    parent_run_id: str,
    phase: str,
    run_id: str | None = None,
) -> dict[str, str]:
    if phase not in {"validation", "full"}:
        raise ValueError(f"Unsupported linked run phase: {phase!r}")
    return materialize_run(
        request,
        workflow_id=workflow_id,
        run_id=run_id or new_id(f"run_{phase}"),
        phase=phase,
        parent_run_id=parent_run_id,
    )


def stage_manifest_path(run_root: str | Path, stage: str) -> Path:
    return Path(run_root) / "manifests" / f"{stage}.json"


def load_completed_stage(run_root: str | Path, stage: str) -> dict[str, Any] | None:
    path = stage_manifest_path(run_root, stage)
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Stage {stage} has an incomplete manifest; start a new run or clean the run directory.")
    report_path = Path(manifest["report_path"])
    if not report_path.exists():
        raise RuntimeError(f"Stage {stage} manifest exists but report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != manifest.get("report_sha256"):
        raise RuntimeError(f"Stage {stage} report checksum mismatch.")
    return report


def archive_failed_stage_attempt(
    run_root: str | Path,
    report_root: str | Path,
    stage: str,
    *,
    allow_interrupted: bool = False,
) -> None:
    manifest_path = stage_manifest_path(run_root, stage)
    if not manifest_path.exists():
        raise RuntimeError(f"Cannot retry {stage}: failed stage manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = Path(report_root) / f"{stage}_report.json"
    report_status = None
    if report_path.exists():
        try:
            report_status = json.loads(report_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            report_status = None
    retryable_status = manifest.get("status") == "failed" or (
        manifest.get("status") == "completed" and report_status == "failed"
    ) or (allow_interrupted and manifest.get("status") == "started")
    if not retryable_status:
        raise RuntimeError(
            f"Cannot retry {stage}: manifest status={manifest.get('status')!r}, "
            f"report status={report_status!r}."
        )

    execution_id = str(manifest.get("stage_execution_id") or new_id(f"stage_{stage}"))
    archive_root = Path(run_root) / "attempts" / stage / execution_id
    archive_root.mkdir(parents=True, exist_ok=True)
    archived_manifest = archive_root / "manifest.json"
    if archived_manifest.exists():
        raise RuntimeError(f"Cannot retry {stage}: archived manifest already exists for {execution_id}.")
    os.replace(manifest_path, archived_manifest)
    if report_path.exists():
        archived_report = archive_root / report_path.name
        if archived_report.exists():
            raise RuntimeError(f"Cannot retry {stage}: archived report already exists for {execution_id}.")
        os.replace(report_path, archived_report)


def _execute_stage_unlocked(
    request: PipelineRequest,
    stage: str,
    run_root: str,
    config_path: str,
    report_root: str,
    reference_scripts_dir: str,
    python: str,
    retry_failed: bool = False,
    archive_failed: bool = True,
    stage_execution_id: str | None = None,
) -> dict[str, Any]:
    if retry_failed and archive_failed:
        archive_failed_stage_attempt(run_root, report_root, stage)
    completed = load_completed_stage(run_root, stage)
    if completed is not None:
        return completed
    manifest_path = stage_manifest_path(run_root, stage)
    if manifest_path.exists():
        raise RuntimeError(f"Stage {stage} was interrupted before completion; automatic destructive retry is disabled.")
    execution_id = stage_execution_id or new_id(f"stage_{stage}")
    validate_run_id(execution_id, "stage_execution_id")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, {
        "schema_version": "1.0",
        "stage": stage,
        "stage_execution_id": execution_id,
        "status": "started",
        "started_at": utc_now(),
    })
    start = time.time()
    config = load_config_tool(config_path)["config"]
    try:
        if stage == "preflight":
            report = preflight_tool(config, max_files=request.max_files, target_h5=request.target_h5)
            if report.get("status") != "ok":
                raise RuntimeError(f"preflight failed: {report.get('findings', [])}")
        elif stage == "package":
            report = {"stage": stage, "status": "dry_run"} if request.dry_run else package_tool(
                config, max_files=request.max_files, target_h5=request.target_h5
            )
            report.setdefault("status", "ok")
        else:
            report = run_reference_stage_tool(
                stage=stage,
                config_path=config_path,
                report_root=report_root,
                reference_scripts_dir=reference_scripts_dir,
                python=python,
                smoke_test=request.smoke_test,
                overwrite_token_output=request.overwrite_token_output,
                allow_placeholder_readme=request.allow_placeholder_readme,
                strict_validation=request.strict_validation,
                dry_run=request.dry_run,
            )
        report.update({
            "stage": stage,
            "stage_execution_id": execution_id,
            "duration_seconds": round(time.time() - start, 3),
        })
        report_path = Path(report_root) / f"{stage}_report.json"
        atomic_write_json(report_path, report)
        started_at = json.loads(manifest_path.read_text(encoding="utf-8"))["started_at"]
        report_failed = report.get("status") == "failed"
        stage_manifest = {
            "schema_version": "1.0",
            "stage": stage,
            "stage_execution_id": execution_id,
            "status": "failed" if report_failed else "completed",
            "started_at": started_at,
            "report_path": str(report_path),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
        stage_manifest["failed_at" if report_failed else "completed_at"] = utc_now()
        atomic_write_json(manifest_path, stage_manifest)
        return report
    except Exception as exc:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        current.update(status="failed", failed_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(manifest_path, current)
        raise


def execute_stage(
    request: PipelineRequest,
    stage: str,
    run_root: str,
    config_path: str,
    report_root: str,
    reference_scripts_dir: str,
    python: str,
    retry_failed: bool = False,
    stage_execution_id: str | None = None,
) -> dict[str, Any]:
    lock_path = Path(run_root) / "locks" / f"{stage}.lock"
    with exclusive_lock(lock_path):
        return _execute_stage_unlocked(
            request=request,
            stage=stage,
            run_root=run_root,
            config_path=config_path,
            report_root=report_root,
            reference_scripts_dir=reference_scripts_dir,
            python=python,
            retry_failed=retry_failed,
            stage_execution_id=stage_execution_id,
        )


def attempt_root(run_root: str | Path, stage: str, execution_id: str) -> Path:
    if stage not in TMUX_STAGES:
        raise ValueError(f"Stage {stage!r} is not tmux-managed.")
    validate_run_id(execution_id, "stage_execution_id")
    root = Path(run_root).expanduser().resolve()
    path = (root / "attempts" / stage / execution_id).resolve()
    if root not in path.parents:
        raise ValueError("Attempt path escapes run root.")
    return path


def _tmux_socket() -> str:
    configured = os.environ.get("NEUROTOKEN_TMUX_SOCKET")
    return str(Path(configured).expanduser()) if configured else str(Path.home() / ".local/state/neurotoken-runtime/tmux/neurotoken.sock")


def _tmux_command(*arguments: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    tmux = shutil.which("tmux")
    if tmux is None:
        raise RuntimeError("tmux is required for durable package, tokenize, and reconstruct stages.")
    socket = Path(_tmux_socket())
    socket.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run([tmux, "-S", str(socket), *arguments], check=check, capture_output=True, text=True)


def _tmux_session_exists(session: str) -> bool:
    return _tmux_command("has-session", "-t", f"={session}").returncode == 0


def _bounded_log_tail(path: Path, max_bytes: int = 32_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        size = path.stat().st_size
        if size > max_bytes:
            stream.seek(-max_bytes, os.SEEK_END)
        return stream.read().decode("utf-8", errors="replace")


def _attempt_public(job: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    root = Path(job["attempt_root"])
    public_job = {
        key: value for key, value in job.items()
        if key not in {"request", "config_sha256", "python", "worker_python", "reference_scripts_dir"}
    }
    merged = {**public_job, **status}
    merged["reconnectable"] = merged.get("status") in {"submitted", "running", "detached", "reconnecting"}
    merged["stdout_log"] = str(root / "stdout.log")
    merged["stderr_log"] = str(root / "stderr.log")
    merged["log_tail"] = _bounded_log_tail(root / "stdout.log") or _bounded_log_tail(root / "stderr.log")
    return merged


def submit_tmux_stage(
    request: PipelineRequest,
    stage: str,
    run_root: str,
    config_path: str,
    report_root: str,
    reference_scripts_dir: str,
    python: str,
    retry_failed: bool = False,
    worker_python: str | None = None,
) -> dict[str, Any]:
    if stage not in TMUX_STAGES:
        raise ValueError(f"Stage {stage!r} is not tmux-managed.")
    run_path = Path(run_root).expanduser().resolve()
    submit_lock = run_path / "locks" / f"{stage}.submit.lock"
    with exclusive_lock(submit_lock):
        pointer = run_path / "attempts" / stage / "current.json"
        if pointer.exists():
            current = json.loads(pointer.read_text(encoding="utf-8"))
            inspected = inspect_stage_job(run_path, stage, current.get("stage_execution_id"))
            if inspected.get("status") not in TERMINAL_ATTEMPT_STATUSES:
                return inspected
            if not retry_failed:
                return inspected
            if _tmux_session_exists(str(inspected["tmux_session"])):
                raise RuntimeError(
                    f"Cannot retry {stage}: previous attempt {inspected['stage_execution_id']} is still active."
                )
            if stage_manifest_path(run_path, stage).exists():
                archive_failed_stage_attempt(run_path, report_root, stage, allow_interrupted=True)
        elif retry_failed and stage_manifest_path(run_path, stage).exists():
            archive_failed_stage_attempt(run_path, report_root, stage, allow_interrupted=True)
        execution_id = new_id(f"stage_{stage}")
        root = attempt_root(run_path, stage, execution_id)
        root.mkdir(parents=True, exist_ok=False)
        session = f"nt-{hashlib.sha256(str(run_path).encode()).hexdigest()[:8]}-{stage}-{execution_id[-8:]}"
        config = Path(config_path).expanduser().resolve()
        run_manifest_path = run_path / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.is_file() else {}
        job = {
            "schema_version": "1.0",
            "stage_execution_id": execution_id,
            "stage": stage,
            "workflow_id": run_manifest.get("workflow_id"),
            "run_id": run_manifest.get("run_id"),
            "phase": run_manifest.get("phase", "single"),
            "run_root": str(run_path),
            "attempt_root": str(root),
            "config_path": str(config),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "report_root": str(Path(report_root).expanduser().resolve()),
            "reference_scripts_dir": str(Path(reference_scripts_dir).expanduser().resolve()),
            "python": str(Path(python).expanduser().resolve()),
            "worker_python": str(Path(worker_python or sys.executable).expanduser().resolve()),
            "request": request.model_dump(mode="json"),
            "retry_failed": False,
            "tmux_socket": _tmux_socket(),
            "tmux_session": session,
            "submitted_at": utc_now(),
        }
        atomic_write_json(root / "job.json", job)
        status = {"status": "submitted", "updated_at": utc_now()}
        atomic_write_json(root / "status.json", status)
        atomic_write_json(pointer, {"stage_execution_id": execution_id, "updated_at": utc_now()})
        command = [worker_python or sys.executable, "-m", "agent_pipeline.stage_worker", "--job", str(root / "job.json")]
        completed = _tmux_command(
            "new-session", "-d", "-s", session, "-c", str(Path(__file__).resolve().parents[1]),
            *command,
        )
        if completed.returncode != 0:
            status = {"status": "failed", "updated_at": utc_now(), "error": completed.stderr.strip() or "tmux failed to start"}
            atomic_write_json(root / "status.json", status)
            raise RuntimeError(status["error"])
        return _attempt_public(job, status)


def inspect_stage_job(run_root: str | Path, stage: str, execution_id: str | None = None) -> dict[str, Any]:
    run_path = Path(run_root).expanduser().resolve()
    if execution_id is None:
        pointer = run_path / "attempts" / stage / "current.json"
        if not pointer.exists():
            raise RuntimeError(f"No durable attempt exists for {stage}.")
        execution_id = json.loads(pointer.read_text(encoding="utf-8"))["stage_execution_id"]
    root = attempt_root(run_path, stage, execution_id)
    job = json.loads((root / "job.json").read_text(encoding="utf-8"))
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "submitted"}
    if status.get("status") not in TERMINAL_ATTEMPT_STATUSES:
        if _tmux_session_exists(str(job["tmux_session"])):
            status = {**status, "status": "running" if status.get("status") == "running" else "detached", "updated_at": utc_now()}
        else:
            try:
                completed = load_completed_stage(run_path, stage)
            except RuntimeError:
                completed = None
            if completed is not None and completed.get("stage_execution_id") == execution_id:
                status = {**status, "status": "completed", "completed_at": status.get("completed_at") or utc_now(), "updated_at": utc_now()}
            else:
                status = {**status, "status": "orphaned", "updated_at": utc_now(), "error": "tmux session disappeared before a terminal status was written."}
            atomic_write_json(status_path, status)
    return _attempt_public(job, status)


def reconcile_stage_job(run_root: str | Path, stage: str, execution_id: str | None = None) -> dict[str, Any]:
    inspected = inspect_stage_job(run_root, stage, execution_id)
    if inspected.get("status") == "completed":
        report = load_completed_stage(run_root, stage)
        if report is None:
            raise RuntimeError(f"Durable {stage} attempt completed without a valid stage manifest.")
        if report.get("stage_execution_id") != inspected.get("stage_execution_id"):
            raise RuntimeError(f"Durable {stage} report execution identity does not match the current attempt.")
        inspected["report"] = report
    return inspected


def cancel_stage_job(run_root: str | Path, stage: str, execution_id: str | None = None) -> dict[str, Any]:
    inspected = inspect_stage_job(run_root, stage, execution_id)
    if inspected.get("status") in TERMINAL_ATTEMPT_STATUSES:
        return inspected
    session = str(inspected["tmux_session"])
    current = inspect_stage_job(run_root, stage, str(inspected["stage_execution_id"]))
    if current["tmux_session"] != session:
        raise RuntimeError("Refusing to cancel a tmux session with mismatched execution identity.")
    killed = _tmux_command("kill-session", "-t", f"={session}")
    if killed.returncode != 0:
        raise RuntimeError(killed.stderr.strip() or f"Failed to cancel tmux session {session}.")
    try:
        completed = load_completed_stage(run_root, stage)
    except RuntimeError:
        completed = None
    if completed is not None and completed.get("stage_execution_id") == inspected.get("stage_execution_id"):
        status = {"status": "completed", "completed_at": utc_now(), "updated_at": utc_now()}
        atomic_write_json(Path(inspected["attempt_root"]) / "status.json", status)
        return _attempt_public(inspected, status)
    status = {"status": "cancelled", "cancelled_at": utc_now(), "updated_at": utc_now()}
    atomic_write_json(Path(inspected["attempt_root"]) / "status.json", status)
    return _attempt_public(inspected, status)
