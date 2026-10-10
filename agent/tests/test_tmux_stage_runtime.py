from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.run_runtime import cancel_stage_job, inspect_stage_job, stage_manifest_path, submit_tmux_stage


pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")


def test_tmux_attempt_is_idempotent_and_cancellable(tmp_path: Path, monkeypatch):
    socket = tmp_path / "tmux" / "neurotoken.sock"
    monkeypatch.setenv("NEUROTOKEN_TMUX_SOCKET", str(socket))
    config = tmp_path / "config.toml"
    config.write_text("[paths]\nreport_root = '/tmp'\n", encoding="utf-8")
    report_root = tmp_path / "reports"
    report_root.mkdir()
    request = PipelineRequest(config_path=str(config), stages=["package"], target_h5=str(tmp_path / "sample.h5"))

    first = submit_tmux_stage(
        request=request,
        stage="package",
        run_root=str(tmp_path),
        config_path=str(config),
        report_root=str(report_root),
        reference_scripts_dir=str(tmp_path),
        python=sys.executable,
        worker_python=sys.executable,
    )
    second = submit_tmux_stage(
        request=request,
        stage="package",
        run_root=str(tmp_path),
        config_path=str(config),
        report_root=str(report_root),
        reference_scripts_dir=str(tmp_path),
        python=sys.executable,
        worker_python=sys.executable,
    )

    assert second["stage_execution_id"] == first["stage_execution_id"]
    assert (Path(first["attempt_root"]) / "job.json").is_file()
    assert (Path(first["attempt_root"]) / "status.json").is_file()
    inspected = inspect_stage_job(tmp_path, "package", first["stage_execution_id"])
    assert inspected["stage_execution_id"] == first["stage_execution_id"]

    if inspected["status"] not in {"completed", "failed", "cancelled", "orphaned"}:
        cancelled = cancel_stage_job(tmp_path, "package", first["stage_execution_id"])
        assert cancelled["status"] == "cancelled"


def test_missing_session_becomes_orphaned(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEUROTOKEN_TMUX_SOCKET", str(tmp_path / "missing.sock"))
    execution_id = "stage_tokenize_deadbeef"
    root = tmp_path / "attempts" / "tokenize" / execution_id
    root.mkdir(parents=True)
    job = {
        "stage_execution_id": execution_id,
        "stage": "tokenize",
        "run_root": str(tmp_path),
        "attempt_root": str(root),
        "tmux_session": "nt-missing-tokenize-deadbeef",
    }
    (root / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (root / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")

    inspected = inspect_stage_job(tmp_path, "tokenize", execution_id)

    assert inspected["status"] == "orphaned"
    assert "disappeared" in inspected["error"]


def test_new_session_argv_uses_worker_python_without_separator(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEUROTOKEN_TMUX_SOCKET", str(tmp_path / "tmux.sock"))
    config = tmp_path / "config.toml"
    config.write_text("[paths]\nreport_root = '/tmp'\n", encoding="utf-8")
    report_root = tmp_path / "reports"
    request = PipelineRequest(config_path=str(config), stages=["package"], target_h5=str(tmp_path / "sample.h5"))

    with patch("agent_pipeline.run_runtime._tmux_command") as tmux:
        tmux.return_value.returncode = 0
        tmux.return_value.stderr = ""
        submitted = submit_tmux_stage(
            request=request,
            stage="package",
            run_root=str(tmp_path),
            config_path=str(config),
            report_root=str(report_root),
            reference_scripts_dir=str(tmp_path),
            python="/scientific/bin/python",
            worker_python="/server/bin/python",
        )

    argv = tmux.call_args.args
    assert argv[:6] == ("new-session", "-d", "-s", submitted["tmux_session"], "-c", str(Path(__file__).resolve().parents[1]))
    assert argv[6:9] == ("/server/bin/python", "-m", "agent_pipeline.stage_worker")
    assert "--" not in argv
    job = json.loads((Path(submitted["attempt_root"]) / "job.json").read_text(encoding="utf-8"))
    pointer = json.loads((tmp_path / "attempts" / "package" / "current.json").read_text(encoding="utf-8"))
    assert pointer["stage_execution_id"] == submitted["stage_execution_id"]
    assert job["python"] == "/scientific/bin/python"
    assert job["worker_python"] == "/server/bin/python"


def test_retry_archives_failed_outputs_and_updates_current_pointer(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEUROTOKEN_TMUX_SOCKET", str(tmp_path / "tmux.sock"))
    config = tmp_path / "config.toml"
    config.write_text("[paths]\nreport_root = '/tmp'\n", encoding="utf-8")
    report_root = tmp_path / "reports"
    report_root.mkdir()
    old_execution_id = "stage_tokenize_old"
    old_root = tmp_path / "attempts" / "tokenize" / old_execution_id
    old_root.mkdir(parents=True)
    (old_root / "job.json").write_text(json.dumps({
        "stage_execution_id": old_execution_id,
        "stage": "tokenize",
        "run_root": str(tmp_path),
        "attempt_root": str(old_root),
        "tmux_session": "nt-old-tokenize",
    }), encoding="utf-8")
    (old_root / "status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    pointer = tmp_path / "attempts" / "tokenize" / "current.json"
    pointer.write_text(json.dumps({"stage_execution_id": old_execution_id}), encoding="utf-8")
    stage_manifest_path(tmp_path, "tokenize").parent.mkdir()
    stage_manifest_path(tmp_path, "tokenize").write_text(json.dumps({
        "stage": "tokenize", "stage_execution_id": old_execution_id, "status": "failed",
    }), encoding="utf-8")
    (report_root / "tokenize_report.json").write_text(json.dumps({"stage": "tokenize", "status": "failed"}), encoding="utf-8")
    request = PipelineRequest(config_path=str(config), stages=["tokenize"], target_h5=str(tmp_path / "sample.h5"))

    with (
        patch("agent_pipeline.run_runtime._tmux_session_exists", return_value=False),
        patch("agent_pipeline.run_runtime._tmux_command") as tmux,
    ):
        tmux.return_value.returncode = 0
        tmux.return_value.stderr = ""
        retried = submit_tmux_stage(
            request=request,
            stage="tokenize",
            run_root=str(tmp_path),
            config_path=str(config),
            report_root=str(report_root),
            reference_scripts_dir=str(tmp_path),
            python="/scientific/bin/python",
            worker_python="/server/bin/python",
            retry_failed=True,
        )

    assert retried["stage_execution_id"] != old_execution_id
    assert json.loads(pointer.read_text(encoding="utf-8"))["stage_execution_id"] == retried["stage_execution_id"]
    assert (old_root / "manifest.json").exists()
    assert (old_root / "tokenize_report.json").exists()
    assert (old_root / "job.json").exists()
    assert (old_root / "status.json").exists()


def test_cancel_race_preserves_matching_completed_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEUROTOKEN_TMUX_SOCKET", str(tmp_path / "tmux.sock"))
    execution_id = "stage_reconstruct_completed"
    root = tmp_path / "attempts" / "reconstruct" / execution_id
    root.mkdir(parents=True)
    report = tmp_path / "reports" / "reconstruct_report.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"stage": "reconstruct", "status": "ok", "stage_execution_id": execution_id}), encoding="utf-8")
    manifest = stage_manifest_path(tmp_path, "reconstruct")
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "stage": "reconstruct",
        "stage_execution_id": execution_id,
        "status": "completed",
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    job = {
        "stage_execution_id": execution_id,
        "stage": "reconstruct",
        "run_root": str(tmp_path),
        "attempt_root": str(root),
        "tmux_session": "nt-reconstruct-completed",
    }
    (root / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (root / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")

    with patch("agent_pipeline.run_runtime._tmux_session_exists", return_value=True), patch("agent_pipeline.run_runtime._tmux_command") as tmux:
        tmux.return_value.returncode = 0
        tmux.return_value.stderr = ""
        cancelled = cancel_stage_job(tmp_path, "reconstruct", execution_id)

    assert cancelled["status"] == "completed"
    assert json.loads((root / "status.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_retry_refuses_terminal_status_while_old_tmux_session_is_still_alive(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEUROTOKEN_TMUX_SOCKET", str(tmp_path / "tmux.sock"))
    config = tmp_path / "config.toml"
    config.write_text("[paths]\nreport_root = '/tmp'\n", encoding="utf-8")
    report_root = tmp_path / "reports"
    report_root.mkdir()
    execution_id = "stage_package_old"
    root = tmp_path / "attempts" / "package" / execution_id
    root.mkdir(parents=True)
    (root / "job.json").write_text(json.dumps({
        "stage_execution_id": execution_id,
        "stage": "package",
        "run_root": str(tmp_path),
        "attempt_root": str(root),
        "tmux_session": "nt-package-old",
    }), encoding="utf-8")
    (root / "status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    (root.parent / "current.json").write_text(json.dumps({"stage_execution_id": execution_id}), encoding="utf-8")
    request = PipelineRequest(config_path=str(config), stages=["package"], target_h5=str(tmp_path / "sample.h5"))

    with patch("agent_pipeline.run_runtime._tmux_session_exists", return_value=True):
        with pytest.raises(RuntimeError, match="still active"):
            submit_tmux_stage(
                request=request,
                stage="package",
                run_root=str(tmp_path),
                config_path=str(config),
                report_root=str(report_root),
                reference_scripts_dir=str(tmp_path),
                python="/scientific/bin/python",
                worker_python="/server/bin/python",
                retry_failed=True,
            )


def test_missing_session_accepts_matching_completed_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEUROTOKEN_TMUX_SOCKET", str(tmp_path / "missing.sock"))
    execution_id = "stage_tokenize_completed"
    root = tmp_path / "attempts" / "tokenize" / execution_id
    root.mkdir(parents=True)
    report = tmp_path / "reports" / "tokenize_report.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"stage": "tokenize", "status": "ok", "stage_execution_id": execution_id}), encoding="utf-8")
    manifest = tmp_path / "manifests" / "tokenize.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "stage": "tokenize",
        "stage_execution_id": execution_id,
        "status": "completed",
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    (root / "job.json").write_text(json.dumps({
        "stage_execution_id": execution_id,
        "stage": "tokenize",
        "run_root": str(tmp_path),
        "attempt_root": str(root),
        "tmux_session": "nt-finished-tokenize",
    }), encoding="utf-8")
    (root / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")

    inspected = inspect_stage_job(tmp_path, "tokenize", execution_id)

    assert inspected["status"] == "completed"
