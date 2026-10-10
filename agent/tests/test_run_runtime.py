import fcntl
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_pipeline.config import load_config
from agent_pipeline.dynamic_config import build_effective_config, create_dynamic_config
from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.run_runtime import (
    _execute_stage_unlocked,
    archive_failed_stage_attempt,
    exclusive_lock,
    load_completed_stage,
    materialize_child_run,
    materialize_run,
    safe_run_root,
    stage_manifest_path,
)
from agent_pipeline.tools import write_json


def test_effective_config_preview_matches_materialized_config(tmp_path: Path):
    base_config = tmp_path / "base.toml"
    base_config.write_text(
        """
[dataset]
name = "Example"
modality = "meg"
task_type = "MEG"

[source]
adapter = "meg_h5"

[chunking]
strategy = "rest"
split_points = 30000

[paths]
h5_root = "/input"
""".strip(),
        encoding="utf-8",
    )
    target = tmp_path / "SMN_MEG-sub-02.h5"
    output_root = tmp_path / "outputs"
    arguments = {
        "base_config_path": base_config,
        "output_root": output_root,
        "target_h5": target,
        "source_adapter": "custom_smn",
        "config_overrides": {
            "chunking": {"strategy": "event"},
            "runtime": {"device": "cpu"},
        },
    }

    preview = build_effective_config(**arguments)
    materialized_path = create_dynamic_config(**arguments)
    materialized = load_config(materialized_path)
    materialized.pop("__config_dir__")
    materialized.pop("__config_path__")

    assert materialized == preview
    assert preview["source"]["adapter"] == "custom_smn"
    assert preview["chunking"]["strategy"] == "event"
    assert preview["runtime"]["device"] == "cpu"


def test_run_roots_are_isolated(tmp_path: Path):
    first = safe_run_root(tmp_path, "wf_a", "run_a")
    second = safe_run_root(tmp_path, "wf_a", "run_b")
    third = safe_run_root(tmp_path, "wf_b", "run_a")
    assert len({first, second, third}) == 3
    assert first == tmp_path.resolve() / "runs" / "wf_a" / "run_a"


def test_linked_child_runs_are_isolated_and_selection_is_auditable(tmp_path: Path):
    base_config = tmp_path / "base.toml"
    base_config.write_text('[dataset]\nname = "Example"\n[paths]\nh5_root = "/input"\n', encoding="utf-8")
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    target = input_root / "sample.h5"
    target.touch()
    output_root = tmp_path / "outputs"
    linked = PipelineRequest(
        config_path=str(base_config),
        stages=["preflight"],
        execution_strategy="linked",
        validation_target_h5=str(target),
        full_input_root=str(input_root),
        report_root=str(output_root),
    )
    validation_request = linked.request_for_phase("validation")
    validation = materialize_run(
        validation_request,
        workflow_id="wf_linked",
        run_id="run_validation",
        phase="validation",
    )
    full = materialize_child_run(
        linked.request_for_phase("full"),
        workflow_id="wf_linked",
        parent_run_id=validation["run_id"],
        phase="full",
        run_id="run_full",
    )

    assert validation["run_root"] != full["run_root"]
    assert validation["config_path"] != full["config_path"]
    validation_manifest = json.loads((Path(validation["run_root"]) / "run_manifest.json").read_text(encoding="utf-8"))
    full_manifest = json.loads((Path(full["run_root"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert validation_manifest["selection"]["target_h5"] == str(target)
    assert full_manifest["selection"]["input_root"] == str(input_root)
    assert full_manifest["parent_run_id"] == "run_validation"


def test_materialize_run_is_idempotent_for_same_identity(tmp_path: Path):
    base_config = tmp_path / "base.toml"
    base_config.write_text('[dataset]\nname = "Example"\n[paths]\nh5_root = "/input"\n', encoding="utf-8")
    target = tmp_path / "sample.h5"
    request = PipelineRequest(
        config_path=str(base_config),
        stages=["preflight"],
        target_h5=str(target),
        report_root=str(tmp_path / "outputs"),
    )
    first = materialize_run(request, workflow_id="wf_same", run_id="run_same")
    second = materialize_run(request, workflow_id="wf_same", run_id="run_same")
    assert second == first


def test_run_root_rejects_path_traversal_and_absolute_ids(tmp_path: Path):
    with pytest.raises(ValueError, match="workflow_id"):
        safe_run_root(tmp_path, "../../escape", "run_a")
    with pytest.raises(ValueError, match="run_id"):
        safe_run_root(tmp_path, "wf_a", "/tmp/escape")


def test_exclusive_lock_rejects_an_active_owner(tmp_path: Path):
    lock_path = tmp_path / "repository.lock"
    with lock_path.open("a+", encoding="utf-8") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Run is locked"):
            with exclusive_lock(lock_path):
                raise AssertionError("active lock must not be acquired")


def test_exclusive_lock_is_reusable_after_failure(tmp_path: Path):
    lock_path = tmp_path / "repository.lock"
    with pytest.raises(ValueError, match="simulated failure"):
        with exclusive_lock(lock_path):
            raise ValueError("simulated failure")
    with exclusive_lock(lock_path):
        assert "pid=" in lock_path.read_text(encoding="utf-8")


def test_completed_stage_requires_report_checksum(tmp_path: Path):
    report = tmp_path / "reports" / "preflight_report.json"
    write_json(report, {"stage": "preflight", "status": "ok"})
    import hashlib
    manifest = stage_manifest_path(tmp_path, "preflight")
    write_json(manifest, {
        "status": "completed",
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    })
    assert load_completed_stage(tmp_path, "preflight")["status"] == "ok"
    report.write_text(json.dumps({"stage": "preflight", "status": "changed"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_completed_stage(tmp_path, "preflight")


def test_incomplete_manifest_blocks_unsafe_resume(tmp_path: Path):
    write_json(stage_manifest_path(tmp_path, "tokenize"), {"status": "started"})
    with pytest.raises(RuntimeError, match="incomplete manifest"):
        load_completed_stage(tmp_path, "tokenize")


def test_started_stage_attempt_requires_explicit_interrupted_retry(tmp_path: Path):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    write_json(stage_manifest_path(tmp_path, "package"), {
        "stage": "package",
        "stage_execution_id": "stage_package_interrupted",
        "status": "started",
    })

    with pytest.raises(RuntimeError, match="manifest status='started'"):
        archive_failed_stage_attempt(tmp_path, report_root, "package")

    archive_failed_stage_attempt(tmp_path, report_root, "package", allow_interrupted=True)

    assert (tmp_path / "attempts" / "package" / "stage_package_interrupted" / "manifest.json").is_file()
    assert not stage_manifest_path(tmp_path, "package").exists()


def test_failed_stage_attempt_is_archived_before_retry(tmp_path: Path):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    manifest_path = stage_manifest_path(tmp_path, "tokenize")
    write_json(manifest_path, {
        "stage": "tokenize",
        "stage_execution_id": "stage_tokenize_old",
        "status": "failed",
    })
    report_path = report_root / "tokenize_report.json"
    write_json(report_path, {"stage": "tokenize", "status": "failed"})

    archive_failed_stage_attempt(tmp_path, report_root, "tokenize")

    archive = tmp_path / "attempts" / "tokenize" / "stage_tokenize_old"
    assert not manifest_path.exists()
    assert not report_path.exists()
    assert json.loads((archive / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads((archive / "tokenize_report.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_legacy_completed_manifest_with_failed_report_is_retryable(tmp_path: Path):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    write_json(stage_manifest_path(tmp_path, "tokenize"), {
        "stage": "tokenize",
        "stage_execution_id": "stage_tokenize_legacy",
        "status": "completed",
    })
    write_json(report_root / "tokenize_report.json", {"stage": "tokenize", "status": "failed"})

    archive_failed_stage_attempt(tmp_path, report_root, "tokenize")

    archive = tmp_path / "attempts" / "tokenize" / "stage_tokenize_legacy"
    assert (archive / "manifest.json").exists()
    assert (archive / "tokenize_report.json").exists()


def test_completed_manifest_with_successful_report_cannot_be_retried(tmp_path: Path):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    write_json(stage_manifest_path(tmp_path, "tokenize"), {
        "stage": "tokenize",
        "stage_execution_id": "stage_tokenize_ok",
        "status": "completed",
    })
    write_json(report_root / "tokenize_report.json", {"stage": "tokenize", "status": "ok"})

    with pytest.raises(RuntimeError, match="manifest status='completed', report status='ok'"):
        archive_failed_stage_attempt(tmp_path, report_root, "tokenize")


def test_retry_executes_after_archiving_failed_attempt(tmp_path: Path):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    write_json(stage_manifest_path(tmp_path, "tokenize"), {
        "stage": "tokenize",
        "stage_execution_id": "stage_tokenize_old",
        "status": "failed",
    })
    write_json(report_root / "tokenize_report.json", {"stage": "tokenize", "status": "failed"})
    request = PipelineRequest(
        config_path=str(tmp_path / "config.toml"),
        stages=["tokenize"],
        target_h5=str(tmp_path / "sample.h5"),
    )

    with (
        patch("agent_pipeline.run_runtime.load_config_tool", return_value={"config": {}}),
        patch(
            "agent_pipeline.run_runtime.run_reference_stage_tool",
            return_value={"stage": "tokenize", "status": "ok", "returncode": 0},
        ) as worker,
    ):
        report = _execute_stage_unlocked(
            request=request,
            stage="tokenize",
            run_root=str(tmp_path),
            config_path=request.config_path,
            report_root=str(report_root),
            reference_scripts_dir=str(tmp_path),
            python="python",
            retry_failed=True,
        )

    assert report["status"] == "ok"
    assert worker.call_count == 1
    manifest = json.loads(stage_manifest_path(tmp_path, "tokenize").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["stage_execution_id"] != "stage_tokenize_old"
    assert (tmp_path / "attempts" / "tokenize" / "stage_tokenize_old" / "manifest.json").exists()


def test_structured_stage_failure_is_persisted_as_failed(tmp_path: Path):
    report_root = tmp_path / "reports"
    failed_report = {
        "stage": "tokenize",
        "status": "failed",
        "returncode": 1,
        "stderr": "CUDA out of memory",
    }
    request = PipelineRequest(
        config_path=str(tmp_path / "config.toml"),
        stages=["tokenize"],
        target_h5=str(tmp_path / "sample.h5"),
    )

    with (
        patch("agent_pipeline.run_runtime.load_config_tool", return_value={"config": {}}),
        patch("agent_pipeline.run_runtime.run_reference_stage_tool", return_value=failed_report),
    ):
        returned = _execute_stage_unlocked(
            request=request,
            stage="tokenize",
            run_root=str(tmp_path),
            config_path=request.config_path,
            report_root=str(report_root),
            reference_scripts_dir=str(tmp_path),
            python="python",
        )

    manifest = json.loads(stage_manifest_path(tmp_path, "tokenize").read_text(encoding="utf-8"))
    persisted_report = json.loads((report_root / "tokenize_report.json").read_text(encoding="utf-8"))
    assert returned["status"] == "failed"
    assert persisted_report["status"] == "failed"
    assert manifest["status"] == "failed"
    assert "failed_at" in manifest
    assert "completed_at" not in manifest
    assert manifest["report_path"] == str(report_root / "tokenize_report.json")
    with pytest.raises(RuntimeError, match="incomplete manifest"):
        load_completed_stage(tmp_path, "tokenize")
