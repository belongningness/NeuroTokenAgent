import pytest

from agent_pipeline.workspace_projection import build_workspace_snapshot, with_workspace


def state_for(mode: str, **updates):
    state = {
        "mode": mode,
        "completed_stages": [],
        "pending_stages": [],
        "stage_reports": [],
        "artifacts": [],
    }
    state.update(updates)
    return state


def test_workspace_snapshot_projects_running_pipeline():
    snapshot = build_workspace_snapshot(state_for(
        "running",
        intent="execute",
        workflow_id="wf_1",
        run_id="run_1",
        report_root="/output/reports",
        completed_stages=["preflight"],
        pending_stages=["package", "tokenize"],
        current_stage="package",
        stage_reports=[{"stage": "preflight", "duration_seconds": 1.25}],
    ))
    assert snapshot.is_working
    assert snapshot.run.run_id == "run_1"
    assert [stage.status for stage in snapshot.stages] == ["completed", "running", "pending"]
    assert snapshot.stages[0].duration_seconds == 1.25


@pytest.mark.parametrize(
    ("mode", "approval_kind", "is_working"),
    [
        ("idle", None, False),
        ("awaiting_plan_approval", "plan_approval", False),
        ("paused", None, False),
        ("completed", None, False),
        ("cancelled", None, False),
        ("failed", None, False),
    ],
)
def test_workspace_modes(mode, approval_kind, is_working):
    snapshot = build_workspace_snapshot(state_for(mode, current_plan={"stages": ["preflight"]}))
    assert snapshot.mode == mode
    assert snapshot.is_working is is_working
    assert (snapshot.approval.kind if snapshot.approval else None) == approval_kind


def test_workspace_snapshot_projects_adapter_review_without_diff():
    snapshot = build_workspace_snapshot(state_for(
        "awaiting_patch_approval",
        pending_adapter={"name": "eeg_h5", "request_id": "adapter_1", "diff": "large diff"},
    ))
    assert snapshot.approval is not None
    assert snapshot.approval.kind == "adapter_patch_approval"
    assert snapshot.approval.request_id == "adapter_1"
    assert "diff" not in snapshot.model_dump_json()


def test_workspace_snapshot_projects_detached_attempt_without_private_job_spec():
    snapshot = build_workspace_snapshot(state_for(
        "detached",
        pending_stages=["package"],
        current_stage="package",
        external_stage_job={
            "stage_execution_id": "stage_package_123",
            "stage": "package",
            "status": "detached",
            "tmux_session": "nt-package-123",
            "stdout_log": "/run/stdout.log",
            "stderr_log": "/run/stderr.log",
            "log_tail": "writing tar",
            "updated_at": "2026-08-17T09:00:00Z",
            "reconnectable": True,
        },
    ))
    assert snapshot.is_working
    assert snapshot.external_stage_job is not None
    assert snapshot.external_stage_job.stage_execution_id == "stage_package_123"
    assert snapshot.external_stage_job.log_tail == "writing tar"


def test_workspace_snapshot_projects_linked_gate_state():
    snapshot = build_workspace_snapshot(state_for(
        "awaiting_validation_gate",
        active_phase="gate",
        workflow_id="wf_1",
        run_id="run_validation",
        linked_runs={
            "validation": {"run_id": "run_validation", "run_root": "/runs/validation"},
            "full": {"run_id": "run_full", "run_root": "/runs/full"},
        },
        gate_id="gate_1",
        gate_status="passed",
        validation_verdict="pass",
        validation_report={"verdict": "pass", "metrics": {"pearson": 0.9}},
    ))
    assert snapshot.run.active_phase == "gate"
    assert snapshot.run.linked_run_ids == {"validation": "run_validation", "full": "run_full"}
    assert snapshot.approval is not None
    assert snapshot.approval.kind == "validation_gate"
    assert snapshot.validation_gate is not None
    assert snapshot.validation_gate.verdict == "pass"


def test_workspace_snapshot_projects_failure_to_current_stage():
    snapshot = build_workspace_snapshot(state_for(
        "failed",
        pending_stages=["tokenize"],
        current_stage="tokenize",
        last_error={"code": "stage_failed", "message": "boom", "stage": "tokenize"},
    ))
    assert snapshot.error is not None
    assert snapshot.stages[0].status == "failed"


def test_workspace_snapshot_filters_foreign_and_duplicate_artifacts():
    snapshot = build_workspace_snapshot(state_for(
        "completed",
        owner="owner_1",
        thread_id="thread_1",
        artifacts=[
            {"artifact_id": "a1", "kind": "report", "summary": "ok", "owner": "owner_1", "thread_id": "thread_1"},
            {"artifact_id": "a1", "kind": "report", "summary": "duplicate"},
            {"artifact_id": "a2", "kind": "report", "summary": "foreign", "owner": "owner_2", "thread_id": "thread_1"},
        ],
    ))
    assert [artifact.artifact_id for artifact in snapshot.artifacts] == ["a1"]


def test_workspace_snapshot_projects_clickable_file_metadata():
    snapshot = build_workspace_snapshot(state_for(
        "reviewing_stage",
        owner="owner_1",
        thread_id="thread_1",
        artifacts=[{
            "artifact_id": "docs-readme",
            "kind": "stage_document",
            "stage": "docs",
            "summary": "docs output: README.md",
            "file_name": "README.md",
            "storage_uri": "/output/neurotoken/README.md",
            "media_type": "text/markdown",
        }],
    ))
    assert snapshot.artifacts[0].file_name == "README.md"
    assert snapshot.artifacts[0].media_type == "text/markdown"


def test_workspace_snapshot_normalizes_unknown_mode_paths_and_summary():
    snapshot = build_workspace_snapshot(state_for(
        "surprise",
        report_root="bad\x00path",
        current_plan={
            "stages": ["preflight", "unknown"],
            "target_h5": "/input/data.h5",
            "report_root": "/output",
            "source_adapter": "eeg_h5",
            "secret": "not projected",
        },
        artifacts=[{"artifact_id": "a1", "kind": "report", "summary": "x" * 800}],
    ))
    assert snapshot.mode == "idle"
    assert snapshot.run.report_root is None
    assert snapshot.plan is not None
    assert snapshot.plan.stages == ["preflight"]
    assert snapshot.plan.adapter == "eeg_h5"
    assert len(snapshot.artifacts[0].summary) == 500


def test_workspace_snapshot_projects_linked_phase_runs_gate_and_duplicate_stage_names():
    snapshot = build_workspace_snapshot(state_for(
        "awaiting_validation_gate",
        active_phase="gate",
        workflow_id="wf_linked",
        run_id="run_validation",
        current_plan={
            "stages": ["preflight", "package", "tokenize", "reconstruct"],
            "execution_strategy": "linked",
            "validation_target_h5": "/input/session-01.h5",
            "full_input_root": "/input",
            "validation_stages": ["preflight", "package", "tokenize", "reconstruct"],
            "full_stages": ["preflight", "package", "tokenize"],
        },
        linked_runs={
            "validation": {"workflow_id": "wf_linked", "run_id": "run_validation", "run_root": "/out/validation"},
            "full": {"workflow_id": "wf_linked", "run_id": "run_full", "run_root": "/out/full"},
        },
        completed_stages=["preflight", "package", "tokenize", "reconstruct"],
        stage_reports=[
            {"stage": "tokenize", "status": "ok", "scope": "validation", "run_id": "run_validation", "stage_id": "validation:run_validation:tokenize"},
            {"stage": "tokenize", "status": "ok", "scope": "full", "run_id": "run_full", "stage_id": "full:run_full:tokenize"},
        ],
        gate_id="gate_1",
        gate_status="passed",
        validation_verdict="pass",
        validation_report={"metrics": {"pearson": 0.9}},
    ))

    assert snapshot.active_phase == "gate"
    assert snapshot.validation_target_h5 == "/input/session-01.h5"
    assert {run["scope"] for run in snapshot.linked_runs} == {"validation", "full"}
    tokenize_stages = [stage for stage in snapshot.stages if stage.name == "tokenize"]
    assert {stage.stage_id for stage in tokenize_stages} == {
        "validation:run_validation:tokenize", "full:run_full:tokenize"
    }
    assert snapshot.validation_gate is not None
    assert snapshot.validation_gate.gate_id == "gate_1"


def test_with_workspace_projects_merged_update():
    update = with_workspace(
        state_for("idle", pending_stages=["preflight"]),
        {"mode": "awaiting_plan_approval", "current_plan": {"stages": ["preflight"]}},
    )
    assert update["workspace"]["mode"] == "awaiting_plan_approval"
    assert update["workspace"]["approval"]["kind"] == "plan_approval"
