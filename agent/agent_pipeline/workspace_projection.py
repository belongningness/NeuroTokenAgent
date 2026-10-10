"""Stable workspace view projected from internal AgentState fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_pipeline.pipeline_stages import STAGE_ORDER
from agent_pipeline.server_state import (
    AgentError,
    WorkspaceApprovalRef,
    WorkspaceArtifactRef,
    WorkspaceExternalStageJob,
    WorkspacePlanSummary,
    WorkspaceRun,
    WorkspaceSnapshot,
    WorkspaceValidationGate,
    WorkspaceStage,
)

KNOWN_MODES = {
    "idle", "planned", "awaiting_plan_approval", "approved", "running", "reviewing_stage",
    "paused", "completed", "cancelled", "failed", "authoring_adapter",
    "ready_for_coding", "awaiting_patch_approval", "adapter_question", "adapter_ready",
    "adapter_revision_required", "detached", "reconnecting", "orphaned", "awaiting_validation_gate",
}
WORKING_MODES = {"approved", "running", "detached", "reconnecting", "authoring_adapter"}


def _stage_duration(
    reports: list[dict[str, Any]], stage: str, run_id: str | None = None
) -> float | None:
    for report in reversed(reports):
        if report.get("stage") != stage:
            continue
        if run_id is not None and report.get("run_id") not in {None, run_id}:
            continue
        if isinstance(report.get("duration_seconds"), (int, float)):
            return float(report["duration_seconds"])
    return None


def _approval(state: Mapping[str, Any], mode: str) -> WorkspaceApprovalRef | None:
    if mode == "awaiting_plan_approval":
        return WorkspaceApprovalRef(kind="plan_approval", title="Confirm pipeline plan")
    if mode == "awaiting_validation_gate":
        return WorkspaceApprovalRef(
            kind="validation_gate",
            title="Review validation before full run",
            stage="reconstruct",
            request_id=state.get("gate_id"),
        )
    if mode == "reviewing_stage":
        stage = state.get("current_stage")
        return WorkspaceApprovalRef(kind="stage_review", title=f"Review {stage or 'stage'}", stage=stage)
    if mode == "awaiting_patch_approval":
        adapter = state.get("pending_adapter") or {}
        name = adapter.get("name") or "adapter"
        return WorkspaceApprovalRef(
            kind="adapter_patch_approval",
            title=f"Review adapter: {name}",
            request_id=adapter.get("request_id"),
        )
    return None


def _safe_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or "\x00" in value:
        return None
    return value[:4096]


def _plan(state: Mapping[str, Any]) -> WorkspacePlanSummary | None:
    raw = state.get("current_plan") or state.get("approved_plan")
    if not isinstance(raw, Mapping):
        return None
    stages = [stage for stage in raw.get("stages") or [] if isinstance(stage, str) and stage in STAGE_ORDER]
    execution_strategy = "linked" if raw.get("execution_strategy") == "linked" else "single"
    return WorkspacePlanSummary(
        stages=stages,
        input_path=_safe_path(
            raw.get("validation_target_h5") or raw.get("target_h5") or raw.get("full_input_root") or raw.get("input_root")
        ),
        output_path=_safe_path(raw.get("report_root")),
        adapter=str(raw["source_adapter"])[:120] if raw.get("source_adapter") else None,
        execution_strategy=execution_strategy,
        validation_target_h5=_safe_path(raw.get("validation_target_h5")),
        full_input_root=_safe_path(raw.get("full_input_root")),
        validation_stages=[stage for stage in raw.get("validation_stages") or [] if stage in STAGE_ORDER],
        full_stages=[stage for stage in raw.get("full_stages") or [] if stage in STAGE_ORDER],
    )


def _artifacts(state: Mapping[str, Any]) -> list[WorkspaceArtifactRef]:
    owner = state.get("owner")
    thread_id = state.get("thread_id")
    seen: set[str] = set()
    result: list[WorkspaceArtifactRef] = []
    for raw in state.get("artifacts") or []:
        if not isinstance(raw, Mapping):
            continue
        if owner and raw.get("owner") not in {None, owner}:
            continue
        if thread_id and raw.get("thread_id") not in {None, thread_id}:
            continue
        artifact_id = str(raw.get("artifact_id") or "").strip()
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        result.append(
            WorkspaceArtifactRef(
                artifact_id=artifact_id,
                kind=str(raw.get("kind") or "artifact"),
                stage=raw.get("stage"),
                summary=str(raw.get("summary") or "Generated artifact")[:500],
                file_name=_safe_path(raw.get("file_name")),
                storage_uri=_safe_path(raw.get("storage_uri")),
                media_type=str(raw.get("media_type") or "application/json"),
                created_at=raw.get("created_at"),
            )
        )
    return result


def build_workspace_snapshot(state: Mapping[str, Any]) -> WorkspaceSnapshot:
    raw_mode = str(state.get("mode") or "idle")
    mode = raw_mode if raw_mode in KNOWN_MODES else "idle"
    completed = set(state.get("completed_stages") or [])
    pending = set(state.get("pending_stages") or [])
    current = state.get("current_stage")
    error = AgentError.model_validate(state["last_error"]) if state.get("last_error") else None
    reports = [report for report in state.get("stage_reports") or [] if isinstance(report, dict)]

    involved = completed | pending | ({current} if current else set())
    active_phase = str(state.get("active_phase") or "single")
    active_scope = "validation" if active_phase == "gate" else active_phase
    current_run_id = str(state.get("run_id")) if state.get("run_id") else None
    stage_map: dict[str, WorkspaceStage] = {}
    for report in reports:
        scope = report.get("scope")
        run_id = str(report.get("run_id")) if report.get("run_id") else None
        stage = str(report.get("stage") or "")
        if scope not in {"validation", "full"} or stage not in STAGE_ORDER:
            continue
        stage_id = str(report.get("stage_id") or f"{scope}:{run_id or 'unknown'}:{stage}")
        stage_map[stage_id] = WorkspaceStage(
            name=stage,
            status="failed" if report.get("status") == "failed" else "completed",
            duration_seconds=_stage_duration(reports, stage, run_id),
            stage_id=stage_id,
            scope=scope,
            run_id=run_id,
        )
    for stage in STAGE_ORDER:
        if involved and stage not in involved:
            continue
        status = "completed" if stage in completed else "running" if stage == current else "pending"
        external = state.get("external_stage_job") or {}
        if stage == current and external.get("status") in {"failed", "cancelled", "orphaned"}:
            status = "failed"
        if error and error.stage == stage:
            status = "failed"
        scope = active_scope if active_scope in {"single", "validation", "full"} else "single"
        stage_id = f"{scope}:{current_run_id or 'unmaterialized'}:{stage}"
        stage_map[stage_id] = WorkspaceStage(
            name=stage,
            status=status,
            duration_seconds=_stage_duration(reports, stage, current_run_id),
            stage_id=stage_id,
            scope=scope,
            run_id=current_run_id,
        )
    stages = list(stage_map.values())

    plan_summary = _plan(state)
    linked_runs = [
        {**dict(identity), "scope": str(phase)}
        for phase, identity in (state.get("linked_runs") or {}).items()
        if phase in {"validation", "full"} and isinstance(identity, Mapping)
    ]
    return WorkspaceSnapshot(
        mode=mode,
        intent=state.get("intent"),
        is_working=mode in WORKING_MODES,
        plan=plan_summary,
        active_phase=active_phase,
        validation_target_h5=plan_summary.validation_target_h5 if plan_summary else None,
        full_input_root=plan_summary.full_input_root if plan_summary else None,
        linked_runs=linked_runs,
        run=WorkspaceRun(
            workflow_id=state.get("workflow_id"),
            run_id=state.get("run_id"),
            report_root=_safe_path(state.get("report_root")),
            config_path=_safe_path(state.get("config_path")),
            active_phase=str(state.get("active_phase") or "single"),
            linked_run_ids={
                str(phase): str(identity.get("run_id"))
                for phase, identity in (state.get("linked_runs") or {}).items()
                if isinstance(identity, Mapping) and identity.get("run_id")
            },
        ),
        stages=stages,
        artifacts=_artifacts(state),
        external_stage_job=(
            WorkspaceExternalStageJob.model_validate(state["external_stage_job"])
            if state.get("external_stage_job") else None
        ),
        approval=_approval(state, mode),
        validation_gate=(
            WorkspaceValidationGate(
                gate_id=state.get("gate_id"),
                status=state.get("gate_status"),
                verdict=state.get("validation_verdict"),
                report=state.get("validation_report"),
            )
            if state.get("gate_id") or state.get("validation_report") else None
        ),
        error=error,
    )


def with_workspace(state: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**state, **update}
    return {**update, "workspace": build_workspace_snapshot(merged).model_dump(mode="json")}
