"""Serializable state and payload models for the NeuroToken Agent Server."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class ArtifactEnvelope(BaseModel):
    schema_version: str = "1.0"
    artifact_id: str
    kind: str
    owner: str
    thread_id: str
    workflow_id: str | None = None
    run_id: str | None = None
    stage: str | None = None
    summary: str
    file_name: str | None = None
    storage_uri: str | None = None
    media_type: str = "application/json"
    size: int | None = None
    created_at: str
    expires_at: str | None = None
    payload: Any = Field(default_factory=dict)


class AgentError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    stage: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    action: Literal[
        "approve",
        "reject",
        "edit",
        "continue",
        "pause",
        "cancel",
        "apply",
        "apply_and_commit",
        "ask",
        "refine",
        "start_full_run",
        "rerun_smoke",
    ]
    plan: dict[str, Any] | None = None
    message: str | None = None
    task_id: str | None = None
    revision: int | None = Field(default=None, ge=1)
    commit_message: str | None = None
    gate_id: str | None = None


class WorkspaceStage(BaseModel):
    name: str
    status: Literal["pending", "running", "completed", "failed"]
    duration_seconds: float | None = None
    stage_id: str | None = None
    scope: Literal["single", "validation", "full"] | None = None
    run_id: str | None = None


class WorkspaceArtifactRef(BaseModel):
    artifact_id: str
    kind: str
    stage: str | None = None
    summary: str
    file_name: str | None = None
    storage_uri: str | None = None
    media_type: str = "application/json"
    created_at: str | None = None


class WorkspaceRun(BaseModel):
    workflow_id: str | None = None
    run_id: str | None = None
    report_root: str | None = None
    config_path: str | None = None
    active_phase: Literal["single", "validation", "gate", "full"] = "single"
    linked_run_ids: dict[str, str] = Field(default_factory=dict)


class WorkspaceValidationGate(BaseModel):
    gate_id: str | None = None
    status: Literal["pending", "passed", "failed", "started", "cancelled"] | None = None
    verdict: Literal["pass", "fail"] | None = None
    report: dict[str, Any] | None = None


class WorkspaceExternalStageJob(BaseModel):
    stage_execution_id: str
    stage: str
    workflow_id: str | None = None
    run_id: str | None = None
    phase: Literal["single", "validation", "full"] | None = None
    status: Literal[
        "submitted", "running", "detached", "reconnecting", "completed", "failed",
        "cancelling", "cancelled", "orphaned",
    ]
    tmux_session: str | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    log_tail: str = ""
    updated_at: str | None = None
    reconnectable: bool = False


class WorkspaceApprovalRef(BaseModel):
    kind: str
    title: str
    stage: str | None = None
    request_id: str | None = None


class WorkspacePlanSummary(BaseModel):
    stages: list[str] = Field(default_factory=list)
    input_path: str | None = None
    output_path: str | None = None
    adapter: str | None = None
    execution_strategy: Literal["single", "linked"] = "single"
    validation_target_h5: str | None = None
    full_input_root: str | None = None
    validation_stages: list[str] = Field(default_factory=list)
    full_stages: list[str] = Field(default_factory=list)


class WorkspaceSnapshot(BaseModel):
    schema_version: str = "1.0"
    mode: str = "idle"
    intent: str | None = None
    is_working: bool = False
    plan: WorkspacePlanSummary | None = None
    active_phase: Literal["single", "validation", "gate", "full"] = "single"
    validation_target_h5: str | None = None
    full_input_root: str | None = None
    linked_runs: list[dict[str, Any]] = Field(default_factory=list)
    run: WorkspaceRun = Field(default_factory=WorkspaceRun)
    stages: list[WorkspaceStage] = Field(default_factory=list)
    artifacts: list[WorkspaceArtifactRef] = Field(default_factory=list)
    external_stage_job: WorkspaceExternalStageJob | None = None
    approval: WorkspaceApprovalRef | None = None
    validation_gate: WorkspaceValidationGate | None = None
    error: AgentError | None = None


class AgentState(MessagesState):
    owner: str
    thread_id: str
    mode: str
    intent: str | None
    requirements: dict[str, Any]
    other_requirements: list[str]
    current_plan: dict[str, Any] | None
    approved_plan: dict[str, Any] | None
    plan_revision: int
    workflow_id: str | None
    run_id: str | None
    run_root: str | None
    config_path: str | None
    report_root: str | None
    active_phase: str
    linked_runs: dict[str, dict[str, Any]]
    validation_report: dict[str, Any] | None
    validation_verdict: str | None
    gate_id: str | None
    gate_status: str | None
    completed_stages: list[str]
    pending_stages: list[str]
    current_stage: str | None
    current_tool_call_id: str | None
    retry_requested: bool
    retrying_stage: bool
    reconcile_requested: bool
    external_stage_job: dict[str, Any] | None
    stage_reports: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    pending_adapter: dict[str, Any] | None
    adapter_collaboration_action: str | None
    adapter_collaboration_message: str | None
    last_adapter: dict[str, Any] | None
    last_error: dict[str, Any] | None
    workspace: dict[str, Any]
