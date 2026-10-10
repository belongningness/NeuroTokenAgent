"""LangGraph Agent Server graph for NeuroToken Desktop."""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_pipeline.chat_services import (
    answer_adapter_question,
    answer_question,
    append_package_sample_summary,
    build_pipeline_request,
    classify_intent,
    merge_requirements,
    parse_adapter_request,
    summarize_stage,
)
from agent_pipeline.adapter_authoring.h5_inspector import format_h5_summary, inspect_h5
from agent_pipeline.adapter_authoring.models import (
    AdapterAuthoringResult,
    AdapterCandidateSubmission,
    AdapterCodingDescriptor,
    AdapterSpec,
    CodingAgentResult,
)
from agent_pipeline.adapter_authoring.prompts import build_adapter_prompt, build_refinement_prompt
from agent_pipeline.adapter_authoring.review import review_adapter_candidate, review_feedback, run_candidate_preflight
from agent_pipeline.adapter_authoring.task_runtime import AdapterCodingTaskRuntime
from agent_pipeline.adapter_authoring.workflow import apply_authoring_result, authoring_history_dir
from agent_pipeline.dynamic_config import build_effective_config
from agent_pipeline.llm_runtime import (
    DEFAULT_CONFIG,
    DEFAULT_LLM_ENV_FILE,
    DEFAULT_MODEL,
    configure_llm_env,
    load_llm_env_config,
    make_llm,
)
from agent_pipeline.package import ensure_package_report_previews
from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.pipeline_planning import explicitly_removed_stages, refine_request, select_config
from agent_pipeline.pipeline_stages import DEFAULT_REFERENCE_SCRIPTS, resolve_stage_dependencies
from agent_pipeline.run_runtime import (
    TERMINAL_ATTEMPT_STATUSES,
    TMUX_STAGES,
    cancel_stage_job,
    execute_stage,
    materialize_child_run,
    materialize_run,
    new_id,
    reconcile_stage_job,
    submit_tmux_stage,
    atomic_write_json,
    utc_now,
)
from agent_pipeline.server_policy import select_linked_validation_target, validate_adapter_source, validate_pipeline_paths
from agent_pipeline.server_state import AgentState, ArtifactEnvelope, ApprovalDecision
from agent_pipeline.coding.worktree import changed_worktree_paths, diff_worktree, paths_allowed
from agent_pipeline.workspace_projection import with_workspace


ROOT = Path(__file__).resolve().parents[1]


def runtime_config() -> dict[str, Any]:
    config = load_llm_env_config(os.environ.get("NEUROTOKEN_LLM_ENV") or DEFAULT_LLM_ENV_FILE)
    model = os.environ.get("NEUROTOKEN_MODEL") or config.get("model") or DEFAULT_MODEL
    api_key = os.environ.get("OPENAI_API_KEY") or config.get("api_key")
    base_url = os.environ.get("OPENAI_BASE_URL") or config.get("base_url")
    proxy = os.environ.get("NEUROTOKEN_LLM_PROXY") or config.get("proxy") or os.environ.get("HTTPS_PROXY")
    configure_llm_env(api_key, base_url, proxy)
    return {"model": model, "default_config": os.environ.get("NEUROTOKEN_DEFAULT_CONFIG", DEFAULT_CONFIG)}


def llm():
    return make_llm(runtime_config()["model"])


def adapter_task_runtime() -> AdapterCodingTaskRuntime:
    registry = os.environ.get("NEUROTOKEN_ADAPTER_TASK_REGISTRY")
    registry_path = Path(registry).expanduser() if registry else authoring_history_dir(ROOT) / "tasks.json"
    worktree_root = os.environ.get("NEUROTOKEN_ADAPTER_WORKTREE_ROOT", "/tmp/neurotoken-adapter-worktrees")
    return AdapterCodingTaskRuntime(ROOT, registry_path, worktree_root)


def latest_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) == "human":
            content = message.content
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def initialize(state: AgentState, config) -> dict[str, Any]:
    configurable = config.get("configurable", {}) if config else {}
    metadata = config.get("metadata", {}) if config else {}
    owner = state.get("owner") or configurable.get("owner") or metadata.get("owner")
    if os.environ.get("NEUROTOKEN_REQUIRE_AUTH") == "1" and not owner:
        raise PermissionError("Authenticated owner is required.")
    return {
        "owner": str(owner or "local-user"),
        "thread_id": state.get("thread_id") or str(configurable.get("thread_id") or metadata.get("thread_id") or "local-thread"),
        "mode": state.get("mode") or "idle",
        "requirements": state.get("requirements") or {},
        "other_requirements": state.get("other_requirements") or [],
        "plan_revision": state.get("plan_revision") or 0,
        "completed_stages": state.get("completed_stages") or [],
        "pending_stages": state.get("pending_stages") or [],
        "stage_reports": state.get("stage_reports") or [],
        "artifacts": state.get("artifacts") or [],
        "retry_requested": bool(state.get("retry_requested")),
        "retrying_stage": False,
        "reconcile_requested": bool(state.get("reconcile_requested")),
        "external_stage_job": state.get("external_stage_job"),
        "active_phase": state.get("active_phase") or "single",
        "linked_runs": state.get("linked_runs") or {},
        "validation_report": state.get("validation_report"),
        "validation_verdict": state.get("validation_verdict"),
        "gate_id": state.get("gate_id"),
        "gate_status": state.get("gate_status"),
    }


def retry_failed_stage(state: AgentState) -> dict[str, Any]:
    error = state.get("last_error") or {}
    stage = state.get("current_stage")
    active_identity = (state.get("linked_runs") or {}).get(state.get("active_phase") or "single")
    if active_identity and active_identity.get("run_root") != state.get("run_root"):
        raise RuntimeError("Active phase run identity does not match current run paths.")
    if state.get("mode") != "failed" or not error.get("retryable") or not stage:
        return {
            "retry_requested": False,
            "retrying_stage": False,
            "messages": [AIMessage(content="当前没有可重试的失败阶段。")],
        }
    if error.get("stage") not in {None, stage}:
        raise RuntimeError(
            f"Retry stage mismatch: current_stage={stage!r}, error.stage={error.get('stage')!r}"
        )
    call_id = f"tool_{uuid.uuid4().hex[:12]}"
    tool_call = {
        "name": f"run_{stage}",
        "args": {"stage": stage, "run_id": state.get("run_id"), "retry": True},
        "id": call_id,
        "type": "tool_call",
    }
    return {
        "retry_requested": False,
        "retrying_stage": True,
        "current_tool_call_id": call_id,
        "last_error": None,
        "mode": "running",
        "messages": [AIMessage(content=f"正在重新执行失败的 {stage} 阶段。", tool_calls=[tool_call])],
    }


def route_after_initialize(state: AgentState) -> str:
    user_text = latest_user_text(state).strip().lower()
    if state.get("external_stage_job") and user_text in {"cancel", "stop", "取消", "停止", "终止"}:
        return "cancel"
    job_status = (state.get("external_stage_job") or {}).get("status")
    if state.get("retry_requested") and job_status in TERMINAL_ATTEMPT_STATUSES:
        return "retry_failed_stage"
    if state.get("reconcile_requested") or state.get("external_stage_job"):
        return "reconcile_stage"
    if state.get("mode") == "awaiting_validation_gate" and state.get("gate_id"):
        return "validation_gate"
    if state.get("mode") == "paused" and state.get("active_phase") == "gate" and state.get("gate_id"):
        if user_text in {"continue", "继续", "resume", "恢复"}:
            return "validation_gate"
    if state.get("retry_requested"):
        return "retry_failed_stage"
    return "understand"


def understand(state: AgentState) -> dict[str, Any]:
    user_input = latest_user_text(state)
    intent = classify_intent(llm(), dict(state), user_input)
    requirements, extra = merge_requirements(
        state.get("requirements", {}), state.get("other_requirements", []), intent
    )
    return {
        "intent": intent.intent,
        "requirements": requirements,
        "other_requirements": extra,
        "last_error": None,
    }


def route_intent(state: AgentState) -> str:
    intent = state.get("intent") or "question"
    if intent == "continue_coding":
        return "continue_coding"
    if intent in {"plan", "execute", "edit_plan"}:
        return "build_plan"
    if intent == "continue" and state.get("mode") == "paused" and state.get("approved_plan"):
        return "select_stage"
    if intent == "sdk_cleanup":
        return "cleanup_sdk"
    if intent == "cancel":
        return "cancel"
    if intent == "report":
        return "report"
    if intent == "adapter_authoring":
        return "author_adapter"
    if intent == "exit":
        return "exit"
    return "question"


def question(state: AgentState) -> dict[str, Any]:
    text = answer_question(llm(), dict(state), latest_user_text(state))
    return {"messages": [AIMessage(content=text)], "mode": "idle"}


def build_plan(state: AgentState) -> dict[str, Any]:
    request, added = build_pipeline_request(
        llm(),
        latest_user_text(state),
        runtime_config()["default_config"],
        state.get("requirements", {}),
        state.get("other_requirements", []),
        state.get("last_adapter"),
    )
    request = select_linked_validation_target(request)
    summary = ["已生成 NeuroToken 执行计划：", f"- 阶段：{', '.join(request.stages)}"]
    if request.execution_strategy == "linked":
        summary.extend([
            f"- 单文件验证：{request.validation_target_h5}",
            f"- 全量输入目录：{request.full_input_root}",
            f"- 验证阶段：{', '.join(request.validation_stages)}",
            f"- 全量阶段：{', '.join(request.full_stages)}",
        ])
    elif request.target_h5:
        summary.append(f"- 输入文件：{request.target_h5}")
    elif request.input_root:
        summary.append(f"- 输入目录：{request.input_root}")
    summary.append(f"- 输出根目录：{request.report_root}")
    if added:
        summary.append(f"- 自动补齐前置阶段：{', '.join(added)}")
    if state.get("intent") == "plan":
        summary.append("当前只展示计划；你可以继续要求执行或修改。")
        mode = "planned"
    else:
        summary.append("请在审批卡中确认后执行。")
        mode = "awaiting_plan_approval"
    return {
        "current_plan": request.model_dump(mode="json"),
        "approved_plan": None,
        "plan_revision": (state.get("plan_revision") or 0) + 1,
        "workflow_id": None,
        "run_id": None,
        "run_root": None,
        "config_path": None,
        "report_root": None,
        "pending_stages": list(request.stages),
        "completed_stages": [],
        "current_stage": None,
        "current_tool_call_id": None,
        "stage_reports": [],
        "artifacts": [],
        "external_stage_job": None,
        "reconcile_requested": False,
        "active_phase": "validation" if request.execution_strategy == "linked" else "single",
        "linked_runs": {},
        "validation_report": None,
        "validation_verdict": None,
        "gate_id": None,
        "gate_status": None,
        "last_error": None,
        "mode": mode,
        "messages": [AIMessage(content="\n".join(summary))],
    }


def route_after_plan(state: AgentState) -> str:
    return "plan_approval" if state.get("intent") != "plan" else END


def plan_approval(state: AgentState) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "kind": "plan_approval",
        "owner": state["owner"],
        "thread_id": state["thread_id"],
        "plan": state["current_plan"],
        "plan_revision": state.get("plan_revision") or 1,
        "allowed_actions": ["approve", "refine", "reject"],
    }
    decision = ApprovalDecision.model_validate(interrupt(payload))
    if decision.action == "reject":
        return {
            "mode": "idle",
            "current_plan": None,
            "approved_plan": None,
            "pending_stages": [],
            "current_stage": None,
            "current_tool_call_id": None,
            "messages": [AIMessage(content="已取消执行计划。")],
        }
    if decision.action == "refine":
        instruction = (decision.message or "").strip()
        if not instruction:
            raise ValueError("A natural-language plan refinement instruction is required.")
        previous_plan = PipelineRequest.model_validate(state["current_plan"])
        plan = select_linked_validation_target(refine_request(llm(), previous_plan, instruction))
        requested_removals = explicitly_removed_stages(instruction)
        added = resolve_stage_dependencies(plan)
        removed = [stage for stage in previous_plan.stages if stage not in plan.stages]
        retained_dependencies = [stage for stage in requested_removals if stage in plan.stages]
        summary = f"已根据你的要求优化计划：{instruction}"
        if removed:
            summary += f"\n已移除阶段：{', '.join(removed)}"
        if retained_dependencies:
            summary += f"\n无法移除的依赖阶段：{', '.join(retained_dependencies)}"
        if added:
            summary += f"\n自动补齐前置阶段：{', '.join(added)}"
        if requested_removals and not removed:
            summary += "\n计划阶段没有变化；请检查仍保留阶段的依赖说明。"
        return {
            "current_plan": plan.model_dump(mode="json"),
            "approved_plan": None,
            "plan_revision": (state.get("plan_revision") or 1) + 1,
            "pending_stages": list(plan.stages),
            "mode": "awaiting_plan_approval",
            "messages": [AIMessage(content=f"{summary}\n请再次确认新版计划。")],
        }
    plan = select_linked_validation_target(PipelineRequest.model_validate(decision.plan or state["current_plan"]))
    return {
        "current_plan": plan.model_dump(mode="json"),
        "approved_plan": plan.model_dump(mode="json"),
        "pending_stages": list(plan.stages),
        "mode": "approved",
        "messages": [AIMessage(content="计划已批准，开始创建隔离运行目录。")],
    }


def route_after_approval(state: AgentState) -> str:
    if state.get("mode") == "awaiting_plan_approval" and state.get("current_plan"):
        return "plan_approval"
    return "materialize" if state.get("mode") == "approved" and state.get("approved_plan") else END


def materialize(state: AgentState) -> dict[str, Any]:
    plan = select_linked_validation_target(PipelineRequest.model_validate(state["approved_plan"]))
    validate_pipeline_paths(plan)
    phase = "validation" if plan.execution_strategy == "linked" else "single"
    phase_request = plan.request_for_phase(phase)
    resolve_stage_dependencies(phase_request)
    requested_workflow_id = new_id("wf") if phase == "validation" else None
    requested_run_id = new_id("run_validation") if phase == "validation" else None
    run = materialize_run(
        phase_request,
        workflow_id=requested_workflow_id,
        run_id=requested_run_id,
        phase=phase,
    )
    linked_runs = dict(state.get("linked_runs") or {})
    if phase == "validation":
        linked_runs["validation"] = {**run, "phase": "validation", "parent_run_id": None}
    return {
        **run,
        "active_phase": phase,
        "linked_runs": linked_runs,
        "pending_stages": list(phase_request.stages),
        "completed_stages": [],
        "mode": "running",
    }


def select_stage(state: AgentState) -> dict[str, Any]:
    pending = state.get("pending_stages", [])
    if not pending:
        return {"current_stage": None}
    stage = pending[0]
    call_id = f"tool_{uuid.uuid4().hex[:12]}"
    tool_call = {"name": f"run_{stage}", "args": {"stage": stage, "run_id": state.get("run_id")}, "id": call_id, "type": "tool_call"}
    return {
        "current_stage": stage,
        "current_tool_call_id": call_id,
        "messages": [AIMessage(content=f"正在执行 {stage} 阶段。", tool_calls=[tool_call])],
    }


def route_selected_stage(state: AgentState) -> str:
    if state.get("current_stage") is not None:
        return "execute_stage"
    if state.get("active_phase") == "validation" and state.get("approved_plan"):
        plan = PipelineRequest.model_validate(state["approved_plan"])
        if plan.execution_strategy == "linked":
            return "validation_gate"
    return "finalize"


def _validation_gate_result(plan: PipelineRequest, report: dict[str, Any]) -> dict[str, Any]:
    thresholds = plan.reconstruction_gate.model_dump(mode="json")
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    failures: list[dict[str, Any]] = []
    checks = (
        ("pearson", "min_pearson", lambda value, limit: value >= limit),
        ("ssim", "min_ssim", lambda value, limit: value >= limit),
        ("mse", "max_mse", lambda value, limit: value <= limit),
        ("mae", "max_mae", lambda value, limit: value <= limit),
        ("relative_l2", "max_relative_l2", lambda value, limit: value <= limit),
    )
    for metric, threshold_name, predicate in checks:
        threshold = thresholds.get(threshold_name)
        if threshold is None:
            continue
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append({"check": threshold_name, "reason": "metric_missing", "value": value, "threshold": threshold})
        elif not predicate(float(value), float(threshold)):
            failures.append({"check": threshold_name, "reason": "threshold_not_met", "value": float(value), "threshold": threshold})
    for report_key, threshold_name in (
        ("evaluated_sessions", "min_evaluated_sessions"),
        ("evaluated_windows", "min_evaluated_windows"),
    ):
        value = int(report.get(report_key) or 0)
        threshold = int(thresholds[threshold_name])
        if value < threshold:
            failures.append({"check": threshold_name, "reason": "threshold_not_met", "value": value, "threshold": threshold})
    if report.get("status") != "ok":
        failures.append({
            "check": "status",
            "reason": "reconstruction_not_ok",
            "value": report.get("status"),
            "threshold": "ok",
        })
    verdict = "fail" if failures else "pass"
    return {
        "schema_version": "1.0",
        "verdict": verdict,
        "metrics": metrics,
        "thresholds": thresholds,
        "evaluated_sessions": int(report.get("evaluated_sessions") or 0),
        "evaluated_windows": int(report.get("evaluated_windows") or 0),
        "failures": failures,
        "reconstruction_report": report,
    }


def _enter_validation_gate(state: AgentState, report: dict[str, Any]) -> dict[str, Any]:
    plan = PipelineRequest.model_validate(state["approved_plan"])
    if plan.execution_strategy != "linked" or state.get("active_phase") != "validation":
        return {"mode": "reviewing_stage"}
    validation_report = _validation_gate_result(plan, report)
    return {
        "validation_report": validation_report,
        "validation_verdict": validation_report["verdict"],
        "gate_id": state.get("gate_id") or new_id("gate"),
        "gate_status": "passed" if validation_report["verdict"] == "pass" else "failed",
        "active_phase": "gate",
        "mode": "awaiting_validation_gate",
    }


def _artifact_file_payload(path: Path) -> tuple[str, str] | None:
    """Return a bounded text artifact for files produced by the current stage."""
    if not path.is_file() or path.stat().st_size > 1_000_000:
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    media_type = "text/markdown" if path.suffix.lower() in {".md", ".markdown"} else "application/json" if path.suffix.lower() == ".json" else "text/plain"
    return content, media_type


def _stage_artifacts(state: AgentState, stage: str, report: dict[str, Any], report_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = [report_artifact]
    report_root = Path(state["report_root"])
    output_root = report_root.parent / "neurotoken"
    candidates = [report_root / f"{stage}_report.json"]
    if stage in {"docs", "readme"}:
        candidates.extend([output_root / "README.md", output_root / "DATA_DICTIONARY.md", output_root / "manifest.json"])
        candidates.extend(sorted((output_root / "schemas").glob("*.json")) if (output_root / "schemas").is_dir() else [])
    seen = {report_artifact["artifact_id"]}
    for path in candidates:
        if not path.is_file() or path == report_root / f"{stage}_report.json":
            continue
        loaded = _artifact_file_payload(path)
        if loaded is None:
            continue
        content, media_type = loaded
        artifact_id = f"artifact_{uuid.uuid4().hex[:12]}"
        if artifact_id in seen:
            continue
        artifacts.append(ArtifactEnvelope(
            artifact_id=artifact_id,
            kind="stage_document",
            owner=state["owner"],
            thread_id=state["thread_id"],
            workflow_id=state.get("workflow_id"),
            run_id=state.get("run_id"),
            stage=stage,
            summary=f"{stage} output: {path.name}",
            file_name=path.name,
            storage_uri=str(path),
            media_type=media_type,
            size=len(content.encode("utf-8")),
            created_at=utc_now(),
            payload=content,
        ).model_dump(mode="json"))
    return artifacts


def execute_stage_node(state: AgentState) -> dict[str, Any]:
    stage = state["current_stage"]
    active_identity = (state.get("linked_runs") or {}).get(state.get("active_phase") or "single")
    if active_identity and active_identity.get("run_root") != state.get("run_root"):
        raise RuntimeError("Active phase run identity does not match current run paths.")
    plan = PipelineRequest.model_validate(state["approved_plan"])
    phase = state.get("active_phase") or "single"
    if phase not in {"single", "validation", "full"}:
        raise RuntimeError(f"Cannot execute a pipeline stage while active_phase={phase!r}.")
    plan = plan.request_for_phase(phase)
    if state.get("retrying_stage") and stage == "tokenize":
        plan = plan.with_updates(overwrite_token_output=True)
    if stage in TMUX_STAGES:
        try:
            job = submit_tmux_stage(
                request=plan,
                stage=stage,
                run_root=state["run_root"],
                config_path=state["config_path"],
                report_root=state["report_root"],
                reference_scripts_dir=str((ROOT / DEFAULT_REFERENCE_SCRIPTS).resolve()),
                python=os.environ.get("NEUROTOKEN_PIPELINE_PYTHON", sys.executable),
                retry_failed=bool(state.get("retrying_stage")),
            )
        except Exception as exc:
            error = {"code": "stage_submit_failed", "message": f"{type(exc).__name__}: {exc}", "stage": stage, "retryable": True}
            return {"last_error": error, "retrying_stage": False, "mode": "failed"}
        return {
            "external_stage_job": job,
            "reconcile_requested": False,
            "retrying_stage": False,
            "last_error": None,
            "mode": "detached",
            "messages": [AIMessage(content=f"{stage} 已提交到持久 tmux attempt；关闭 Desktop 或重启 Agent Server 后仍可恢复。")],
        }
    try:
        report = execute_stage(
            request=plan,
            stage=stage,
            run_root=state["run_root"],
            config_path=state["config_path"],
            report_root=state["report_root"],
            reference_scripts_dir=str((ROOT / DEFAULT_REFERENCE_SCRIPTS).resolve()),
            python=os.environ.get("NEUROTOKEN_PIPELINE_PYTHON", sys.executable),
            retry_failed=bool(state.get("retrying_stage")),
        )
    except Exception as exc:
        error = {"code": "stage_failed", "message": f"{type(exc).__name__}: {exc}", "stage": stage, "retryable": False}
        return {
            "last_error": error,
            "retrying_stage": False,
            "mode": "failed",
            "messages": [ToolMessage(content=json.dumps(error, ensure_ascii=False), tool_call_id=state["current_tool_call_id"], name=f"run_{stage}")],
        }
    if report.get("status") == "failed":
        machine_report = report.get("report") if isinstance(report.get("report"), dict) else {}
        findings = machine_report.get("findings") if isinstance(machine_report.get("findings"), list) else []
        errors = [
            str(finding.get("message"))
            for finding in findings
            if isinstance(finding, dict) and finding.get("severity") == "error" and finding.get("message")
        ]
        summary = machine_report.get("summary") if isinstance(machine_report.get("summary"), dict) else {}
        message = "; ".join(errors) or report.get("stderr") or report.get("stdout") or f"{stage} exited with status {report.get('returncode')}"
        error = {
            "code": "stage_validation_failed" if stage == "validate" else "stage_failed",
            "message": message,
            "stage": stage,
            "retryable": True,
            "details": {
                "returncode": report.get("returncode"),
                "summary": summary,
                "findings": findings,
                "stdout": report.get("stdout"),
                "stderr": report.get("stderr"),
                "report_path": report.get("report_path"),
            },
        }
        return {
            "last_error": error,
            "retrying_stage": False,
            "stage_reports": [*state.get("stage_reports", []), report],
            "mode": "failed",
            "messages": [ToolMessage(content=json.dumps(error, ensure_ascii=False), tool_call_id=state["current_tool_call_id"], name=f"run_{stage}")],
        }

    if stage == "readme":
        readme_path = Path(state["report_root"]).parent / "neurotoken" / "README.md"
        loaded_readme = _artifact_file_payload(readme_path)
        if loaded_readme is not None:
            readme_content, _ = loaded_readme
            report = {
                **report,
                "readme_path": str(readme_path),
                "readme_content": readme_content,
            }

    artifact_payload = report
    if stage == "reconstruct":
        sessions = report.get("sessions") if isinstance(report.get("sessions"), list) else []
        metric_keys = ("pearson", "ssim", "mse", "mae", "relative_l2")
        source_metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        metrics: dict[str, float] = {}
        for key in metric_keys:
            source_value = source_metrics.get(key)
            if isinstance(source_value, (int, float)) and math.isfinite(float(source_value)):
                metrics[key] = float(source_value)
                continue
            weighted_sum = 0.0
            total_weight = 0
            for session in sessions:
                if not isinstance(session, dict) or not isinstance(session.get("metrics_mean"), dict):
                    continue
                value = session["metrics_mean"].get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    weight = max(int(session.get("evaluated_windows", 0)), 1)
                    weighted_sum += float(value) * weight
                    total_weight += weight
            if total_weight:
                metrics[key] = weighted_sum / total_weight
        report = {
            **report,
            "metrics": metrics,
            "evaluated_sessions": len(sessions),
            "evaluated_windows": sum(
                int(session.get("evaluated_windows", 0))
                for session in sessions
                if isinstance(session, dict)
            ),
        }
        artifact_payload = {
            "schema_version": "1.0",
            "metrics": metrics,
            "interpretation": {
                "higher_is_better": ["pearson", "ssim"],
                "lower_is_better": ["mse", "mae", "relative_l2"],
            },
            "report": report,
        }
    artifact = ArtifactEnvelope(
        artifact_id=f"artifact_{uuid.uuid4().hex[:12]}",
        kind="reconstruct_quality" if stage == "reconstruct" else "stage_report",
        owner=state["owner"],
        thread_id=state["thread_id"],
        workflow_id=state.get("workflow_id"),
        run_id=state.get("run_id"),
        stage=stage,
        summary=f"{stage} completed with status={report.get('status')}",
        file_name=f"{stage}_report.json",
        storage_uri=str(Path(state["report_root"]) / f"{stage}_report.json"),
        created_at=utc_now(),
        payload=artifact_payload,
    ).model_dump(mode="json")
    completed = list(dict.fromkeys([*state.get("completed_stages", []), stage]))
    pending = [item for item in state.get("pending_stages", []) if item != stage]
    report = {
        **report,
        "run_id": state.get("run_id"),
        "scope": state.get("active_phase") or "single",
        "stage_id": f"{state.get('active_phase') or 'single'}:{state.get('run_id') or 'unknown'}:{stage}",
    }
    update = {
        "stage_reports": [*state.get("stage_reports", []), report],
        "artifacts": [*state.get("artifacts", []), *_stage_artifacts(state, stage, report, artifact)],
        "completed_stages": completed,
        "pending_stages": pending,
        "messages": [ToolMessage(content=json.dumps({"stage": stage, "status": report.get("status"), "artifact": artifact}, ensure_ascii=False), tool_call_id=state["current_tool_call_id"], name=f"run_{stage}")],
        "retrying_stage": False,
        "mode": "reviewing_stage",
    }
    return update


def route_after_stage(state: AgentState) -> str:
    if state.get("external_stage_job") and state.get("mode") in {"detached", "reconnecting"}:
        return END
    return END if state.get("last_error") else "stage_review"


def reconcile_stage(state: AgentState) -> dict[str, Any]:
    job = state.get("external_stage_job") or {}
    active_identity = (state.get("linked_runs") or {}).get(state.get("active_phase") or "single")
    if active_identity and active_identity.get("run_root") != state.get("run_root"):
        raise RuntimeError("Active phase run identity does not match current run paths.")
    stage = str(job.get("stage") or state.get("current_stage") or "")
    execution_id = job.get("stage_execution_id")
    expected_job_phase = "validation" if state.get("active_phase") == "gate" else state.get("active_phase")
    if job.get("run_id") not in {None, state.get("run_id")} or job.get("phase") not in {None, expected_job_phase}:
        error = {
            "code": "stale_stage_job",
            "message": "Durable stage identity does not match the active linked run.",
            "stage": stage or None,
            "retryable": False,
            "details": {"job_run_id": job.get("run_id"), "active_run_id": state.get("run_id")},
        }
        return {"reconcile_requested": False, "external_stage_job": job or None, "last_error": error, "mode": "failed"}
    if not stage or not state.get("run_root"):
        return {"reconcile_requested": False, "external_stage_job": None, "mode": state.get("mode") or "idle"}
    try:
        reconciled = reconcile_stage_job(state["run_root"], stage, execution_id)
    except Exception as exc:
        error = {"code": "stage_reconcile_failed", "message": f"{type(exc).__name__}: {exc}", "stage": stage, "retryable": True}
        return {"reconcile_requested": False, "external_stage_job": job or None, "last_error": error, "mode": "failed"}
    status = reconciled.get("status")
    if status in {"submitted", "running", "detached", "reconnecting"}:
        return {"reconcile_requested": False, "external_stage_job": reconciled, "last_error": None, "mode": "detached"}
    if status == "completed":
        report = reconciled.get("report") or {}
        report_execution_id = report.get("stage_execution_id") or reconciled.get("stage_execution_id")
        already_harvested = any(
            item.get("stage_execution_id") == report_execution_id for item in state.get("stage_reports", [])
            if isinstance(item, dict)
        )
        if already_harvested:
            return {"reconcile_requested": False, "external_stage_job": None, "last_error": None, "mode": "reviewing_stage"}
        report_path = Path(state["report_root"]) / f"{stage}_report.json"
        artifact_payload: dict[str, Any] = report
        artifact_kind = "stage_report"
        if stage == "reconstruct":
            sessions = report.get("sessions") if isinstance(report.get("sessions"), list) else []
            source_metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
            metrics: dict[str, float] = {}
            for key in ("pearson", "ssim", "mse", "mae", "relative_l2"):
                source_value = source_metrics.get(key)
                if isinstance(source_value, (int, float)) and math.isfinite(float(source_value)):
                    metrics[key] = float(source_value)
                    continue
                weighted_sum = 0.0
                total_weight = 0
                for session in sessions:
                    if not isinstance(session, dict) or not isinstance(session.get("metrics_mean"), dict):
                        continue
                    value = session["metrics_mean"].get(key)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        weight = max(int(session.get("evaluated_windows", 0)), 1)
                        weighted_sum += float(value) * weight
                        total_weight += weight
                if total_weight:
                    metrics[key] = weighted_sum / total_weight
            report = {
                **report,
                "metrics": metrics,
                "evaluated_sessions": len(sessions),
                "evaluated_windows": sum(
                    int(session.get("evaluated_windows", 0))
                    for session in sessions
                    if isinstance(session, dict)
                ),
            }
            artifact_payload = {
                "schema_version": "1.0",
                "metrics": metrics,
                "interpretation": {
                    "higher_is_better": ["pearson", "ssim"],
                    "lower_is_better": ["mse", "mae", "relative_l2"],
                },
                "report": report,
            }
            artifact_kind = "reconstruct_quality"
        artifact = ArtifactEnvelope(
            artifact_id=f"artifact_{uuid.uuid4().hex[:12]}",
            kind=artifact_kind,
            owner=state["owner"],
            thread_id=state["thread_id"],
            workflow_id=state.get("workflow_id"),
            run_id=state.get("run_id"),
            stage=stage,
            summary=f"{stage} stage report",
            file_name=report_path.name,
            storage_uri=str(report_path),
            media_type="application/json",
            size=report_path.stat().st_size if report_path.exists() else None,
            created_at=utc_now(),
            payload=artifact_payload,
        ).model_dump(mode="json")
        completed = list(dict.fromkeys([*state.get("completed_stages", []), stage]))
        pending = [item for item in state.get("pending_stages", []) if item != stage]
        harvested_report = {
            **report,
            "stage_execution_id": report_execution_id,
            "run_id": state.get("run_id"),
            "scope": state.get("active_phase") or "single",
            "stage_id": f"{state.get('active_phase') or 'single'}:{state.get('run_id') or 'unknown'}:{stage}",
        }
        update = {
            "reconcile_requested": False,
            "external_stage_job": None,
            "stage_reports": [*state.get("stage_reports", []), harvested_report],
            "artifacts": [*state.get("artifacts", []), artifact],
            "completed_stages": completed,
            "pending_stages": pending,
            "last_error": None,
            "mode": "reviewing_stage",
        }
        return update
    error = {
        "code": "stage_orphaned" if status == "orphaned" else "stage_failed",
        "message": str(reconciled.get("error") or f"{stage} attempt ended with {status}"),
        "stage": stage,
        "retryable": status in {"failed", "orphaned", "cancelled"},
        "details": {"stage_execution_id": execution_id, "log_tail": reconciled.get("log_tail", "")},
    }
    return {"reconcile_requested": False, "external_stage_job": reconciled, "last_error": error, "mode": "failed"}


def route_after_reconcile(state: AgentState) -> str:
    return "stage_review" if state.get("mode") == "reviewing_stage" and state.get("stage_reports") else END


def _linked_child_run_id(phase: str, workflow_id: str, gate_id: str, parent_run_id: str) -> str:
    identity = uuid.uuid5(uuid.NAMESPACE_URL, f"neurotoken:{workflow_id}:{gate_id}:{phase}:{parent_run_id}")
    return f"run_{phase}_{identity.hex[:12]}"


def validation_gate(state: AgentState) -> dict[str, Any]:
    gate_id = state.get("gate_id")
    validation_report = state.get("validation_report") or {}
    verdict = state.get("validation_verdict")
    allowed_actions = ["rerun_smoke", "pause", "cancel"]
    if verdict == "pass":
        allowed_actions.insert(0, "start_full_run")
    plan = PipelineRequest.model_validate(state["approved_plan"])
    decision = ApprovalDecision.model_validate(interrupt({
        "schema_version": "1.0",
        "kind": "smoke_validation_review",
        "owner": state["owner"],
        "thread_id": state["thread_id"],
        "workflow_id": state.get("workflow_id"),
        "gate_id": gate_id,
        "validation_target_h5": plan.validation_target_h5,
        "full_input_root": plan.full_input_root,
        "status": state.get("gate_status"),
        "verdict": verdict,
        "report": validation_report,
        "allowed_actions": allowed_actions,
    }))
    if decision.gate_id != gate_id:
        raise ValueError("Validation gate decision does not match the active gate_id.")
    if decision.action == "pause":
        return {"mode": "paused", "active_phase": "gate"}
    if decision.action == "cancel":
        return {
            "mode": "cancelled",
            "gate_status": "cancelled",
            "pending_stages": [],
            "current_stage": None,
            "current_tool_call_id": None,
        }
    if decision.action == "rerun_smoke":
        validation_request = plan.request_for_phase("validation")
        resolve_stage_dependencies(validation_request)
        validation_identity = (state.get("linked_runs") or {}).get("validation") or {}
        parent_run_id = validation_identity.get("run_id") or state["run_id"]
        run = materialize_child_run(
            validation_request,
            workflow_id=state["workflow_id"],
            parent_run_id=parent_run_id,
            phase="validation",
            run_id=_linked_child_run_id("validation", state["workflow_id"], str(gate_id), parent_run_id),
        )
        linked_runs = {
            **(state.get("linked_runs") or {}),
            "validation": {**run, "phase": "validation", "parent_run_id": validation_identity.get("run_id") or state["run_id"]},
        }
        return {
            **run,
            "active_phase": "validation",
            "linked_runs": linked_runs,
            "validation_report": None,
            "validation_verdict": None,
            "gate_id": None,
            "gate_status": None,
            "completed_stages": [],
            "pending_stages": list(validation_request.stages),
            "current_stage": None,
            "current_tool_call_id": None,
            "external_stage_job": None,
            "last_error": None,
            "mode": "running",
        }
    if decision.action != "start_full_run":
        raise ValueError(f"Unsupported validation gate action: {decision.action}")
    if verdict != "pass":
        return {
            "mode": "awaiting_validation_gate",
            "last_error": {
                "code": "validation_gate_failed",
                "message": "Full run cannot start until reconstruction gate passes.",
                "stage": "reconstruct",
                "retryable": True,
            },
        }
    full_request = plan.request_for_phase("full")
    resolve_stage_dependencies(full_request)
    existing = (state.get("linked_runs") or {}).get("full")
    if existing:
        pending = state.get("pending_stages", [])
        if state.get("active_phase") != "full":
            pending = list(full_request.stages)
        run_fields = {
            key: existing[key]
            for key in ("workflow_id", "run_id", "run_root", "config_path", "report_root")
            if key in existing
        }
        return {
            **run_fields,
            "active_phase": "full",
            "gate_status": "started",
            "completed_stages": state.get("completed_stages", []) if state.get("active_phase") == "full" else [],
            "pending_stages": pending,
            "current_stage": state.get("current_stage") if state.get("active_phase") == "full" else None,
            "current_tool_call_id": state.get("current_tool_call_id") if state.get("active_phase") == "full" else None,
            "mode": "running",
        }
    validation_identity = (state.get("linked_runs") or {}).get("validation") or {}
    parent_run_id = validation_identity.get("run_id") or state["run_id"]
    run = materialize_child_run(
        full_request,
        workflow_id=state["workflow_id"],
        parent_run_id=parent_run_id,
        phase="full",
        run_id=_linked_child_run_id("full", state["workflow_id"], str(gate_id), parent_run_id),
    )
    return {
        **run,
        "active_phase": "full",
        "linked_runs": {
            **(state.get("linked_runs") or {}),
            "full": {**run, "phase": "full", "parent_run_id": validation_identity.get("run_id") or state["run_id"]},
        },
        "gate_status": "started",
        "completed_stages": [],
        "pending_stages": list(full_request.stages),
        "current_stage": None,
        "current_tool_call_id": None,
        "external_stage_job": None,
        "last_error": None,
        "mode": "running",
    }


def route_after_gate(state: AgentState) -> str:
    return "select_stage" if state.get("mode") == "running" else END


def stage_review(state: AgentState) -> dict[str, Any]:
    report = ensure_package_report_previews(state.get("stage_reports", [])[-1])
    explanation = summarize_stage(llm(), report)
    decision = ApprovalDecision.model_validate(interrupt({
        "schema_version": "1.0",
        "kind": "stage_review",
        "owner": state["owner"],
        "thread_id": state["thread_id"],
        "workflow_id": state["workflow_id"],
        "stage": report["stage"],
        "summary": explanation,
        "report": report,
        "remaining_stages": state.get("pending_stages", []),
        "allowed_actions": ["continue", "pause", "cancel"],
    }))
    if decision.action == "pause":
        return {"mode": "paused", "messages": [AIMessage(content="Pipeline 已暂停。")]} 
    if decision.action == "cancel":
        return {
            "mode": "cancelled",
            "approved_plan": None,
            "pending_stages": [],
            "current_stage": None,
            "current_tool_call_id": None,
            "messages": [AIMessage(content="Pipeline 已取消。")],
        }
    completed_message = append_package_sample_summary(explanation, report)
    if (
        report.get("stage") == "reconstruct"
        and not state.get("pending_stages")
        and state.get("active_phase") == "validation"
    ):
        return {
            **_enter_validation_gate(state, report),
            "messages": [AIMessage(content=completed_message)],
        }
    return {"mode": "running", "messages": [AIMessage(content=completed_message)]}


def route_after_review(state: AgentState) -> str:
    if state.get("mode") == "awaiting_validation_gate":
        return "validation_gate"
    return "select_stage" if state.get("mode") == "running" else END


def finalize(state: AgentState) -> dict[str, Any]:
    plan_data = state.get("approved_plan") or state.get("current_plan")
    linked_plan = bool(isinstance(plan_data, dict) and plan_data.get("execution_strategy") == "linked")
    if linked_plan and state.get("active_phase") != "full":
        raise RuntimeError("Linked execution can finalize only after the full child run completes.")
    report = {
        "schema_version": "1.0",
        "workflow_id": state.get("workflow_id"),
        "run_id": state.get("run_id"),
        "active_phase": state.get("active_phase") or "single",
        "linked_runs": state.get("linked_runs") or {},
        "validation_report": state.get("validation_report"),
        "validation_verdict": state.get("validation_verdict"),
        "gate_id": state.get("gate_id"),
        "gate_status": state.get("gate_status"),
        "status": "ok",
        "config": state.get("config_path"),
        "stages": state.get("stage_reports", []),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(Path(state["report_root"]) / "pipeline_report.json", report)
    return {"mode": "completed", "messages": [AIMessage(content=f"Pipeline 已完成，共执行 {len(report['stages'])} 个阶段。报告：{state['report_root']}/pipeline_report.json")]}


def report(state: AgentState) -> dict[str, Any]:
    if not state.get("stage_reports"):
        text = "当前 thread 还没有 pipeline 阶段报告。"
    else:
        text = f"当前状态：{state.get('mode')}；已完成阶段：{', '.join(state.get('completed_stages', []))}；报告目录：{state.get('report_root')}"
    return {"messages": [AIMessage(content=text)]}


def cancel(state: AgentState) -> dict[str, Any]:
    job = state.get("external_stage_job") or {}
    active_identity = (state.get("linked_runs") or {}).get(state.get("active_phase") or "single")
    if active_identity and active_identity.get("run_root") != state.get("run_root"):
        raise RuntimeError("Active phase run identity does not match current run paths.")
    expected_job_phase = "validation" if state.get("active_phase") == "gate" else state.get("active_phase")
    if job.get("run_id") not in {None, state.get("run_id")} or job.get("phase") not in {None, expected_job_phase}:
        return {
            "external_stage_job": job,
            "last_error": {
                "code": "stale_stage_job",
                "message": "Refusing to cancel a durable job from another linked run.",
                "stage": job.get("stage"),
                "retryable": False,
            },
            "mode": "failed",
        }
    if job and state.get("run_root") and job.get("stage"):
        try:
            cancelled_job = cancel_stage_job(state["run_root"], job["stage"], job.get("stage_execution_id"))
        except Exception as exc:
            return {
                "external_stage_job": job,
                "last_error": {"code": "stage_cancel_failed", "message": f"{type(exc).__name__}: {exc}", "stage": job.get("stage"), "retryable": True},
                "mode": "failed",
            }
    else:
        cancelled_job = None
    return {
        "mode": "cancelled",
        "external_stage_job": cancelled_job,
        "approved_plan": None,
        "pending_stages": [],
        "current_stage": None,
        "current_tool_call_id": None,
        "messages": [AIMessage(content="已取消当前 workflow。历史报告和 artifact 保留。")],
    }


def existing_adapter_names() -> list[str]:
    root = Path(__file__).resolve().parent / "adapters"
    return sorted(path.stem for path in root.glob("*.py") if path.stem not in {"__init__", "base"})


def recommended_chunking_strategy(source_path: str | None) -> str:
    """Prefer event chunks when the source declares usable event records."""
    if not source_path:
        return "rest"
    try:
        import h5py

        with h5py.File(source_path, "r") as source:
            event_count = int(source.attrs.get("n_events", 0))
            return "event" if event_count > 0 and "events_pkl_u8" in source else "rest"
    except (OSError, TypeError, ValueError):
        return "rest"


def author_adapter_node(state: AgentState) -> dict[str, Any]:
    original_user_request = latest_user_text(state)
    request = parse_adapter_request(llm(), original_user_request, existing_adapter_names())
    task_id = new_id("adapter")
    validate_adapter_source(request.source_path)
    spec = AdapterSpec(request.name, Path(request.source_path).expanduser().resolve(), request.modality, request.notes)
    h5_summary = format_h5_summary(inspect_h5(spec.source_path))
    requirements_snapshot = {
        "requirements": state.get("requirements") or {},
        "other_requirements": state.get("other_requirements") or [],
    }
    task = adapter_task_runtime().create_task(
        task_id=task_id,
        owner=state["owner"],
        thread_id=state["thread_id"],
        adapter=AdapterCodingDescriptor(name=spec.name, modality=spec.modality, notes=spec.notes),
        instruction=build_adapter_prompt(spec, h5_summary),
        original_user_request=original_user_request,
        requirements_snapshot=requirements_snapshot,
    )
    pending = {
        "task_id": task.task_id,
        "request_id": task.task_id,
        "name": spec.name,
        "source_path": str(spec.source_path),
        "modality": spec.modality,
        "notes": spec.notes,
        "h5_summary": h5_summary,
        "coding_summary": "",
        "instruction": task.instruction,
        "original_instruction": task.instruction,
        "original_user_request": task.original_user_request,
        "requirements_snapshot": task.requirements_snapshot,
        "diff": "",
        "created_at": task.created_at,
        "revision": task.revision,
        "collaboration": [],
        "design_document": "",
        "knowledge_base_path": str((ROOT / "agent_pipeline/adapter_authoring/knowledge_base.md").resolve()),
        "template_adapters": ["agent_pipeline/adapters/meg_h5.py", "agent_pipeline/adapters/eeg_h5.py"],
    }
    return {
        "pending_adapter": pending,
        "mode": "ready_for_coding",
        "messages": [AIMessage(content=f"已创建 `{spec.name}` 的隔离 coding task。Codex 将读取 adapter 知识库、对比模板，并在提交前生成 ADAPTER_DESIGN.md；请在 Desktop 中打开并完成候选实现。")],
    }


def adapter_candidate(state: AgentState) -> dict[str, Any]:
    pending = state.get("pending_adapter")
    if not pending:
        return {
            "mode": "adapter_ready" if state.get("last_adapter") else "idle",
            "last_error": None,
            "messages": [AIMessage(content="当前没有待提交的 adapter coding task。若要处理数据，请直接确认开始 pipeline；若要修改代码，请明确要求继续编辑 adapter。")],
        }
    runtime = adapter_task_runtime()
    task = runtime.require_task(
        pending["task_id"],
        owner=state["owner"],
        thread_id=state["thread_id"],
    )
    if task.status == "needs_revision":
        task = runtime.begin_revision(
            task.task_id,
            expected_revision=task.revision,
            instruction=f"{getattr(task, 'instruction', pending.get('instruction', 'Continue improving the adapter candidate.'))}\n\nValidation feedback:\n{task.error or pending.get('coding_summary', 'Review and correct the candidate.')}",
        )
    submission = AdapterCandidateSubmission.model_validate(interrupt({
        "schema_version": "1.0",
        "kind": "adapter_coding_handoff",
        "handoff": runtime.handoff(task.task_id, owner=state["owner"]).model_dump(mode="json"),
        "allowed_actions": ["submit_candidate"],
    }))
    if submission.task_id != task.task_id or submission.thread_id != task.thread_id:
        raise PermissionError("Candidate submission does not match the active adapter task.")
    if submission.revision != task.revision:
        raise RuntimeError(
            f"Stale adapter candidate revision: expected {task.revision}, received {submission.revision}"
        )
    if task.status == "ready_for_coding":
        task = runtime.update_status(
            task.task_id,
            "coding",
            expected_revision=task.revision,
            codex_thread_id=submission.codex_thread_id,
        )
    elif task.status == "coding":
        task = runtime.bind_coding_thread(
            task.task_id,
            expected_revision=task.revision,
            codex_thread_id=submission.codex_thread_id,
            replaces_codex_thread_id=submission.replaces_codex_thread_id,
        )
    elif task.status == "validating":
        if task.codex_thread_id != submission.codex_thread_id:
            raise PermissionError("Candidate submission does not match the validating Codex thread.")
    else:
        raise RuntimeError(f"Adapter task is not accepting a candidate: {task.status}")
    if task.status == "coding":
        task = runtime.update_status(task.task_id, "validating", expected_revision=task.revision)
    injected_support_paths = {"agent_pipeline/adapter_authoring/knowledge_base.md"}
    changed_paths = [
        path for path in changed_worktree_paths(task.worktree_path)
        if path not in injected_support_paths
    ]
    diff = diff_worktree(task.worktree_path, task.allowed_paths)
    design_path = Path(task.worktree_path) / "ADAPTER_DESIGN.md"
    design_document = design_path.read_text(encoding="utf-8") if design_path.is_file() else ""
    validation_error = None
    previous_candidate_diff = str(pending.get("previous_candidate_diff") or "").strip()
    if "ADAPTER_DESIGN.md" in task.allowed_paths and not design_document.strip():
        validation_error = "Candidate must include ADAPTER_DESIGN.md explaining its logic, chunking strategy, and template differences."
    if not diff.strip():
        validation_error = "Candidate produced no adapter changes."
    elif task.revision > 1 and previous_candidate_diff and diff.strip() == previous_candidate_diff:
        validation_error = "Revision produced no changes relative to the previous candidate. Apply the requested refinement before submitting again."
    elif not paths_allowed(changed_paths, task.allowed_paths):
        validation_error = "Candidate changed files outside allowed paths."
    if validation_error:
        task = runtime.update_status(
            task.task_id,
            "needs_revision",
            expected_revision=task.revision,
            error=validation_error,
        )
        return {
            "pending_adapter": {
                **pending,
                "coding_summary": validation_error,
                "diff": diff,
                "revision": task.revision,
                "codex_thread_id": submission.codex_thread_id,
                "validation_error": validation_error,
                "intent_review": None,
                "changed_paths": changed_paths,
            },
            "mode": "adapter_revision_required",
            "messages": [AIMessage(content=f"候选验证未通过：{validation_error} 已停止自动修正。请检查变更后显式选择修正、重试或拒绝；不会自动创建下一版本。")],
        }

    spec = AdapterSpec(
        pending["name"],
        Path(pending["source_path"]),
        pending["modality"],
        pending.get("notes", ""),
    )
    intent_context = str(pending.get("instruction") or task.instruction)
    original_user_request = str(getattr(task, "original_user_request", "") or pending.get("original_user_request") or "").strip()
    requirements_snapshot = getattr(task, "requirements_snapshot", None) or pending.get("requirements_snapshot") or {}
    captured_requirements = requirements_snapshot.get("requirements", {})
    explicit_overrides = captured_requirements.get("config_overrides", {})
    proposed_overrides = {
        section: dict(values)
        for section, values in explicit_overrides.items()
        if isinstance(values, dict)
    }
    proposed_overrides.setdefault("chunking", {}).setdefault(
        "strategy", recommended_chunking_strategy(str(spec.source_path))
    )
    base_config = select_config(spec.modality, runtime_config()["default_config"])
    requested_output = captured_requirements.get("output_path") or captured_requirements.get("report_root")
    review_output_root = Path(requested_output).expanduser().resolve() if requested_output else ROOT / "outputs" / "adapter-review"
    preflight_evidence: dict[str, Any]
    try:
        effective_config = build_effective_config(
            base_config_path=base_config,
            output_root=review_output_root,
            target_h5=spec.source_path,
            source_adapter=spec.name,
            config_overrides=proposed_overrides,
        )
        preflight_evidence = run_candidate_preflight(
            worktree=task.worktree_path,
            effective_config=effective_config,
            target_h5=spec.source_path,
        )
    except (OSError, TypeError, ValueError) as exc:
        effective_config = {
            "_review_error": f"Unable to build proposed effective config: {type(exc).__name__}: {exc}",
            "_base_config": str(base_config),
            "_requested_output_root": str(review_output_root),
            "source": {"adapter": spec.name},
            "chunking": proposed_overrides.get("chunking", {}),
        }
        preflight_evidence = {
            "status": "failed",
            "source_session_loaded": False,
            "target_h5": str(spec.source_path),
            "error": effective_config["_review_error"],
        }
    original_instruction = str(pending.get("original_instruction") or "").strip()
    if original_instruction and original_instruction != intent_context:
        intent_context = f"Original coding instruction:\n{original_instruction}\n\nCurrent coding instruction:\n{intent_context}"
    collaboration = pending.get("collaboration") or []
    if collaboration:
        intent_context = (
            f"{intent_context}\n\nRecorded human collaboration and revision history:\n"
            f"{json.dumps(collaboration, ensure_ascii=False, indent=2)}"
        )
    try:
        intent_review = review_adapter_candidate(
            llm(),
            spec=spec,
            h5_summary=pending["h5_summary"],
            instruction=intent_context,
            design_document=design_document,
            diff=diff,
            worktree=task.worktree_path,
            original_user_request=original_user_request,
            requirements_snapshot=requirements_snapshot,
            effective_config=effective_config,
            preflight_evidence=preflight_evidence,
            allowed_paths=task.allowed_paths,
        )
        intent_review_payload = intent_review.model_dump(mode="json")
    except Exception as exc:
        intent_review = None
        intent_review_payload = {
            "verdict": "unavailable",
            "summary": f"Independent intent review was unavailable: {type(exc).__name__}: {str(exc)[:500]}",
            "findings": [],
        }

    if intent_review is not None and intent_review.verdict in {"needs_revision", "blocked_by_scope"}:
        validation_error = review_feedback(intent_review)
        task = runtime.update_status(
            task.task_id,
            "needs_revision",
            expected_revision=task.revision,
            error=validation_error,
        )
        return {
            "pending_adapter": {
                **pending,
                "coding_summary": validation_error,
                "diff": diff,
                "design_document": design_document,
                "revision": task.revision,
                "codex_thread_id": submission.codex_thread_id,
                "validation_error": validation_error,
                "intent_review": intent_review_payload,
                "original_user_request": original_user_request,
                "requirements_snapshot": requirements_snapshot,
                "effective_config": effective_config,
                "preflight_evidence": preflight_evidence,
                "changed_paths": changed_paths,
            },
            "mode": "adapter_revision_required",
            "messages": [AIMessage(content=f"独立用户意图审查未通过：{validation_error} 已阻止进入 patch approval；不会自动创建下一版本。")],
        }

    task = runtime.update_status(
        task.task_id,
        "awaiting_patch_approval",
        expected_revision=task.revision,
    )
    review_message = (
        "独立用户意图审查通过"
        if intent_review is not None
        else "独立用户意图审查暂不可用，请在审批中人工核对"
    )
    return {
        "pending_adapter": {
            **pending,
            "coding_summary": "Candidate submitted from Codex Desktop.",
            "diff": diff,
            "design_document": design_document,
            "revision": task.revision,
            "codex_thread_id": submission.codex_thread_id,
            "intent_review": intent_review_payload,
            "original_user_request": original_user_request,
            "requirements_snapshot": requirements_snapshot,
            "effective_config": effective_config,
            "preflight_evidence": preflight_evidence,
        },
        "mode": "awaiting_patch_approval",
        "messages": [AIMessage(content=f"已验证 `{task.adapter.name}` 候选变更；{review_message}。请检查 Codex 设计说明与独立审查后决定应用、修正或拒绝。")],
    }


def patch_approval(state: AgentState) -> dict[str, Any]:
    pending = state["pending_adapter"]
    runtime = adapter_task_runtime()
    task = runtime.require_task(
        pending["task_id"],
        owner=state["owner"],
        thread_id=state["thread_id"],
    )
    if task.status != "awaiting_patch_approval":
        raise RuntimeError(f"Adapter task is not awaiting patch approval: {task.status}")
    if not str(pending.get("diff") or "").strip():
        task = runtime.update_status(
            task.task_id,
            "needs_revision",
            expected_revision=task.revision,
            error="Approval checkpoint is missing the frozen candidate diff.",
        )
        return {
            "pending_adapter": {
                **pending,
                "coding_summary": "Approval checkpoint is missing the frozen candidate diff.",
                "validation_error": "Approval checkpoint is missing the frozen candidate diff.",
            },
            "mode": "adapter_revision_required",
            "messages": [AIMessage(content="候选审批缺少冻结 diff，已阻止应用。请重新打开 coding task、提交候选后再审批。")],
        }
    raw_decision = interrupt({
        "schema_version": "1.0",
        "kind": "adapter_patch_approval",
        "owner": state["owner"],
        "thread_id": state["thread_id"],
        "task_id": task.task_id,
        "revision": task.revision,
        "adapter": {key: pending.get(key) for key in ["request_id", "task_id", "name", "source_path", "modality", "notes", "h5_summary", "coding_summary", "design_document", "knowledge_base_path", "template_adapters", "diff", "revision", "collaboration", "intent_review", "original_user_request", "requirements_snapshot", "effective_config", "preflight_evidence"]},
        "allowed_actions": ["apply", "apply_and_commit", "ask", "refine", "reject"],
    })
    if isinstance(raw_decision, dict) and raw_decision.get("action") == "submit_candidate":
        return {
            "mode": "awaiting_patch_approval",
            "last_error": None,
            "messages": [AIMessage(content="候选已提交并等待审批；忽略重复的 Codex completion 通知。")],
        }
    decision = ApprovalDecision.model_validate(raw_decision)
    if decision.task_id not in {None, task.task_id}:
        return {
            "adapter_collaboration_action": None,
            "adapter_collaboration_message": None,
            "mode": "awaiting_patch_approval",
            "messages": [AIMessage(content="审批未提交：所选 adapter task 与当前候选不匹配，请刷新后重试。")],
        }
    if decision.revision not in {None, task.revision}:
        return {
            "adapter_collaboration_action": None,
            "adapter_collaboration_message": None,
            "mode": "awaiting_patch_approval",
            "messages": [AIMessage(content=f"审批未提交：候选版本已更新为 {task.revision}，请检查最新 diff 后重试。")],
        }
    if decision.action in {"ask", "refine"}:
        message = (decision.message or "").strip()
        if not message:
            return {
                "adapter_collaboration_action": None,
                "adapter_collaboration_message": None,
                "mode": "awaiting_patch_approval",
                "messages": [AIMessage(content="审批未提交：提问或修改请求不能为空。")],
            }
        return {
            "adapter_collaboration_action": decision.action,
            "adapter_collaboration_message": message,
            "mode": "adapter_question" if decision.action == "ask" else "authoring_adapter",
        }
    if decision.action == "reject":
        runtime.cleanup_task(task.task_id, status="rejected", owner=state["owner"])
        return {
            "pending_adapter": None,
            "adapter_collaboration_action": None,
            "adapter_collaboration_message": None,
            "approved_plan": None,
            "pending_stages": [],
            "current_stage": None,
            "current_tool_call_id": None,
            "last_error": None,
            "mode": "cancelled",
            "messages": [AIMessage(content="已拒绝并终止当前 adapter 任务。你可以在此对话中发送新的要求。")],
        }
    spec = AdapterSpec(pending["name"], Path(pending["source_path"]), pending["modality"], pending.get("notes", ""))
    result = AdapterAuthoringResult(
        spec=spec,
        h5_summary=pending["h5_summary"],
        coding_result=CodingAgentResult(summary=pending.get("coding_summary", ""), diff=pending["diff"]),
    )
    commit_message = None
    if decision.action == "apply_and_commit":
        commit_message = (decision.commit_message or "").strip()
        if not commit_message:
            return {
                "adapter_collaboration_action": None,
                "adapter_collaboration_message": None,
                "mode": "awaiting_patch_approval",
                "messages": [AIMessage(content="审批未提交：Apply & Commit 必须填写 commit message。")],
            }
    try:
        patch_record = apply_authoring_result(
            ROOT,
            result,
            owner=state["owner"],
            thread_id=state["thread_id"],
            request_id=pending["task_id"],
            commit_message=commit_message,
        )
        runtime.cleanup_task(task.task_id, status="applied", owner=state["owner"])
    except Exception as exc:
        try:
            runtime.cleanup_task(task.task_id, status="failed", owner=state["owner"], error=str(exc))
        except Exception:
            pass
        raise
    import importlib
    import agent_pipeline.adapters as adapters
    importlib.invalidate_caches()
    importlib.reload(adapters)
    adapter = {"name": spec.name, "source_path": str(spec.source_path), "modality": spec.modality, "patch_record": patch_record.name}
    requirements = dict(state.get("requirements", {}))
    config_overrides = {
        section: dict(values)
        for section, values in (requirements.get("config_overrides") or {}).items()
        if isinstance(values, dict)
    }
    config_overrides.setdefault("chunking", {}).setdefault(
        "strategy", recommended_chunking_strategy(str(spec.source_path))
    )
    chunking_strategy = str(config_overrides["chunking"]["strategy"])
    requirements.update(
        adapter_name=spec.name,
        input_path=str(spec.source_path),
        modality=spec.modality,
        config_overrides=config_overrides,
    )
    return {
        "pending_adapter": None,
        "adapter_collaboration_action": None,
        "adapter_collaboration_message": None,
        "last_adapter": adapter,
        "requirements": requirements,
        "mode": "adapter_ready",
        "messages": [AIMessage(content=f"Adapter `{spec.name}` 已应用、通过 import 验证并激活。后续 pipeline 将使用 `{chunking_strategy}` 切分；你仍可在计划审批中改为 `rest` 或 `event`。")],
    }


def adapter_execution_confirmation(state: AgentState) -> dict[str, Any]:
    adapter = state.get("last_adapter") or {}
    requirements = state.get("requirements") or {}
    source_path = requirements.get("input_path") or adapter.get("source_path")
    output_path = requirements.get("output_path")
    decision = ApprovalDecision.model_validate(interrupt({
        "schema_version": "1.0",
        "kind": "adapter_execution_confirmation",
        "owner": state["owner"],
        "thread_id": state["thread_id"],
        "adapter_name": adapter.get("name"),
        "recommended_chunking_strategy": requirements.get("config_overrides", {}).get("chunking", {}).get("strategy", "rest"),
        "input_path": source_path,
        "output_path": output_path,
        "allowed_actions": ["continue", "pause"],
    }))
    if decision.action == "continue":
        return {
            "intent": "execute",
            "mode": "planning",
            "messages": [AIMessage(content=f"已确认使用 `{adapter.get('name')}` 处理数据，正在生成 pipeline 计划。")],
        }
    return {
        "mode": "adapter_ready",
        "messages": [AIMessage(content="已保留 adapter，暂不处理数据。你可以继续修改代码，或稍后在此对话中要求开始处理。")],
    }


def route_after_patch_decision(state: AgentState) -> str:
    action = state.get("adapter_collaboration_action")
    if action in {"ask", "refine"}:
        return "adapter_collaboration"
    if state.get("mode") == "adapter_revision_required":
        return "adapter_candidate"
    if state.get("mode") == "awaiting_patch_approval":
        return "patch_approval"
    if state.get("mode") == "adapter_ready":
        return "adapter_execution_confirmation"
    return END


def route_after_execution_confirmation(state: AgentState) -> str:
    return "build_plan" if state.get("intent") == "execute" and state.get("mode") == "planning" else END


def route_after_candidate(state: AgentState) -> str:
    if state.get("mode") == "awaiting_patch_approval":
        return "patch_approval"
    return END


def adapter_collaboration(state: AgentState) -> dict[str, Any]:
    pending = state["pending_adapter"]
    action = state.get("adapter_collaboration_action")
    message = state.get("adapter_collaboration_message") or ""
    history = list(pending.get("collaboration") or [])

    if action == "ask":
        answer = answer_adapter_question(llm(), pending, message)
        history.append({"kind": "question", "message": message, "answer": answer, "created_at": utc_now()})
        return {
            "pending_adapter": {**pending, "collaboration": history},
            "adapter_collaboration_action": None,
            "adapter_collaboration_message": None,
            "mode": "awaiting_patch_approval",
            "messages": [AIMessage(content=answer)],
        }

    spec = AdapterSpec(pending["name"], Path(pending["source_path"]), pending["modality"], pending.get("notes", ""))
    task = adapter_task_runtime().begin_revision(
        pending["task_id"],
        expected_revision=int(pending["revision"]),
        instruction=build_refinement_prompt(spec, pending["h5_summary"], message),
    )
    history.append({
        "kind": "refinement",
        "message": message,
        "summary": "Revision requested in Desktop coding task.",
        "revision": task.revision,
        "created_at": utc_now(),
    })
    return {
        "pending_adapter": {
            **pending,
            "coding_summary": "",
            "instruction": task.instruction,
            "previous_candidate_diff": str(pending.get("diff") or ""),
            "diff": "",
            "revision": task.revision,
            "codex_thread_id": task.codex_thread_id,
            "collaboration": history,
            "intent_review": None,
        },
        "adapter_collaboration_action": None,
        "adapter_collaboration_message": None,
        "mode": "ready_for_coding",
        "messages": [AIMessage(content=f"已创建第 {task.revision} 版修正任务，请回到 Desktop coding workspace 继续编辑。")],
    }


def cleanup_sdk_node(state: AgentState) -> dict[str, Any]:
    from agent_pipeline.adapter_authoring.cleanup import cleanup

    adapter = state.get("last_adapter")
    if not adapter or not adapter.get("patch_record"):
        return {"mode": "idle", "messages": [AIMessage(content="当前 thread 没有可清理的 SDK adapter 补丁。")]} 
    removed = cleanup(
        ROOT,
        patch_id=adapter["patch_record"],
        owner=state["owner"],
        thread_id=state["thread_id"],
    )
    return {
        "last_adapter": None,
        "mode": "idle",
        "messages": [AIMessage(content=f"已清理当前 thread 的 SDK adapter 补丁，共回滚 {len(removed)} 个记录。")],
    }


def exit_node(state: AgentState) -> dict[str, Any]:
    return {"messages": [AIMessage(content="Web 会话无需退出；你可以关闭页面或开始新的 thread。")], "mode": "idle"}


def workspace_node(node):
    parameters = inspect.signature(node).parameters
    accepts_config = "config" in parameters

    def projected(state: AgentState, config=None):
        update = node(state, config) if accepts_config else node(state)
        return with_workspace(state, update)

    projected.__name__ = node.__name__
    return projected


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("initialize", workspace_node(initialize))
    builder.add_node("retry_failed_stage", workspace_node(retry_failed_stage))
    builder.add_node("reconcile_stage", workspace_node(reconcile_stage))
    builder.add_node("understand", workspace_node(understand))
    builder.add_node("question", workspace_node(question))
    builder.add_node("build_plan", workspace_node(build_plan))
    builder.add_node("plan_approval", workspace_node(plan_approval))
    builder.add_node("materialize", workspace_node(materialize))
    builder.add_node("select_stage", workspace_node(select_stage))
    builder.add_node("execute_stage", workspace_node(execute_stage_node))
    builder.add_node("stage_review", workspace_node(stage_review))
    builder.add_node("validation_gate", workspace_node(validation_gate))
    builder.add_node("finalize", workspace_node(finalize))
    builder.add_node("report", workspace_node(report))
    builder.add_node("cancel", workspace_node(cancel))
    builder.add_node("author_adapter", workspace_node(author_adapter_node))
    builder.add_node("adapter_candidate", workspace_node(adapter_candidate))
    builder.add_node("patch_approval", workspace_node(patch_approval))
    builder.add_node("adapter_execution_confirmation", workspace_node(adapter_execution_confirmation))
    builder.add_node("adapter_collaboration", workspace_node(adapter_collaboration))
    builder.add_node("cleanup_sdk", workspace_node(cleanup_sdk_node))
    builder.add_node("exit", workspace_node(exit_node))

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        route_after_initialize,
        {"validation_gate": "validation_gate", "retry_failed_stage": "retry_failed_stage", "reconcile_stage": "reconcile_stage", "cancel": "cancel", "understand": "understand"},
    )
    builder.add_conditional_edges(
        "reconcile_stage",
        route_after_reconcile,
        {"stage_review": "stage_review", "validation_gate": "validation_gate", END: END},
    )
    builder.add_conditional_edges(
        "retry_failed_stage",
        lambda state: "execute_stage" if state.get("retrying_stage") else END,
        {"execute_stage": "execute_stage", END: END},
    )
    builder.add_conditional_edges("understand", route_intent, {
        "question": "question",
        "build_plan": "build_plan",
        "select_stage": "select_stage",
        "report": "report",
        "cleanup_sdk": "cleanup_sdk",
        "cancel": "cancel",
        "author_adapter": "author_adapter",
        "continue_coding": "adapter_candidate",
        "exit": "exit",
    })
    builder.add_conditional_edges("build_plan", route_after_plan, {"plan_approval": "plan_approval", END: END})
    builder.add_conditional_edges(
        "plan_approval",
        route_after_approval,
        {"plan_approval": "plan_approval", "materialize": "materialize", END: END},
    )
    builder.add_edge("materialize", "select_stage")
    builder.add_conditional_edges("select_stage", route_selected_stage, {"execute_stage": "execute_stage", "finalize": "finalize"})
    builder.add_conditional_edges(
        "execute_stage",
        route_after_stage,
        {"stage_review": "stage_review", "validation_gate": "validation_gate", END: END},
    )
    builder.add_conditional_edges(
        "stage_review",
        route_after_review,
        {"select_stage": "select_stage", "validation_gate": "validation_gate", END: END},
    )
    builder.add_conditional_edges("validation_gate", route_after_gate, {"select_stage": "select_stage", END: END})
    builder.add_edge("author_adapter", "adapter_candidate")
    builder.add_conditional_edges(
        "adapter_candidate",
        route_after_candidate,
        {"adapter_candidate": "adapter_candidate", "patch_approval": "patch_approval", END: END},
    )
    builder.add_conditional_edges("patch_approval", route_after_patch_decision, {
        "adapter_candidate": "adapter_candidate",
        "adapter_collaboration": "adapter_collaboration",
        "adapter_execution_confirmation": "adapter_execution_confirmation",
        "patch_approval": "patch_approval",
        END: END,
    })
    builder.add_conditional_edges(
        "adapter_execution_confirmation",
        route_after_execution_confirmation,
        {"build_plan": "build_plan", END: END},
    )
    builder.add_conditional_edges(
        "adapter_collaboration",
        lambda state: "adapter_candidate" if state.get("mode") == "ready_for_coding" else "patch_approval",
        {"adapter_candidate": "adapter_candidate", "patch_approval": "patch_approval"},
    )
    for node in ["question", "report", "cancel", "cleanup_sdk", "exit", "finalize"]:
        builder.add_edge(node, END)
    return builder.compile()


graph = build_graph()
