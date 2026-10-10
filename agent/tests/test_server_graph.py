import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agent_pipeline.chat_services import classify_intent, parse_adapter_request
from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.pipeline_planning import explicitly_removed_stages
from agent_pipeline.adapter_authoring.models import AdapterIntentReview
from agent_pipeline.server_graph import (
    _stage_artifacts,
    _validation_gate_result,
    adapter_candidate,
    adapter_collaboration,
    adapter_execution_confirmation,
    build_plan,
    cancel,
    graph,
    patch_approval,
    plan_approval,
    route_after_candidate,
    route_after_execution_confirmation,
    route_after_patch_decision,
    route_after_approval,
    route_after_review,
    route_after_stage,
    route_intent,
    route_selected_stage,
    stage_review,
    execute_stage_node,
    finalize,
    report,
    reconcile_stage,
    retry_failed_stage,
    route_after_gate,
    route_after_initialize,
    validation_gate,
    workspace_node,
)


def linked_pipeline_plan(**updates):
    plan = PipelineRequest(
        config_path="agent_pipeline/configs/eeg.toml",
        stages=["preflight"],
        execution_strategy="linked",
        validation_target_h5="/input/dataset/example.h5",
        full_input_root="/input/dataset",
        report_root="/output/run",
        reconstruction_gate={"min_pearson": 0.8},
    )
    return plan.with_updates(**updates)


def test_linked_request_defaults_and_root_validation():
    plan = linked_pipeline_plan()
    assert plan.validation_stages == ["preflight", "package", "tokenize", "reconstruct"]
    assert plan.full_stages == ["preflight", "package", "tokenize"]
    assert plan.request_for_phase("validation").target_h5.endswith("example.h5")
    assert plan.request_for_phase("full").input_root == "/input/dataset"
    with pytest.raises(ValueError, match="within full_input_root"):
        linked_pipeline_plan(validation_target_h5="/other/example.h5")


def test_validation_gate_requires_matching_identity_and_materializes_full_once():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "workflow_id": "wf_linked",
        "run_id": "run_validation",
        "run_root": "/output/validation",
        "config_path": "/output/validation/config.toml",
        "report_root": "/output/validation/reports",
        "approved_plan": linked_pipeline_plan().model_dump(mode="json"),
        "active_phase": "gate",
        "linked_runs": {
            "validation": {
                "workflow_id": "wf_linked", "run_id": "run_validation",
                "run_root": "/output/validation", "config_path": "/output/validation/config.toml",
                "report_root": "/output/validation/reports",
            },
        },
        "validation_report": {"metrics": {"pearson": 0.9}, "evaluated_sessions": 1, "evaluated_windows": 2},
        "validation_verdict": "pass",
        "gate_id": "gate_1",
        "gate_status": "passed",
        "pending_stages": [],
    }
    full = {
        "workflow_id": "wf_linked", "run_id": "run_full", "run_root": "/output/full",
        "config_path": "/output/full/config.toml", "report_root": "/output/full/reports",
    }
    with (
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "start_full_run", "gate_id": "gate_1"}),
        patch("agent_pipeline.server_graph.materialize_child_run", return_value=full) as materialize_full,
    ):
        update = validation_gate(state)
    assert update["active_phase"] == "full"
    assert update["pending_stages"] == ["preflight", "package", "tokenize"]
    assert update["linked_runs"]["full"]["run_id"] == "run_full"
    materialize_full.assert_called_once()
    expected_run_id = materialize_full.call_args.kwargs["run_id"]
    assert expected_run_id.startswith("run_full_")

    repeated = {**state, **update}
    with (
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "start_full_run", "gate_id": "gate_1"}),
        patch("agent_pipeline.server_graph.materialize_child_run") as duplicate_materialize,
    ):
        duplicate = validation_gate(repeated)
    duplicate_materialize.assert_not_called()
    assert duplicate["run_id"] == "run_full"

    with patch("agent_pipeline.server_graph.interrupt", return_value={"action": "start_full_run", "gate_id": "stale"}):
        with pytest.raises(ValueError, match="gate_id"):
            validation_gate(state)


def test_reconcile_request_routes_without_llm():
    assert route_after_initialize({"reconcile_requested": True}) == "reconcile_stage"
    assert route_after_initialize({"external_stage_job": {"stage": "tokenize"}}) == "reconcile_stage"
    assert route_after_initialize({
        "retry_requested": True,
        "external_stage_job": {"stage": "tokenize", "status": "failed"},
    }) == "retry_failed_stage"


def test_explicit_cancel_routes_to_durable_cancel_without_llm():
    state = {
        "external_stage_job": {"stage": "package", "stage_execution_id": "stage_package_123"},
        "messages": [SimpleNamespace(type="human", content="取消")],
    }
    assert route_after_initialize(state) == "cancel"


def test_running_external_stage_reconcile_remains_detached():
    state = {
        "run_root": "/run",
        "current_stage": "tokenize",
        "external_stage_job": {"stage": "tokenize", "stage_execution_id": "stage_tokenize_123"},
    }
    with patch("agent_pipeline.server_graph.reconcile_stage_job", return_value={
        "stage": "tokenize", "stage_execution_id": "stage_tokenize_123", "status": "running",
    }):
        update = reconcile_stage(state)
    assert update["mode"] == "detached"
    assert update["external_stage_job"]["status"] == "running"


def test_completed_external_stage_is_harvested_once():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "workflow_id": "wf",
        "run_id": "run",
        "run_root": "/run",
        "report_root": "/reports",
        "current_stage": "tokenize",
        "pending_stages": ["tokenize", "reconstruct"],
        "completed_stages": [],
        "stage_reports": [],
        "artifacts": [],
        "external_stage_job": {"stage": "tokenize", "stage_execution_id": "stage_tokenize_123"},
    }
    durable = {
        "stage": "tokenize",
        "stage_execution_id": "stage_tokenize_123",
        "status": "completed",
        "report": {"stage": "tokenize", "status": "ok", "stage_execution_id": "stage_tokenize_123"},
    }
    with patch("agent_pipeline.server_graph.reconcile_stage_job", return_value=durable):
        update = reconcile_stage(state)
    assert update["mode"] == "reviewing_stage"
    assert update["completed_stages"] == ["tokenize"]
    assert update["pending_stages"] == ["reconstruct"]
    assert update["stage_reports"][0]["stage_execution_id"] == "stage_tokenize_123"

    repeated = {**state, **update, "external_stage_job": state["external_stage_job"]}
    with patch("agent_pipeline.server_graph.reconcile_stage_job", return_value=durable):
        duplicate = reconcile_stage(repeated)
    assert "stage_reports" not in duplicate
    assert duplicate["mode"] == "reviewing_stage"


def test_reconcile_rejects_a_job_from_another_linked_run():
    state = {
        "run_id": "run_full", "run_root": "/output/full", "active_phase": "full",
        "linked_runs": {"full": {"run_id": "run_full", "run_root": "/output/full"}},
        "external_stage_job": {
            "stage": "tokenize", "stage_execution_id": "stage_old",
            "run_id": "run_validation", "phase": "validation",
        },
    }
    update = reconcile_stage(state)
    assert update["mode"] == "failed"
    assert update["last_error"]["code"] == "stale_stage_job"


def test_gate_phase_accepts_its_validation_job_identity_during_recovery():
    state = {
        "run_id": "run_validation", "run_root": "/output/validation", "active_phase": "gate",
        "linked_runs": {"validation": {"run_id": "run_validation", "run_root": "/output/validation"}},
        "external_stage_job": {
            "stage": "reconstruct", "stage_execution_id": "stage_reconstruct",
            "run_id": "run_validation", "phase": "validation",
        },
    }
    with patch("agent_pipeline.server_graph.reconcile_stage_job", return_value={
        **state["external_stage_job"], "status": "running",
    }):
        update = reconcile_stage(state)
    assert update["mode"] == "detached"
    assert update["last_error"] is None


def test_orphaned_external_stage_is_retryable():
    state = {
        "run_root": "/run",
        "current_stage": "reconstruct",
        "external_stage_job": {"stage": "reconstruct", "stage_execution_id": "stage_reconstruct_123"},
    }
    with patch("agent_pipeline.server_graph.reconcile_stage_job", return_value={
        "stage": "reconstruct",
        "stage_execution_id": "stage_reconstruct_123",
        "status": "orphaned",
        "error": "tmux session disappeared",
        "log_tail": "last batch",
    }):
        update = reconcile_stage(state)
    assert update["mode"] == "failed"
    assert update["last_error"]["retryable"] is True
    assert update["last_error"]["details"]["log_tail"] == "last batch"


def test_reconstruction_report_enters_explicit_gate_with_threshold_verdict():
    plan = linked_pipeline_plan()
    passed = _validation_gate_result(plan, {
        "status": "ok",
        "metrics": {"pearson": 0.85},
        "evaluated_sessions": 1,
        "evaluated_windows": 4,
    })
    failed = _validation_gate_result(plan, {
        "status": "ok",
        "metrics": {"pearson": 0.75},
        "evaluated_sessions": 1,
        "evaluated_windows": 4,
    })
    assert passed["verdict"] == "pass"
    invalid_status = _validation_gate_result(plan, {
        "status": "failed",
        "metrics": {"pearson": 0.95},
        "evaluated_sessions": 1,
        "evaluated_windows": 4,
    })
    missing_windows = _validation_gate_result(plan, {
        "status": "ok",
        "metrics": {"pearson": 0.95},
        "evaluated_sessions": 1,
        "evaluated_windows": 0,
    })
    assert failed["verdict"] == "fail"
    assert failed["failures"][0]["check"] == "min_pearson"
    assert invalid_status["verdict"] == "fail"
    assert any(item["check"] == "status" for item in invalid_status["failures"])
    assert missing_windows["verdict"] == "fail"
    assert any(item["check"] == "min_evaluated_windows" for item in missing_windows["failures"])


def test_start_full_run_materializes_once_and_switches_phase():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "workflow_id": "wf_linked",
        "run_id": "run_validation",
        "run_root": "/output/validation",
        "config_path": "/output/validation/config.toml",
        "report_root": "/output/validation/reports",
        "approved_plan": linked_pipeline_plan().model_dump(mode="json"),
        "active_phase": "gate",
        "linked_runs": {"validation": {"run_id": "run_validation", "run_root": "/output/validation"}},
        "validation_report": {"verdict": "pass"},
        "validation_verdict": "pass",
        "gate_id": "gate_1",
        "gate_status": "passed",
        "completed_stages": ["preflight", "package", "tokenize", "reconstruct"],
        "pending_stages": [],
    }
    full_run = {
        "workflow_id": "wf_linked",
        "run_id": "run_full",
        "run_root": "/output/full",
        "config_path": "/output/full/config.toml",
        "report_root": "/output/full/reports",
    }
    with (
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "start_full_run", "gate_id": "gate_1"}),
        patch("agent_pipeline.server_graph.materialize_child_run", return_value=full_run) as materialize_child,
    ):
        update = validation_gate(state)
    assert update["active_phase"] == "full"
    assert update["pending_stages"] == ["preflight", "package", "tokenize"]
    assert update["linked_runs"]["full"]["run_id"] == "run_full"
    assert route_after_gate(update) == "select_stage"
    materialize_child.assert_called_once()

    repeated_state = {**state, **update}
    with (
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "start_full_run", "gate_id": "gate_1"}),
        patch("agent_pipeline.server_graph.materialize_child_run") as duplicate_materialization,
    ):
        repeated = validation_gate(repeated_state)
    duplicate_materialization.assert_not_called()
    assert repeated["run_id"] == "run_full"
    assert repeated["pending_stages"] == ["preflight", "package", "tokenize"]


def test_failed_gate_allows_rerun_pause_cancel_but_not_full_start():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "workflow_id": "wf_linked",
        "run_id": "run_validation",
        "approved_plan": linked_pipeline_plan().model_dump(mode="json"),
        "active_phase": "gate",
        "linked_runs": {"validation": {"run_id": "run_validation"}},
        "validation_report": {"verdict": "fail"},
        "validation_verdict": "fail",
        "gate_id": "gate_1",
        "gate_status": "failed",
    }
    with patch("agent_pipeline.server_graph.interrupt", return_value={"action": "start_full_run", "gate_id": "gate_1"}):
        blocked = validation_gate(state)
    assert blocked["mode"] == "awaiting_validation_gate"
    assert blocked["last_error"]["code"] == "validation_gate_failed"

    rerun = {"workflow_id": "wf_linked", "run_id": "run_validation_2", "run_root": "/output/v2", "config_path": "/output/v2/config.toml", "report_root": "/output/v2/reports"}
    with (
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "rerun_smoke", "gate_id": "gate_1"}),
        patch("agent_pipeline.server_graph.materialize_child_run", return_value=rerun),
    ):
        restarted = validation_gate(state)
    assert restarted["active_phase"] == "validation"
    assert restarted["validation_report"] is None
    assert restarted["pending_stages"] == ["preflight", "package", "tokenize", "reconstruct"]


def test_workspace_node_projects_lifecycle_update():
    node = workspace_node(lambda state: {"mode": "running", "current_stage": "preflight"})
    update = node({"pending_stages": ["preflight"], "artifacts": []})
    assert update["workspace"]["mode"] == "running"
    assert update["workspace"]["stages"][0]["status"] == "running"


def test_workspace_node_forwards_runtime_config_when_requested():
    def configured_node(state, config):
        return {"owner": config["configurable"]["owner"], "mode": "idle"}

    node = workspace_node(configured_node)
    update = node({"artifacts": []}, {"configurable": {"owner": "alice"}})
    assert update["owner"] == "alice"
    assert update["workspace"]["mode"] == "idle"


def test_workspace_node_never_exposes_cross_thread_artifacts():
    node = workspace_node(lambda state: {"mode": "completed"})
    update = node({
        "owner": "owner_1",
        "thread_id": "thread_1",
        "artifacts": [
            {"artifact_id": "owned", "kind": "report", "summary": "ok", "owner": "owner_1", "thread_id": "thread_1"},
            {"artifact_id": "foreign", "kind": "report", "summary": "no", "owner": "owner_2", "thread_id": "thread_1"},
        ],
    })
    assert [artifact["artifact_id"] for artifact in update["workspace"]["artifacts"]] == ["owned"]


def test_continue_coding_intent_routes_to_adapter_candidate():
    state = {"mode": "ready_for_coding", "intent": "continue_coding"}
    assert route_intent(state) == "continue_coding"


def test_existing_adapter_execution_request_does_not_enter_coding():
    llm = Mock()
    intent = classify_intent(
        llm,
        {"mode": "idle"},
        "使用 smn_meg_sub_02_ses_01_rdr_57_meg_adapter_3 处理 /home/luyilin/smn_h5/test/RDR/sub-02/ses-01/SMN_MEG-sub-02_ses-01_RDR_57.h5 输出到 /home/luyilin/braintoken/agent/outputs2",
    )

    assert intent.intent == "execute"
    assert intent.adapter_name == "smn_meg_sub_02_ses_01_rdr_57_meg_adapter_3"
    assert intent.input_path.endswith("SMN_MEG-sub-02_ses-01_RDR_57.h5")
    assert intent.output_path == "/home/luyilin/braintoken/agent/outputs2"
    assert intent.modality == "meg"
    llm.with_structured_output.assert_not_called()


def test_adapter_authoring_fast_path_preserves_requested_output_path():
    llm = Mock()
    intent = classify_intent(
        llm,
        {"mode": "idle"},
        "针对 /input/example.h5 编写 EEG adapter 然后输出到 /output/neurotoken",
    )

    assert intent.intent == "adapter_authoring"
    assert intent.output_path == "/output/neurotoken"
    llm.with_structured_output.assert_not_called()


def test_plain_adapter_word_does_not_trigger_authoring_fast_path():
    parser = Mock()
    parser.invoke.return_value = SimpleNamespace(intent="execute")
    llm = Mock()
    llm.with_structured_output.return_value = parser

    classify_intent(llm, {"mode": "idle"}, "用 adapter 处理 /data/example_meg.h5")

    llm.with_structured_output.assert_called_once()


def test_explicit_h5_coding_request_skips_intent_llm():
    llm = Mock()
    intent = classify_intent(
        llm,
        {"mode": "idle"},
        "请为 /data/SMN_MEG-sub-02_ses-01_RDR_57.h5 编写 adapter 代码",
    )
    assert intent.intent == "adapter_authoring"
    assert intent.input_path == "/data/SMN_MEG-sub-02_ses-01_RDR_57.h5"
    assert intent.modality == "meg"
    llm.with_structured_output.assert_not_called()


def test_explicit_h5_adapter_request_skips_parser_llm_and_uses_unique_name():
    llm = Mock()
    request = parse_adapter_request(
        llm,
        "请写代码处理 /data/subject-01_eeg.h5，并保留原始采样率",
        ["subject_01_eeg_adapter"],
    )
    assert request.source_path == "/data/subject-01_eeg.h5"
    assert request.modality == "eeg"
    assert request.name == "subject_01_eeg_adapter_2"
    assert "保留原始采样率" in request.notes
    llm.with_structured_output.assert_not_called()


def test_ambiguous_adapter_request_still_uses_llm_parser():
    parsed = SimpleNamespace(name="custom_eeg", source_path="/data/example.h5", modality="eeg", notes="")
    parser = Mock()
    parser.invoke.return_value = parsed
    llm = Mock()
    llm.with_structured_output.return_value = parser
    assert parse_adapter_request(llm, "请帮我写一个 adapter", []).name == "custom_eeg"
    llm.with_structured_output.assert_called_once()


def test_extracts_single_and_multiple_explicit_stage_removals():
    assert explicitly_removed_stages("去掉 readme") == ["readme"]
    assert explicitly_removed_stages("Remove docs, validate and reconstruct stages") == [
        "docs",
        "validate",
        "reconstruct",
    ]
    assert explicitly_removed_stages("保留 preflight 和 package") == []


def test_server_graph_exposes_chat_and_approval_nodes():
    nodes = set(graph.get_graph().nodes)
    assert {
        "understand",
        "build_plan",
        "plan_approval",
        "materialize",
        "execute_stage",
        "stage_review",
        "author_adapter",
        "adapter_candidate",
        "patch_approval",
        "adapter_collaboration",
        "cleanup_sdk",
    } <= nodes


def pipeline_plan(**updates):
    plan = PipelineRequest(
        config_path="agent_pipeline/configs/eeg.toml",
        stages=["preflight", "package"],
        target_h5="/input/example.h5",
        report_root="/output/run",
    )
    return plan.model_copy(update=updates)


def test_plan_refinement_updates_plan_and_requires_another_approval():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "current_plan": pipeline_plan().model_dump(mode="json"),
    }
    refined = pipeline_plan(stages=["validate"], strict_validation=True)
    with (
        patch(
            "agent_pipeline.server_graph.interrupt",
            return_value={"action": "refine", "message": "增加严格验证"},
        ) as interrupt_mock,
        patch("agent_pipeline.server_graph.llm", return_value=Mock()),
        patch("agent_pipeline.server_graph.refine_request", return_value=refined) as refine,
        patch(
            "agent_pipeline.server_graph.resolve_stage_dependencies",
            side_effect=lambda plan: plan.stages.insert(0, "preflight") or ["preflight"],
        ),
    ):
        update = plan_approval(state)

    assert update["mode"] == "awaiting_plan_approval"
    assert update["approved_plan"] is None
    assert update["plan_revision"] == 2
    assert update["current_plan"]["stages"] == ["preflight", "validate"]
    assert update["current_plan"]["strict_validation"] is True
    assert route_after_approval(update) == "plan_approval"
    assert interrupt_mock.call_args.args[0]["plan_revision"] == 1
    refine.assert_called_once()


def test_plan_refinement_reports_removed_and_dependency_retained_stages():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "current_plan": pipeline_plan(stages=["preflight", "package", "readme", "docs"]).model_dump(mode="json"),
    }
    refined = pipeline_plan(stages=["package"])
    with (
        patch(
            "agent_pipeline.server_graph.interrupt",
            return_value={"action": "refine", "message": "去掉 preflight、readme 和 docs"},
        ),
        patch("agent_pipeline.server_graph.llm", return_value=Mock()),
        patch("agent_pipeline.server_graph.refine_request", return_value=refined),
    ):
        update = plan_approval(state)

    assert update["current_plan"]["stages"] == ["preflight", "package"]
    message = update["messages"][0].content
    assert "已移除阶段：readme, docs" in message
    assert "无法移除的依赖阶段：preflight" in message
    assert "自动补齐前置阶段：preflight" in message


def test_plan_refinement_requires_natural_language_instruction():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "current_plan": pipeline_plan().model_dump(mode="json"),
    }
    with patch("agent_pipeline.server_graph.interrupt", return_value={"action": "refine", "message": "  "}):
        try:
            plan_approval(state)
        except ValueError as exc:
            assert "refinement instruction" in str(exc)
        else:
            raise AssertionError("Expected an empty refinement instruction to be rejected")


def pending_adapter_state():
    return {
        "owner": "owner",
        "thread_id": "thread",
        "pending_adapter": {
            "task_id": "adapter_1",
            "request_id": "adapter_1",
            "name": "custom_eeg",
            "source_path": "/input/example.h5",
            "modality": "eeg",
            "notes": "",
            "h5_summary": "datasets: signal",
            "coding_summary": "initial",
            "original_user_request": "Create an EEG adapter and preserve explicit event chunking.",
            "requirements_snapshot": {
                "requirements": {
                    "modality": "eeg",
                    "output_path": "/output/requested",
                    "config_overrides": {"chunking": {"strategy": "event"}},
                },
                "other_requirements": [],
            },
            "diff": "diff --git a/a b/a",
            "revision": 1,
            "collaboration": [],
        },
    }


def active_task(status="awaiting_patch_approval", revision=1, codex_thread_id=None):
    return SimpleNamespace(
        task_id="adapter_1",
        thread_id="thread",
        revision=revision,
        status=status,
        adapter=SimpleNamespace(name="custom_eeg"),
        allowed_paths=["agent_pipeline/adapters"],
        instruction="Refine the adapter candidate.",
        worktree_path="/tmp/worktree",
        codex_thread_id=codex_thread_id,
    )


def test_patch_approval_exposes_intent_review():
    state = pending_adapter_state()
    state["pending_adapter"]["intent_review"] = passing_intent_review().model_dump(mode="json")
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "ask", "message": "为什么这样读取？"}) as interrupt_mock,
    ):
        patch_approval(state)

    adapter = interrupt_mock.call_args.args[0]["adapter"]
    assert adapter["intent_review"]["verdict"] == "pass"
    assert adapter["original_user_request"].startswith("Create an EEG adapter")
    assert adapter["requirements_snapshot"]["requirements"]["modality"] == "eeg"


def test_patch_approval_routes_questions_without_applying():
    state = pending_adapter_state()
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "ask", "message": "为什么这样读取？"}),
    ):
        update = patch_approval(state)
    assert update["adapter_collaboration_action"] == "ask"
    assert update["mode"] == "adapter_question"
    assert route_after_patch_decision(update) == "adapter_collaboration"


def test_patch_approval_reopens_candidate_when_frozen_diff_is_missing():
    state = pending_adapter_state()
    state["pending_adapter"]["diff"] = ""
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    runtime.update_status.return_value = active_task(status="needs_revision")

    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt") as interrupt_mock,
    ):
        update = patch_approval(state)

    assert update["mode"] == "adapter_revision_required"
    assert "缺少冻结 diff" in update["messages"][0].content
    assert route_after_patch_decision(update) == "adapter_candidate"
    runtime.update_status.assert_called_once_with(
        "adapter_1",
        "needs_revision",
        expected_revision=1,
        error="Approval checkpoint is missing the frozen candidate diff.",
    )
    interrupt_mock.assert_not_called()


def test_patch_approval_ignores_duplicate_candidate_completion():
    state = pending_adapter_state()
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value=candidate_submission()),
    ):
        update = patch_approval(state)
    assert update["mode"] == "awaiting_patch_approval"
    assert update["last_error"] is None
    assert "忽略重复" in update["messages"][0].content
    assert route_after_patch_decision(update) == "patch_approval"


def test_patch_approval_reprompts_after_empty_collaboration_message():
    state = pending_adapter_state()
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "ask", "message": "   "}),
    ):
        update = patch_approval(state)
    assert update["mode"] == "awaiting_patch_approval"
    assert "不能为空" in update["messages"][0].content
    assert route_after_patch_decision(update) == "patch_approval"


def test_patch_approval_reprompts_after_empty_commit_message():
    state = pending_adapter_state()
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch(
            "agent_pipeline.server_graph.interrupt",
            return_value={"action": "apply_and_commit", "commit_message": "   "},
        ),
    ):
        update = patch_approval(state)
    assert update["mode"] == "awaiting_patch_approval"
    assert "commit message" in update["messages"][0].content
    assert route_after_patch_decision(update) == "patch_approval"


def test_patch_approval_reprompts_after_stale_or_wrong_identity():
    state = pending_adapter_state()
    for decision, expected in (
        ({"action": "reject", "task_id": "wrong", "revision": 1}, "不匹配"),
        ({"action": "reject", "task_id": "adapter_1", "revision": 2}, "版本已更新"),
    ):
        runtime = Mock()
        runtime.require_task.return_value = active_task()
        with (
            patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
            patch("agent_pipeline.server_graph.interrupt", return_value=decision),
        ):
            update = patch_approval(state)
        assert update["mode"] == "awaiting_patch_approval"
        assert expected in update["messages"][0].content
        assert route_after_patch_decision(update) == "patch_approval"
        runtime.cleanup_task.assert_not_called()


def test_adapter_question_preserves_candidate_and_returns_to_approval():
    state = {
        **pending_adapter_state(),
        "adapter_collaboration_action": "ask",
        "adapter_collaboration_message": "为什么这样读取？",
    }
    with (
        patch("agent_pipeline.server_graph.llm"),
        patch("agent_pipeline.server_graph.answer_adapter_question", return_value="因为数据布局如此。"),
    ):
        update = adapter_collaboration(state)
    assert update["mode"] == "awaiting_patch_approval"
    assert update["pending_adapter"]["diff"] == state["pending_adapter"]["diff"]
    assert update["pending_adapter"]["collaboration"][0]["kind"] == "question"


def test_adapter_refinement_reopens_durable_task_with_new_revision():
    state = {
        **pending_adapter_state(),
        "adapter_collaboration_action": "refine",
        "adapter_collaboration_message": "改成分块读取",
    }
    runtime = Mock()
    runtime.begin_revision.return_value = active_task(
        status="coding",
        revision=2,
        codex_thread_id="codex-1",
    )
    with patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime):
        update = adapter_collaboration(state)
    assert update["mode"] == "ready_for_coding"
    assert update["pending_adapter"]["diff"] == ""
    assert update["pending_adapter"]["previous_candidate_diff"] == state["pending_adapter"]["diff"]
    assert update["pending_adapter"]["instruction"] == runtime.begin_revision.return_value.instruction
    assert update["pending_adapter"]["revision"] == 2
    assert update["pending_adapter"]["codex_thread_id"] == "codex-1"
    assert update["pending_adapter"]["collaboration"][0]["kind"] == "refinement"
    runtime.begin_revision.assert_called_once()


def candidate_submission(revision=1, replaces_codex_thread_id=None):
    return {
        "action": "submit_candidate",
        "task_id": "adapter_1",
        "thread_id": "thread",
        "codex_thread_id": "codex_1",
        "replaces_codex_thread_id": replaces_codex_thread_id,
        "revision": revision,
    }


def passing_intent_review():
    return AdapterIntentReview(
        verdict="pass",
        summary="The candidate satisfies the recorded human intent.",
        findings=[],
    )


def test_adapter_candidate_without_pending_task_is_recoverable():
    update = adapter_candidate({"pending_adapter": None, "last_adapter": {"name": "ready_adapter"}})

    assert update["mode"] == "adapter_ready"
    assert update["last_error"] is None
    assert "没有待提交" in update["messages"][0].content


def test_adapter_candidate_rejects_stale_revision_before_diffing():
    state = pending_adapter_state()
    runtime = Mock()
    runtime.require_task.return_value = active_task(status="coding", revision=2)
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value=candidate_submission()),
        patch("agent_pipeline.server_graph.diff_worktree") as diff,
    ):
        try:
            adapter_candidate(state)
        except RuntimeError as exc:
            assert "Stale adapter candidate revision" in str(exc)
        else:
            raise AssertionError("Expected stale candidate rejection")
    diff.assert_not_called()


def test_candidate_routing_stops_after_validation_failure():
    assert route_after_candidate({"mode": "adapter_revision_required"}) == "__end__"
    assert route_after_candidate({"mode": "ready_for_coding"}) == "__end__"
    assert route_after_candidate({"mode": "awaiting_patch_approval"}) == "patch_approval"
    assert route_after_candidate({"mode": "failed"}) == "__end__"


def test_adapter_candidate_validates_entire_worktree_and_enters_approval():
    state = pending_adapter_state()
    coding = active_task(status="coding")
    validating = active_task(status="validating")
    awaiting = active_task(status="awaiting_patch_approval")
    runtime = Mock()
    runtime.require_task.return_value = coding
    runtime.bind_coding_thread.return_value = coding
    runtime.update_status.side_effect = [validating, awaiting]
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value=candidate_submission()),
        patch("agent_pipeline.server_graph.changed_worktree_paths", return_value=["agent_pipeline/adapters/example.py"]),
        patch("agent_pipeline.server_graph.diff_worktree", return_value="adapter diff") as diff,
        patch("agent_pipeline.server_graph.llm", return_value=Mock()),
        patch("agent_pipeline.server_graph.run_candidate_preflight", return_value={"status": "ok", "source_session_loaded": True, "event_count": 2}) as preflight,
        patch("agent_pipeline.server_graph.review_adapter_candidate", return_value=passing_intent_review()) as review,
    ):
        update = adapter_candidate(state)
    assert update["mode"] == "awaiting_patch_approval"
    assert update["pending_adapter"]["diff"] == "adapter diff"
    assert update["pending_adapter"]["intent_review"]["verdict"] == "pass"
    assert update["pending_adapter"]["effective_config"]["source"]["adapter"] == "custom_eeg"
    assert update["pending_adapter"]["effective_config"]["chunking"]["strategy"] == "event"
    assert update["pending_adapter"]["effective_config"]["paths"]["report_root"].startswith("/output/requested/")
    assert update["pending_adapter"]["preflight_evidence"]["source_session_loaded"] is True
    diff.assert_called_once_with(coding.worktree_path, coding.allowed_paths)
    preflight.assert_called_once()
    assert preflight.call_args.kwargs["effective_config"]["paths"]["tar_root"].startswith("/output/requested/")
    review.assert_called_once()
    review_kwargs = review.call_args.kwargs
    assert review_kwargs["original_user_request"].startswith("Create an EEG adapter")
    assert review_kwargs["requirements_snapshot"]["requirements"]["modality"] == "eeg"
    assert review_kwargs["preflight_evidence"]["status"] == "ok"
    assert review_kwargs["allowed_paths"] == coding.allowed_paths
    runtime.bind_coding_thread.assert_called_once_with(
        coding.task_id,
        expected_revision=coding.revision,
        codex_thread_id="codex_1",
        replaces_codex_thread_id=None,
    )


def test_adapter_candidate_blocks_semantic_mismatch_before_approval():
    state = pending_adapter_state()
    state["pending_adapter"]["notes"] = "Split every event and every gap between adjacent events."
    coding = active_task(status="coding")
    coding.instruction = state["pending_adapter"]["notes"]
    validating = active_task(status="validating")
    validating.instruction = coding.instruction
    needs_revision = active_task(status="needs_revision")
    runtime = Mock()
    runtime.require_task.return_value = coding
    runtime.bind_coding_thread.return_value = coding
    runtime.update_status.side_effect = [validating, needs_revision]
    review = AdapterIntentReview(
        verdict="blocked_by_scope",
        summary="The candidate preserves events but shared packaging drops inter-event time ranges.",
        findings=[{
            "severity": "error",
            "title": "Retain inter-event time ranges",
            "evidence": "Shared sample_ranges yields only source events.",
            "required_change": "Update the permitted implementation scope so packaging emits event metadata, an ordinary segment without event metadata, then event metadata.",
        }],
    )
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value=candidate_submission()),
        patch("agent_pipeline.server_graph.changed_worktree_paths", return_value=["agent_pipeline/adapters/example.py"]),
        patch("agent_pipeline.server_graph.diff_worktree", return_value="adapter diff"),
        patch("agent_pipeline.server_graph.llm", return_value=Mock()),
        patch("agent_pipeline.server_graph.review_adapter_candidate", return_value=review),
    ):
        update = adapter_candidate(state)

    assert update["mode"] == "adapter_revision_required"
    assert update["pending_adapter"]["intent_review"]["verdict"] == "blocked_by_scope"
    assert "Retain inter-event time ranges" in update["pending_adapter"]["validation_error"]
    assert runtime.update_status.call_args_list[-1].args[1] == "needs_revision"


def test_adapter_candidate_exposes_unavailable_review_for_human_fallback():
    state = pending_adapter_state()
    coding = active_task(status="coding")
    validating = active_task(status="validating")
    awaiting = active_task(status="awaiting_patch_approval")
    runtime = Mock()
    runtime.require_task.return_value = coding
    runtime.bind_coding_thread.return_value = coding
    runtime.update_status.side_effect = [validating, awaiting]
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value=candidate_submission()),
        patch("agent_pipeline.server_graph.changed_worktree_paths", return_value=["agent_pipeline/adapters/example.py"]),
        patch("agent_pipeline.server_graph.diff_worktree", return_value="adapter diff"),
        patch("agent_pipeline.server_graph.llm", return_value=Mock()),
        patch("agent_pipeline.server_graph.review_adapter_candidate", side_effect=RuntimeError("review provider offline")),
    ):
        update = adapter_candidate(state)

    assert update["mode"] == "awaiting_patch_approval"
    assert update["pending_adapter"]["intent_review"]["verdict"] == "unavailable"
    assert "人工核对" in update["messages"][0].content


def test_unchanged_revision_candidate_is_rejected_before_approval():
    state = pending_adapter_state()
    state["pending_adapter"].update({
        "revision": 2,
        "previous_candidate_diff": "adapter diff",
        "diff": "",
    })
    coding = active_task(status="coding", revision=2)
    validating = active_task(status="validating", revision=2)
    needs_revision = active_task(status="needs_revision", revision=2)
    runtime = Mock()
    runtime.require_task.return_value = coding
    runtime.bind_coding_thread.return_value = coding
    runtime.update_status.side_effect = [validating, needs_revision]
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value=candidate_submission(revision=2)),
        patch("agent_pipeline.server_graph.changed_worktree_paths", return_value=["agent_pipeline/adapters/example.py"]),
        patch("agent_pipeline.server_graph.diff_worktree", return_value="adapter diff"),
    ):
        update = adapter_candidate(state)

    assert update["mode"] == "adapter_revision_required"
    assert "no changes relative to the previous candidate" in update["pending_adapter"]["validation_error"]
    assert update["pending_adapter"]["diff"] == "adapter diff"
    assert runtime.update_status.call_args_list[-1].args[1] == "needs_revision"


def test_invalid_candidate_stops_without_opening_automatic_revision():
    state = pending_adapter_state()
    coding = active_task(status="coding")
    validating = active_task(status="validating")
    needs_revision = active_task(status="needs_revision")
    validating.instruction = "write adapter"
    runtime = Mock()
    runtime.require_task.return_value = coding
    runtime.bind_coding_thread.return_value = coding
    runtime.update_status.side_effect = [validating, needs_revision]
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value=candidate_submission()),
        patch("agent_pipeline.server_graph.changed_worktree_paths", return_value=["README.md"]),
        patch("agent_pipeline.server_graph.diff_worktree", return_value="adapter diff"),
    ):
        update = adapter_candidate(state)
    assert update["mode"] == "adapter_revision_required"
    assert update["pending_adapter"]["revision"] == 1
    assert update["pending_adapter"]["diff"] == "adapter diff"
    assert update["pending_adapter"]["changed_paths"] == ["README.md"]
    runtime.begin_revision.assert_not_called()


def test_patch_approval_reject_cleans_task_and_ends():
    state = pending_adapter_state()
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "reject", "task_id": "adapter_1", "revision": 1}),
    ):
        update = patch_approval(state)

    runtime.cleanup_task.assert_called_once_with("adapter_1", status="rejected", owner="owner")
    assert update["mode"] == "cancelled"
    assert update["pending_adapter"] is None
    assert update["approved_plan"] is None
    assert update["pending_stages"] == []
    assert update["last_error"] is None
    assert "发送新的要求" in update["messages"][0].content
    assert route_after_patch_decision(update) == "__end__"


def test_patch_approval_requires_commit_message_for_apply_and_commit():
    state = pending_adapter_state()
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "apply_and_commit", "task_id": "adapter_1", "revision": 1}),
    ):
        update = patch_approval(state)
    assert update["mode"] == "awaiting_patch_approval"
    assert "commit message" in update["messages"][0].content
    assert route_after_patch_decision(update) == "patch_approval"


def test_patch_approval_apply_and_commit_applies_and_cleans_task():
    state = pending_adapter_state()
    state["requirements"] = {
        "config_overrides": {
            "chunking": {"strategy": "event", "downsample_factor": 4},
            "runtime": {"device": "cpu"},
        }
    }
    runtime = Mock()
    runtime.require_task.return_value = active_task()
    patch_record = SimpleNamespace(name="custom_eeg.py")
    with (
        patch("agent_pipeline.server_graph.adapter_task_runtime", return_value=runtime),
        patch("agent_pipeline.server_graph.interrupt", return_value={"action": "apply_and_commit", "task_id": "adapter_1", "revision": 1, "commit_message": "Add custom EEG adapter"}),
        patch("agent_pipeline.server_graph.apply_authoring_result", return_value=patch_record),
        patch("agent_pipeline.server_graph.ROOT"),
        patch("importlib.reload"),
    ):
        update = patch_approval(state)

    runtime.cleanup_task.assert_called_once_with("adapter_1", status="applied", owner="owner")
    assert update["mode"] == "adapter_ready"
    assert update["last_adapter"]["patch_record"] == "custom_eeg.py"
    assert update["requirements"]["config_overrides"] == {
        "chunking": {"strategy": "event", "downsample_factor": 4},
        "runtime": {"device": "cpu"},
    }
    assert route_after_patch_decision(update) == "adapter_execution_confirmation"


def test_adapter_execution_confirmation_can_start_pipeline():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "last_adapter": {"name": "custom_eeg_adapter", "source_path": "/input/example.h5"},
        "requirements": {"input_path": "/input/example.h5", "output_path": "/output/run"},
    }
    with patch("agent_pipeline.server_graph.interrupt", return_value={"action": "continue"}):
        update = adapter_execution_confirmation(state)

    assert update["intent"] == "execute"
    assert update["mode"] == "planning"
    assert route_after_execution_confirmation(update) == "build_plan"


def test_adapter_execution_confirmation_can_keep_adapter_without_running():
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "last_adapter": {"name": "custom_eeg_adapter", "source_path": "/input/example.h5"},
        "requirements": {},
    }
    with patch("agent_pipeline.server_graph.interrupt", return_value={"action": "pause"}):
        update = adapter_execution_confirmation(state)

    assert update["mode"] == "adapter_ready"
    assert route_after_execution_confirmation(update) == "__end__"


def test_new_plan_clears_previous_execution_identity():
    request = PipelineRequest(
        config_path="agent_pipeline/configs/eeg_h5.toml",
        stages=["preflight"],
        target_h5="/input/new.h5",
        report_root="/output/new",
    )
    state = {
        "intent": "execute",
        "requirements": {},
        "other_requirements": [],
        "last_adapter": None,
        "approved_plan": {"target_h5": "/input/old.h5"},
        "workflow_id": "wf_old",
        "run_id": "run_old",
        "run_root": "/output/old",
        "config_path": "/output/old/config.toml",
        "report_root": "/output/old/reports",
        "current_stage": "tokenize",
        "current_tool_call_id": "tool_old",
        "artifacts": [{"artifact_id": "old"}],
        "last_error": {"code": "old"},
        "messages": [],
    }
    with (
        patch("agent_pipeline.server_graph.latest_user_text", return_value="execute new"),
        patch("agent_pipeline.server_graph.llm"),
        patch("agent_pipeline.server_graph.runtime_config", return_value={"default_config": "unused"}),
        patch("agent_pipeline.server_graph.build_pipeline_request", return_value=(request, [])),
    ):
        update = build_plan(state)

    for field in (
        "approved_plan",
        "workflow_id",
        "run_id",
        "run_root",
        "config_path",
        "report_root",
        "current_stage",
        "current_tool_call_id",
        "last_error",
    ):
        assert update[field] is None
    assert update["artifacts"] == []


def test_rejected_or_idle_approval_never_materializes_stale_plan():
    assert route_after_approval({"mode": "idle", "approved_plan": {"stages": ["preflight"]}}) == "__end__"
    assert route_after_approval({"mode": "approved", "approved_plan": {"stages": ["preflight"]}}) == "materialize"


def test_continue_only_resumes_paused_workflow():
    base = {"intent": "continue", "approved_plan": {"stages": ["preflight"]}, "pending_stages": ["preflight"]}
    assert route_intent({**base, "mode": "cancelled"}) == "question"
    assert route_intent({**base, "mode": "paused"}) == "select_stage"


def test_continue_finalizes_paused_workflow_after_last_stage():
    state = {
        "intent": "continue",
        "mode": "paused",
        "approved_plan": {"stages": ["preflight"]},
        "completed_stages": ["preflight"],
        "pending_stages": [],
    }
    assert route_intent(state) == "select_stage"
    assert route_selected_stage(state) == "finalize"


def test_cancel_is_terminal_and_clears_execution_state():
    update = cancel({})
    assert update["mode"] == "cancelled"
    assert update["approved_plan"] is None
    assert update["pending_stages"] == []
    assert update["current_stage"] is None
    assert update["current_tool_call_id"] is None


def stage_review_state(action):
    return {
        "stage_reports": [{"stage": "preflight", "status": "ok"}],
        "owner": "owner",
        "thread_id": "thread",
        "workflow_id": "wf_test",
        "pending_stages": ["package"],
        "mode": "reviewing_stage",
        "approved_plan": {"stages": ["preflight", "package"]},
    }, {"action": action}


def test_stage_review_exposes_package_sample_in_interrupt():
    state, decision = stage_review_state("continue")
    state["stage_reports"] = [{
        "stage": "package",
        "status": "ok",
        "sessions": [{
            "tar_path": "/tmp/session/data.tar",
            "tar_sample": [{"name": "sample.meg.npy", "size": 4096, "type": "file"}],
            "segmentation_review": {
                "strategy": "rest",
                "sfreq": 100.0,
                "total_samples": 2500,
                "total_duration_seconds": 25.0,
                "segment_count": 3,
                "coverage_percent": 100.0,
                "min_segment_seconds": 5.0,
                "max_segment_seconds": 10.0,
                "split_points": 1000,
                "uncovered_ranges": [],
                "overlaps": [],
                "preview_segments": [{
                    "chunk_index": 0,
                    "start_sample": 0,
                    "end_sample": 1000,
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "duration_seconds": 10.0,
                }],
            },
            "tar_sample_metadata_member": "sample.metadata.json",
            "tar_sample_metadata": {"chunk_index": 0, "n_samples": 1000},
        }],
    }]
    with (
        patch("agent_pipeline.server_graph.llm"),
        patch("agent_pipeline.server_graph.summarize_stage", return_value="ok"),
        patch("agent_pipeline.server_graph.interrupt", return_value=decision) as interrupt_mock,
    ):
        update = stage_review(state)

    payload = interrupt_mock.call_args.args[0]
    session = payload["report"]["sessions"][0]
    assert session["tar_sample"][0]["name"] == "sample.meg.npy"
    assert session["tar_sample_metadata"]["chunk_index"] == 0
    assert session["segmentation_review"]["preview_segments"][0]["end_sample"] == 1000
    assert "## tar 切分审核" in update["messages"][0].content
    assert "每 1000 samples 切一段" in update["messages"][0].content
    assert "samples `[0, 1000)`" in update["messages"][0].content
    assert "## tar 内容样例" in update["messages"][0].content
    assert "`sample.meg.npy`（4.0 KiB）" in update["messages"][0].content
    assert '"chunk_index": 0' in update["messages"][0].content


def test_stage_review_backfills_legacy_package_report_before_interrupt(tmp_path: Path):
    tar_path = tmp_path / "data.tar"
    metadata_path = tmp_path / "sample.metadata.json"
    metadata_path.write_text(json.dumps({
        "chunking_strategy": "rest",
        "sfreq": 100.0,
        "n_samples_total": 1000,
        "chunk_index": 0,
        "chunk_start_sample": 0,
        "chunk_end_sample": 1000,
        "n_samples": 1000,
    }), encoding="utf-8")
    with tarfile.open(tar_path, "w") as archive:
        archive.add(metadata_path, arcname=metadata_path.name)
    state, decision = stage_review_state("continue")
    state["stage_reports"] = [{
        "stage": "package",
        "status": "ok",
        "sessions": [{"tar_path": str(tar_path)}],
    }]
    with (
        patch("agent_pipeline.server_graph.llm"),
        patch("agent_pipeline.server_graph.summarize_stage", return_value="ok"),
        patch("agent_pipeline.server_graph.interrupt", return_value=decision) as interrupt_mock,
    ):
        stage_review(state)

    session = interrupt_mock.call_args.args[0]["report"]["sessions"][0]
    assert session["tar_sample"][0]["name"] == "sample.metadata.json"
    assert session["tar_sample_metadata"]["chunk_index"] == 0
    assert session["segmentation_review"]["segment_count"] == 1
    assert session["segmentation_review"]["coverage_percent"] == 100.0


def test_stage_review_exposes_readme_content_in_interrupt():
    state, decision = stage_review_state("continue")
    state["stage_reports"] = [{
        "stage": "readme",
        "status": "ok",
        "readme_path": "/tmp/neurotoken/README.md",
        "readme_content": "# Example dataset\n\nGenerated README body.",
    }]
    with (
        patch("agent_pipeline.server_graph.llm"),
        patch("agent_pipeline.server_graph.summarize_stage", return_value="ok"),
        patch("agent_pipeline.server_graph.interrupt", return_value=decision) as interrupt_mock,
    ):
        stage_review(state)

    payload = interrupt_mock.call_args.args[0]
    assert payload["report"]["readme_path"] == "/tmp/neurotoken/README.md"
    assert payload["report"]["readme_content"].startswith("# Example dataset")


def test_stage_review_continue_resumes_next_stage():
    state, decision = stage_review_state("continue")
    with (
        patch("agent_pipeline.server_graph.llm"),
        patch("agent_pipeline.server_graph.summarize_stage", return_value="ok"),
        patch("agent_pipeline.server_graph.interrupt", return_value=decision),
    ):
        update = stage_review(state)
    assert update["mode"] == "running"
    assert route_after_review(update) == "select_stage"


def test_stage_review_pause_preserves_remaining_stages():
    state, decision = stage_review_state("pause")
    with (
        patch("agent_pipeline.server_graph.llm"),
        patch("agent_pipeline.server_graph.summarize_stage", return_value="ok"),
        patch("agent_pipeline.server_graph.interrupt", return_value=decision),
    ):
        update = stage_review(state)
    assert update["mode"] == "paused"
    assert route_after_review(update) == "__end__"


def test_stage_review_cancel_clears_remaining_execution_state():
    state, decision = stage_review_state("cancel")
    with (
        patch("agent_pipeline.server_graph.llm"),
        patch("agent_pipeline.server_graph.summarize_stage", return_value="ok"),
        patch("agent_pipeline.server_graph.interrupt", return_value=decision),
    ):
        update = stage_review(state)
    assert update["mode"] == "cancelled"
    assert update["approved_plan"] is None
    assert update["pending_stages"] == []
    assert route_after_review(update) == "__end__"


def test_execute_readme_stage_embeds_generated_content(tmp_path: Path):
    report_root = tmp_path / "reports"
    output_root = tmp_path / "neurotoken"
    report_root.mkdir()
    output_root.mkdir()
    readme_path = output_root / "README.md"
    readme_path.write_text("# Example dataset\n\nREADME body.", encoding="utf-8")
    state = {
        "current_stage": "readme",
        "approved_plan": pipeline_plan(stages=["readme"]).model_dump(mode="json"),
        "run_root": str(tmp_path),
        "config_path": str(tmp_path / "config.toml"),
        "report_root": str(report_root),
        "current_tool_call_id": "tool_1",
        "owner": "owner",
        "thread_id": "thread",
        "workflow_id": "wf",
        "run_id": "run",
        "stage_reports": [],
        "artifacts": [],
        "completed_stages": [],
        "pending_stages": ["readme"],
    }
    with patch("agent_pipeline.server_graph.execute_stage", return_value={"stage": "readme", "status": "ok"}):
        update = execute_stage_node(state)

    report = update["stage_reports"][0]
    assert report["readme_path"] == str(readme_path)
    assert report["readme_content"].startswith("# Example dataset")
    readme_artifact = next(artifact for artifact in update["artifacts"] if artifact.get("file_name") == "README.md")
    assert readme_artifact["payload"] == report["readme_content"]


def test_execute_reconstruct_stage_projects_obtained_metrics(tmp_path: Path):
    state = {
        "current_stage": "reconstruct",
        "approved_plan": pipeline_plan(stages=["reconstruct"]).model_dump(mode="json"),
        "run_root": str(tmp_path),
        "config_path": str(tmp_path / "config.toml"),
        "report_root": str(tmp_path / "reports"),
        "current_tool_call_id": "tool_1",
        "owner": "owner",
        "thread_id": "thread",
        "workflow_id": "wf",
        "run_id": "run",
        "stage_reports": [],
        "artifacts": [],
        "completed_stages": [],
        "pending_stages": ["reconstruct"],
    }
    durable = {
        "stage": "reconstruct",
        "stage_execution_id": "stage_reconstruct_123",
        "status": "completed",
        "report": {
            "stage": "reconstruct",
            "status": "ok",
            "sessions": [
                {"evaluated_windows": 2, "metrics_mean": {"pearson": 0.8, "mse": 0.2}},
                {"evaluated_windows": 1, "metrics_mean": {"pearson": 0.5, "mse": 0.5}},
            ],
        },
    }
    state["external_stage_job"] = {
        "stage": "reconstruct",
        "stage_execution_id": "stage_reconstruct_123",
    }
    with patch("agent_pipeline.server_graph.reconcile_stage_job", return_value=durable):
        update = reconcile_stage(state)

    report = update["stage_reports"][0]
    assert report["stage_execution_id"] == "stage_reconstruct_123"
    assert report["evaluated_sessions"] == 2
    assert report["evaluated_windows"] == 3
    assert report["metrics"]["pearson"] == pytest.approx(0.7)
    assert report["metrics"]["mse"] == pytest.approx(0.3)
    assert update["artifacts"][0]["kind"] == "reconstruct_quality"
    assert update["artifacts"][0]["payload"]["metrics"] == report["metrics"]


def test_stage_artifacts_include_generated_readme(tmp_path: Path):
    report_root = tmp_path / "reports"
    output_root = tmp_path / "neurotoken"
    report_root.mkdir()
    output_root.mkdir()
    readme_path = output_root / "README.md"
    readme_path.write_text("# Example dataset\n\nREADME body.", encoding="utf-8")
    state = {
        "owner": "owner",
        "thread_id": "thread",
        "workflow_id": "wf",
        "run_id": "run",
        "report_root": str(report_root),
    }
    report_artifact = {"artifact_id": "report"}

    artifacts = _stage_artifacts(state, "readme", {"status": "ok"}, report_artifact)

    readme_artifact = next(artifact for artifact in artifacts if artifact.get("file_name") == "README.md")
    assert readme_artifact["media_type"] == "text/markdown"
    assert readme_artifact["storage_uri"] == str(readme_path)
    assert readme_artifact["payload"].startswith("# Example dataset")


def test_retry_request_routes_directly_to_failed_stage_without_llm():
    state = {
        "mode": "failed",
        "current_stage": "tokenize",
        "run_id": "run_1",
        "last_error": {"stage": "tokenize", "retryable": True},
        "retry_requested": True,
    }

    assert route_after_initialize(state) == "retry_failed_stage"
    update = retry_failed_stage(state)

    assert update["mode"] == "running"
    assert update["retrying_stage"] is True
    assert update["retry_requested"] is False
    assert update["last_error"] is None
    assert update["current_tool_call_id"].startswith("tool_")
    assert update["messages"][0].tool_calls[0]["args"]["retry"] is True


def test_retry_request_rejects_nonretryable_failure():
    update = retry_failed_stage({
        "mode": "failed",
        "current_stage": "preflight",
        "last_error": {"stage": "preflight", "retryable": False},
    })
    assert update["retrying_stage"] is False
    assert "没有可重试" in update["messages"][0].content


def test_retry_tokenize_forces_clean_output_overwrite():
    state = {
        "current_stage": "tokenize",
        "approved_plan": pipeline_plan(stages=["tokenize"]).model_dump(mode="json"),
        "run_root": "/tmp/run",
        "config_path": "/tmp/config.toml",
        "report_root": "/tmp/reports",
        "current_tool_call_id": "tool_retry",
        "retrying_stage": True,
        "stage_reports": [],
        "completed_stages": [],
        "pending_stages": ["tokenize"],
        "artifacts": [],
        "owner": "owner",
        "thread_id": "thread",
    }
    with patch("agent_pipeline.server_graph.submit_tmux_stage", return_value={
        "stage": "tokenize", "stage_execution_id": "stage_tokenize_retry", "status": "submitted",
    }) as submit:
        execute_stage_node(state)

    assert submit.call_args.kwargs["retry_failed"] is True
    assert submit.call_args.kwargs["request"].overwrite_token_output is True


def test_execute_stage_failure_is_terminal_with_structured_error():
    state = {
        "current_stage": "tokenize",
        "approved_plan": pipeline_plan(stages=["tokenize"]).model_dump(mode="json"),
        "run_root": "/tmp/run",
        "config_path": "/tmp/config.toml",
        "report_root": "/tmp/reports",
        "current_tool_call_id": "tool_1",
    }
    with patch("agent_pipeline.server_graph.submit_tmux_stage", side_effect=RuntimeError("worker failed")):
        update = execute_stage_node(state)
    assert update["mode"] == "failed"
    assert update["last_error"]["code"] == "stage_submit_failed"
    assert route_after_stage(update) == "__end__"


def test_execute_validate_failure_preserves_machine_readable_findings():
    state = {
        "current_stage": "validate",
        "approved_plan": pipeline_plan(stages=["validate"]).model_dump(mode="json"),
        "run_root": "/tmp/run",
        "config_path": "/tmp/config.toml",
        "report_root": "/tmp/reports",
        "current_tool_call_id": "tool_1",
        "stage_reports": [],
    }
    failed_report = {
        "stage": "validate",
        "status": "failed",
        "returncode": 1,
        "stderr": "",
        "report_path": "/tmp/reports/validate_report.json",
        "report": {
            "summary": {"errors": 2, "warnings": 8},
            "findings": [
                {"severity": "error", "category": "format_error", "message": "Missing required file: DATA_DICTIONARY.md"},
                {"severity": "error", "category": "format_error", "message": "Missing required directory: schemas"},
            ],
        },
    }
    with patch("agent_pipeline.server_graph.execute_stage", return_value=failed_report):
        update = execute_stage_node(state)
    assert update["mode"] == "failed"
    assert update["last_error"]["code"] == "stage_validation_failed"
    assert update["last_error"]["retryable"] is True
    assert update["last_error"]["details"]["summary"]["errors"] == 2
    assert "DATA_DICTIONARY.md" in update["last_error"]["message"]
    assert update["stage_reports"] == [failed_report]


def test_finalize_refuses_linked_validation_before_full_run():
    state = {
        "active_phase": "validation",
        "approved_plan": linked_pipeline_plan().model_dump(mode="json"),
    }
    with pytest.raises(RuntimeError, match="full child run"):
        finalize(state)


def test_finalize_writes_completed_pipeline_report(tmp_path):
    state = {
        "workflow_id": "wf_1",
        "run_id": "run_1",
        "config_path": str(tmp_path / "config.toml"),
        "report_root": str(tmp_path / "reports"),
        "stage_reports": [{"stage": "preflight", "status": "ok"}],
    }
    update = finalize(state)
    assert update["mode"] == "completed"
    report_path = tmp_path / "reports" / "pipeline_report.json"
    assert report_path.exists()
    assert '"status": "ok"' in report_path.read_text(encoding="utf-8")


def test_report_handles_empty_and_completed_runs():
    assert "还没有 pipeline 阶段报告" in report({"stage_reports": []})["messages"][0].content
    message = report({"mode": "completed", "stage_reports": [{"stage": "preflight"}], "completed_stages": ["preflight"], "report_root": "/reports"})["messages"][0].content
    assert "preflight" in message
