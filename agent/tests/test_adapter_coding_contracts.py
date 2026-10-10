from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_pipeline.adapter_authoring.models import (
    AdapterCandidateSubmission,
    AdapterCodingDescriptor,
    AdapterCodingHandoff,
    AdapterPatchDecision,
)
from agent_pipeline.server_state import ApprovalDecision


def handoff_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "task_id": "adapter-task-001",
        "thread_id": "langgraph-thread-001",
        "revision": 1,
        "status": "ready_for_coding",
        "adapter": {
            "name": "example_eeg",
            "modality": "eeg",
            "notes": "Read channel metadata from HDF5 attributes.",
        },
        "allowed_paths": ["agent_pipeline/adapters/"],
        "instruction": "Implement the example EEG adapter.",
        "codex_thread_id": None,
        "created_at": "2026-08-04T07:00:00Z",
    }


def test_handoff_round_trip_matches_desktop_contract():
    handoff = AdapterCodingHandoff.model_validate(handoff_payload())

    assert handoff.model_dump(mode="json") == handoff_payload()
    assert isinstance(handoff.adapter, AdapterCodingDescriptor)


def test_handoff_rejects_unknown_protocol_version():
    payload = {**handoff_payload(), "schema_version": "2.0"}

    with pytest.raises(ValidationError):
        AdapterCodingHandoff.model_validate(payload)


def test_handoff_never_serializes_server_paths():
    payload = handoff_payload()
    payload["worktree"] = Path("/srv/neurotoken/worktrees/adapter-task-001")

    with pytest.raises(ValidationError):
        AdapterCodingHandoff.model_validate(payload)

    serialized = AdapterCodingHandoff.model_validate(handoff_payload()).model_dump_json()
    assert "/srv/" not in serialized
    assert "worktree" not in serialized
    assert "source_path" not in serialized


def test_candidate_submission_is_identity_only():
    payload = {
        "schema_version": "1.0",
        "action": "submit_candidate",
        "task_id": "adapter-task-001",
        "thread_id": "langgraph-thread-001",
        "codex_thread_id": "codex-thread-002",
        "replaces_codex_thread_id": "codex-thread-001",
        "revision": 2,
    }

    submission = AdapterCandidateSubmission.model_validate(payload)
    assert submission.model_dump(mode="json") == payload

    with pytest.raises(ValidationError):
        AdapterCandidateSubmission.model_validate({**payload, "diff": "untrusted diff"})


def test_patch_decision_supports_apply_and_commit_metadata():
    payload = {
        "schema_version": "1.0",
        "action": "apply_and_commit",
        "task_id": "adapter-task-001",
        "revision": 2,
        "message": None,
        "commit_message": "Add example EEG adapter",
    }

    decision = AdapterPatchDecision.model_validate(payload)
    approval = ApprovalDecision.model_validate(decision.model_dump(mode="json"))

    assert approval.action == "apply_and_commit"
    assert approval.task_id == "adapter-task-001"
    assert approval.revision == 2
    assert approval.commit_message == "Add example EEG adapter"
