import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_pipeline.adapter_authoring.models import AdapterCodingDescriptor
from agent_pipeline.adapter_authoring.task_registry import (
    AdapterCodingTaskRecord,
    TaskRegistryStore,
    canonical_allowed_path,
    path_is_allowed,
)
from agent_pipeline.adapter_authoring.task_runtime import AdapterCodingTaskRuntime


def task_record(tmp_path: Path, task_id: str = "adapter-task-001") -> AdapterCodingTaskRecord:
    now = "2026-08-04T08:00:00+00:00"
    return AdapterCodingTaskRecord(
        task_id=task_id,
        owner="alice",
        thread_id="thread-001",
        revision=1,
        status="ready_for_coding",
        adapter=AdapterCodingDescriptor(name="example_eeg", modality="eeg"),
        allowed_paths=["agent_pipeline/adapters/"],
        instruction="Implement the EEG adapter.",
        repo_root=str(tmp_path / "repo"),
        worktree_path=str(tmp_path / "worktree"),
        baseline_path=str(tmp_path / "baseline"),
        created_at=now,
        updated_at=now,
    )


def test_registry_writes_versioned_atomic_json(tmp_path: Path):
    registry_path = tmp_path / "registry" / "tasks.json"
    store = TaskRegistryStore(registry_path)
    record = task_record(tmp_path)

    created = store.create(record)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))

    assert created == record
    assert payload["schema_version"] == "1.0"
    assert payload["tasks"][record.task_id]["worktree_path"] == record.worktree_path
    assert not list(registry_path.parent.glob("*.tmp"))
    assert store.lock_path.exists()


def test_registry_rejects_duplicates_stale_updates_and_immutable_paths(tmp_path: Path):
    store = TaskRegistryStore(tmp_path / "tasks.json")
    record = task_record(tmp_path)
    store.create(record)

    with pytest.raises(FileExistsError):
        store.create(record)
    with pytest.raises(RuntimeError, match="Stale"):
        store.transition(record.task_id, "coding", expected_revision=2)
    with pytest.raises(ValueError, match="Immutable"):
        store.update(record.task_id, worktree_path="/untrusted")
    with pytest.raises(ValueError, match="Immutable"):
        store.update(record.task_id, status="coding")
    with pytest.raises(ValueError, match="Immutable"):
        store.update(record.task_id, original_user_request="replace intent")
    with pytest.raises(ValueError, match="Immutable"):
        store.update(record.task_id, requirements_snapshot={"requirements": {}})

    updated = store.transition(record.task_id, "coding", expected_revision=1, codex_thread_id="codex-1")
    assert updated.status == "coding"
    assert updated.codex_thread_id == "codex-1"


def test_registry_rejects_invalid_task_identifiers(tmp_path: Path):
    record = task_record(tmp_path).model_copy(update={"task_id": "../../escape"})

    with pytest.raises(ValueError, match="Invalid"):
        TaskRegistryStore(tmp_path / "tasks.json").create(record)


def test_allowed_paths_are_canonical_and_component_aware(tmp_path: Path):
    record = task_record(tmp_path)

    assert record.allowed_paths == ["agent_pipeline/adapters"]
    assert canonical_allowed_path("agent_pipeline/adapters/") == "agent_pipeline/adapters"
    assert path_is_allowed("agent_pipeline/adapters/example.py", record.allowed_paths)
    assert not path_is_allowed("agent_pipeline/adapters_evil/example.py", record.allowed_paths)
    for invalid in ["/absolute", "../escape", "agent_pipeline/../escape", "windows\\path"]:
        with pytest.raises(ValueError, match="Invalid"):
            canonical_allowed_path(invalid)


def test_registry_enforces_status_transitions_and_terminal_states(tmp_path: Path):
    store = TaskRegistryStore(tmp_path / "tasks.json")
    record = task_record(tmp_path)
    store.create(record)

    coding = store.transition(record.task_id, "coding", expected_revision=1)
    validating = store.transition(record.task_id, "validating", expected_revision=coding.revision)
    approval = store.transition(record.task_id, "awaiting_patch_approval", expected_revision=validating.revision)
    applied = store.transition(record.task_id, "applied", expected_revision=approval.revision)

    assert applied.status == "applied"
    with pytest.raises(RuntimeError, match="Invalid"):
        store.transition(record.task_id, "coding", expected_revision=applied.revision)


def test_revision_increments_only_when_revision_coding_begins(tmp_path: Path):
    store = TaskRegistryStore(tmp_path / "tasks.json")
    record = task_record(tmp_path)
    store.create(record)
    store.transition(record.task_id, "coding", expected_revision=1)
    store.transition(record.task_id, "validating", expected_revision=1)
    store.transition(record.task_id, "needs_revision", expected_revision=1)

    with pytest.raises(RuntimeError, match="Revision must increment"):
        store.transition(record.task_id, "coding", expected_revision=1)
    revised = store.transition(record.task_id, "coding", expected_revision=1, increment_revision=True)
    assert revised.revision == 2


def test_runtime_owns_worktree_and_public_handoff_hides_server_paths(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path / "worktrees" / "repo-adapter-authoring-adapter-task-001"
    baseline = worktree.with_name(f"{worktree.name}-baseline")
    worktree.mkdir(parents=True)
    baseline.mkdir()

    with (
        patch("agent_pipeline.adapter_authoring.task_runtime.git_root", return_value=repo),
        patch("agent_pipeline.adapter_authoring.task_runtime.create_worktree", return_value=worktree) as create,
    ):
        runtime = AdapterCodingTaskRuntime(repo, tmp_path / "tasks.json", tmp_path / "worktrees")
        task = runtime.create_task(
            task_id="adapter-task-001",
            owner="alice",
            thread_id="thread-001",
            adapter=AdapterCodingDescriptor(name="example_eeg", modality="eeg"),
            instruction="Implement the EEG adapter.",
            original_user_request="为这个文件编写 EEG adapter，并保留原始采样率。",
            requirements_snapshot={"requirements": {"modality": "eeg"}},
        )
        handoff = runtime.handoff(task.task_id, owner="alice")

    create.assert_called_once_with(repo.resolve(), root=(tmp_path / "worktrees").resolve(), request_id=task.task_id)
    assert task.worktree_path == str(worktree.resolve())
    assert task.original_user_request.startswith("为这个文件")
    assert task.requirements_snapshot["requirements"]["modality"] == "eeg"
    assert TaskRegistryStore(tmp_path / "tasks.json").get(task.task_id).original_user_request == task.original_user_request
    assert "worktree" not in handoff.model_dump(mode="json")
    assert str(tmp_path) not in handoff.model_dump_json()
    with pytest.raises(PermissionError):
        runtime.handoff(task.task_id, owner="bob")


def test_runtime_binds_coding_thread_without_changing_status_or_revision(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path / "worktree"
    baseline = worktree.with_name(f"{worktree.name}-baseline")
    worktree.mkdir()
    baseline.mkdir()

    with (
        patch("agent_pipeline.adapter_authoring.task_runtime.git_root", return_value=repo),
        patch("agent_pipeline.adapter_authoring.task_runtime.create_worktree", return_value=worktree),
    ):
        runtime = AdapterCodingTaskRuntime(repo, tmp_path / "tasks.json", tmp_path / "worktrees")
        task = runtime.create_task(
            task_id="adapter-task-bind",
            owner="alice",
            thread_id="thread-001",
            adapter=AdapterCodingDescriptor(name="example_eeg", modality="eeg"),
            instruction="Implement it",
        )
        task = runtime.update_status(task.task_id, "coding", expected_revision=1)
        bound = runtime.bind_coding_thread(
            task.task_id,
            expected_revision=task.revision,
            codex_thread_id="codex-1",
        )

    assert bound.status == "coding"
    assert bound.revision == 1
    assert bound.codex_thread_id == "codex-1"
    with pytest.raises(PermissionError):
        runtime.bind_coding_thread(
            task.task_id,
            expected_revision=task.revision,
            codex_thread_id="codex-2",
        )
    rebound = runtime.bind_coding_thread(
        task.task_id,
        expected_revision=task.revision,
        codex_thread_id="codex-2",
        replaces_codex_thread_id="codex-1",
    )
    assert rebound.codex_thread_id == "codex-2"
    with pytest.raises(PermissionError):
        runtime.bind_coding_thread(
            task.task_id,
            expected_revision=task.revision,
            codex_thread_id="codex-3",
            replaces_codex_thread_id="codex-1",
        )


def test_runtime_revision_and_cleanup_are_durable(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path / "worktree"
    baseline = worktree.with_name(f"{worktree.name}-baseline")
    worktree.mkdir()
    baseline.mkdir()

    with (
        patch("agent_pipeline.adapter_authoring.task_runtime.git_root", return_value=repo),
        patch("agent_pipeline.adapter_authoring.task_runtime.create_worktree", return_value=worktree),
        patch("agent_pipeline.adapter_authoring.task_runtime.remove_authoring_worktree") as remove,
    ):
        runtime = AdapterCodingTaskRuntime(repo, tmp_path / "tasks.json", tmp_path / "worktrees")
        task = runtime.create_task(
            task_id="adapter-task-002",
            owner="alice",
            thread_id="thread-001",
            adapter=AdapterCodingDescriptor(name="example_eeg", modality="eeg"),
            instruction="Initial instruction",
        )
        runtime.update_status(
            task.task_id,
            "coding",
            expected_revision=1,
            codex_thread_id="codex-1",
        )
        runtime.update_status(task.task_id, "validating", expected_revision=1)
        runtime.update_status(task.task_id, "awaiting_patch_approval", expected_revision=1)
        revised = runtime.begin_revision(task.task_id, expected_revision=1, instruction="Refine channel mapping")
        terminal = runtime.cleanup_task(task.task_id, status="cancelled", owner="alice")

    assert revised.revision == 2
    assert revised.status == "coding"
    assert revised.codex_thread_id == "codex-1"
    assert terminal.status == "cancelled"
    remove.assert_called_once_with(repo.resolve(), root=(tmp_path / "worktrees").resolve(), request_id=task.task_id)
    assert TaskRegistryStore(tmp_path / "tasks.json").get(task.task_id).status == "cancelled"


def test_runtime_rejects_invalid_cleanup_before_removing_worktree(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    with (
        patch("agent_pipeline.adapter_authoring.task_runtime.git_root", return_value=repo),
        patch("agent_pipeline.adapter_authoring.task_runtime.create_worktree", return_value=worktree),
        patch("agent_pipeline.adapter_authoring.task_runtime.remove_authoring_worktree") as remove,
    ):
        runtime = AdapterCodingTaskRuntime(repo, tmp_path / "tasks.json", tmp_path / "worktrees")
        task = runtime.create_task(
            task_id="adapter-task-invalid-cleanup",
            owner="alice",
            thread_id="thread-001",
            adapter=AdapterCodingDescriptor(name="example_eeg", modality="eeg"),
            instruction="Initial instruction",
        )
        with pytest.raises(RuntimeError, match="ready_for_coding -> rejected"):
            runtime.cleanup_task(task.task_id, status="rejected", owner="alice")

    remove.assert_not_called()
    assert TaskRegistryStore(tmp_path / "tasks.json").get(task.task_id).status == "ready_for_coding"


def test_runtime_cleans_worktree_if_registry_create_fails(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    with (
        patch("agent_pipeline.adapter_authoring.task_runtime.git_root", return_value=repo),
        patch("agent_pipeline.adapter_authoring.task_runtime.create_worktree", return_value=worktree),
        patch.object(TaskRegistryStore, "create", side_effect=RuntimeError("disk failure")),
        patch("agent_pipeline.adapter_authoring.task_runtime.remove_authoring_worktree") as remove,
    ):
        runtime = AdapterCodingTaskRuntime(repo, tmp_path / "tasks.json", tmp_path / "worktrees")
        with pytest.raises(RuntimeError, match="disk failure"):
            runtime.create_task(
                task_id="adapter-task-003",
                owner="alice",
                thread_id="thread-001",
                adapter=AdapterCodingDescriptor(name="example_eeg", modality="eeg"),
                instruction="Implement it",
            )

    remove.assert_called_once()
