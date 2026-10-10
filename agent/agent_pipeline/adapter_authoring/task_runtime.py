"""Server-owned worktree lifecycle for durable adapter coding tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from agent_pipeline.adapter_authoring.models import (
    AdapterCodingDescriptor,
    AdapterCodingHandoff,
    AdapterCodingStatus,
)
from agent_pipeline.adapter_authoring.task_registry import (
    ALLOWED_STATUS_TRANSITIONS,
    AdapterCodingTaskRecord,
    TaskRegistryStore,
    canonical_allowed_path,
    utc_now,
    validate_task_id,
)
from agent_pipeline.coding.worktree import (
    baseline_path,
    create_worktree,
    git_root,
    remove_authoring_worktree,
)


DEFAULT_ALLOWED_PATHS = ["agent_pipeline/adapters", "ADAPTER_DESIGN.md"]
TerminalTaskStatus = Literal["applied", "rejected", "cancelled", "failed"]


class AdapterCodingTaskRuntime:
    def __init__(
        self,
        repo: str | Path,
        registry_path: str | Path,
        worktree_root: str | Path = "/tmp/neurotoken-adapter-worktrees",
    ):
        self.repo_root = git_root(repo).resolve()
        self.worktree_root = Path(worktree_root).expanduser().resolve()
        self.registry = TaskRegistryStore(registry_path)

    def create_task(
        self,
        *,
        task_id: str,
        owner: str,
        thread_id: str,
        adapter: AdapterCodingDescriptor,
        instruction: str,
        original_user_request: str = "",
        requirements_snapshot: dict[str, Any] | None = None,
        allowed_paths: list[str] | None = None,
    ) -> AdapterCodingTaskRecord:
        validate_task_id(task_id)
        if self.registry.get(task_id) is not None:
            raise FileExistsError(f"Adapter coding task already exists: {task_id}")
        paths = [canonical_allowed_path(path) for path in (allowed_paths or DEFAULT_ALLOWED_PATHS)]

        created_at = utc_now()
        worktree = create_worktree(
            self.repo_root,
            root=self.worktree_root,
            request_id=task_id,
        ).resolve()
        task = AdapterCodingTaskRecord(
            task_id=task_id,
            owner=owner,
            thread_id=thread_id,
            revision=1,
            status="ready_for_coding",
            adapter=adapter,
            allowed_paths=paths,
            instruction=instruction,
            original_user_request=original_user_request,
            requirements_snapshot=requirements_snapshot or {},
            repo_root=str(self.repo_root),
            worktree_path=str(worktree),
            baseline_path=str(baseline_path(worktree).resolve()),
            created_at=created_at,
            updated_at=created_at,
        )
        try:
            return self.registry.create(task)
        except Exception:
            remove_authoring_worktree(
                self.repo_root,
                root=self.worktree_root,
                request_id=task_id,
            )
            raise

    def require_task(
        self,
        task_id: str,
        *,
        owner: str | None = None,
        thread_id: str | None = None,
        require_worktree: bool = True,
    ) -> AdapterCodingTaskRecord:
        task = self.registry.get(task_id)
        if task is None:
            raise KeyError(f"Unknown adapter coding task: {task_id}")
        if owner is not None and task.owner != owner:
            raise PermissionError(f"Adapter coding task is not owned by {owner}: {task_id}")
        if thread_id is not None and task.thread_id != thread_id:
            raise PermissionError(f"Adapter coding task does not belong to thread {thread_id}: {task_id}")
        if Path(task.repo_root).resolve() != self.repo_root:
            raise RuntimeError(f"Adapter coding task belongs to another repository: {task_id}")
        if require_worktree:
            if not Path(task.worktree_path).is_dir() or not Path(task.baseline_path).is_dir():
                raise RuntimeError(f"Adapter coding task worktree is missing: {task_id}")
        return task

    def handoff(self, task_id: str, *, owner: str | None = None) -> AdapterCodingHandoff:
        task = self.require_task(task_id, owner=owner)
        return AdapterCodingHandoff(
            task_id=task.task_id,
            thread_id=task.thread_id,
            revision=task.revision,
            status=task.status,
            adapter=task.adapter,
            allowed_paths=task.allowed_paths,
            instruction=task.instruction,
            codex_thread_id=task.codex_thread_id,
            created_at=task.created_at,
        )

    def update_status(
        self,
        task_id: str,
        status: AdapterCodingStatus,
        *,
        expected_revision: int,
        codex_thread_id: str | None = None,
        error: str | None = None,
        increment_revision: bool = False,
    ) -> AdapterCodingTaskRecord:
        self.require_task(task_id, require_worktree=status not in {"applied", "rejected", "cancelled", "failed"})
        changes: dict[str, object] = {"error": error}
        if codex_thread_id is not None:
            changes["codex_thread_id"] = codex_thread_id
        return self.registry.transition(
            task_id,
            status,
            expected_revision=expected_revision,
            increment_revision=increment_revision,
            **changes,
        )

    def bind_coding_thread(
        self,
        task_id: str,
        *,
        expected_revision: int,
        codex_thread_id: str,
        replaces_codex_thread_id: str | None = None,
    ) -> AdapterCodingTaskRecord:
        task = self.require_task(task_id)
        if task.status != "coding":
            raise RuntimeError(f"Adapter coding task is not accepting a coding thread: {task.status}")
        if task.codex_thread_id not in {None, codex_thread_id}:
            if not replaces_codex_thread_id or replaces_codex_thread_id != task.codex_thread_id:
                raise PermissionError("Adapter coding task is already bound to another Codex thread.")
        return self.registry.update(
            task_id,
            expected_revision=expected_revision,
            expected_codex_thread_id=task.codex_thread_id,
            codex_thread_id=codex_thread_id,
        )

    def begin_revision(
        self,
        task_id: str,
        *,
        expected_revision: int,
        instruction: str,
    ) -> AdapterCodingTaskRecord:
        task = self.require_task(task_id)
        if task.status == "awaiting_patch_approval":
            task = self.registry.transition(
                task_id,
                "needs_revision",
                expected_revision=expected_revision,
                instruction=instruction,
                error=None,
            )
        if task.status != "needs_revision":
            raise RuntimeError(f"Adapter coding task is not ready for revision: {task.status}")
        return self.registry.transition(
            task_id,
            "coding",
            expected_revision=task.revision,
            increment_revision=True,
            instruction=instruction,
            error=None,
        )

    def cleanup_task(
        self,
        task_id: str,
        *,
        status: TerminalTaskStatus,
        owner: str | None = None,
        error: str | None = None,
    ) -> AdapterCodingTaskRecord:
        task = self.require_task(task_id, owner=owner, require_worktree=False)
        if status not in ALLOWED_STATUS_TRANSITIONS[task.status]:
            raise RuntimeError(f"Invalid adapter coding transition: {task.status} -> {status}")
        remove_authoring_worktree(
            self.repo_root,
            root=self.worktree_root,
            request_id=task.task_id,
        )
        return self.registry.transition(
            task_id,
            status,
            expected_revision=task.revision,
            error=error,
        )
