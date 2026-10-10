"""Durable server-owned registry for adapter coding tasks."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_pipeline.adapter_authoring.models import AdapterCodingDescriptor, AdapterCodingStatus


TASK_REGISTRY_VERSION = "1.0"
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
TERMINAL_STATUSES = {"applied", "rejected", "cancelled", "failed"}
_UNSET = object()
ALLOWED_STATUS_TRANSITIONS: dict[AdapterCodingStatus, set[AdapterCodingStatus]] = {
    "ready_for_coding": {"coding", "cancelled", "failed"},
    "coding": {"validating", "cancelled", "failed"},
    "validating": {"awaiting_patch_approval", "needs_revision", "cancelled", "failed"},
    "awaiting_patch_approval": {"applied", "needs_revision", "rejected", "cancelled", "failed"},
    "needs_revision": {"coding", "cancelled", "failed"},
    "applied": set(),
    "rejected": set(),
    "cancelled": set(),
    "failed": set(),
}


def canonical_allowed_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid allowed path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Invalid allowed path: {value!r}")
    return path.as_posix().rstrip("/")


def path_is_allowed(candidate: str, allowed_paths: list[str]) -> bool:
    candidate_path = PurePosixPath(canonical_allowed_path(candidate))
    return any(
        candidate_path == PurePosixPath(allowed)
        or PurePosixPath(allowed) in candidate_path.parents
        for allowed in allowed_paths
    )


class AdapterCodingTaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = TASK_REGISTRY_VERSION
    task_id: str
    owner: str
    thread_id: str
    revision: int = Field(ge=1)
    status: AdapterCodingStatus
    adapter: AdapterCodingDescriptor
    allowed_paths: list[str]
    instruction: str
    original_user_request: str = ""
    requirements_snapshot: dict[str, Any] = Field(default_factory=dict)
    repo_root: str
    worktree_path: str
    baseline_path: str
    codex_thread_id: str | None = None
    created_at: str
    updated_at: str
    error: str | None = None

    @field_validator("task_id")
    @classmethod
    def valid_task_id(cls, value: str) -> str:
        return validate_task_id(value)

    @field_validator("allowed_paths")
    @classmethod
    def canonical_paths(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("allowed_paths must not be empty")
        normalized = [canonical_allowed_path(value) for value in values]
        return list(dict.fromkeys(normalized))

    @field_validator("repo_root", "worktree_path", "baseline_path")
    @classmethod
    def absolute_server_path(cls, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"Server path must be absolute: {value!r}")
        return str(path.resolve())


class AdapterCodingTaskRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = TASK_REGISTRY_VERSION
    tasks: dict[str, AdapterCodingTaskRecord] = Field(default_factory=dict)

    @model_validator(mode="after")
    def matching_task_keys(self) -> "AdapterCodingTaskRegistry":
        mismatched = [key for key, task in self.tasks.items() if key != task.task_id]
        if mismatched:
            raise ValueError(f"Registry task keys do not match task_id: {mismatched}")
        return self


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_task_id(task_id: str) -> str:
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"Invalid adapter coding task_id: {task_id!r}")
    return task_id


class TaskRegistryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()} created_at={utc_now()}\n".encode())
            os.fsync(fd)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_unlocked(self) -> AdapterCodingTaskRegistry:
        if not self.path.exists():
            return AdapterCodingTaskRegistry()
        return AdapterCodingTaskRegistry.model_validate_json(self.path.read_text(encoding="utf-8"))

    def _write_unlocked(self, registry: AdapterCodingTaskRegistry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(registry.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)

    def snapshot(self) -> AdapterCodingTaskRegistry:
        with self._lock():
            return self._read_unlocked().model_copy(deep=True)

    def get(self, task_id: str) -> AdapterCodingTaskRecord | None:
        validate_task_id(task_id)
        with self._lock():
            task = self._read_unlocked().tasks.get(task_id)
            return task.model_copy(deep=True) if task else None

    def create(self, task: AdapterCodingTaskRecord) -> AdapterCodingTaskRecord:
        validate_task_id(task.task_id)
        with self._lock():
            registry = self._read_unlocked()
            if task.task_id in registry.tasks:
                raise FileExistsError(f"Adapter coding task already exists: {task.task_id}")
            registry.tasks[task.task_id] = task
            self._write_unlocked(registry)
            return task.model_copy(deep=True)

    def update(
        self,
        task_id: str,
        *,
        expected_revision: int | None = None,
        expected_codex_thread_id: str | None | object = _UNSET,
        **changes: object,
    ) -> AdapterCodingTaskRecord:
        validate_task_id(task_id)
        forbidden = {"schema_version", "task_id", "owner", "thread_id", "original_user_request", "requirements_snapshot", "repo_root", "worktree_path", "baseline_path", "status", "revision"}
        invalid = forbidden.intersection(changes)
        if invalid:
            raise ValueError(f"Immutable adapter coding task fields: {sorted(invalid)}")
        with self._lock():
            registry = self._read_unlocked()
            current = registry.tasks.get(task_id)
            if current is None:
                raise KeyError(f"Unknown adapter coding task: {task_id}")
            if expected_revision is not None and current.revision != expected_revision:
                raise RuntimeError(
                    f"Stale adapter coding task revision: expected {expected_revision}, current {current.revision}"
                )
            if expected_codex_thread_id is not _UNSET and current.codex_thread_id != expected_codex_thread_id:
                raise RuntimeError("Stale adapter coding task Codex thread binding.")
            updated = current.model_copy(update={**changes, "updated_at": utc_now()})
            updated = AdapterCodingTaskRecord.model_validate(updated.model_dump())
            registry.tasks[task_id] = updated
            self._write_unlocked(registry)
            return updated.model_copy(deep=True)

    def transition(
        self,
        task_id: str,
        status: AdapterCodingStatus,
        *,
        expected_revision: int,
        increment_revision: bool = False,
        **changes: object,
    ) -> AdapterCodingTaskRecord:
        validate_task_id(task_id)
        forbidden = {"schema_version", "task_id", "owner", "thread_id", "original_user_request", "requirements_snapshot", "repo_root", "worktree_path", "baseline_path", "status", "revision"}
        invalid = forbidden.intersection(changes)
        if invalid:
            raise ValueError(f"Immutable adapter coding task fields: {sorted(invalid)}")
        with self._lock():
            registry = self._read_unlocked()
            current = registry.tasks.get(task_id)
            if current is None:
                raise KeyError(f"Unknown adapter coding task: {task_id}")
            if current.revision != expected_revision:
                raise RuntimeError(
                    f"Stale adapter coding task revision: expected {expected_revision}, current {current.revision}"
                )
            if status not in ALLOWED_STATUS_TRANSITIONS[current.status]:
                raise RuntimeError(f"Invalid adapter coding transition: {current.status} -> {status}")
            if increment_revision != (current.status == "needs_revision" and status == "coding"):
                raise RuntimeError("Revision must increment exactly when needs_revision transitions to coding")
            revision = current.revision + 1 if increment_revision else current.revision
            updated = current.model_copy(
                update={**changes, "status": status, "revision": revision, "updated_at": utc_now()}
            )
            updated = AdapterCodingTaskRecord.model_validate(updated.model_dump())
            registry.tasks[task_id] = updated
            self._write_unlocked(registry)
            return updated.model_copy(deep=True)

    def delete(self, task_id: str) -> AdapterCodingTaskRecord | None:
        validate_task_id(task_id)
        with self._lock():
            registry = self._read_unlocked()
            removed = registry.tasks.pop(task_id, None)
            if removed is not None:
                self._write_unlocked(registry)
            return removed.model_copy(deep=True) if removed else None
