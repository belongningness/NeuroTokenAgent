"""Rollback services for applied adapter-authoring patches."""

from __future__ import annotations

import json
from pathlib import Path

from agent_pipeline.adapter_authoring.workflow import authoring_history_dir
from agent_pipeline.coding.worktree import apply_diff, git_root, remove_authoring_worktree
from agent_pipeline.run_runtime import exclusive_lock


def patch_records(repo: str | Path) -> list[Path]:
    history = authoring_history_dir(repo)
    return sorted(history.glob("*.patch")) if history.exists() else []


def patch_metadata(patch_path: Path) -> dict[str, str]:
    metadata_path = patch_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def cleanup(
    repo: str | Path,
    *,
    patch_id: str | None = None,
    owner: str | None = None,
    thread_id: str | None = None,
    latest: bool = False,
) -> list[Path]:
    root = git_root(repo)
    history = authoring_history_dir(root)
    lock_path = history / "repository.lock"
    removed: list[Path] = []
    with exclusive_lock(lock_path):
        records = patch_records(root)
        if patch_id:
            selected = [path for path in records if path.name == patch_id or path.stem == patch_id]
            if not selected:
                raise FileNotFoundError(f"Unknown SDK patch: {patch_id}")
        else:
            selected = records[-1:] if latest else records
        for patch_path in reversed(selected):
            metadata = patch_metadata(patch_path)
            if owner is not None and metadata.get("owner") != owner:
                raise PermissionError(f"SDK patch is not owned by {owner}: {patch_path.name}")
            if thread_id is not None and metadata.get("thread_id") != thread_id:
                raise PermissionError(
                    f"SDK patch does not belong to thread {thread_id}: {patch_path.name}"
                )
            diff = patch_path.read_text(encoding="utf-8")
            apply_diff(root, diff, reverse=True, check=True)
            apply_diff(root, diff, reverse=True)
            patch_path.unlink()
            metadata_path = patch_path.with_suffix(".json")
            if metadata_path.exists():
                metadata_path.unlink()
            removed.append(patch_path)
    remove_authoring_worktree(root)
    if history.exists() and not any(history.iterdir()):
        history.rmdir()
    return removed
