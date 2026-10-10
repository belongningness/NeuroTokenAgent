"""End-to-end adapter authoring workflow."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from agent_pipeline.adapter_authoring.h5_inspector import format_h5_summary, inspect_h5
from agent_pipeline.adapter_authoring.models import AdapterAuthoringResult, AdapterSpec, CodingTask
from agent_pipeline.adapter_authoring.prompts import build_adapter_prompt, build_refinement_prompt
from agent_pipeline.coding.cursor_provider import CursorSdkProvider
from agent_pipeline.coding.worktree import apply_diff, create_worktree, git_root, remove_authoring_worktree


@contextmanager
def repository_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def author_adapter(
    spec: AdapterSpec,
    repo: str | Path,
    model: str,
    cursor_api_key: str | None = None,
    proxy: str | None = None,
    request_id: str | None = None,
) -> AdapterAuthoringResult:
    repo_root = git_root(repo)
    h5_summary = format_h5_summary(inspect_h5(spec.source_path))
    worktree = create_worktree(repo_root, request_id=request_id)
    try:
        task = CodingTask(
            instruction=build_adapter_prompt(spec, h5_summary),
            worktree=worktree,
            allowed_paths=["agent_pipeline/adapters/", "ADAPTER_DESIGN.md"],
        )
        coding_result = CursorSdkProvider(model, cursor_api_key=cursor_api_key, proxy=proxy).run(task)
        return AdapterAuthoringResult(spec, h5_summary, coding_result)
    finally:
        remove_authoring_worktree(repo_root, request_id=request_id)


def refine_adapter(
    result: AdapterAuthoringResult,
    repo: str | Path,
    request: str,
    model: str,
    cursor_api_key: str | None = None,
    proxy: str | None = None,
    request_id: str | None = None,
) -> AdapterAuthoringResult:
    repo_root = git_root(repo)
    worktree = create_worktree(repo_root, request_id=request_id)
    try:
        apply_diff(worktree, result.coding_result.diff)
        task = CodingTask(
            instruction=build_refinement_prompt(result.spec, result.h5_summary, request),
            worktree=worktree,
            allowed_paths=["agent_pipeline/adapters/", "ADAPTER_DESIGN.md"],
        )
        coding_result = CursorSdkProvider(model, cursor_api_key=cursor_api_key, proxy=proxy).run(task)
        return AdapterAuthoringResult(result.spec, result.h5_summary, coding_result)
    finally:
        remove_authoring_worktree(repo_root, request_id=request_id)


def authoring_history_dir(repo: str | Path) -> Path:
    root = git_root(repo)
    git_dir = Path(subprocess.check_output(["git", "rev-parse", "--git-dir"], cwd=root, text=True).strip())
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    return git_dir / "neurotoken-sdk-authoring"


def record_authoring_result(
    repo: str | Path,
    result: AdapterAuthoringResult,
    *,
    owner: str,
    thread_id: str,
    request_id: str,
) -> Path:
    history = authoring_history_dir(repo)
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    patch_id = f"{stamp}-{request_id}-{result.spec.name}"
    patch_path = history / f"{patch_id}.patch"
    patch_path.write_text(result.coding_result.diff, encoding="utf-8")
    metadata = {
        "patch_id": patch_id,
        "owner": owner,
        "thread_id": thread_id,
        "request_id": request_id,
        "adapter_name": result.spec.name,
        "source_path": str(result.spec.source_path),
        "modality": result.spec.modality,
        "patch": patch_path.name,
    }
    patch_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return patch_path


def patch_paths(diff: str) -> list[str]:
    paths: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:]
            if path != "/dev/null":
                paths.add(path)
    if not paths:
        raise ValueError("Adapter patch does not contain any repository paths.")
    return sorted(paths)


def apply_authoring_result(
    repo: str | Path,
    result: AdapterAuthoringResult,
    *,
    owner: str = "local-user",
    thread_id: str = "local-thread",
    request_id: str = "manual",
    commit_message: str | None = None,
) -> Path:
    root = git_root(repo)
    lock_path = authoring_history_dir(root) / "repository.lock"
    with repository_lock(lock_path):
        affected_paths = patch_paths(result.coding_result.diff)
        apply_diff(root, result.coding_result.diff, check=True)
        apply_diff(root, result.coding_result.diff)
        patch_record: Path | None = None
        head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        try:
            subprocess.run(
                [sys.executable, "-c", "from agent_pipeline.adapters import build_adapter"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            patch_record = record_authoring_result(
                root,
                result,
                owner=owner,
                thread_id=thread_id,
                request_id=request_id,
            )
            if commit_message is not None:
                with tempfile.NamedTemporaryFile(prefix="neurotoken-index-", delete=False) as index_file:
                    temporary_index = Path(index_file.name)
                try:
                    temporary_index.unlink()
                    environment = {**os.environ, "GIT_INDEX_FILE": str(temporary_index)}
                    subprocess.run(
                        ["git", "read-tree", "HEAD"],
                        cwd=root,
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "apply", "--cached", "--whitespace=nowarn", str(patch_record)],
                        cwd=root,
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "commit", "-m", commit_message],
                        cwd=root,
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "apply", "--cached", "--whitespace=nowarn", str(patch_record)],
                        cwd=root,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                finally:
                    temporary_index.unlink(missing_ok=True)
            return patch_record
        except Exception:
            head_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            if head_after == head_before:
                subprocess.run(
                    ["git", "reset", "--quiet", "--", *affected_paths],
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                apply_diff(root, result.coding_result.diff, reverse=True)
                if patch_record is not None:
                    patch_record.unlink(missing_ok=True)
                    patch_record.with_suffix(".json").unlink(missing_ok=True)
            elif patch_record is not None:
                return patch_record
            raise
