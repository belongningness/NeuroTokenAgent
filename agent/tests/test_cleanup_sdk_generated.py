import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_pipeline.adapter_authoring.cleanup import cleanup
from agent_pipeline.run_runtime import exclusive_lock


def write_record(history: Path, patch_id: str, owner: str, thread_id: str) -> Path:
    history.mkdir(parents=True, exist_ok=True)
    patch_path = history / f"{patch_id}.patch"
    patch_path.write_text("diff", encoding="utf-8")
    patch_path.with_suffix(".json").write_text(
        json.dumps({"patch_id": patch_id, "owner": owner, "thread_id": thread_id}),
        encoding="utf-8",
    )
    return patch_path


def test_cleanup_requires_exact_owned_patch(tmp_path: Path):
    history = tmp_path / "history"
    owned = write_record(history, "owned", "alice", "thread-a")
    write_record(history, "other", "bob", "thread-b")
    lock = history / "repository.lock"

    with (
        patch("agent_pipeline.adapter_authoring.cleanup.git_root", return_value=tmp_path),
        patch("agent_pipeline.adapter_authoring.cleanup.authoring_history_dir", return_value=history),
        patch("agent_pipeline.adapter_authoring.cleanup.apply_diff") as apply_diff,
        patch("agent_pipeline.adapter_authoring.cleanup.remove_authoring_worktree"),
    ):
        removed = cleanup(tmp_path, patch_id="owned", owner="alice", thread_id="thread-a")

    assert removed == [owned]
    assert apply_diff.call_count == 2
    assert not owned.exists()
    assert not owned.with_suffix(".json").exists()
    assert (history / "other.patch").exists()
    assert lock.exists()
    with exclusive_lock(lock):
        pass


def test_cleanup_rejects_cross_owner_patch(tmp_path: Path):
    history = tmp_path / "history"
    write_record(history, "owned", "alice", "thread-a")

    with (
        patch("agent_pipeline.adapter_authoring.cleanup.git_root", return_value=tmp_path),
        patch("agent_pipeline.adapter_authoring.cleanup.authoring_history_dir", return_value=history),
        patch("agent_pipeline.adapter_authoring.cleanup.apply_diff") as apply_diff,
    ):
        with pytest.raises(PermissionError, match="not owned"):
            cleanup(tmp_path, patch_id="owned", owner="bob", thread_id="thread-a")

    apply_diff.assert_not_called()
