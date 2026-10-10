from pathlib import Path

from agent_pipeline.coding.worktree import apply_diff, baseline_path, diff_worktree


def candidate_fixture(tmp_path: Path) -> tuple[Path, Path]:
    worktree = tmp_path / "candidate"
    baseline = baseline_path(worktree)
    for root in (worktree, baseline):
        adapters = root / "agent_pipeline" / "adapters"
        adapters.mkdir(parents=True)
        (adapters / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
    return worktree, baseline


def test_diff_worktree_includes_adapter_and_new_design_document(tmp_path: Path):
    worktree, _baseline = candidate_fixture(tmp_path)
    (worktree / "agent_pipeline" / "adapters" / "base.py").write_text("VALUE = 2\n", encoding="utf-8")
    (worktree / "ADAPTER_DESIGN.md").write_text("# Adapter Design\n", encoding="utf-8")

    diff = diff_worktree(worktree, ["agent_pipeline/adapters", "ADAPTER_DESIGN.md"])

    assert "diff --git a/agent_pipeline/adapters/base.py b/agent_pipeline/adapters/base.py" in diff
    assert "diff --git a/ADAPTER_DESIGN.md b/ADAPTER_DESIGN.md" in diff
    assert "--- /dev/null" in diff
    assert "+# Adapter Design" in diff

    target = tmp_path / "target"
    target_adapter = target / "agent_pipeline" / "adapters"
    target_adapter.mkdir(parents=True)
    (target_adapter / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
    apply_diff(target, diff, check=True)
    apply_diff(target, diff)
    assert (target_adapter / "base.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (target / "ADAPTER_DESIGN.md").read_text(encoding="utf-8") == "# Adapter Design\n"


def test_diff_worktree_treats_existing_design_as_modified(tmp_path: Path):
    worktree, baseline = candidate_fixture(tmp_path)
    for root in (worktree, baseline):
        (root / "ADAPTER_DESIGN.md").write_text("# Original\n", encoding="utf-8")
    (worktree / "ADAPTER_DESIGN.md").write_text("# Revised\n", encoding="utf-8")

    diff = diff_worktree(worktree, ["ADAPTER_DESIGN.md"])

    assert "--- a/ADAPTER_DESIGN.md" in diff
    assert "+++ b/ADAPTER_DESIGN.md" in diff
    assert "--- /dev/null" not in diff
    assert "-# Original" in diff
    assert "+# Revised" in diff
