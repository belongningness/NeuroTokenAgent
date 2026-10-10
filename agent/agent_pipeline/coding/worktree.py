"""Tiny git worktree helpers for coding-agent edits."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath


ADAPTER_PATH = Path("agent_pipeline/adapters")
DESIGN_PATH = Path("ADAPTER_DESIGN.md")
DEFAULT_CANDIDATE_PATHS = [ADAPTER_PATH.as_posix(), DESIGN_PATH.as_posix()]
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def git_root(start: str | Path) -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=Path(start), text=True)
    return Path(out.strip())


def baseline_path(worktree: str | Path) -> Path:
    worktree = Path(worktree)
    return worktree.with_name(f"{worktree.name}-baseline")


def create_worktree(
    repo: str | Path,
    root: str | Path = "/tmp/neurotoken-adapter-worktrees",
    request_id: str | None = None,
) -> Path:
    repo = git_root(repo)
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    suffix = f"-{request_id}" if request_id else ""
    worktree = base / f"{repo.name}-adapter-authoring{suffix}"
    baseline = baseline_path(worktree)
    subprocess.run(["git", "worktree", "prune"], cwd=repo, check=True)
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, check=False, capture_output=True)
    shutil.rmtree(worktree, ignore_errors=True)
    shutil.rmtree(baseline, ignore_errors=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=repo, check=True)

    source = repo / ADAPTER_PATH
    design_source = repo / DESIGN_PATH
    worktree_design = worktree / DESIGN_PATH
    baseline_design = baseline / DESIGN_PATH
    baseline_design.parent.mkdir(parents=True, exist_ok=True)
    if design_source.is_file():
        shutil.copy2(design_source, worktree_design)
        shutil.copy2(design_source, baseline_design)
    else:
        worktree_design.unlink(missing_ok=True)
    knowledge_source = repo / "agent_pipeline/adapter_authoring/knowledge_base.md"
    knowledge_target = worktree / "agent_pipeline/adapter_authoring/knowledge_base.md"
    knowledge_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(knowledge_source, knowledge_target)
    worktree_adapters = worktree / ADAPTER_PATH
    shutil.rmtree(worktree_adapters)
    shutil.copytree(source, worktree_adapters, ignore=COPY_IGNORE)
    shutil.copytree(source, baseline / ADAPTER_PATH, ignore=COPY_IGNORE)
    return worktree


def diff_worktree(
    worktree: str | Path,
    allowed_paths: list[str] | None = None,
) -> str:
    worktree = Path(worktree)
    baseline = baseline_path(worktree)
    for root in (worktree / ADAPTER_PATH, baseline / ADAPTER_PATH):
        if not root.is_dir():
            continue
        for cache in root.rglob("__pycache__"):
            shutil.rmtree(cache)
        for pyc in root.rglob("*.pyc"):
            pyc.unlink()

    parts: list[str] = []
    for allowed_path in allowed_paths or DEFAULT_CANDIDATE_PATHS:
        relative = Path(allowed_path.rstrip("/"))
        before = baseline / relative
        after = worktree / relative
        if not before.exists() and not after.exists():
            continue
        before_arg = str(Path(baseline.name) / relative) if before.exists() else "/dev/null"
        after_arg = str(Path(worktree.name) / relative) if after.exists() else "/dev/null"
        proc = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", before_arg, after_arg],
            cwd=worktree.parent,
            text=True,
            capture_output=True,
        )
        if proc.returncode not in {0, 1}:
            raise RuntimeError(proc.stderr.strip())
        if proc.returncode == 1 and not proc.stdout and proc.stderr.strip():
            raise RuntimeError(proc.stderr.strip())
        parts.append(proc.stdout)

    diff = "".join(parts)
    for prefix in ("a", "b"):
        diff = diff.replace(f"{prefix}/{baseline.name}/", f"{prefix}/")
        diff = diff.replace(f"{prefix}/{worktree.name}/", f"{prefix}/")
    return diff


def paths_allowed(paths: list[str], allowed_paths: list[str]) -> bool:
    allowed = [PurePosixPath(prefix.rstrip("/")) for prefix in allowed_paths]
    return all(
        any(PurePosixPath(path) == root or root in PurePosixPath(path).parents for root in allowed)
        for path in paths
    )


def changed_allowed(diff: str, allowed_paths: list[str]) -> bool:
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:]
            if path != "/dev/null":
                files.append(path)
    return paths_allowed(files, allowed_paths)


def changed_worktree_paths(worktree: str | Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=worktree,
        check=True,
        capture_output=True,
    )
    entries = proc.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    paths: list[str] = []
    index = 0
    while index < len(entries) and entries[index]:
        entry = entries[index]
        status = entry[:2]
        paths.append(entry[3:])
        if "R" in status or "C" in status:
            index += 1
            if index < len(entries) and entries[index]:
                paths.append(entries[index])
        index += 1
    return paths


def apply_diff(repo: str | Path, diff: str, reverse: bool = False, check: bool = False) -> None:
    cmd = ["git", "apply"]
    if reverse:
        cmd.append("--reverse")
    if check:
        cmd.append("--check")
    proc = subprocess.run(cmd, cwd=repo, input=diff, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())


def remove_authoring_worktree(
    repo: str | Path,
    root: str | Path = "/tmp/neurotoken-adapter-worktrees",
    request_id: str | None = None,
) -> None:
    repo = git_root(repo)
    suffix = f"-{request_id}" if request_id else ""
    worktree = Path(root) / f"{repo.name}-adapter-authoring{suffix}"
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, check=False, capture_output=True)
    shutil.rmtree(worktree, ignore_errors=True)
    shutil.rmtree(baseline_path(worktree), ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=repo, check=True)
