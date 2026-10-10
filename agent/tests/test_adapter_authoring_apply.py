import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_pipeline.adapter_authoring.models import (
    AdapterAuthoringResult,
    AdapterSpec,
    CodingAgentResult,
)
from agent_pipeline.adapter_authoring.workflow import apply_authoring_result


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    adapters = repo / "agent_pipeline" / "adapters"
    adapters.mkdir(parents=True)
    (repo / "agent_pipeline" / "__init__.py").write_text("", encoding="utf-8")
    (adapters / "__init__.py").write_text("def build_adapter(config):\n    return config\n", encoding="utf-8")
    (adapters / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "NeuroToken Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@neurotoken.local"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    return repo


def result_for(repo: Path, name: str, content: str) -> AdapterAuthoringResult:
    relative = f"agent_pipeline/adapters/{name}.py"
    patch_text = (
        f"diff --git a/{relative} b/{relative}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{relative}\n"
        f"@@ -0,0 +1 @@\n+{content}\n"
    )
    return AdapterAuthoringResult(
        spec=AdapterSpec(name=name, source_path=repo / "source.h5", modality="eeg"),
        h5_summary="datasets: signal",
        coding_result=CodingAgentResult(summary="generated", diff=patch_text),
    )


def test_apply_only_leaves_verified_change_uncommitted(tmp_path: Path):
    repo = init_repo(tmp_path)
    head = git(repo, "rev-parse", "HEAD")

    record = apply_authoring_result(repo, result_for(repo, "apply_only", "VALUE = 2"), request_id="apply-only")

    assert (repo / "agent_pipeline/adapters/apply_only.py").read_text() == "VALUE = 2\n"
    assert git(repo, "rev-parse", "HEAD") == head
    assert "?? agent_pipeline/adapters/apply_only.py" in git(repo, "status", "--short")
    assert record.exists()
    assert record.with_suffix(".json").exists()


def test_apply_and_commit_creates_scoped_commit(tmp_path: Path):
    repo = init_repo(tmp_path)
    unrelated = repo / "notes.txt"
    unrelated.write_text("keep uncommitted\n", encoding="utf-8")

    apply_authoring_result(
        repo,
        result_for(repo, "committed", "VALUE = 3"),
        request_id="commit",
        commit_message="Add committed adapter",
    )

    assert git(repo, "log", "-1", "--format=%s") == "Add committed adapter"
    assert "agent_pipeline/adapters/committed.py" in git(repo, "show", "--name-only", "--format=")
    assert "?? notes.txt" in git(repo, "status", "--short")


def test_apply_and_commit_excludes_preexisting_changes_in_shared_file(tmp_path: Path):
    repo = init_repo(tmp_path)
    registry = repo / "agent_pipeline" / "adapters" / "__init__.py"
    original = registry.read_text(encoding="utf-8")
    registry.write_text(
        original
        + "\nKEEP_1 = True\nKEEP_2 = True\nKEEP_3 = True\n"
        + "KEEP_4 = True\nKEEP_5 = True\nKEEP_6 = True\nBASELINE = True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", str(registry)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add registry baseline"], cwd=repo, check=True, capture_output=True)
    registry.write_text(
        registry.read_text(encoding="utf-8").replace(
            "def build_adapter(config):", "def build_adapter(config):  # PREEXISTING"
        ),
        encoding="utf-8",
    )
    patch_text = (
        "diff --git a/agent_pipeline/adapters/__init__.py b/agent_pipeline/adapters/__init__.py\n"
        "--- a/agent_pipeline/adapters/__init__.py\n"
        "+++ b/agent_pipeline/adapters/__init__.py\n"
        "@@ -7,4 +7,4 @@\n"
        " KEEP_4 = True\n"
        " KEEP_5 = True\n"
        " KEEP_6 = True\n"
        "-BASELINE = True\n"
        "+COMMITTED = True\n"
    )
    result = AdapterAuthoringResult(
        spec=AdapterSpec(name="committed", source_path=repo / "source.h5", modality="eeg"),
        h5_summary="datasets: signal",
        coding_result=CodingAgentResult(summary="generated", diff=patch_text),
    )

    apply_authoring_result(repo, result, request_id="shared-file", commit_message="Scoped shared-file change")

    committed = git(repo, "show", "HEAD:agent_pipeline/adapters/__init__.py")
    assert "COMMITTED = True" in committed
    assert "# PREEXISTING" not in committed
    assert "# PREEXISTING" in registry.read_text(encoding="utf-8")
    status = git(repo, "status", "--short")
    assert "MM agent_pipeline/adapters/__init__.py" not in status
    assert "M agent_pipeline/adapters/__init__.py" in status



def test_import_failure_reverses_patch_and_removes_record(tmp_path: Path):
    repo = init_repo(tmp_path)
    authored = result_for(repo, "broken", "VALUE = 4")

    real_run = subprocess.run
    with patch("agent_pipeline.adapter_authoring.workflow.subprocess.run") as run:
        def fail_validation(command, *args, **kwargs):
            if command[:2] == [sys.executable, "-c"]:
                raise subprocess.CalledProcessError(1, command)
            return real_run(command, *args, **kwargs)

        run.side_effect = fail_validation
        with pytest.raises(subprocess.CalledProcessError):
            apply_authoring_result(repo, authored, request_id="rollback")

    assert not (repo / "agent_pipeline/adapters/broken.py").exists()
    history = repo / ".git" / "neurotoken-sdk-authoring"
    assert not list(history.glob("*.patch"))
    assert not list(history.glob("*.json"))


def test_concurrent_applies_are_serialized(tmp_path: Path):
    repo = init_repo(tmp_path)
    active = 0
    maximum = 0
    guard = threading.Lock()
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []
    real_record = __import__(
        "agent_pipeline.adapter_authoring.workflow",
        fromlist=["record_authoring_result"],
    ).record_authoring_result

    def observed_record(*args, **kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.15)
        try:
            return real_record(*args, **kwargs)
        finally:
            with guard:
                active -= 1

    def apply(name: str):
        try:
            barrier.wait()
            apply_authoring_result(repo, result_for(repo, name, f"NAME = '{name}'"), request_id=name)
        except BaseException as exc:
            errors.append(exc)

    with patch("agent_pipeline.adapter_authoring.workflow.record_authoring_result", side_effect=observed_record):
        threads = [threading.Thread(target=apply, args=(name,)) for name in ("one", "two")]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)

    assert errors == []
    assert maximum == 1
    assert (repo / "agent_pipeline/adapters/one.py").exists()
    assert (repo / "agent_pipeline/adapters/two.py").exists()
