import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.pipeline_stages import REFERENCE_STAGE_SCRIPTS, STAGE_ORDER, resolve_stage_dependencies
from agent_pipeline.tools import run_reference_stage_tool, stage_command, tokenize_postcondition_failure


def request(*stages: str) -> PipelineRequest:
    return PipelineRequest(
        config_path="agent_pipeline/configs/eeg_h5.toml",
        stages=list(stages),
        target_h5="/input/sample.h5",
        report_root="/output",
    )


def test_stage_order_and_internal_workers_are_stable():
    assert STAGE_ORDER == (
        "preflight",
        "package",
        "tokenize",
        "readme",
        "docs",
        "validate",
        "reconstruct",
    )
    assert set(REFERENCE_STAGE_SCRIPTS) == {
        "tokenize",
        "readme",
        "docs",
        "validate",
        "reconstruct",
    }


def test_tokenize_adds_required_predecessors():
    pipeline_request = request("tokenize")
    added = resolve_stage_dependencies(pipeline_request)
    assert pipeline_request.stages == ["preflight", "package", "tokenize"]
    assert added == ["preflight", "package"]


def test_validate_adds_release_document_dependencies():
    pipeline_request = request("validate")
    added = resolve_stage_dependencies(pipeline_request)
    assert pipeline_request.stages == ["readme", "docs", "validate"]
    assert added == ["readme", "docs"]


def test_reconstruct_adds_explicit_package_and_tokenize_closure():
    pipeline_request = request("reconstruct")
    added = resolve_stage_dependencies(pipeline_request)
    assert pipeline_request.stages == ["preflight", "package", "tokenize", "reconstruct"]
    assert added == ["preflight", "package", "tokenize"]


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("tokenize", "2_neural2token_human_new.py"),
        ("readme", "3_make_readme.py"),
        ("docs", "4_generate_release_docs_and_schemas.py"),
        ("validate", "5_validate_neurotoken_dataset.py"),
        ("reconstruct", "6_verify_token_reconstruction.py"),
    ],
)
def test_stage_command_targets_internal_worker(stage: str, expected: str):
    command = stage_command(
        stage=stage,
        config_path="/run/config.toml",
        report_root=Path("/run/reports"),
        reference_scripts_dir=Path("/workers"),
        python="/env/bin/python",
        smoke_test=True,
        overwrite_token_output=True,
        allow_placeholder_readme=True,
        strict_validation=True,
    )
    assert command[:3] == ["/env/bin/python", "-B", f"/workers/{expected}"]
    assert command[3:5] == ["--config", "/run/config.toml"]


def write_tokenize_config(tmp_path: Path) -> tuple[Path, Path]:
    token_root = tmp_path / "neurotoken"
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(f'[paths]\ntoken_root = "{token_root}"\n', encoding="utf-8")
    return config_path, token_root


def test_tokenize_postcondition_rejects_zero_successful_sessions(tmp_path: Path):
    config_path, token_root = write_tokenize_config(tmp_path)
    token_root.mkdir()
    (token_root / "manifest.json").write_text(
        json.dumps({"num_sessions": 0, "sessions": [], "failed_sessions": ["session-1"]}),
        encoding="utf-8",
    )

    finding = tokenize_postcondition_failure(str(config_path))

    assert finding is not None
    assert finding["category"] == "tokenization_no_sessions"
    assert "zero successful sessions" in finding["message"]
    assert "1 session(s) failed" in finding["message"]


def test_tokenize_postcondition_accepts_nonempty_consistent_manifest(tmp_path: Path):
    config_path, token_root = write_tokenize_config(tmp_path)
    token_root.mkdir()
    (token_root / "manifest.json").write_text(
        json.dumps({"num_sessions": 1, "sessions": [{"session_id": "session-1"}]}),
        encoding="utf-8",
    )

    assert tokenize_postcondition_failure(str(config_path)) is None


def test_durable_worker_logs_are_redirected_and_only_bounded_tails_enter_report(tmp_path: Path, monkeypatch):
    config_path, token_root = write_tokenize_config(tmp_path)
    token_root.mkdir()
    (token_root / "manifest.json").write_text(
        json.dumps({"num_sessions": 1, "sessions": [{"session_id": "session-1"}]}),
        encoding="utf-8",
    )
    report_root = tmp_path / "reports"
    report_root.mkdir()
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    stdout_log.write_text("x" * 40_000, encoding="utf-8")
    stderr_log.write_text("y" * 40_000, encoding="utf-8")
    monkeypatch.setenv("NEUROTOKEN_STAGE_STDOUT_LOG", str(stdout_log))
    monkeypatch.setenv("NEUROTOKEN_STAGE_STDERR_LOG", str(stderr_log))

    with patch(
        "agent_pipeline.tools.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0),
    ) as runner:
        report = run_reference_stage_tool(
            stage="tokenize",
            config_path=str(config_path),
            report_root=str(report_root),
            reference_scripts_dir=str(tmp_path),
        )

    assert runner.call_args.kwargs["stdout"].name == str(stdout_log)
    assert runner.call_args.kwargs["stderr"].name == str(stderr_log)
    assert len(report["stdout"].encode("utf-8")) <= 32_000
    assert len(report["stderr"].encode("utf-8")) <= 32_000


def test_tokenize_stage_is_failed_when_worker_returns_zero_but_manifest_is_empty(tmp_path: Path):
    config_path, token_root = write_tokenize_config(tmp_path)
    token_root.mkdir()
    (token_root / "manifest.json").write_text(
        json.dumps({"num_sessions": 0, "sessions": [], "failed_sessions": ["session-1"]}),
        encoding="utf-8",
    )
    report_root = tmp_path / "reports"
    report_root.mkdir()

    with patch(
        "agent_pipeline.tools.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="CUDA out of memory"),
    ):
        report = run_reference_stage_tool(
            stage="tokenize",
            config_path=str(config_path),
            report_root=str(report_root),
            reference_scripts_dir=str(tmp_path),
        )

    assert report["status"] == "failed"
    assert report["returncode"] == 0
    assert report["findings"][0]["category"] == "tokenization_no_sessions"
    assert report["report"]["status"] == "failed"
