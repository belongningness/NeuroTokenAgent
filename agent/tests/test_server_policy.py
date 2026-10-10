from pathlib import Path

import pytest

from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.server_policy import is_within, validate_pipeline_paths


def test_is_within_rejects_sibling_prefix(tmp_path: Path):
    allowed = tmp_path / "data"
    assert is_within(allowed / "sample.h5", [allowed.resolve()])
    assert not is_within(tmp_path / "database" / "sample.h5", [allowed.resolve()])


def test_pipeline_paths_require_allowed_roots(tmp_path: Path, monkeypatch):
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"
    input_root.mkdir()
    output_root.mkdir()
    source = input_root / "sample.h5"
    source.touch()
    monkeypatch.setenv("NEUROTOKEN_ALLOWED_INPUT_ROOTS", str(input_root))
    monkeypatch.setenv("NEUROTOKEN_ALLOWED_OUTPUT_ROOTS", str(output_root))
    request = PipelineRequest(
        config_path="agent_pipeline/configs/eeg_h5.toml",
        stages=["preflight"],
        target_h5=str(source),
        report_root=str(output_root / "run"),
    )
    validate_pipeline_paths(request)
    request.report_root = str(tmp_path / "outside")
    with pytest.raises(PermissionError):
        validate_pipeline_paths(request)
