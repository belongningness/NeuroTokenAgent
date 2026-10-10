from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import pytest

from agent_pipeline.adapters import build_adapter
from agent_pipeline.adapters.meg_h5 import MEGH5Adapter
from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.server_policy import select_linked_validation_target


class RuntimeGeneratedAdapter:
    def __init__(self, config):
        self.config = config


def test_build_adapter_discovers_generated_module_not_statically_registered():
    module = SimpleNamespace(RuntimeGeneratedAdapterAdapter=RuntimeGeneratedAdapter)
    config = {"source": {"adapter": "runtime_generated_adapter"}}

    with patch("agent_pipeline.adapters.import_module", return_value=module) as import_module:
        adapter = build_adapter(config)

    import_module.assert_called_once_with("agent_pipeline.adapters.runtime_generated_adapter")
    assert isinstance(adapter, RuntimeGeneratedAdapter)
    assert adapter.config is config


@pytest.mark.parametrize(
    ("adapter_name", "class_name"),
    [
        ("smn_meg_h5", "SmnMegH5Adapter"),
        ("nod_meg_h5", "NodMegH5Adapter"),
    ],
)
def test_build_adapter_loads_dataset_adapters_dynamically(adapter_name: str, class_name: str):
    adapter = build_adapter({
        "source": {"adapter": adapter_name},
        "dataset": {"modality": "meg"},
        "chunking": {"strategy": "rest"},
    })

    assert adapter.__class__.__name__ == class_name


@pytest.mark.parametrize(
    "adapter_name",
    [
        "smn_meg_sub_02_ses_01_rdr_57_meg_adapter_7",
        "brainmu_sub001_eeg_h5",
        "camcan_h5",
    ],
)
def test_build_adapter_rejects_removed_dataset_specific_adapters(adapter_name: str):
    with pytest.raises(ValueError, match="Unsupported source.adapter"):
        build_adapter({"source": {"adapter": adapter_name}})


def test_build_adapter_rejects_unsafe_or_missing_dynamic_module_names():
    with pytest.raises(ValueError, match="Unsupported source.adapter"):
        build_adapter({"source": {"adapter": "../outside"}})


def write_meg_candidate(path: Path, *, valid: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as source:
        source.create_dataset("meg", data=[[1.0, 2.0]])
        if valid:
            source.create_dataset("meg_timestamps", data=[0.0, 0.1])
            source.create_dataset("metadata_json_u8", data=[])
            source.create_dataset("meg_channels_pkl_u8", data=[])


def test_meg_discovery_is_recursive_and_schema_aware(tmp_path: Path):
    invalid = tmp_path / "a-invalid.h5"
    valid = tmp_path / "nested" / "b-valid.hdf5"
    write_meg_candidate(invalid, valid=False)
    write_meg_candidate(valid, valid=True)
    adapter = MEGH5Adapter({"dataset": {"modality": "meg"}})

    assert adapter.discover(tmp_path) == [valid]


def test_linked_validation_selection_uses_adapter_discovery(tmp_path: Path):
    invalid = tmp_path / "a-invalid.h5"
    valid = tmp_path / "nested" / "b-valid.h5"
    write_meg_candidate(invalid, valid=False)
    write_meg_candidate(valid, valid=True)
    request = PipelineRequest(
        config_path="agent_pipeline/configs/camcan_meg_h5.toml",
        stages=["preflight"],
        execution_strategy="linked",
        full_input_root=str(tmp_path),
        report_root=str(tmp_path / "output"),
        source_adapter="meg_h5",
    )

    selected = select_linked_validation_target(request)

    assert selected.validation_target_h5 == str(valid.resolve())
    assert selected.request_for_phase("full").input_root == str(tmp_path)


def test_linked_validation_rejects_explicit_h5_not_discovered_by_adapter(tmp_path: Path):
    invalid = tmp_path / "invalid.h5"
    write_meg_candidate(invalid, valid=False)
    request = PipelineRequest(
        config_path="agent_pipeline/configs/camcan_meg_h5.toml",
        stages=["preflight"],
        execution_strategy="linked",
        validation_target_h5=str(invalid),
        full_input_root=str(tmp_path),
        report_root=str(tmp_path / "output"),
        source_adapter="meg_h5",
    )

    with pytest.raises(ValueError, match="was not discovered"):
        select_linked_validation_target(request)
