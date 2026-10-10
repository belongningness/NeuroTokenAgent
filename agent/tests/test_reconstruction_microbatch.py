from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = ModuleType("torch")

    def no_grad():
        return lambda function: function

    torch_stub.no_grad = no_grad
    torch_stub.Tensor = object
    torch_stub.device = object
    sys.modules["torch"] = torch_stub


SCRIPT_PATH = Path(__file__).parents[1] / "agent_pipeline" / "reference_scripts" / "6_verify_token_reconstruction.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("neurotoken_stage6", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
stage6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage6)


def test_event_tokens_are_split_by_model_window_in_time_order():
    rows = 2
    window_time_lengths = [8, 8, 4]
    time_stride = 2
    grid = np.empty((rows, 10, 2), dtype=np.int32)
    grid[:, :4] = 1
    grid[:, 4:8] = 2
    grid[:, 8:] = 3

    windows = stage6.split_event_tokens_by_model_window(
        grid.reshape(-1, 2),
        [rows, 10],
        window_time_lengths,
        time_stride,
    )

    assert [window.shape for window in windows] == [(8, 2), (8, 2), (4, 2)]
    np.testing.assert_array_equal(windows[0], np.ones((8, 2), dtype=np.int32))
    np.testing.assert_array_equal(windows[1], np.full((8, 2), 2, dtype=np.int32))
    np.testing.assert_array_equal(windows[2], np.full((4, 2), 3, dtype=np.int32))


def test_reconstruction_batches_respect_limit_and_token_shape():
    windows = [
        np.zeros((8, 2), dtype=np.int32),
        np.zeros((8, 2), dtype=np.int32),
        np.zeros((8, 2), dtype=np.int32),
        np.zeros((4, 2), dtype=np.int32),
    ]

    assert stage6.reconstruction_microbatch_slices(windows, 2) == [(0, 2), (2, 3), (3, 4)]


def test_event_token_split_rejects_grid_mismatch():
    with pytest.raises(RuntimeError, match="model windows require"):
        stage6.split_event_tokens_by_model_window(
            np.zeros((20, 2), dtype=np.int32),
            [2, 10],
            [8, 8],
            2,
        )
