from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "agent_pipeline" / "reference_scripts" / "2_neural2token_human_new.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("neurotoken_stage2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
stage2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage2)


class RecordingEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, model, signals, modality, config, channel_masks, time_masks):
        del model, modality, config, channel_masks
        self.batch_sizes.append(len(signals))
        outputs = []
        for signal, time_mask in zip(signals, time_masks):
            valid_time_cols = int(np.count_nonzero(np.max(time_mask.reshape(4, 2), axis=1)))
            token_count = 4 * valid_time_cols
            window_id = int(signal[0, 0])
            outputs.append(np.full((token_count, 2), window_id, dtype=np.int32))
        return np.concatenate(outputs, axis=0)


def runtime_config(max_chunk_batch_size: int):
    config = stage2.RuntimeConfig(
        modality="meg",
        device="cpu",
        use_bfloat16=False,
        checkpoint_path=Path("checkpoint"),
        train_config_path=Path("train.yaml"),
        use_ema=True,
        data_seed=42,
        storage_dtype="fp32",
        max_chunk_batch_size=max_chunk_batch_size,
    )
    config.quality_cfg = {}
    return config


def test_event_windows_are_microbatched_and_restored_in_time_order():
    signal = np.concatenate(
        [
            np.full((8, 8), 1, dtype=np.float32),
            np.full((8, 8), 2, dtype=np.float32),
            np.full((8, 4), 3, dtype=np.float32),
        ],
        axis=1,
    )
    transform = SimpleNamespace(
        time_size=8,
        cell_num=8,
        cell_stride=2,
        time_stride=2,
        padding_loss_weight=0.1,
    )
    clean_info = {
        "channel_count_raw": 8,
        "channel_count_kept": 8,
        "bad_channel_count": 0,
        "bad_channel_source": "test",
        "good_channel_indices": list(range(8)),
    }
    encoder = RecordingEncoder()

    with (
        patch.object(stage2, "clean_2d_signal", return_value=(signal, clean_info)),
        patch.object(stage2, "encode_2d_window_batch", side_effect=encoder),
    ):
        token_ids, info = stage2.tokenize_neural_sample(
            object(),
            transform,
            signal,
            "meg",
            runtime_config(max_chunk_batch_size=2),
            channels=[],
            sample_key="event-1",
            chunking_strategy="event",
        )

    assert encoder.batch_sizes == [2, 1]
    assert max(encoder.batch_sizes) <= 2
    assert info["chunking_strategy"] == "event"
    assert info["num_time_windows"] == 3
    assert info["max_chunk_batch_size"] == 2
    assert len(info["windows"]) == 1
    assert info["windows"][0]["time_len"] == 20
    assert info["windows"][0]["grid_shape"] == [4, 10]
    expected_one_codebook = np.tile(
        np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3], dtype=np.int32),
        4,
    )
    np.testing.assert_array_equal(token_ids[:, 0], expected_one_codebook)
    np.testing.assert_array_equal(token_ids[:, 1], expected_one_codebook)


def test_runtime_config_rejects_nonpositive_microbatch_limit():
    try:
        runtime_config(max_chunk_batch_size=0)
    except ValueError as exc:
        assert "max_chunk_batch_size must be positive" in str(exc)
    else:
        raise AssertionError("Expected a nonpositive micro-batch limit to be rejected")
