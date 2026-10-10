#!/usr/bin/env python3
"""Convert tar-packaged human neural sessions into NeuroToken dataset format.

This is the ``2.5`` variant of ``2_neural2token_human_rel_dir.py``. The dataset
structure, tar iteration, session output layout and manifest are kept *identical*
to the original script. The only thing that changed is the **tokenization
scheme**: it is now made to match ``tok_real_data_evaluation.ipynb`` exactly,
reusing that notebook's code as much as possible.

Concretely, the notebook-style tokenization means:

* The model is built via the notebook's ``build_unitok_for_inference``:
  - notebook ``TOK_MODEL_ARGS`` (detail-refiner / STFT / wavelet / EMA-friendly args)
  - model sizes / strides resolved from ``hdata/configs/tok_human.yaml`` via
    ``infer_brain_model_sizes_from_dataset_meta`` (after the fixed-shape patch)
  - EMA generator weights loaded through ``DDPCheckpoint._find_checkpoint_files``
* Pre-processing produces the same fixed shapes / alignment-padding masks as the
  notebook (EEG ``256x640`` / MEG ``384x640`` / fMRI ``2x96x96x96``; real ``1.0``
  / alignment ``padding_loss_weight`` / pure-pad ``0.0``) but is made **fully
  deterministic** — no random time crop, channel permutation or sr resampling:
  - EEG/MEG: ``preprocess_2d_fixed_shape`` keeps every channel and time sample in
    its original order and only zero-pads to the model shape. Rest chunks are
    split into consecutive model windows; event chunks remain one logical token
    segment even when several model windows are needed. Nothing is cropped or
    resampled.
  - fMRI: the real ``FmriVolumeTransform`` is reused for its 96³ spatial resize
    but forced deterministic (center crop + nearest selection + no augment +
    temporal window pinned to the chunk start).

The single function that performs this tokenization is
``tokenize_neural_sample``. Everything before/after it (loading tar members,
writing ``<modality>_token_ids.npy`` + sidecars + manifest) is unchanged from stage 2.

Output structure (unchanged):

neurotoken_dataset/
├── README.md                      # Stage 3 (3_make_readme.py) generates this
├── manifest.json
├── subjects.json
├── tokenizer/
│   ├── codebook.npz
│   └── codebook_metadata.json
├── stimuli/
└── sessions/
    ├── <subject>_<original_session_name>/
    │   ├── <modality>_token_ids.npy
    │   ├── token_metadata.json
    │   ├── session_metadata.json
    │   └── sidecars copied from the source session
    └── ...
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import math
import os
import random
import shutil
import sys
import tarfile
import types
import yaml
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neurotoken_config import cfg_bool, cfg_get, cfg_path, load_config

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# session-level metadata sidecar filename, produced by stage1 (1_make_tar_packages...).
SESSION_METADATA_FILENAME = 'session_metadata.json'


PROJECT_ROOT: Optional[Path] = None

# UniTok project imports are loaded lazily after PROJECT_ROOT is resolved from
# config. This keeps --help, py_compile and preflight diagnostics from failing at
# import time on machines where the tokenizer project has not been mounted yet.
open_clip = None
build_neural_transform = None
UniTok = None
UniTokConfig = None
build_quality_filter_config = None
compute_zscore_std_floor = None
detect_static_bad_channels = None
HumanFMRIDataset = None
_load_required_fmri_stats = None
_normalise_fmri_volume = None

SUPPORTED_MODALITIES = {'eeg', 'meg', 'fmri'}
CELL_STRIDE = 8
TIME_STRIDE = 8

# ---------------------------------------------------------------------------
# Inline replacements for train/pretrain_tok to avoid pulling in its import
# chain: pretrain_tok → hdata.dataset_base → hdata.video_utils → decord.
# These three items are kept byte-identical to the originals.
# ---------------------------------------------------------------------------


def load_grouped_dataset_meta(config_path: str) -> dict:
    """Read the tok_human.yaml grouped-dataset config (same as pretrain_tok)."""
    with open(config_path, "r") as stream:
        dataset_meta = yaml.safe_load(stream) or {}
    if not isinstance(dataset_meta, dict):
        raise ValueError(
            f"Dataset config at {config_path} must load to a dict, "
            f"got {type(dataset_meta).__name__}"
        )
    return dataset_meta


def infer_brain_model_sizes_from_dataset_meta(
    dataset_meta: dict,
    require_fixed_shape: bool = False,
) -> dict:
    """Resolve model sizes/strides from the grouped dataset meta (same as pretrain_tok)."""
    resolved_sizes: dict = {}

    def _assign_unique(key: str, value: tuple, group_name: str):
        if key in resolved_sizes and resolved_sizes[key] != value:
            raise ValueError(
                f"Conflicting {key} definitions between grouped datasets: "
                f"existing={resolved_sizes[key]}, new={value} from {group_name}"
            )
        resolved_sizes[key] = value

    def _normalize_int_sequence(value, field_name: str, group_name: str, expected_len: int):
        if not isinstance(value, (list, tuple)) or len(value) != expected_len:
            raise ValueError(
                f"{group_name} must provide {field_name} as a length-{expected_len} list/tuple, got {value}"
            )
        return tuple(int(v) for v in value)

    def _resolve_candidate(value, field_name: str, group_name: str, modality_name: str) -> int:
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            raise ValueError(f"{group_name} must provide a non-empty {field_name} for {modality_name}")
        if field_name == "time_size_candidates":
            resolved = max(int(v) for v in value)
            if len(value) > 1:
                logger.info(
                    "[time-padding] %s.%s has %d candidates %s for %s; "
                    "model size uses max=%d. Time padding pads shorter samples to this size.",
                    group_name, field_name, len(value), list(value), modality_name, resolved,
                )
            return resolved
        if require_fixed_shape and len(value) != 1:
            resolved = max(int(v) for v in value)
            logger.info(
                "[fixed-shape] %s.%s has %d candidates %s for %s; "
                "using max=%d. Data will be padded to this size during tok training. "
                "NaViT can still use all candidates via pos_embed interpolation.",
                group_name, field_name, len(value), list(value), modality_name, resolved,
            )
            return resolved
        return int(value[0])

    def _normalize_stage_strides(value, field_name: str, group_name: str, expected_len: int):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(
                f"{group_name} must provide {field_name} as a length-3 list/tuple, got {value}"
            )
        return tuple(
            _normalize_int_sequence(v, f"{field_name}[{idx}]", group_name, expected_len)
            for idx, v in enumerate(value)
        )

    def _resolve_stride_args(grouped_config, group_name: str, expected_len: int):
        encoder_args = grouped_config.get("encoder_stride_args")
        decoder_args = grouped_config.get("decoder_stride_args")
        resolved = {}
        if encoder_args is not None:
            if not isinstance(encoder_args, dict):
                raise ValueError(f"{group_name} encoder_stride_args must be a dict, got {encoder_args}")
            resolved["encoder_stem_stride"] = _normalize_int_sequence(
                encoder_args.get("stem_stride"), "encoder_stride_args.stem_stride", group_name, expected_len
            )
            resolved["encoder_stage_strides"] = _normalize_stage_strides(
                encoder_args.get("stage_strides"), "encoder_stride_args.stage_strides", group_name, expected_len
            )
        if decoder_args is not None:
            if not isinstance(decoder_args, dict):
                raise ValueError(f"{group_name} decoder_stride_args must be a dict, got {decoder_args}")
            resolved["decoder_stem_stride"] = _normalize_int_sequence(
                decoder_args.get("stem_stride"), "decoder_stride_args.stem_stride", group_name, expected_len
            )
            resolved["decoder_stage_strides"] = _normalize_stage_strides(
                decoder_args.get("stage_strides"), "decoder_stride_args.stage_strides", group_name, expected_len
            )
        return resolved

    for grouped_dataset_name, grouped_config in dataset_meta.items():
        if not isinstance(grouped_config, dict):
            continue

        transform_args = grouped_config.get("neural_transform_args") or {}
        if not isinstance(transform_args, dict):
            continue

        group_name_lower = str(grouped_dataset_name).lower()
        dataset_names = grouped_config.get("dataset_names") or []
        dataset_names_lower = [str(name).lower() for name in dataset_names]

        def _matches(modality_name: str) -> bool:
            return modality_name in group_name_lower or any(modality_name in name for name in dataset_names_lower)

        transform_type = transform_args.get("transform_type")

        if transform_type == "continuous_2d" and _matches("eeg"):
            eeg_size = (
                _resolve_candidate(transform_args.get("cell_num_candidates"), "cell_num_candidates", grouped_dataset_name, "eeg"),
                _resolve_candidate(transform_args.get("time_size_candidates"), "time_size_candidates", grouped_dataset_name, "eeg"),
            )
            _assign_unique("eeg_size", eeg_size, grouped_dataset_name)
            stride_args = _resolve_stride_args(grouped_config, grouped_dataset_name, 2)
            for key, value in stride_args.items():
                _assign_unique(f"eeg_{key}", value, grouped_dataset_name)

        if transform_type == "continuous_2d" and _matches("meg"):
            meg_size = (
                _resolve_candidate(transform_args.get("cell_num_candidates"), "cell_num_candidates", grouped_dataset_name, "meg"),
                _resolve_candidate(transform_args.get("time_size_candidates"), "time_size_candidates", grouped_dataset_name, "meg"),
            )
            _assign_unique("meg_size", meg_size, grouped_dataset_name)
            stride_args = _resolve_stride_args(grouped_config, grouped_dataset_name, 2)
            for key, value in stride_args.items():
                _assign_unique(f"meg_{key}", value, grouped_dataset_name)

        if transform_type == "fmri_volume_4d" or _matches("fmri"):
            fmri_size = (
                _resolve_candidate(transform_args.get("time_size_candidates"), "time_size_candidates", grouped_dataset_name, "fmri"),
                *_normalize_int_sequence(transform_args.get("target_volume_shape"), "target_volume_shape", grouped_dataset_name, 3),
            )
            fmri_patch_size = _normalize_int_sequence(
                transform_args.get("patch_size"), "patch_size", grouped_dataset_name, 4
            )
            _assign_unique("fmri_size", fmri_size, grouped_dataset_name)
            _assign_unique("fmri_patch_size", fmri_patch_size, grouped_dataset_name)
            stride_args = _resolve_stride_args(grouped_config, grouped_dataset_name, 4)
            for key, value in stride_args.items():
                _assign_unique(f"fmri_{key}", value, grouped_dataset_name)

    return resolved_sizes


class DDPCheckpoint:
    """Minimal checkpoint-file discovery extracted from train/pretrain_tok.py.

    Only ``_find_checkpoint_files`` is kept — the full class (save, load, etc.)
    is not needed at inference time and would pull in torch.distributed + the
    hdata.dataset_base → video_utils → decord chain.
    """

    @staticmethod
    def _find_checkpoint_files(ckpt_path: str) -> dict:
        files: dict = {}
        if os.path.isdir(ckpt_path):
            file_map = {
                'generator': 'generator.pt',
                'ema_generator': 'ema_generator.pt',
                'discriminator': 'discriminator.pt',
                'train_state': 'train_state.pt',
                'checkpoint': 'checkpoint.pt',
                'legacy_model': 'model.pt',
                'legacy_ema_model': 'ema_model.pt',
            }
            for key, filename in file_map.items():
                full_path = os.path.join(ckpt_path, filename)
                if os.path.exists(full_path):
                    files[key] = full_path
        elif os.path.exists(ckpt_path):
            files['checkpoint'] = ckpt_path
        return files


def configure_unitok_project_root(project_root: Path) -> None:
    global PROJECT_ROOT
    PROJECT_ROOT = Path(project_root).expanduser().resolve()


def register_lightweight_hdata_package() -> None:
    """Expose hdata submodules without executing the project's heavy __init__."""
    if PROJECT_ROOT is None:
        raise RuntimeError('PROJECT_ROOT must be configured before loading hdata modules')
    hdata_path = PROJECT_ROOT / 'hdata'
    module = sys.modules.get('hdata')
    if module is None:
        module = types.ModuleType('hdata')
        module.__path__ = [str(hdata_path)]
        sys.modules['hdata'] = module
    elif not hasattr(module, '__path__'):
        raise RuntimeError(f"Existing sys.modules['hdata'] is not a package: {module!r}")


def ensure_unitok_imports() -> None:
    global open_clip, build_neural_transform, UniTok, UniTokConfig
    global build_quality_filter_config, compute_zscore_std_floor, detect_static_bad_channels
    global HumanFMRIDataset, _load_required_fmri_stats, _normalise_fmri_volume

    if UniTok is not None:
        return
    if PROJECT_ROOT is None or not (PROJECT_ROOT / 'modeling').exists():
        raise FileNotFoundError(
            f'Cannot find human UniTok project root: {PROJECT_ROOT}. '
            'Set [tokenizer].project_root in the pipeline config or pass --project-root.'
        )
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    register_lightweight_hdata_package()
    open_clip = importlib.import_module('open_clip')
    build_neural_transform = importlib.import_module('hdata.nerual_utils').build_neural_transform
    unitok_module = importlib.import_module('modeling.unitok.unitok')
    UniTok = unitok_module.UniTok
    UniTokConfig = unitok_module.UniTokConfig
    # load_grouped_dataset_meta / infer_brain_model_sizes_from_dataset_meta /
    # DDPCheckpoint are inlined below (see "Inline replacements for
    # train/pretrain_tok" comment) to avoid pulling in hdata.dataset_base →
    # video_utils → decord.
    quality_filters = importlib.import_module('hdata.quality_filters')
    build_quality_filter_config = quality_filters.build_quality_filter_config
    compute_zscore_std_floor = quality_filters.compute_zscore_std_floor
    detect_static_bad_channels = quality_filters.detect_static_bad_channels


def ensure_fmri_imports() -> None:
    global HumanFMRIDataset, _load_required_fmri_stats, _normalise_fmri_volume
    ensure_unitok_imports()
    if HumanFMRIDataset is not None:
        return
    fmri_module = importlib.import_module('hdata.fmri')
    HumanFMRIDataset = fmri_module.HumanFMRIDataset
    _load_required_fmri_stats = fmri_module._load_required_fmri_stats
    _normalise_fmri_volume = fmri_module._normalise_fmri_volume

# ---------------------------------------------------------------------------
# Tokenization model args — copied verbatim from tok_real_data_evaluation.ipynb.
# Sizes/strides are NOT listed here; they are resolved from the train config via
# infer_brain_model_sizes_from_dataset_meta (matching the notebook exactly).
# ---------------------------------------------------------------------------
TOK_MODEL_ARGS = dict(
    backbone='local-dir:/home/fanguofan/cache/checkpoint/ViT-L-16-SigLIP2-256/ViT-L-16-SigLIP2-256',
    disc_backbone='local-dir:/home/fanguofan/cache/checkpoint/ViT-B-16-SigLIP2-256',
    drop_path=0.1,
    quant_proj='attn',
    vocab_size=32768,  # tok.sh
    vq_beta=0.25,
    num_codebooks=8,   # tok.sh
    vocab_width=64,    # tok.sh
    embed_dim=1024,
    grad_ckpt=False,
    rec_loss=True,
    contrast_loss=False,
    use_image=False,
    use_behavior=False,
    gan_loss=True,
    rec_loss_weight=1.0,
    vq_loss_weight=1.0,
    contrast_loss_weight=1.0,
    gan_loss_weight=0.05,
    gradient_loss_weight=0.08,
    ssim_loss_weight=0.0,
    stft_loss_weight=0.004,
    wavelet_loss_weight=0.003,
    stft_fft_sizes='16_32_64_128',
    stft_hop_sizes='4_8_16_32',
    stft_win_lengths='16_32_64_128',
    stft_sc_weight=0.35,
    stft_logmag_weight=0.9,
    stft_valid_channel_threshold=0.5,
    stft_highfreq_weight_mode='linear',
    stft_highfreq_start_ratio=0.6,
    stft_highfreq_weight_max=2.5,
    stft_highfreq_weight_power=1.0,
    wavelet_type='haar',
    wavelet_levels=1,
    wavelet_approx_weight=0.25,
    wavelet_detail_weight=0.5,
    disc_kernel_size=3,
    disc_norm='gn',
    disc_norm_eps=1e-6,
    disc_specnorm=True,
    pearson_threshold_eeg=0.75,
    pearson_threshold_meg=0.75,
    pearson_threshold_fmri=0.75,
    disc_label_smoothing=0.0,
    use_prob_clip_loss=False,
    prob_a_init=1.0,
    prob_b_init=0.0,
    shortcut_downsample='conv',
    disc_shortcut_downsample='conv',
    use_detail_refiner=True,  # True
    detail_refiner_channels=16,
    detail_refiner_depth=2,
    detail_refiner_kernel_size=3,
    detail_refiner_alpha=0.2,
    detail_refiner_learnable_alpha=True,
    detail_loss_weight=0.04,
    detail_loss_kernel_sizes='3_7',
    detail_loss_start_steps=0,
    detail_loss_ramp_steps=6000,
    eeg_pos_mode='rope_axial',
    meg_pos_mode='rope_axial',
    fmri_pos_mode='rope_axial',
    rope_base_eeg_h=50.0,
    rope_base_eeg_w=50.0,
    rope_base_meg_h=50.0,
    rope_base_meg_w=50.0,
    rope_base_fmri_time=10.0,
    rope_base_fmri_space=10.0,
)


class RuntimeConfig:
    def __init__(
        self,
        *,
        modality: str,
        device: torch.device,
        use_bfloat16: bool,
        checkpoint_path: Path,
        train_config_path: Path,
        use_ema: bool,
        data_seed: int,
        storage_dtype: str = 'fp16',
        max_chunk_batch_size: int = 8,
    ):
        self.modality = modality
        self.device = device
        self.use_bfloat16 = use_bfloat16
        self.checkpoint_path = Path(checkpoint_path)
        self.train_config_path = Path(train_config_path)
        self.use_ema = bool(use_ema)
        self.data_seed = int(data_seed)
        # Precision the offline repack stage stored cleaned data with (fp16 by
        # default, matching repack_human's DTYPE). The eval notebook therefore
        # tokenizes fp16-rounded inputs; we round-trip our cleaned signal through
        # this dtype before encode so our token ids match what the model saw.
        self.storage_dtype = str(storage_dtype).strip().lower()
        self.max_chunk_batch_size = int(max_chunk_batch_size)
        if self.max_chunk_batch_size <= 0:
            raise ValueError(
                f'max_chunk_batch_size must be positive, got {self.max_chunk_batch_size}'
            )
        # Reference preprocessing config; populated in main() once the dataset
        # meta (tok_human.yaml) is loaded. quality_cfg feeds the reference
        # compute_zscore_std_floor; zscore_mode (fMRI) feeds _normalise_fmri_volume.
        self.quality_cfg: Optional[dict] = None
        self.zscore_mode: str = 'voxel'


def patch_open_clip_local_dir_fallback() -> None:
    ensure_unitok_imports()
    if not hasattr(open_clip, '_brainmu_original_create_model_from_pretrained'):
        open_clip._brainmu_original_create_model_from_pretrained = open_clip.create_model_from_pretrained
    original_create_model_from_pretrained = open_clip._brainmu_original_create_model_from_pretrained

    def fallback_arch_from_local_dir(model_name: str) -> Optional[str]:
        if not isinstance(model_name, str) or not model_name.startswith('local-dir:'):
            return None
        local_path = Path(model_name[len('local-dir:'):])
        if local_path.is_dir():
            return None
        arch = local_path.name
        if open_clip.get_model_config(arch) is None:
            arch = local_path.parent.name if local_path.parent.name == arch else arch
        return arch if open_clip.get_model_config(arch) is not None else None

    def create_model_from_pretrained_with_local_fallback(model_name, *args, **kwargs):
        arch = fallback_arch_from_local_dir(model_name)
        if arch is None:
            return original_create_model_from_pretrained(model_name, *args, **kwargs)
        logger.warning('Backbone local-dir is unavailable; instantiate open_clip architecture without pretrained weights: %s', arch)
        model = open_clip.create_model(arch, pretrained=None)
        return_transform = kwargs.get('return_transform', True)
        if return_transform is False:
            return model
        return model, None

    open_clip.create_model_from_pretrained = create_model_from_pretrained_with_local_fallback


# ---------------------------------------------------------------------------
# Notebook helpers (copied from tok_real_data_evaluation.ipynb) for resolving a
# checkpoint dir, patching the dataset meta to fixed shapes, and building UniTok.
# ---------------------------------------------------------------------------
def has_unitok_weight_files(path: Path) -> bool:
    if not path.exists():
        return False
    expected = ['generator.pt', 'ema_generator.pt', 'checkpoint.pt', 'model.pt', 'ema_model.pt']
    return any((path / name).exists() for name in expected)


def find_latest_unitok_checkpoint(checkpoint_root: Path) -> Path:
    checkpoint_root = Path(checkpoint_root)
    if checkpoint_root.is_file():
        return checkpoint_root
    if checkpoint_root.is_dir() and has_unitok_weight_files(checkpoint_root):
        return checkpoint_root
    if not checkpoint_root.exists():
        raise FileNotFoundError(f'Checkpoint root does not exist: {checkpoint_root}')

    candidates = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        base = child.name.replace('_final', '')
        if not base.isdigit():
            continue
        if not has_unitok_weight_files(child):
            continue
        step = int(base)
        is_final = child.name.endswith('_final')
        candidates.append((step, is_final, child))

    if not candidates:
        raise FileNotFoundError(f'No UniTok checkpoint found under: {checkpoint_root}')

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def patch_dataset_meta_for_fixed_shape(dataset_meta: dict) -> dict:
    patched = deepcopy(dataset_meta)
    for _, grouped_config in patched.items():
        if not isinstance(grouped_config, dict):
            continue
        transform_args = grouped_config.get('neural_transform_args')
        if not isinstance(transform_args, dict):
            continue
        for field_name in ('cell_num_candidates', 'time_size_candidates'):
            candidates = transform_args.get(field_name)
            if isinstance(candidates, (list, tuple)) and len(candidates) > 1:
                transform_args[field_name] = [max(int(v) for v in candidates)]
        if 'padding_loss_weight' not in transform_args:
            transform_args['padding_loss_weight'] = 0.1
    return patched


def build_unitok_for_inference(
    train_config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool = True,
):
    """Identical to the notebook's ``build_unitok_for_inference`` (plus the
    open_clip local-dir fallback so missing backbones don't crash)."""
    ensure_unitok_imports()
    patch_open_clip_local_dir_fallback()

    dataset_meta = load_grouped_dataset_meta(str(train_config_path))
    dataset_meta = patch_dataset_meta_for_fixed_shape(dataset_meta)
    resolved_sizes = infer_brain_model_sizes_from_dataset_meta(dataset_meta, require_fixed_shape=True)

    model_config = UniTokConfig(**TOK_MODEL_ARGS, **resolved_sizes)
    model = UniTok(model_config)

    files = DDPCheckpoint._find_checkpoint_files(str(checkpoint_path))
    source_key = 'ema_generator' if use_ema and 'ema_generator' in files else 'generator'
    if source_key not in files and 'legacy_model' in files:
        source_key = 'legacy_ema_model' if use_ema and 'legacy_ema_model' in files else 'legacy_model'

    if source_key in files:
        logger.info('Loading UniTok weights from: %s', files[source_key])
        state = torch.load(files[source_key], map_location='cpu')
    elif 'checkpoint' in files:
        logger.info('Loading UniTok weights from nested checkpoint: %s', files['checkpoint'])
        ckpt = torch.load(files['checkpoint'], map_location='cpu')
        key = 'ema_generator' if use_ema and 'ema_generator' in ckpt else 'generator'
        if key not in ckpt and 'model' in ckpt:
            key = 'model'
        state = ckpt.get(key, {})
        del ckpt
    else:
        state = {}

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info('Loaded weights: missing=%d, unexpected=%d', len(missing), len(unexpected))
    if missing[:5]:
        logger.info('First missing keys: %s', missing[:5])
    if unexpected[:5]:
        logger.info('First unexpected keys: %s', unexpected[:5])
    del state

    model = model.to(device).eval()
    return model, dataset_meta, resolved_sizes, files, missing, unexpected


def resolve_checkpoint_path(checkpoint_path: Optional[Path], checkpoint_root: Path) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.is_file() or has_unitok_weight_files(path):
            return path
        return find_latest_unitok_checkpoint(path)
    return find_latest_unitok_checkpoint(checkpoint_root)


def build_modality_neural_transform(dataset_meta: dict, modality: str):
    """Build the exact training/notebook transform for ``modality`` from the
    (fixed-shape patched) dataset meta. Reuses ``build_neural_transform``."""
    ensure_unitok_imports()
    group_name = None
    for name, cfg in dataset_meta.items():
        if not isinstance(cfg, dict) or 'neural_transform_args' not in cfg:
            continue
        if modality in str(name).lower():
            group_name = name
            break
    if group_name is None:
        raise ValueError(f'No grouped dataset with neural_transform_args matches modality={modality!r}')

    transform_args = deepcopy(dataset_meta[group_name]['neural_transform_args'])
    transform = build_neural_transform(**transform_args)

    # ------------------------------------------------------------------
    # Make the transform fully deterministic for tokenization.
    #
    # The training config samples randomly (``candidate_selection: random``,
    # random time crop / channel permutation, ``crop_mode: random_crop_or_pad``
    # for fMRI). For tokenizing already-chunked sessions we must NOT introduce
    # any randomness, otherwise re-tokenizing the same data would give different
    # ids and the "session = chunks concatenated in time order" invariant could
    # be broken. EEG/MEG bypass this object entirely (see
    # ``preprocess_2d_fixed_shape``); for fMRI we keep the transform but disable
    # every random branch.
    # ------------------------------------------------------------------
    if hasattr(transform, 'augment'):
        transform.augment = False
    if hasattr(transform, 'candidate_selection'):
        transform.candidate_selection = 'nearest'
    if hasattr(transform, 'channel_order_mode'):
        transform.channel_order_mode = 'sorted'
    if getattr(transform, 'crop_mode', None) == 'random_crop_or_pad':
        # Deterministic spatial crop (center) instead of a random crop.
        transform.crop_mode = 'center_crop_or_pad'

    logger.info('Built %s transform (deterministic) from group %s: %s', modality, group_name, transform_args)
    return transform


def build_modality_quality_cfg(dataset_meta: dict, modality: str) -> Tuple[dict, str]:
    """Build the reference quality-filter config + zscore_mode for ``modality``.

    Reuses the reference ``build_quality_filter_config`` so the z-score floor
    parameters (``zscore_std_floor_ratio`` / ``zscore_std_floor_abs`` /
    ``fallback_reference_stat``) and the fMRI ``zscore_mode`` come straight from
    ``tok_human.yaml`` rather than being hard-coded.
    """
    ensure_unitok_imports()
    for name, cfg in dataset_meta.items():
        if not isinstance(cfg, dict) or 'neural_transform_args' not in cfg:
            continue
        if modality in str(name).lower():
            quality_cfg = build_quality_filter_config(modality, cfg.get('quality_filter_args'))
            zscore_mode = str(cfg.get('zscore_mode', 'voxel')).strip().lower()
            return quality_cfg, zscore_mode
    raise ValueError(f'No grouped dataset matches modality={modality!r} for quality config')


# ---------------------------------------------------------------------------
# Pre-tokenization normalization (z-score), identical to what the notebook's
# dataset classes apply *before* the NeuralTransform.
# ---------------------------------------------------------------------------
def modality_sidecar_names(modality: str) -> Tuple[str, ...]:
    if modality in {'eeg', 'meg'}:
        return (f'{modality}_channels.json', 'events_info.json', SESSION_METADATA_FILENAME)
    if modality == 'fmri':
        return ('events_info.json', 'fmri_info.npz', 'fmri_meta.json', 'fmri_metadata.json', SESSION_METADATA_FILENAME)
    raise ValueError(f'Unsupported modality: {modality}')


def find_session_folders(input_base: Path) -> List[Path]:
    input_base = Path(input_base)
    if not input_base.exists():
        raise FileNotFoundError(input_base)
    if input_base.is_dir() and (input_base / 'data.tar').exists():
        return [input_base]
    folders = sorted(p for p in input_base.iterdir() if p.is_dir() and (p / 'data.tar').exists())
    if not folders:
        raise FileNotFoundError(f'No session folder with data.tar found under {input_base}')
    return folders


def load_channels_json(session_dir: Path, modality: str) -> List[dict]:
    if modality not in {'eeg', 'meg'}:
        return []
    path = session_dir / f'{modality}_channels.json'
    if not path.exists() and modality == 'eeg':
        path = session_dir / 'eeg_channels.json'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {modality}_channels.json in {session_dir}; channel mean/std are required for {modality.upper()}'
        )
    with path.open('r', encoding='utf-8') as f:
        channels = json.load(f)
    if not channels:
        raise ValueError(f'Empty channel info in {path}; channel mean/std are required for {modality.upper()}')
    return channels


def load_fmri_info(session_dir: Path) -> Optional[dict]:
    """Load the session-level ``fmri_info.npz`` (mean/std/absmax volumes) as a
    plain dict so the reference fMRI helpers can read it via ``.get(...)``."""
    path = Path(session_dir) / 'fmri_info.npz'
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as npz:
        info = {}
        for key in npz.files:
            value = npz[key]
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            info[key] = value
    return info


def channel_stats_from_sidecar(channels: List[dict], expected_channels: int, modality: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read raw per-channel mean/std from the sidecar.

    The std floor is *not* applied here; it is delegated to the reference
    ``compute_zscore_std_floor`` so the z-score exactly matches the human UniTok
    dataset classes (eeg.py / meg.py).
    """
    if not channels:
        raise ValueError(f'Missing channel mean/std info for {modality.upper()}; channels sidecar is required')
    means = np.asarray([row.get('mean', 0.0) for row in channels], dtype=np.float32)
    stds = np.asarray([row.get('std', 1.0) for row in channels], dtype=np.float32)
    if means.shape[0] != expected_channels or stds.shape[0] != expected_channels:
        raise ValueError(f'Channel sidecar length mismatch: sidecar={len(channels)}, {modality.upper()} channels={expected_channels}')
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError(f'Invalid mean/std values in {modality}_channels.json')
    return means.astype(np.float32, copy=False), stds.astype(np.float32, copy=False)


def apply_storage_dtype(arr: np.ndarray, storage_dtype: str) -> np.ndarray:
    """Round ``arr`` through the offline repack storage precision.

    The repack stage (repack_human) writes cleaned EEG/MEG/fMRI arrays as fp16
    by default, so the eval notebook tokenizes fp16-rounded inputs. We mirror
    that by rounding to ``storage_dtype`` and casting back to fp32 (the model's
    working dtype) so our token ids match what the reference model saw.
    """
    dtype = str(storage_dtype).strip().lower()
    out = np.asarray(arr, dtype=np.float32)
    if dtype in ('fp16', 'float16'):
        return out.astype(np.float16).astype(np.float32, copy=False)
    if dtype in ('fp32', 'float32'):
        return out
    raise ValueError(f"Unsupported storage_dtype={storage_dtype!r}; expected 'fp16' or 'fp32'")


def clean_2d_signal(
    raw_signal: np.ndarray,
    channels: List[dict],
    timestamps: Optional[np.ndarray],
    modality: str,
    quality_cfg: dict,
) -> Tuple[np.ndarray, dict]:
    """Reproduce the offline repack EEG/MEG cleaning: bad-channel removal + z-score.

    This mirrors the bad-channel-removal + z-score portion of
    ``human_offline_filter.classify_continuous_sample`` by reusing the same
    reference primitives, so the cleaned signal matches the ``*_clean`` tars the
    eval pipeline consumes:

    1. ``detect_static_bad_channels`` flags bad channels. With our sidecars this
       takes the ``is_bad`` path (``use_is_bad: true``); the variance fallback
       (``enable_bad_channel_fallback: true``) only kicks in when ``is_bad`` is
       absent. Flagged channels are dropped — exactly the repack behaviour.
    2. ``compute_zscore_std_floor`` floors the per-channel std, then we z-score
       the kept channels with ``(raw - mean) / std_used``.

    Per the chosen scope we deliberately **do not** run the low-information
    sample gate (samples are never dropped nor marked). As a safety net, if every
    channel is flagged bad we keep all channels instead of producing an empty
    sample.

    Returns ``(z_signal, clean_info)`` where ``z_signal`` keeps only the good
    channels in their original order.
    """
    signal = np.asarray(raw_signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f'{modality.upper()} array must be 2D (C, T), got {signal.shape}')
    if not np.isfinite(signal).all():
        raise ValueError(f'{modality.upper()} array contains NaN/Inf')

    n_channels = int(signal.shape[0])
    mean, std = channel_stats_from_sidecar(channels, n_channels, modality)

    # Reference bad-channel detection. detect_static_bad_channels reads
    # use_is_bad / enable_bad_channel_fallback from quality_cfg (both true in
    # tok_human.yaml and the repack config). The is_bad path ignores timestamps;
    # the fallback needs them, so pass a monotonic axis when none is available.
    channels_df = pd.DataFrame(channels)
    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    else:
        ts = np.arange(signal.shape[1], dtype=np.float64)
    bad_mask, bad_source, bad_details = detect_static_bad_channels(signal, ts, channels_df, quality_cfg or {})
    bad_mask = np.asarray(bad_mask, dtype=bool).reshape(-1)
    good_mask = ~bad_mask
    if not good_mask.any():
        logger.warning(
            '%s: all %d channels flagged bad (source=%s); keeping all channels to avoid an empty sample.',
            modality.upper(), n_channels, bad_source,
        )
        good_mask = np.ones(n_channels, dtype=bool)
    good_indices = np.nonzero(good_mask)[0]

    signal = signal[good_mask]
    mean = mean[good_mask]
    std = std[good_mask]

    std_used, _ = compute_zscore_std_floor(std, quality_cfg or {})
    signal = (signal - mean[:, None]) / std_used[:, None]
    if not np.isfinite(signal).all():
        raise ValueError(f'{modality.upper()} array contains NaN/Inf after z-score')

    clean_info = {
        'channel_count_raw': int(n_channels),
        'channel_count_kept': int(good_indices.size),
        'bad_channel_count': int(n_channels - good_indices.size),
        'bad_channel_source': str(bad_source),
        'bad_channel_indices': [int(i) for i in np.nonzero(bad_mask)[0]],
        # Original-channel indices kept (in order), used to filter the output
        # *_channels.json so the sidecar rows stay aligned with the token rows.
        'good_channel_indices': [int(i) for i in good_indices],
    }
    return signal.astype(np.float32, copy=False), clean_info


def validate_fmri(raw_fmri: np.ndarray) -> np.ndarray:
    fmri = np.asarray(raw_fmri, dtype=np.float32)
    if fmri.ndim != 4:
        raise ValueError(f'FMRI array must be 4D (T, X, Y, Z), got {fmri.shape}')
    if not np.isfinite(fmri).all():
        raise ValueError('FMRI array contains NaN/Inf')
    return fmri.astype(np.float32, copy=False)


def normalize_fmri_volume(
    volume_xyzt: np.ndarray,
    fmri_info: Optional[dict],
    sample_key: str,
    zscore_mode: str,
) -> np.ndarray:
    """fMRI z-score, reusing the reference ``_normalise_fmri_volume``.

    ``mean_volume`` / ``std_volume`` are read from the session ``fmri_info.npz``
    via the reference ``_load_required_fmri_stats`` (which returns an identity
    transform when the stats are absent — the documented "already z-scored
    upstream" contract). The brain mask and background fill match ``fmri.py``.
    If stats are missing or shape-mismatched, the volume is returned unchanged
    (== identity), exactly like the reference dataset.
    """
    ensure_fmri_imports()
    if fmri_info is None:
        logger.warning('fMRI %s: no fmri_info.npz; skipping z-score (assume pre-z-scored upstream).', sample_key)
        return volume_xyzt
    stats = _load_required_fmri_stats(fmri_info, sample_key, 'openneural_fmri')
    if stats is None:
        logger.warning('fMRI %s: invalid mean/std_volume in fmri_info.npz; skipping z-score.', sample_key)
        return volume_xyzt
    mean_vol, std_vol = stats
    if tuple(mean_vol.shape) != tuple(volume_xyzt.shape[:3]):
        logger.warning(
            'fMRI %s: stats spatial shape %s != volume spatial %s; skipping z-score.',
            sample_key, tuple(mean_vol.shape), tuple(volume_xyzt.shape[:3]),
        )
        return volume_xyzt
    brain_mask = HumanFMRIDataset._extract_brain_mask(
        volume_xyzt, fmri_info, sample_key, 'openneural_fmri', sample=None,
    )
    normed = _normalise_fmri_volume(volume_xyzt, brain_mask, mean_vol, std_vol, zscore_mode=zscore_mode)
    return np.asarray(normed, dtype=np.float32)


def load_npy_from_tar_member(tar: tarfile.TarFile, member_name: str) -> np.ndarray:
    f = tar.extractfile(member_name)
    if f is None:
        raise FileNotFoundError(member_name)
    return np.load(io.BytesIO(f.read()))


def load_json_from_tar_member(tar: tarfile.TarFile, member_name: str) -> dict:
    f = tar.extractfile(member_name)
    if f is None:
        raise FileNotFoundError(member_name)
    return json.loads(f.read().decode('utf-8'))


def modality_suffixes(modality: str) -> Tuple[str, ...]:
    if modality in {'eeg', 'meg'}:
        return (f'.{modality}.npy',)
    if modality == 'fmri':
        return ('.fmri.npy', '.fmri_volume.npy', '.fmri_surface.npy')
    raise ValueError(f'Unsupported modality: {modality}')


def chunk_order_key(sample_key: str) -> Tuple[int, object]:
    """Sort by numeric chunk index so token concatenation == time order."""
    tail = sample_key.rsplit('_', 1)[-1]
    if tail.isdigit():
        return (0, int(tail))
    return (1, sample_key)


def list_modality_samples_in_tar(tar_path: Path, modality: str) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    with tarfile.open(tar_path, 'r') as tar:
        names = tar.getnames()
    suffixes = modality_suffixes(modality)
    name_set = set(names)
    samples = []
    for signal_name in names:
        suffix = next((s for s in suffixes if signal_name.endswith(s)), None)
        if suffix is None:
            continue
        key = signal_name[:-len(suffix)]
        ts_name = key + f'.{modality}_timestamps.npy'
        meta_name = key + '.metadata.json'
        samples.append((key, signal_name, ts_name if ts_name in name_set else None, meta_name if meta_name in name_set else None))
    samples.sort(key=lambda item: chunk_order_key(item[0]))
    return samples


def autocast_context(config: RuntimeConfig):
    if config.use_bfloat16:
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def safe_sample_filename(sample_key: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in sample_key)
    return safe.strip('._') or 'sample'


def encode_one_neural_sample(
    model: UniTok,
    signal: np.ndarray,
    modality: str,
    config: RuntimeConfig,
    channel_mask: Optional[np.ndarray] = None,
    time_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run UniTok's grouped ``encode`` to obtain per-token codebook ids.

    ``encode(..., auto_chunk_encode=True)`` internally uses the same
    valid-token packing path (``_encode_2d_tokens``) and the same nearest-code
    assignment (``quantizer.f_to_idx`` == the argmax used in ``forward``), so
    the resulting ids match the notebook's forward-time quantization.
    """
    tensor = torch.as_tensor(signal, dtype=torch.float32, device=config.device)
    shape_key = tuple(int(v) for v in tensor.shape)
    kwargs = {
        f'packed_{modality}_tensors': {shape_key: tensor.unsqueeze(0)},
        f'packed_{modality}_order': {shape_key: [0]},
        'auto_chunk_encode': True,
    }

    if modality in {'eeg', 'meg'}:
        if channel_mask is None:
            channel_mask = np.ones(tensor.shape[0], dtype=np.float32)
        channel_mask_t = torch.as_tensor(channel_mask, dtype=torch.float32, device=config.device)
        if int(channel_mask_t.shape[0]) != int(tensor.shape[0]):
            raise ValueError(
                f'{modality.upper()} channel_mask length {channel_mask_t.shape[0]} '
                f'does not match padded channel count {tensor.shape[0]}'
            )
        channel_shape_key = (int(channel_mask_t.shape[0]),)
        kwargs[f'packed_{modality}_channel_masks'] = {channel_shape_key: channel_mask_t.unsqueeze(0)}
        kwargs[f'packed_{modality}_channel_mask_order'] = {channel_shape_key: [0]}
        if time_mask is not None:
            time_mask_t = torch.as_tensor(time_mask, dtype=torch.float32, device=config.device)
            time_shape_key = (int(time_mask_t.shape[0]),)
            kwargs[f'packed_{modality}_time_masks'] = {time_shape_key: time_mask_t.unsqueeze(0)}
            kwargs[f'packed_{modality}_time_mask_order'] = {time_shape_key: [0]}

    with torch.inference_mode(), autocast_context(config):
        encoded = model.encode(**kwargs)
    if modality not in encoded:
        raise RuntimeError(f'Model did not return encoded results for modality={modality!r}')
    _, indices = encoded[modality]
    if isinstance(indices, tuple):
        indices = indices[0]
    if indices.dim() != 2:
        raise RuntimeError(f'Expected grouped {modality.upper()} indices shape (L, K), got {tuple(indices.shape)}')
    return indices.detach().cpu().numpy().astype(np.int32, copy=False)


def encode_2d_window_batch(
    model: UniTok,
    signals: List[np.ndarray],
    modality: str,
    config: RuntimeConfig,
    channel_masks: List[np.ndarray],
    time_masks: List[np.ndarray],
) -> np.ndarray:
    """Encode one bounded micro-batch of fixed-shape EEG/MEG windows."""
    if not signals:
        raise ValueError('Cannot encode an empty window batch')
    shape_key = tuple(int(v) for v in signals[0].shape)
    if any(tuple(int(v) for v in signal.shape) != shape_key for signal in signals):
        raise ValueError('All windows in an encode micro-batch must have the same shape')

    tensor = torch.as_tensor(np.stack(signals, axis=0), dtype=torch.float32, device=config.device)
    channel_mask_t = torch.as_tensor(
        np.stack(channel_masks, axis=0), dtype=torch.float32, device=config.device,
    )
    time_mask_t = torch.as_tensor(
        np.stack(time_masks, axis=0), dtype=torch.float32, device=config.device,
    )
    order = list(range(len(signals)))
    kwargs = {
        f'packed_{modality}_tensors': {shape_key: tensor},
        f'packed_{modality}_order': {shape_key: order},
        f'packed_{modality}_channel_masks': {(shape_key[0],): channel_mask_t},
        f'packed_{modality}_channel_mask_order': {(shape_key[0],): order},
        f'packed_{modality}_time_masks': {(shape_key[1],): time_mask_t},
        f'packed_{modality}_time_mask_order': {(shape_key[1],): order},
        'auto_chunk_encode': False,
    }
    with torch.inference_mode(), autocast_context(config):
        encoded = model.encode(**kwargs)
    if modality not in encoded:
        raise RuntimeError(f'Model did not return encoded results for modality={modality!r}')
    _, indices = encoded[modality]
    if isinstance(indices, tuple):
        indices = indices[0]
    if indices.dim() != 2:
        raise RuntimeError(
            f'Expected grouped {modality.upper()} indices shape (L, K), got {tuple(indices.shape)}'
        )
    return indices.detach().cpu().numpy().astype(np.int32, copy=False)


def preprocess_2d_fixed_shape(
    signal: np.ndarray,
    neural_transform,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Deterministic EEG/MEG fixed-shape padding — no randomness whatsoever.

    This reproduces the ``NeuralTransform`` output for the *no-resample /
    no-crop / no-permutation* case that applies to our tar packages. Rest chunks
    are passed in ``time_size`` windows; event chunks may require several windows.
    We keep every channel and time sample in its original order and only zero-pad
    to the
    model's fixed shape using the exact alignment-mask convention of
    ``NeuralTransform`` (real ``1.0`` / stride-alignment ``padding_loss_weight``
    / pure-pad ``0.0``).

    Because the UniTok encoder counts valid patch rows/cols as
    ``ceil(real_count / stride)`` (an all-zero pad region is excluded, an
    alignment region is rounded up to the same patch boundary), this produces
    token ids *identical* to the notebook scheme for already-aligned data, while
    keeping "session = chunks concatenated in time order" perfectly intact and
    removing all randomness (random crop / channel permutation / sr resampling)
    from the final tokenization step.

    Returns ``(processed, channel_valid_mask, time_valid_mask, info)``.
    """
    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim != 2:
        raise ValueError(f'2D neural signal must be (C, T), got {sig.shape}')
    C, T = int(sig.shape[0]), int(sig.shape[1])

    cell_stride = int(neural_transform.cell_stride)
    time_stride = int(neural_transform.time_stride)
    # cell_num / time_size are the max candidates (== the model's fixed shape),
    # mirroring NeuralTransform.__call__'s stride alignment of cell_target.
    cell_target = (int(neural_transform.cell_num) // cell_stride) * cell_stride
    time_size = int(neural_transform.time_size)
    pad_weight = float(getattr(neural_transform, 'padding_loss_weight', 0.1))

    # Channel axis: keep all real channels in original order; pad up to the model
    # target. If a sample somehow has more channels than the target, grow to the
    # next stride multiple instead of randomly subsampling (auto_chunk_encode
    # then tokenizes all channels — still deterministic, no data dropped).
    cell_aligned = int(math.ceil(C / cell_stride) * cell_stride) if C > 0 else 0
    target_C = max(cell_target, cell_aligned)
    channel_valid_mask = np.zeros(target_C, dtype=np.float32)
    channel_valid_mask[:C] = 1.0
    if cell_aligned > C:
        channel_valid_mask[C:cell_aligned] = pad_weight

    # Time axis: keep all real samples in original order (no crop); pad to the
    # fixed window. A chunk longer than the window (should not happen) grows to
    # the next stride multiple rather than being cropped.
    time_aligned = int(math.ceil(T / time_stride) * time_stride) if T > 0 else 0
    target_T = max(time_size, time_aligned)
    time_valid_mask = np.zeros(target_T, dtype=np.float32)
    time_valid_mask[:T] = 1.0
    if time_aligned > T:
        time_valid_mask[T:time_aligned] = pad_weight

    processed = np.zeros((target_C, target_T), dtype=np.float32)
    processed[:C, :T] = sig

    info = {
        'cell_target': int(target_C),
        'cell_real': int(C),
        'cell_padded': int(target_C - C),
        'cell_stride': int(cell_stride),
        'time_stride': int(time_stride),
        'time_real': int(T),
        'time_size_chosen': int(target_T),
        'sr_original': None,
        'sr_target': None,
    }
    return processed, channel_valid_mask, time_valid_mask, info


def time_window_starts(total_len: int, unit: int) -> List[int]:
    """Non-overlapping window start indices covering ``total_len`` in ``unit`` steps.

    Always returns at least ``[0]`` (so an at-or-under-unit sample yields a single
    window). The last window may be shorter than ``unit`` (and is padded to the
    model's fixed length downstream).
    """
    unit = int(unit)
    total_len = int(total_len)
    if unit <= 0:
        raise ValueError(f'time unit must be positive, got {unit}')
    if total_len <= 0:
        return [0]
    return list(range(0, total_len, unit))


def tokenize_neural_sample(
    model: UniTok,
    neural_transform,
    raw_signal: np.ndarray,
    modality: str,
    config: RuntimeConfig,
    channels: Optional[List[dict]] = None,
    timestamps: Optional[np.ndarray] = None,
    fmri_info: Optional[dict] = None,
    sample_key: str = '',
    chunking_strategy: str = 'rest',
) -> Tuple[np.ndarray, dict]:
    """Deterministic, notebook-compatible tokenization of a single neural sample.

    This single function encapsulates the whole tokenization step for one sample:

    1. Offline-repack-equivalent cleaning, reusing the reference functions, then
       rounded through the repack storage precision (``config.storage_dtype``,
       fp16 by default) so the encoder sees the same values as the eval pipeline:
       * EEG/MEG: ``clean_2d_signal`` — ``detect_static_bad_channels``
         (bad-channel removal) + ``compute_zscore_std_floor`` per-channel z-score
         (``config.quality_cfg`` from ``tok_human.yaml``). Low-information samples
         are intentionally neither dropped nor marked.
       * fMRI: ``_normalise_fmri_volume`` (``config.zscore_mode``) with the
         session ``fmri_info.npz`` mean/std volumes; it already applies the
         repack background-fill + clip(-10, 10).
    2. Time-axis windowing. The model tokenizes fixed-length time units
       (``neural_transform.time_size`` — 640 for EEG/MEG, 2 for fMRI). If a sample
       is longer than one unit, it is **split along time into consecutive
       unit-length segments** (a warning is logged), each segment is tokenized
       independently, and the per-segment token ids are concatenated in time
       order. This keeps "session = data concatenated in time order" intact and
       guarantees no time samples / volumes are dropped. A sample at or under one
       unit yields a single segment (original behaviour).
    3. Deterministic fixed-shape preparation per segment (no randomness):
       * EEG/MEG: ``preprocess_2d_fixed_shape`` keeps channels and time samples in
         their original order and only zero-pads to the model's fixed shape with
         the notebook's alignment-padding masks. No random crop / permutation /
         resampling.
       * fMRI: the real ``FmriVolumeTransform`` forced deterministic — center
         spatial crop, ``nearest`` candidate selection, no augmentation, temporal
         window pinned to the segment start (``time_start_index=0``).
    4. Quantize each segment with UniTok and concatenate the per-token codebook ids.

    Returns ``(token_ids, preprocessing_info)`` where ``token_ids`` has shape
    ``(num_tokens, num_codebooks)``.
    """
    if modality in {'eeg', 'meg'}:
        signal, clean_info = clean_2d_signal(
            raw_signal, channels or [], timestamps, modality, config.quality_cfg,
        )
        signal = apply_storage_dtype(signal, config.storage_dtype)

        unit = int(neural_transform.time_size)
        total_time = int(signal.shape[1])
        chunking_strategy = str(chunking_strategy).strip().lower()
        if chunking_strategy not in {'rest', 'event'}:
            raise ValueError(f'Unsupported chunking strategy: {chunking_strategy!r}')
        starts = time_window_starts(total_time, unit)
        if len(starts) > 1:
            logger.warning(
                '%s sample %s: time length %d > unit %d; splitting into %d model '
                'windows, encoding at most %d per micro-batch, and restoring token order.',
                modality.upper(), sample_key or '?', total_time, unit, len(starts),
                config.max_chunk_batch_size,
            )

        prepared_windows = []
        for start in starts:
            seg = signal[:, start:start + unit]
            processed, channel_mask, time_mask, info = preprocess_2d_fixed_shape(seg, neural_transform)
            cs, ts_ = int(info['cell_stride']), int(info['time_stride'])
            n_cell_rows = int(math.ceil(info['cell_real'] / cs)) if info['cell_real'] > 0 else 0
            n_time_cols = int(math.ceil(info['time_real'] / ts_)) if info['time_real'] > 0 else 0
            prepared_windows.append({
                'start': int(start),
                'processed': processed,
                'channel_mask': channel_mask,
                'time_mask': time_mask,
                'info': info,
                'n_cell_rows': n_cell_rows,
                'n_time_cols': n_time_cols,
                'expected_tokens': n_cell_rows * n_time_cols,
            })

        token_ids_list: List[np.ndarray] = []
        for batch_start in range(0, len(prepared_windows), config.max_chunk_batch_size):
            batch = prepared_windows[batch_start:batch_start + config.max_chunk_batch_size]
            batch_ids = encode_2d_window_batch(
                model,
                [window['processed'] for window in batch],
                modality,
                config,
                [window['channel_mask'] for window in batch],
                [window['time_mask'] for window in batch],
            )
            offset = 0
            for window in batch:
                next_offset = offset + int(window['expected_tokens'])
                window_ids = batch_ids[offset:next_offset]
                if int(window_ids.shape[0]) != int(window['expected_tokens']):
                    raise RuntimeError(
                        f"{modality.upper()} sample {sample_key or '?'} window@{window['start']}: "
                        f"expected {window['expected_tokens']} tokens, got {window_ids.shape[0]}"
                    )
                token_ids_list.append(window_ids)
                offset = next_offset
            if offset != int(batch_ids.shape[0]):
                raise RuntimeError(
                    f'{modality.upper()} encode micro-batch returned {batch_ids.shape[0]} tokens; '
                    f'only {offset} were assigned to windows'
                )

        first_info = prepared_windows[0]['info']
        windows: List[dict] = []
        if chunking_strategy == 'event':
            row_grids = [
                token_ids.reshape(window['n_cell_rows'], window['n_time_cols'], token_ids.shape[-1])
                for window, token_ids in zip(prepared_windows, token_ids_list)
            ]
            token_grid = np.concatenate(row_grids, axis=1)
            token_ids = token_grid.reshape(-1, token_grid.shape[-1])
            windows.append({
                'local_token_start': 0,
                'local_token_end': int(token_ids.shape[0]),
                'grid_shape': [int(token_grid.shape[0]), int(token_grid.shape[1])],
                'time_start_in_sample': 0,
                'time_len': int(total_time),
            })
        else:
            local_off = 0
            for window, token_ids_for_window in zip(prepared_windows, token_ids_list):
                n_tok = int(token_ids_for_window.shape[0])
                windows.append({
                    'local_token_start': int(local_off),
                    'local_token_end': int(local_off + n_tok),
                    'grid_shape': [int(window['n_cell_rows']), int(window['n_time_cols'])],
                    'time_start_in_sample': int(window['start']),
                    'time_len': int(window['info']['time_real']),
                })
                local_off += n_tok
            token_ids = token_ids_list[0] if len(token_ids_list) == 1 else np.concatenate(token_ids_list, axis=0)

        preprocessing_info = {
            'raw_shape': [int(v) for v in np.asarray(raw_signal).shape],
            'chunking_strategy': chunking_strategy,
            'num_time_windows': len(starts),
            'time_unit': int(unit),
            'time_total': int(total_time),
            'max_chunk_batch_size': int(config.max_chunk_batch_size),
            'windows': windows,
            'cell_stride': int(first_info['cell_stride']),
            'time_stride': int(first_info['time_stride']),
            'cell_target': int(first_info['cell_target']),
            'channel_count_raw': clean_info['channel_count_raw'],
            'channel_count_kept': clean_info['channel_count_kept'],
            'bad_channel_count': clean_info['bad_channel_count'],
            'bad_channel_source': clean_info['bad_channel_source'],
            'good_channel_indices': clean_info['good_channel_indices'],
            'storage_dtype': config.storage_dtype,
            'deterministic': True,
        }
        return token_ids, preprocessing_info

    if modality == 'fmri':
        volume_txyz = validate_fmri(raw_signal)
        volume_xyzt = np.transpose(volume_txyz, (1, 2, 3, 0)).astype(np.float32, copy=False)
        # Reference global z-score + background-fill + clip(-10, 10) (zscore_mode
        # from tok_human.yaml) before the transform, matching hdata/fmri.py. The
        # repack stage stores the result as fp16, so round-trip through the same
        # precision the eval model actually tokenized. Done once on the full
        # volume so the background-fill is computed over all timepoints.
        volume_xyzt = normalize_fmri_volume(volume_xyzt, fmri_info, sample_key, config.zscore_mode)
        volume_xyzt = apply_storage_dtype(volume_xyzt, config.storage_dtype)

        total_time = int(volume_xyzt.shape[3])
        if timestamps is not None:
            ts_full = np.asarray(timestamps, dtype=np.float32).reshape(-1)
        else:
            sr_candidates = getattr(neural_transform, 'sampling_rate_candidates', None) or [1.0]
            sr = float(sr_candidates[0])
            ts_full = (np.arange(total_time, dtype=np.float64) / max(sr, 1e-6)).astype(np.float32)

        unit = int(neural_transform.time_size)
        starts = time_window_starts(total_time, unit)
        if len(starts) > 1:
            logger.warning(
                'fMRI sample %s: %d volumes > unit %d; splitting into %d temporal '
                'windows and concatenating tokens in time order.',
                sample_key or '?', total_time, unit, len(starts),
            )

        token_ids_list = []
        windows: List[dict] = []
        first_info = None
        local_off = 0
        for start in starts:
            vol_seg = volume_xyzt[..., start:start + unit]
            ts_seg = ts_full[start:start + unit]
            out_tensor, _ts_out, info = neural_transform(
                vol_seg,
                timestamps=ts_seg,
                brain_mask=None,
                return_info=True,
                # Pin temporal window to the segment start (no random crop); the
                # transform was also forced to center spatial crop + nearest
                # selection + no augment in build_modality_neural_transform.
                time_start=float(ts_seg[0]) if ts_seg.size > 0 else None,
                time_start_index=0,
            )
            processed = out_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            tid = encode_one_neural_sample(model, processed, modality, config)
            n_tok = int(tid.shape[0])
            wgrid = info.get('token_grid', {}) if isinstance(info.get('token_grid'), dict) else {}
            grid_shape = [int(v) for v in wgrid.get('grid_shape', [])]
            token_ids_list.append(tid)
            windows.append({
                'local_token_start': int(local_off),
                'local_token_end': int(local_off + n_tok),
                'grid_shape': grid_shape,
                'time_start_in_sample': int(start),
                'time_len': int(vol_seg.shape[3]),
            })
            local_off += n_tok
            if first_info is None:
                first_info = info

        token_ids = token_ids_list[0] if len(token_ids_list) == 1 else np.concatenate(token_ids_list, axis=0)

        preprocessing_info = {
            'raw_shape': [int(v) for v in np.asarray(raw_signal).shape],
            'num_time_windows': len(starts),
            'time_unit': int(unit),
            'time_total': int(total_time),
            'windows': windows,
            'fmri_patch_size': [int(v) for v in getattr(neural_transform, 'patch_size', ())],
            'fmri_token_count_total': int(token_ids.shape[0]),
            'sr_target': first_info.get('sr_target'),
            'storage_dtype': config.storage_dtype,
        }
        return token_ids, preprocessing_info

    raise ValueError(f'Unsupported modality: {modality}')


def export_codebook(model: UniTok, tokenizer_dir: Path, config: RuntimeConfig) -> Tuple[Path, Path]:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    raw_weights = []
    normalized_weights = []
    for codebook in model.quantizer.codebooks:
        weight = codebook.codebook.weight.detach().cpu().float().numpy()
        raw_weights.append(weight)
        denom = np.maximum(np.linalg.norm(weight, axis=-1, keepdims=True), 1e-12)
        normalized_weights.append(weight / denom)
    raw_weights = np.stack(raw_weights, axis=0)
    normalized_weights = np.stack(normalized_weights, axis=0)

    codebook_path = tokenizer_dir / 'codebook.npz'
    np.savez_compressed(
        codebook_path,
        codebook_raw=raw_weights.astype(np.float32),
        codebook_normalized=normalized_weights.astype(np.float32),
    )
    metadata = {
        'format': 'UniTok VectorQuantizerM codebooks',
        'num_codebooks': int(model.config.num_codebooks),
        'vocab_size': int(model.config.vocab_size),
        'vocab_size_per_codebook': int(model.config.vocab_size // model.config.num_codebooks),
        'vocab_width': int(model.config.vocab_width),
        'width_per_codebook': int(model.config.vocab_width // model.config.num_codebooks),
        'token_id_layout': '<modality>_token_ids.npy stores local per-codebook ids with shape (total_tokens, num_codebooks)',
        'embedding_lookup': 'embedding = concatenate(codebook_normalized[k, token_ids[:, k]] for k in range(num_codebooks))',
        'checkpoint': '/'.join(Path(config.checkpoint_path).parts[-2:]),
        'modality': config.modality,
        'tokenization_scheme': 'tok_real_data_evaluation.ipynb (NeuralTransform fixed-shape + UniTok EMA)',
    }
    metadata_path = tokenizer_dir / 'codebook_metadata.json'
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return codebook_path, metadata_path


def build_spatial_layout(modality: str, layout_info: Optional[dict]) -> dict:
    """User-facing spatial/token-grid layout for interpreting flattened tokens."""
    layout_info = layout_info or {}
    if modality in {'eeg', 'meg'}:
        cell_stride = int(layout_info.get('cell_stride', CELL_STRIDE))
        return {
            'type': 'channels',
            'channels_file': f'{modality}_channels.json',
            'channel_count': int(layout_info.get('channel_count_kept', 0)),
            'channel_stride': cell_stride,
            'flatten_order': 'cell_row_major',
            'token_grid': {
                'grid_shape_meaning': ['channel_group_index', 'time_token_index'],
                'local_token_index': 'For a segment with grid_shape=[n_channel_groups, n_time_tokens], local index j maps to channel_group_index=j//n_time_tokens and time_token_index=j%n_time_tokens.',
            },
            'channel_mapping': {
                'description': f'Channel groups are consecutive groups of kept channels in {modality}_channels.json order; the final group may be zero-padded.',
                'channels_per_group': cell_stride,
            },
        }
    if modality == 'fnirs':
        # TODO: fNIRS spatial layout — placeholder, currently mirrors channel-based layout.
        # fNIRS has source-detector pairs and may need distinct channel grouping logic.
        cell_stride = int(layout_info.get('cell_stride', CELL_STRIDE))
        return {
            'type': 'channels',
            'channels_file': f'{modality}_channels.json',
            'channel_count': int(layout_info.get('channel_count_kept', 0)),
            'channel_stride': cell_stride,
            'flatten_order': 'cell_row_major',
            'token_grid': {
                'grid_shape_meaning': ['channel_group_index', 'time_token_index'],
                'local_token_index': 'For a segment with grid_shape=[n_channel_groups, n_time_tokens], local index j maps to channel_group_index=j//n_time_tokens and time_token_index=j%n_time_tokens.',
            },
            'channel_mapping': {
                'description': f'Channel groups for {modality} source-detector pairs. The final group may be zero-padded.',
                'channels_per_group': cell_stride,
            },
        }
    if modality == 'fmri':
        return {
            'type': 'volume_patches',
            'patch_size': layout_info.get('fmri_patch_size', []),
            'grid_axes': ['t', 'x', 'y', 'z'],
            'flatten_order': 't_x_y_z_major',
            'token_grid': {
                'grid_shape_meaning': ['time_patch_index', 'x_patch_index', 'y_patch_index', 'z_patch_index'],
                'local_token_index': 'For a segment with grid_shape=[n_t, n_x, n_y, n_z], local index j maps to row-major (t, x, y, z), with z as the fastest axis.',
            },
        }
    return {'type': modality}


def build_temporal_index(modality: str, layout_info: Optional[dict], session_sfreq: Optional[float], segments: List[dict]) -> dict:
    """User-facing token-range to source-time index.

    This is optional for plain sequence modeling, but lets users slice tokens by
    chunk/window or align tokens with events without reading the source tar.
    """
    layout_info = layout_info or {}
    if modality in {'eeg', 'meg'}:
        return {
            'type': 'continuous_time',
            'description': 'Compact mapping from token ranges to source time windows. Users doing only session-level sequence modeling can ignore this section.',
            'time_unit': 'samples',
            'sampling_rate_hz': session_sfreq,
            'time_stride_samples': int(layout_info.get('time_stride', TIME_STRIDE)),
            'num_segments': len(segments),
            'segments': segments,
        }
    if modality == 'fnirs':
        # TODO: fNIRS temporal index — placeholder, currently mirrors continuous_time.
        # fNIRS may have distinct sampling rates per wavelength; adjust as needed.
        return {
            'type': 'continuous_time',
            'description': 'Compact mapping from token ranges to source time windows.',
            'time_unit': 'samples',
            'sampling_rate_hz': session_sfreq,
            'time_stride_samples': int(layout_info.get('time_stride', TIME_STRIDE)),
            'num_segments': len(segments),
            'segments': segments,
        }
    if modality == 'fmri':
        return {
            'type': 'volume_time',
            'description': 'Compact mapping from token ranges to source fMRI volume windows. Users doing only session-level sequence modeling can ignore this section.',
            'time_unit': 'volumes',
            'tr_seconds': session_sfreq,
            'num_segments': len(segments),
            'segments': segments,
        }
    return {
        'type': modality,
        'description': 'Compact mapping from token ranges to source time windows.',
        'num_segments': len(segments),
        'segments': segments,
    }


def build_user_segment(
    *,
    segment_id: int,
    modality: str,
    sample_key: str,
    offset: int,
    window: dict,
    chunk_index,
    chunk_start_sample,
    session_sfreq: Optional[float],
) -> dict:
    local_start = int(window['local_token_start'])
    local_end = int(window['local_token_end'])
    time_len = int(window['time_len'])
    time_start_in_sample = int(window.get('time_start_in_sample', 0))
    if chunk_start_sample is not None:
        time_start = int(chunk_start_sample) + time_start_in_sample
    else:
        time_start = time_start_in_sample
    time_end = time_start + time_len
    time_unit = 'volumes' if modality == 'fmri' else 'samples'

    if modality == 'fmri':
        token_grid_axes = ['time_patch', 'x_patch', 'y_patch', 'z_patch']
    else:
        token_grid_axes = ['channel_group', 'time_token']

    segment = {
        'segment_id': int(segment_id),
        'token': {
            'start_index': int(offset + local_start),
            'end_index': int(offset + local_end),
            'count': int(local_end - local_start),
            'grid_shape': window.get('grid_shape', []),
            'grid_axes': token_grid_axes,
        },
        'source_time': {
            'start': int(time_start),
            'end': int(time_end),
            'unit': time_unit,
        },
    }
    if chunk_index is not None:
        segment['source_chunk_index'] = int(chunk_index)
    if session_sfreq:
        if modality == 'fmri':
            segment['source_time']['start_seconds'] = float(time_start * session_sfreq)
            segment['source_time']['end_seconds'] = float(time_end * session_sfreq)
        else:
            segment['source_time']['start_seconds'] = float(time_start / session_sfreq)
            segment['source_time']['end_seconds'] = float(time_end / session_sfreq)
    return segment


def copy_sidecars(session_dir: Path, output_session_dir: Path, modality: str, skip_names: Optional[set] = None) -> None:
    skip_names = skip_names or set()
    for name in modality_sidecar_names(modality):
        if name in skip_names:
            continue
        src = session_dir / name
        if src.exists():
            shutil.copy2(src, output_session_dir / name)


def write_filtered_channels_json(
    channels: List[dict],
    good_indices: List[int],
    output_session_dir: Path,
    modality: str,
) -> None:
    """Write ``{modality}_channels.json`` filtered to the kept (good) channels.

    Mirrors the reference repack contract (``human_offline_filter.py``: the clean
    tar's ``*_channels.pkl`` is sliced to ``good_indices``) so the sidecar rows
    stay aligned 1:1 with the tokenized data rows. Unlike the reference we keep
    every column (incl. ``mean``/``std``/``is_bad``) as provenance, since the
    output stores token ids — not raw data — so there is no double-normalization
    risk; ``is_bad`` is all-false after filtering.
    """
    filtered = [channels[i] for i in good_indices]
    path = Path(output_session_dir) / f'{modality}_channels.json'
    path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding='utf-8')


def safe_path_name(value: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in value)
    return safe.strip('._') or 'unknown'


def to_relative_str(path: Path, base: Path) -> str:
    path = Path(path)
    base = Path(base)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


def short_checkpoint_id(path: Path) -> str:
    path = Path(path)
    parts = path.parts[-2:] if len(path.parts) >= 2 else path.parts
    return "/".join(parts)


def build_tokenizer_info(config: RuntimeConfig, output_base: Path, codebook_path: Optional[Path] = None, codebook_metadata_path: Optional[Path] = None) -> dict:
    info = {
        'checkpoint': short_checkpoint_id(config.checkpoint_path),
        'modality': config.modality,
        'use_ema': bool(config.use_ema),
        'train_config': config.train_config_path.name,
        'tokenization_scheme': 'tok_real_data_evaluation.ipynb',
    }
    if codebook_path is not None:
        info['codebook'] = to_relative_str(codebook_path, output_base)
    if codebook_metadata_path is not None:
        info['codebook_metadata'] = to_relative_str(codebook_metadata_path, output_base)
    return info


def infer_subject_name_from_path(session_name: str) -> str:
    parts = session_name.split('_')
    for part in parts:
        if part.startswith('sub-') and len(part) > len('sub-'):
            return safe_path_name(part)
    for part in parts:
        if part.startswith('sub') and len(part) > len('sub'):
            return safe_path_name(part)
    return 'sub-unknown'


def subject_id_from_metadata(metadata: dict) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    for container_key in ('Subject', 'subject', 'Participant', 'participant'):
        value = metadata.get(container_key)
        if isinstance(value, dict):
            for id_key in ('ID', 'id', 'Id', 'subject_id', 'participant_id'):
                subject_id = value.get(id_key)
                if subject_id:
                    return safe_path_name(str(subject_id))
        elif value:
            return safe_path_name(str(value))
    for key in ('subject_id', 'SubjectID', 'subjectID', 'participant_id', 'ParticipantID', 'participantID'):
        subject_id = metadata.get(key)
        if subject_id:
            return safe_path_name(str(subject_id))
    return None


def infer_subject_name_from_tar_metadata(session_dir: Path, modality: str) -> Optional[str]:
    tar_path = Path(session_dir) / 'data.tar'
    sample_refs = list_modality_samples_in_tar(tar_path, modality)
    subject_ids = set()
    with tarfile.open(tar_path, 'r') as tar:
        for _, _, _, meta_member in sample_refs:
            if meta_member is None:
                continue
            metadata = load_json_from_tar_member(tar, meta_member)
            subject_id = subject_id_from_metadata(metadata)
            if subject_id:
                subject_ids.add(subject_id)
    if len(subject_ids) > 1:
        raise ValueError(f'Inconsistent subject IDs in {tar_path}: {sorted(subject_ids)}')
    return next(iter(subject_ids)) if subject_ids else None


def session_output_name(source_session_dir: Path, modality: str) -> str:
    original_name = safe_path_name(Path(source_session_dir).name)
    subject_name = infer_subject_name_from_tar_metadata(source_session_dir, modality) or infer_subject_name_from_path(original_name)
    prefix = f'{subject_name}_'
    if original_name.startswith(prefix):
        return original_name
    return f'{subject_name}_{original_name}'


def ensure_unique_session_name(base_name: str, used_names: set[str]) -> str:
    candidate = base_name
    suffix = 2
    while candidate in used_names:
        candidate = f'{base_name}_{suffix:02d}'
        suffix += 1
    used_names.add(candidate)
    return candidate


def convert_session_tar_to_token_ids(
    session_dir: Path,
    output_session_dir: Path,
    model: UniTok,
    config: RuntimeConfig,
    neural_transform,
    max_samples: Optional[int] = None,
    input_base: Optional[Path] = None,
    output_base: Optional[Path] = None,
    codebook_path: Optional[Path] = None,
    codebook_metadata_path: Optional[Path] = None,
) -> dict:
    modality = config.modality
    session_dir = Path(session_dir)
    output_session_dir = Path(output_session_dir)
    input_base = Path(input_base) if input_base is not None else session_dir.parent
    output_base = Path(output_base) if output_base is not None else output_session_dir.parent
    tmp_token_chunk_dir = output_session_dir / '.tmp_token_chunks'
    output_session_dir.mkdir(parents=True, exist_ok=True)
    tmp_token_chunk_dir.mkdir(parents=True, exist_ok=True)

    channels = load_channels_json(session_dir, modality)
    fmri_info = load_fmri_info(session_dir) if modality == 'fmri' else None
    tar_path = session_dir / 'data.tar'
    sample_refs = list_modality_samples_in_tar(tar_path, modality)
    if max_samples is not None:
        sample_refs = sample_refs[:int(max_samples)]

    segments: List[dict] = []
    sample_token_paths = []
    offset = 0
    num_codebooks = None
    # Session-level kept-channel set (EEG/MEG). With the is_bad path this is
    # identical across all chunks, so a single filtered channels.json stays
    # aligned with every chunk's token rows.
    session_good_indices: Optional[List[int]] = None
    # Session-constant token layout (captured once from the first sample), so the
    # per-segment index stays compact.
    layout_info: Optional[dict] = None
    session_sfreq: Optional[float] = None

    with tarfile.open(tar_path, 'r') as tar:
        for key, signal_member, ts_member, meta_member in tqdm(sample_refs, desc=f'Tokenizing {session_dir.name} [{modality}]'):
            raw_signal = load_npy_from_tar_member(tar, signal_member)
            metadata = load_json_from_tar_member(tar, meta_member) if meta_member is not None else {}
            timestamps = load_npy_from_tar_member(tar, ts_member) if ts_member is not None else None

            token_ids, preprocessing_info = tokenize_neural_sample(
                model,
                neural_transform,
                raw_signal,
                modality,
                config,
                channels=channels,
                timestamps=timestamps,
                fmri_info=fmri_info,
                sample_key=key,
                chunking_strategy=str(metadata.get('chunking_strategy', 'rest')).strip().lower(),
            )
            if num_codebooks is None:
                num_codebooks = int(token_ids.shape[1])
            elif int(token_ids.shape[1]) != num_codebooks:
                raise RuntimeError(f'Inconsistent num_codebooks: got {token_ids.shape[1]}, expected {num_codebooks}')

            chunk_good_indices = preprocessing_info.get('good_channel_indices')
            if chunk_good_indices is not None:
                if session_good_indices is None:
                    session_good_indices = chunk_good_indices
                elif chunk_good_indices != session_good_indices:
                    logger.warning(
                        'Session %s [%s]: chunk %s kept channels differ from earlier chunks '
                        '(%d vs %d kept). The session channels.json is filtered to the first '
                        'chunk\'s set and may not align with this chunk; this only happens via the '
                        'variance fallback when is_bad is absent.',
                        session_dir.name, modality, key,
                        len(chunk_good_indices), len(session_good_indices),
                    )

            sample_token_path = tmp_token_chunk_dir / f'{safe_sample_filename(key)}.token_ids.npy'
            np.save(sample_token_path, token_ids)
            sample_token_paths.append(sample_token_path)

            if layout_info is None:
                layout_info = preprocessing_info
            if session_sfreq is None and metadata.get('sfreq') is not None:
                session_sfreq = float(metadata.get('sfreq'))

            # Expand each time window into a compact, user-facing temporal index.
            chunk_start_sample = metadata.get('chunk_start_sample')
            chunk_index = metadata.get('chunk_index')
            for w in preprocessing_info.get('windows', []):
                segment = build_user_segment(
                    segment_id=len(segments),
                    modality=modality,
                    sample_key=key,
                    offset=offset,
                    window=w,
                    chunk_index=chunk_index,
                    chunk_start_sample=chunk_start_sample,
                    session_sfreq=session_sfreq,
                )
                segment['chunking_strategy'] = str(metadata.get('chunking_strategy', 'rest'))
                if metadata.get('event_index') is not None:
                    segment['source_event'] = {
                        'index': int(metadata['event_index']),
                        'type': str(metadata.get('event_type', '')),
                        'value': str(metadata.get('event_value', '')),
                        'start_time': float(metadata.get('event_start_time', 0.0)),
                        'end_time': float(metadata.get('event_end_time', 0.0)),
                    }
                segments.append(segment)

            offset += int(token_ids.shape[0])
            del token_ids, raw_signal

    if not sample_token_paths:
        raise RuntimeError(f'No {modality.upper()} samples were tokenized in {session_dir}')

    token_path = output_session_dir / f'{modality}_token_ids.npy'
    token_ids_concat = np.lib.format.open_memmap(token_path, mode='w+', dtype=np.int32, shape=(int(offset), int(num_codebooks)))
    write_offset = 0
    for sample_token_path in sample_token_paths:
        token_chunk_ids = np.load(sample_token_path)
        next_offset = write_offset + int(token_chunk_ids.shape[0])
        token_ids_concat[write_offset:next_offset] = token_chunk_ids
        write_offset = next_offset
    token_ids_concat.flush()
    token_ids_shape = [int(v) for v in token_ids_concat.shape]
    token_ids_dtype = str(token_ids_concat.dtype)
    del token_ids_concat

    for sample_token_path in sample_token_paths:
        sample_token_path.unlink(missing_ok=True)
    tmp_token_chunk_dir.rmdir()

    token_metadata = {
        'format': 'neurotoken_session',
        'version': '1.0',
        'session': {
            # Filled with the final unique output name in convert_dataset().
            'session_id': None,
            'modality': modality,
        },
        'tokens': {
            'file': token_path.name,
            'path': to_relative_str(token_path, output_base),
            'shape': token_ids_shape,
            'dtype': token_ids_dtype,
            'num_codebooks': int(num_codebooks),
            'id_space': 'local_per_codebook',
            'array_layout': {
                'axis_0': 'token_index',
                'axis_1': 'codebook_index',
                'row_meaning': 'one neural token',
                'column_meaning': 'local token id in each codebook',
            },
        },
        'tokenizer': build_tokenizer_info(config, output_base, codebook_path=codebook_path, codebook_metadata_path=codebook_metadata_path),
        'temporal_index': build_temporal_index(modality, layout_info, session_sfreq, segments),
        'spatial_layout': build_spatial_layout(modality, layout_info),
        'provenance': {
            'source_session_name': session_dir.name,
            'output_session_dir': to_relative_str(output_session_dir, output_base),
            'input_base': Path(input_base).name,
            'tokenization_scheme': 'tok_real_data_evaluation.ipynb',
            'checkpoint': short_checkpoint_id(config.checkpoint_path),
            'train_config': config.train_config_path.name,
            'use_ema': bool(config.use_ema),
            'storage_dtype': config.storage_dtype,
            'deterministic': True,
            'preprocessing': {
                'bad_channel_removal': bool(modality in {'eeg', 'meg'}),
                'zscore': True,
                'channels_file_filtered_to_kept_channels': bool(modality in {'eeg', 'meg'} and session_good_indices is not None),
            },
        },
    }
    metadata_path = output_session_dir / 'token_metadata.json'
    metadata_path.write_text(json.dumps(token_metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    # For EEG/MEG, write a channels.json filtered to the kept channels (aligned
    # with the token rows) instead of copying the raw sidecar verbatim.
    if modality in {'eeg', 'meg'} and session_good_indices is not None:
        copy_sidecars(session_dir, output_session_dir, modality, skip_names={f'{modality}_channels.json'})
        write_filtered_channels_json(channels, session_good_indices, output_session_dir, modality)
        n_bad = len(channels) - len(session_good_indices)
        if n_bad > 0:
            logger.info('Session %s [%s]: channels.json filtered, removed %d bad channel(s), kept %d.',
                        session_dir.name, modality, n_bad, len(session_good_indices))
    else:
        copy_sidecars(session_dir, output_session_dir, modality)

    logger.info('Saved token ids: %s shape=%s dtype=%s', token_path, token_ids_shape, token_ids_dtype)
    return token_metadata


def load_session_metadata(session_dir: Path) -> dict:
    path = Path(session_dir) / SESSION_METADATA_FILENAME
    if not path.exists():
        logger.warning('Session %s is missing %s; the manifest summary will omit Subject info.',
                       Path(session_dir).name, SESSION_METADATA_FILENAME)
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def summarize_session(token_metadata: dict, session_metadata: dict, session_id: str, source_session_name: str, output_session_dir: Path) -> dict:
    tokens = token_metadata.get('tokens', {}) if isinstance(token_metadata.get('tokens'), dict) else {}
    provenance = token_metadata.get('provenance', {}) if isinstance(token_metadata.get('provenance'), dict) else {}
    output_session_rel = provenance.get('output_session_dir')
    subject = session_metadata.get('Subject', {})
    return {
        'session_id': session_id,
        'modality': token_metadata.get('session', {}).get('modality'),
        'token_file': tokens.get('path'),
        'token_metadata': f'{output_session_rel}/token_metadata.json' if output_session_rel else 'token_metadata.json',
        'token_shape': tokens.get('shape'),
        'token_dtype': tokens.get('dtype'),
        'num_codebooks': tokens.get('num_codebooks'),
        'dataset': session_metadata.get('Dataset', ''),
        'task_type': session_metadata.get('Task_type', ''),
        'subject': subject,
        'has_temporal_index': bool(token_metadata.get('temporal_index', {}).get('num_segments', 0)),
        'has_events': (Path(output_session_dir) / 'events_info.json').exists(),
        'provenance': {
            'source_session_name': source_session_name,
        },
    }


def normalize_subject_for_index(subject: dict) -> dict:
    return {
        'ID': str(subject.get('ID', '') or ''),
        'Age': str(subject.get('Age', '') or ''),
        'Sex': str(subject.get('Sex', '') or ''),
        'Health_group': str(subject.get('Health_group', '') or ''),
    }


def build_subjects_index(session_summaries: List[dict]) -> List[dict]:
    subjects_by_key: Dict[str, dict] = {}
    for idx, session in enumerate(session_summaries):
        subject = normalize_subject_for_index(session.get('subject', {}) if isinstance(session.get('subject'), dict) else {})
        subject_id = subject.get('ID', '')
        key = subject_id if subject_id else f'__missing_subject_id_{idx:08d}'
        row = subjects_by_key.setdefault(key, {**subject, 'session_ids': []})
        row['session_ids'].append(str(session.get('session_id', '')))
    return list(subjects_by_key.values())


def initialize_output_dataset(output_base: Path, overwrite: bool = False) -> Tuple[Path, Path, Path]:
    output_base = Path(output_base)
    if output_base.exists() and overwrite:
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_base / 'tokenizer'
    stimuli_dir = output_base / 'stimuli'
    sessions_dir = output_base / 'sessions'
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return tokenizer_dir, stimuli_dir, sessions_dir


def convert_dataset(
    input_base: Path,
    output_base: Path,
    model: UniTok,
    config: RuntimeConfig,
    neural_transform,
    max_samples: Optional[int] = None,
    max_sessions: Optional[int] = None,
    overwrite: bool = False,
) -> dict:
    session_dirs = find_session_folders(input_base)
    if max_sessions is not None:
        session_dirs = session_dirs[:int(max_sessions)]

    tokenizer_dir, stimuli_dir, sessions_dir = initialize_output_dataset(output_base, overwrite=overwrite)
    codebook_path, codebook_metadata_path = export_codebook(model, tokenizer_dir, config)
    logger.info('Exported codebook: %s', codebook_path)
    logger.info('Found %d session(s) under %s', len(session_dirs), input_base)

    session_summaries = []
    failed_sessions = []
    failed_session_errors = []
    used_session_names = set()
    for idx, source_session_dir in enumerate(session_dirs, start=1):
        output_session_name = ensure_unique_session_name(session_output_name(source_session_dir, config.modality), used_session_names)
        output_session_dir = sessions_dir / output_session_name
        logger.info('=== Session %d/%d: %s -> %s ===', idx, len(session_dirs), source_session_dir.name, output_session_name)
        try:
            token_metadata = convert_session_tar_to_token_ids(
                source_session_dir,
                output_session_dir,
                model,
                config,
                neural_transform,
                max_samples=max_samples,
                input_base=input_base,
                output_base=output_base,
                codebook_path=codebook_path,
                codebook_metadata_path=codebook_metadata_path,
            )
        except Exception as exc:
            logger.exception('Session %s failed to convert; skipping this session.', source_session_dir.name)
            failed_sessions.append(source_session_dir.name)
            failed_session_errors.append({
                'session': source_session_dir.name,
                'error_type': type(exc).__name__,
                'message': str(exc),
            })
            continue
        token_metadata.setdefault('session', {})['session_id'] = output_session_name
        token_meta_path = output_session_dir / 'token_metadata.json'
        token_meta_path.write_text(json.dumps(token_metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        session_metadata = load_session_metadata(source_session_dir)
        session_summaries.append(
            summarize_session(token_metadata, session_metadata, output_session_name, source_session_dir.name, output_session_dir)
        )

    output_base = Path(output_base)
    manifest = {
        'format': 'neurotoken_dataset',
        'version': '1.0',
        'modality': config.modality,
        'input_base': Path(input_base).name,
        'output_base': output_base.name,
        'readme': 'README.md',
        'tokenizer': {
            'codebook': to_relative_str(codebook_path, output_base),
            'codebook_metadata': to_relative_str(codebook_metadata_path, output_base),
        },
        'stimuli': {
            'path': to_relative_str(stimuli_dir, output_base),
            'status': 'empty_placeholder; user should move/copy real stimuli files here later',
        },
        'sessions_dir': to_relative_str(sessions_dir, output_base),
        'num_sessions': len(session_summaries),
        'split_policy': 'not_provided; users should create task-appropriate splits from the session/subject metadata',
        'sessions': session_summaries,
    }
    if failed_sessions:
        manifest['failed_sessions'] = failed_sessions
        manifest['failed_session_errors'] = failed_session_errors
        logger.warning('%d session(s) failed to convert: %s', len(failed_sessions), failed_sessions)
    manifest_path = output_base / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Saved dataset manifest: %s (%d session(s) succeeded)', manifest_path, len(session_summaries))
    subjects = build_subjects_index(session_summaries)
    subjects_path = output_base / 'subjects.json'
    subjects_path.write_text(json.dumps(subjects, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Saved subject index: %s (%d subject entr%s)', subjects_path, len(subjects), 'y' if len(subjects) == 1 else 'ies')
    logger.info('README not generated yet; run Stage 3 (3_make_readme.py) to fill in dataset name/description/tokenizer version and produce README.md.')
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert tar-packaged neural sessions into NeuroToken token-id dataset format (notebook-identical tokenization).')
    parser.add_argument('--config', type=Path, required=True, help='TOML pipeline config file.')
    parser.add_argument('--input-base', type=Path, default=None, help='Tar-packaged dataset root containing session folders with data.tar. Overrides config paths.tar_root.')
    parser.add_argument('--output-base', type=Path, default=None, help='Output neurotoken_dataset directory. Overrides config paths.token_root.')
    parser.add_argument('--project-root', type=Path, default=None, help='Human UniTok project root. Overrides config tokenizer.project_root.')
    parser.add_argument('--checkpoint-root', type=Path, default=None, help='UniTok checkpoint root; latest step is auto-selected. Overrides config tokenizer.checkpoint_root.')
    parser.add_argument('--checkpoint-path', type=Path, default=None, help='Explicit UniTok checkpoint dir/file. Overrides --checkpoint-root when set.')
    parser.add_argument('--train-config', type=Path, default=None, help='Train config used to resolve model sizes and neural transform. Overrides config tokenizer.train_config.')
    parser.add_argument('--no-ema', action='store_true', help='Load the raw generator instead of the EMA generator (notebook uses EMA).')
    parser.add_argument('--data-seed', type=int, default=None, help='Seed for the NeuralTransform RNG. Overrides config tokenizer.data_seed.')
    parser.add_argument('--storage-dtype', choices=['fp16', 'fp32'], default=None, help='Repack storage precision the cleaned signal is rounded through before encode (fp16 matches repack_human / the eval data).')
    parser.add_argument('--max-chunk-batch-size', type=int, default=None, help='Maximum number of fixed-shape EEG/MEG model windows encoded together. Overrides config tokenizer.max_chunk_batch_size.')
    parser.add_argument('--modality', choices=sorted(SUPPORTED_MODALITIES), default=None, help='Neural modality to tokenize. Overrides config dataset.modality.')
    parser.add_argument('--device', default=None, help='Torch device, e.g. cuda or cpu. Use config runtime.device="auto" for cuda-if-available.')
    parser.add_argument('--use-bfloat16', action='store_true', help='Use CUDA bfloat16 autocast during encode.')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional max samples per session for smoke tests.')
    parser.add_argument('--max-sessions', type=int, default=None, help='Optional max sessions for smoke tests.')
    parser.add_argument('--overwrite', action='store_true', help='Delete output-base before writing.')
    return parser.parse_args()


def require_cfg(config: dict, dotted_key: str):
    value = cfg_get(config, dotted_key, None)
    if value is None or (isinstance(value, str) and value.strip() == ''):
        raise KeyError(f'Missing required config field: {dotted_key}')
    return value


def require_cfg_path(config: dict, dotted_key: str) -> Path:
    value = cfg_path(config, dotted_key, None)
    if value is None:
        raise KeyError(f'Missing required config field: {dotted_key}')
    return value


def main() -> None:
    args = parse_args()
    pipeline_config = load_config(args.config)

    input_base = args.input_base or require_cfg_path(pipeline_config, 'paths.tar_root')
    output_base = args.output_base or require_cfg_path(pipeline_config, 'paths.token_root')
    project_root = args.project_root or require_cfg_path(pipeline_config, 'tokenizer.project_root')
    configure_unitok_project_root(project_root)
    checkpoint_root = args.checkpoint_root or require_cfg_path(pipeline_config, 'tokenizer.checkpoint_root')
    checkpoint_path = args.checkpoint_path or cfg_path(pipeline_config, 'tokenizer.checkpoint_path', None)
    train_config = args.train_config or require_cfg_path(pipeline_config, 'tokenizer.train_config')
    modality = args.modality or str(require_cfg(pipeline_config, 'dataset.modality'))
    data_seed = int(args.data_seed if args.data_seed is not None else require_cfg(pipeline_config, 'tokenizer.data_seed'))
    storage_dtype = str(args.storage_dtype or require_cfg(pipeline_config, 'tokenizer.storage_dtype'))
    max_chunk_batch_size = int(
        args.max_chunk_batch_size
        if args.max_chunk_batch_size is not None
        else cfg_get(pipeline_config, 'tokenizer.max_chunk_batch_size', 8)
    )
    require_cfg(pipeline_config, 'tokenizer.use_ema')
    use_ema = bool(cfg_bool(pipeline_config, 'tokenizer.use_ema', False)) and not args.no_ema

    device_name = args.device or str(require_cfg(pipeline_config, 'runtime.device'))
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    require_cfg(pipeline_config, 'runtime.use_bfloat16')
    use_bfloat16 = bool(args.use_bfloat16 or cfg_bool(pipeline_config, 'runtime.use_bfloat16', False))
    if use_bfloat16 and device.type != 'cuda':
        raise ValueError('--use-bfloat16 requires a CUDA device')

    resolved_checkpoint = resolve_checkpoint_path(checkpoint_path, checkpoint_root)
    config = RuntimeConfig(
        modality=modality,
        device=device,
        use_bfloat16=use_bfloat16,
        checkpoint_path=resolved_checkpoint,
        train_config_path=train_config,
        use_ema=use_ema,
        data_seed=data_seed,
        storage_dtype=storage_dtype,
        max_chunk_batch_size=max_chunk_batch_size,
    )

    # Tokenization is fully deterministic (no random crop/permutation/resample),
    # so seeding is not required for reproducibility; we still seed to pin any
    # incidental RNG use and keep runs byte-identical.
    random.seed(config.data_seed)
    np.random.seed(config.data_seed)
    torch.manual_seed(config.data_seed)

    logger.info('Project root: %s', PROJECT_ROOT)
    logger.info('Torch: %s | CUDA available: %s', torch.__version__, torch.cuda.is_available())
    logger.info('Input base: %s', input_base)
    logger.info('Output base: %s', output_base)
    logger.info('Train config: %s', config.train_config_path)
    logger.info('Resolved checkpoint: %s', resolved_checkpoint)
    logger.info('Modality: %s | Device: %s | bfloat16 autocast: %s | EMA: %s', modality, device, config.use_bfloat16, config.use_ema)

    model, dataset_meta, resolved_sizes, _ckpt_files, _missing, _unexpected = build_unitok_for_inference(
        train_config_path=config.train_config_path,
        checkpoint_path=resolved_checkpoint,
        device=device,
        use_ema=config.use_ema,
    )
    logger.info('Resolved model sizes: %s', resolved_sizes)
    neural_transform = build_modality_neural_transform(dataset_meta, config.modality)
    # Reuse the reference preprocessing config (z-score floor params / fMRI
    # zscore_mode) straight from tok_human.yaml.
    config.quality_cfg, config.zscore_mode = build_modality_quality_cfg(dataset_meta, config.modality)
    logger.info('Quality cfg (z-score): floor_ratio=%s floor_abs=%s reference=%s | fMRI zscore_mode=%s',
                config.quality_cfg.get('zscore_std_floor_ratio'),
                config.quality_cfg.get('zscore_std_floor_abs'),
                config.quality_cfg.get('fallback_reference_stat'),
                config.zscore_mode)
    logger.info('Bad-channel removal: use_is_bad=%s fallback=%s | storage_dtype=%s (low-info samples: ignored)',
                config.quality_cfg.get('use_is_bad'),
                config.quality_cfg.get('enable_bad_channel_fallback'),
                config.storage_dtype)

    manifest = convert_dataset(
        input_base=input_base,
        output_base=output_base,
        model=model,
        config=config,
        neural_transform=neural_transform,
        max_samples=args.max_samples,
        max_sessions=args.max_sessions,
        overwrite=args.overwrite,
    )
    if int(manifest.get('num_sessions', 0)) == 0:
        failed_sessions = manifest.get('failed_sessions', [])
        raise RuntimeError(
            'Tokenization produced zero successful sessions '
            f'({len(failed_sessions)} session(s) failed). See failed_session_errors in manifest.json.'
        )


if __name__ == '__main__':
    main()
