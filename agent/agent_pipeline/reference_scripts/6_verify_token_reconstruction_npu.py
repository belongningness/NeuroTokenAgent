#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6: verify stored tokens by decoding them back to model inputs.

This is the script version of 6_verify_token_reconstruction_new.ipynb, adapted
to the current public NeuroToken dataset format:

- token ids are read from token_metadata["tokens"]["path"], e.g. eeg_token_ids.npy;
- token ranges come from token_metadata["temporal_index"]["segments"];
- source chunks are matched by source_chunk_index against the source data.tar.

The check is intentionally separate from Stage 5. Stage 5 validates release
structure; this stage loads the tokenizer model and checks token/decode behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from neurotoken_config import cfg_bool, cfg_get, cfg_path, load_config
from neurotoken_report import default_report_path, make_report, write_report

import importlib.util


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_PATH = SCRIPT_DIR / "2_neural2token_human_npu.py"


def _detect_best_device() -> str:
    """Return the best available accelerator device type string.

    Tries NPU (Ascend) first since it is the preferred target for this
    deployment, then CUDA, then falls back to CPU.
    """
    try:
        if torch.npu.is_available():
            return 'npu'
    except (AttributeError, RuntimeError):
        pass
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_stage2_module():
    spec = importlib.util.spec_from_file_location("neural2token_stage2", str(STAGE2_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Stage 2 module from {STAGE2_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rel_path(value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return None if path.is_absolute() else path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def autocast_context(device: torch.device, use_bfloat16: bool):
    if use_bfloat16 and device.type in ("cuda", "npu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def pearson_corr(orig: np.ndarray, recon: np.ndarray) -> float:
    a = np.asarray(orig, dtype=np.float64).reshape(-1)
    b = np.asarray(recon, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / denom)


def compute_metrics(orig: np.ndarray, recon: np.ndarray, mod, ssim_3d: bool = False) -> dict:
    err = np.asarray(recon, dtype=np.float32) - np.asarray(orig, dtype=np.float32)
    denom = max(float((np.asarray(orig, dtype=np.float32) ** 2).mean()), 1e-12)
    result = {
        "mse": float((err ** 2).mean()),
        "mae": float(np.abs(err).mean()),
        "relative_l2": float(np.sqrt(float((err ** 2).mean()) / denom)),
        "pearson": pearson_corr(orig, recon),
    }
    unitok_cls = getattr(mod, "UniTok", None)
    if unitok_cls is not None:
        o = torch.as_tensor(orig[None]).float()
        r = torch.as_tensor(recon[None]).float()
        ssim_fn = unitok_cls._ssim_loss_3d if ssim_3d else unitok_cls._ssim_loss_2d
        result["ssim"] = float(1.0 - ssim_fn(r, o).item())
    return result


def expand_compact_2d_tokens_for_decode(
    quantized: torch.Tensor,
    patch_info: dict,
    target_len: int,
    modality: str,
) -> torch.Tensor:
    """Restore compact valid-grid tokens to the packed canvas expected by decode."""
    if int(quantized.shape[1]) == int(target_len):
        return quantized

    full_h, full_w = tuple(int(v) for v in patch_info["full_grid_shape"])
    valid_rows = patch_info.get("valid_patch_rows")
    valid_cols = patch_info.get("valid_time_cols")
    if valid_rows is None:
        raise RuntimeError(
            f"{modality.upper()} compact token length {quantized.shape[1]} does not match "
            f"packed mask length {target_len}, and valid_patch_rows is missing"
        )

    expanded = quantized.new_zeros(quantized.shape[0], target_len, quantized.shape[2])
    for b in range(int(quantized.shape[0])):
        vr = int(valid_rows[b].item())
        vw = int(valid_cols[b].item()) if valid_cols is not None else full_w
        expected = vr * vw
        if expected != int(quantized.shape[1]):
            raise RuntimeError(
                f"{modality.upper()} compact token length {quantized.shape[1]} does not match "
                f"valid grid {vr}x{vw}={expected}"
            )
        expanded[b, :vr * full_w].view(vr, full_w, -1)[:, :vw, :] = quantized[b].view(vr, vw, -1)
    return expanded


@torch.no_grad()
def reconstruct_2d(model, modality: str, device: torch.device, use_bfloat16: bool, processed, ch_mask, tm_mask, stored_tokens):
    encoder = getattr(model, f"{modality}_encoder")
    decoder = getattr(model, f"{modality}_decoder")
    post_quant_proj = getattr(model, f"{modality}_post_quant_proj")
    detail_refiner = getattr(model, f"{modality}_detail_refiner", None)

    t = torch.as_tensor(processed, dtype=torch.float32, device=device).unsqueeze(0)
    cm = torch.as_tensor(ch_mask, dtype=torch.float32, device=device).unsqueeze(0)
    tmk = torch.as_tensor(tm_mask, dtype=torch.float32, device=device).unsqueeze(0)

    with autocast_context(device, use_bfloat16):
        _patch_tokens, patch_info = model._encode_2d_tokens(
            encoder, t, output_layer=-1, channel_masks=cm, time_masks=tmk,
        )
        token_mask = patch_info.get("packed_token_mask")
        dense_token_mask = patch_info.get("dense_token_mask")
        full_grid = tuple(int(v) for v in patch_info["full_grid_shape"])

        length, num_codebooks = int(stored_tokens.shape[0]), int(stored_tokens.shape[1])
        idx = torch.as_tensor(stored_tokens, dtype=torch.long, device=device).t().reshape(1, num_codebooks, length)
        if device.type in ("cuda", "npu"):
            amp_ctx = torch.amp.autocast(device.type, enabled=True, dtype=torch.float32)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            quantized = model.quantizer.idx_to_f(idx)
            quantized = post_quant_proj(quantized)
        if token_mask is not None:
            quantized = expand_compact_2d_tokens_for_decode(
                quantized, patch_info, int(token_mask.shape[1]), modality,
            )
        if token_mask is not None:
            quantized = quantized * token_mask.unsqueeze(-1).to(quantized.dtype)

        dense = model._scatter_2d_quantized_output(encoder, quantized, patch_info)
        if dense_token_mask is not None:
            dense = dense * dense_token_mask.unsqueeze(-1).to(dense.dtype)

        attn = None
        if dense_token_mask is not None and not bool(dense_token_mask.all().item()):
            attn = model._build_sdpa_attn_mask(dense_token_mask, prepend_cls=False)

        rec = decoder(dense, grid_shape=full_grid, attn_mask=attn).squeeze(1)
        if detail_refiner is not None:
            rec = detail_refiner(rec)
        rec = rec * (cm > 0).to(rec.dtype).unsqueeze(-1)
        rec = rec * (tmk > 0).to(rec.dtype).unsqueeze(1)
    return rec[0].float().cpu().numpy()


@torch.no_grad()
def reconstruct_fmri(model, device: torch.device, use_bfloat16: bool, processed_vol, stored_tokens):
    encoder = model.fmri_encoder
    t = torch.as_tensor(processed_vol, dtype=torch.float32, device=device).unsqueeze(0)
    with autocast_context(device, use_bfloat16):
        _full = encoder(t)
        grid = getattr(encoder, "_cur_grid_size", encoder.grid_size)
        length, num_codebooks = int(stored_tokens.shape[0]), int(stored_tokens.shape[1])
        idx = torch.as_tensor(stored_tokens, dtype=torch.long, device=device).t().reshape(1, num_codebooks, length)
        if device.type in ("cuda", "npu"):
            amp_ctx = torch.amp.autocast(device.type, enabled=True, dtype=torch.float32)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            q = model.quantizer.idx_to_f(idx)
            q = model.fmri_post_quant_proj(q)
        rec = model.fmri_decoder(q, grid_shape=grid)
        if model.fmri_detail_refiner is not None:
            rec = model.fmri_detail_refiner(rec)
    return rec[0].float().cpu().numpy()


def segments_by_chunk(token_metadata: dict) -> dict[int, list[dict]]:
    temporal_index = token_metadata.get("temporal_index", {})
    result: dict[int, list[dict]] = {}
    for segment in temporal_index.get("segments", []):
        chunk_index = segment.get("source_chunk_index")
        if chunk_index is None:
            continue
        result.setdefault(int(chunk_index), []).append(segment)
    for rows in result.values():
        rows.sort(key=lambda s: int(s.get("token", {}).get("start_index", 0)))
    return result


def find_source_session_dir(input_base: Path, source_session_name: str, session_id: str) -> Path:
    for candidate_name in [source_session_name, session_id]:
        if candidate_name:
            candidate = input_base / candidate_name
            if (candidate / "data.tar").exists():
                return candidate
    matches = sorted(p for p in input_base.iterdir() if p.is_dir() and (p / "data.tar").exists() and p.name in {source_session_name, session_id})
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find source data.tar for session={session_id!r} source={source_session_name!r} under {input_base}")


def verify_session(
    *,
    mod,
    session_summary: dict,
    input_base: Path,
    output_base: Path,
    model,
    config,
    neural_transform,
    max_samples: Optional[int],
    device: torch.device,
    use_bfloat16: bool,
) -> dict:
    session_id = str(session_summary.get("session_id", ""))
    token_meta_rel = rel_path(session_summary.get("token_metadata"))
    if token_meta_rel is None:
        raise ValueError(f"{session_id}: token_metadata path is missing or absolute")
    token_metadata = load_json(output_base / token_meta_rel)
    modality = token_metadata.get("session", {}).get("modality") or session_summary.get("modality")
    is_2d = modality in {"eeg", "meg"}
    token_rel = rel_path(token_metadata.get("tokens", {}).get("path") or session_summary.get("token_file"))
    if token_rel is None:
        raise ValueError(f"{session_id}: token path is missing or absolute")
    token_ids = np.load(output_base / token_rel, mmap_mode="r")

    source_session_name = token_metadata.get("provenance", {}).get("source_session_name") or session_summary.get("provenance", {}).get("source_session_name", "")
    source_session_dir = find_source_session_dir(input_base, str(source_session_name), session_id)
    segment_map = segments_by_chunk(token_metadata)
    if not segment_map:
        raise ValueError(f"{session_id}: no source_chunk_index entries found in temporal_index")

    channels = mod.load_channels_json(source_session_dir, modality) if is_2d else []
    fmri_info = mod.load_fmri_info(source_session_dir) if modality == "fmri" else None
    rows: list[dict] = []

    sample_refs = mod.list_modality_samples_in_tar(source_session_dir / "data.tar", modality)
    if max_samples is not None:
        sample_refs = sample_refs[: int(max_samples)]

    with tarfile.open(source_session_dir / "data.tar", "r") as tar:
        for chunk_pos, (key, sig_member, ts_member, meta_member) in enumerate(sample_refs):
            raw = mod.load_npy_from_tar_member(tar, sig_member)
            ts = mod.load_npy_from_tar_member(tar, ts_member) if ts_member else None
            metadata = mod.load_json_from_tar_member(tar, meta_member) if meta_member else {}
            chunk_index = int(metadata.get("chunk_index", chunk_pos))
            chunk_segments = segment_map.get(chunk_index)
            if not chunk_segments:
                continue

            if is_2d:
                unit = int(neural_transform.time_size)
                signal, clean_info = mod.clean_2d_signal(raw, channels, ts, modality, config.quality_cfg)
                signal = mod.apply_storage_dtype(signal, config.storage_dtype)
                n_kept = int(clean_info["channel_count_kept"])
                starts = mod.time_window_starts(int(signal.shape[1]), unit)
                for window_index, start in enumerate(starts):
                    if window_index >= len(chunk_segments):
                        break
                    segment = chunk_segments[window_index]
                    processed, ch_mask, tm_mask, _info = mod.preprocess_2d_fixed_shape(signal[:, start:start + unit], neural_transform)
                    token_info = segment["token"]
                    stored = np.array(token_ids[int(token_info["start_index"]):int(token_info["end_index"])], copy=True)
                    recon = reconstruct_2d(model, modality, device, use_bfloat16, processed, ch_mask, tm_mask, stored)
                    time_len = int(segment["source_time"]["end"] - segment["source_time"]["start"])
                    orig_r = processed[:n_kept, :time_len]
                    recon_r = recon[:n_kept, :time_len]
                    metric = compute_metrics(orig_r, recon_r, mod, ssim_3d=False)
                    metric.update({"sample_key": key, "chunk_index": chunk_index, "window": window_index, "time_len": time_len})
                    rows.append(metric)
            else:
                # Tar data is already (X, Y, Z, T); no transpose needed.
                volume_xyzt = mod.validate_fmri(raw).astype(np.float32, copy=False)
                volume_xyzt = mod.normalize_fmri_volume(volume_xyzt, fmri_info, key, config.zscore_mode)
                volume_xyzt = mod.apply_storage_dtype(volume_xyzt, config.storage_dtype)
                unit = int(max(neural_transform.time_size_candidates))
                total_t = int(volume_xyzt.shape[3])
                ts_full = np.asarray(ts, np.float32).reshape(-1) if ts is not None else np.arange(total_t, dtype=np.float32)
                starts = mod.time_window_starts(total_t, unit)
                for window_index, start in enumerate(starts):
                    if window_index >= len(chunk_segments):
                        break
                    segment = chunk_segments[window_index]
                    vol_seg = volume_xyzt[..., start:start + unit]
                    ts_seg = ts_full[start:start + unit]
                    out_tensor, _ts, _info = neural_transform(
                        vol_seg,
                        timestamps=ts_seg,
                        brain_mask=None,
                        return_info=True,
                        time_start=float(ts_seg[0]) if ts_seg.size else None,
                        time_start_index=0,
                    )
                    processed = out_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                    token_info = segment["token"]
                    stored = np.array(token_ids[int(token_info["start_index"]):int(token_info["end_index"])], copy=True)
                    recon = reconstruct_fmri(model, device, use_bfloat16, processed, stored)
                    metric = compute_metrics(processed, recon, mod, ssim_3d=True)
                    metric.update({"sample_key": key, "chunk_index": chunk_index, "window": window_index})
                    rows.append(metric)

    if not rows:
        raise RuntimeError(f"{session_id}: no reconstruction windows evaluated")
    df = pd.DataFrame(rows)
    metric_cols = [c for c in ["pearson", "mse", "mae", "relative_l2", "ssim"] if c in df.columns]
    means = {col: float(df[col].mean()) for col in metric_cols}
    return {
        "session_id": session_id,
        "modality": modality,
        "source_session_dir": source_session_dir.name,
        "token_file": str(token_rel),
        "evaluated_windows": int(len(df)),
        "metrics_mean": means,
        "metrics_first_rows": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
    }


def require_cfg(config: dict, dotted_key: str):
    value = cfg_get(config, dotted_key, None)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def require_cfg_path(config: dict, dotted_key: str) -> Path:
    value = cfg_path(config, dotted_key, None)
    if value is None:
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify token reconstruction quality for a NeuroToken dataset.")
    parser.add_argument("--config", type=Path, required=True, help="TOML pipeline config file.")
    parser.add_argument("--input-base", type=Path, default=None, help="Tar package root. Overrides config paths.tar_root.")
    parser.add_argument("--output-base", type=Path, default=None, help="NeuroToken dataset root. Overrides config paths.token_root.")
    parser.add_argument("--report-path", type=Path, default=None, help="Write machine-readable JSON report.")
    parser.add_argument("--project-root", type=Path, default=None, help="Human UniTok project root.")
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="UniTok checkpoint root.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Explicit UniTok checkpoint dir/file.")
    parser.add_argument("--train-config", type=Path, default=None, help="Tokenizer train config.")
    parser.add_argument("--modality", choices=["eeg", "meg", "fmri"], default=None, help="Neural modality. Overrides config dataset.modality.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda/cpu/auto.")
    parser.add_argument("--use-bfloat16", action="store_true", help="Use CUDA bfloat16 autocast during decode.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Max sessions to verify.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max source chunks per session.")
    parser.add_argument("--min-mean-pearson", type=float, default=None, help="Fail when session mean pearson is below this value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_config = load_config(args.config)
    mod = load_stage2_module()

    input_base = Path(args.input_base or require_cfg_path(pipeline_config, "paths.tar_root"))
    output_base = Path(args.output_base or require_cfg_path(pipeline_config, "paths.token_root"))
    project_root = args.project_root or require_cfg_path(pipeline_config, "tokenizer.project_root")
    mod.configure_unitok_project_root(project_root)
    checkpoint_root = args.checkpoint_root or require_cfg_path(pipeline_config, "tokenizer.checkpoint_root")
    checkpoint_path = args.checkpoint_path or cfg_path(pipeline_config, "tokenizer.checkpoint_path", None)
    train_config = args.train_config or require_cfg_path(pipeline_config, "tokenizer.train_config")
    modality = args.modality or str(require_cfg(pipeline_config, "dataset.modality"))
    max_sessions = int(args.max_sessions if args.max_sessions is not None else require_cfg(pipeline_config, "reconstruction.max_sessions"))
    max_samples_value = args.max_samples if args.max_samples is not None else require_cfg(pipeline_config, "reconstruction.max_samples")
    max_samples = None if max_samples_value in (None, "", "None") else int(max_samples_value)
    min_mean_pearson_value = args.min_mean_pearson if args.min_mean_pearson is not None else require_cfg(pipeline_config, "reconstruction.min_mean_pearson")
    min_mean_pearson = None if min_mean_pearson_value in (None, "", "None") else float(min_mean_pearson_value)

    device_name = args.device or str(require_cfg(pipeline_config, "runtime.device"))
    if device_name == "auto":
        device_name = _detect_best_device()
    device = torch.device(device_name)
    require_cfg(pipeline_config, "runtime.use_bfloat16")
    use_bfloat16 = bool(args.use_bfloat16 or cfg_bool(pipeline_config, "runtime.use_bfloat16", False))
    if use_bfloat16 and device.type not in ("cuda", "npu"):
        raise ValueError("--use-bfloat16 requires a CUDA or NPU device")

    resolved_checkpoint = mod.resolve_checkpoint_path(checkpoint_path, checkpoint_root)
    require_cfg(pipeline_config, "tokenizer.use_ema")
    config = mod.RuntimeConfig(
        modality=modality,
        device=device,
        use_bfloat16=use_bfloat16,
        checkpoint_path=resolved_checkpoint,
        train_config_path=train_config,
        use_ema=cfg_bool(pipeline_config, "tokenizer.use_ema", False),
        data_seed=int(require_cfg(pipeline_config, "tokenizer.data_seed")),
        storage_dtype=str(require_cfg(pipeline_config, "tokenizer.storage_dtype")),
    )

    manifest = load_json(output_base / "manifest.json")
    sessions = manifest.get("sessions", [])[:max_sessions]
    model, dataset_meta, resolved_sizes, *_ = mod.build_unitok_for_inference(
        train_config_path=config.train_config_path,
        checkpoint_path=resolved_checkpoint,
        device=device,
        use_ema=config.use_ema,
    )
    neural_transform = mod.build_modality_neural_transform(dataset_meta, config.modality)
    config.quality_cfg, config.zscore_mode = mod.build_modality_quality_cfg(dataset_meta, config.modality)

    session_reports = []
    findings = []
    for session in sessions:
        try:
            item = verify_session(
                mod=mod,
                session_summary=session,
                input_base=input_base,
                output_base=output_base,
                model=model,
                config=config,
                neural_transform=neural_transform,
                max_samples=max_samples,
                device=device,
                use_bfloat16=use_bfloat16,
            )
            pearson = item.get("metrics_mean", {}).get("pearson")
            if min_mean_pearson is not None and pearson is not None and not math.isnan(float(pearson)) and float(pearson) < min_mean_pearson:
                findings.append({
                    "severity": "error",
                    "category": "reconstruction_quality",
                    "message": f"{item['session_id']}: mean pearson {pearson:.6f} < threshold {min_mean_pearson:.6f}",
                })
            session_reports.append(item)
        except Exception as exc:
            findings.append({
                "severity": "error",
                "category": "reconstruction_runtime",
                "message": f"{session.get('session_id', '<unknown>')}: {exc}",
            })

    status = "failed" if any(x["severity"] == "error" for x in findings) else "ok"
    report = make_report(
        "reconstruct",
        status=status,
        input_base=str(input_base),
        output_base=str(output_base),
        checkpoint="/".join(Path(resolved_checkpoint).parts[-2:]),
        device=str(device),
        use_bfloat16=bool(use_bfloat16),
        max_sessions=max_sessions,
        max_samples=max_samples,
        min_mean_pearson=min_mean_pearson,
        sessions=session_reports,
        findings=findings,
    )
    report_path = args.report_path or default_report_path(pipeline_config, "reconstruct", Path("reports"))
    write_report(report_path, report)

    print(f"Reconstruction status: {status}")
    for item in session_reports:
        print(item["session_id"], "windows=", item["evaluated_windows"], "metrics=", item["metrics_mean"])
    print(f"Report: {report_path}")
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
