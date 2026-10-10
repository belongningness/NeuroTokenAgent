#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: package source sessions into deterministic WebDataset tar files.

Dataset-specific source parsing lives in ``demo/adapters``. This script is the
generic tar writer: it receives a standard ``SourceSession`` from an adapter,
splits the neural signal into chronological chunks, and writes ``data.tar`` plus
session-level sidecars.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import webdataset as wds

from adapters import build_adapter
from adapters.base import SourceSession, safe_filename
from neurotoken_config import cfg_chunking_strategy, cfg_get, cfg_path, load_config


TAR_FILENAME = "data.tar"
SESSION_METADATA_FILENAME = "session_metadata.json"
CHUNK_INDEX_PAD = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage1RuntimeConfig:
    input_root: Path
    output_base: Path
    dataset_name: str
    chunking_strategy: str     # "rest" = fixed-length chunks; "event" = event-boundary chunks
    split_points: int          # used for rest chunks and event fallback
    downsample_factor: int


def require_cfg(config: dict, dotted_key: str) -> Any:
    value = cfg_get(config, dotted_key, None)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def require_cfg_path(config: dict, dotted_key: str) -> Path:
    value = cfg_path(config, dotted_key, None)
    if value is None:
        raise KeyError(f"Missing required config field: {dotted_key}")
    return value


def build_runtime_config(config: dict, input_root_override: str | None = None, output_base_override: str | None = None) -> Stage1RuntimeConfig:
    """Build required generic Stage 1 config from TOML.

    Source-specific fields are consumed by the selected adapter, not by this
    writer. No dataset/runtime defaults are hidden in this script; missing
    required fields fail fast.
    """
    if not config:
        raise ValueError("Stage 1 requires --config; no fallback config is provided by the writer.")

    input_root = Path(input_root_override).expanduser() if input_root_override else require_cfg_path(config, "paths.h5_root")
    output_base = Path(output_base_override).expanduser() if output_base_override else require_cfg_path(config, "paths.tar_root")

    runtime = Stage1RuntimeConfig(
        input_root=input_root,
        output_base=output_base,
        dataset_name=str(require_cfg(config, "dataset.name")),
        chunking_strategy=cfg_chunking_strategy(config),
        split_points=int(require_cfg(config, "chunking.split_points")),
        downsample_factor=int(require_cfg(config, "chunking.downsample_factor")),
    )
    if runtime.downsample_factor <= 0 or runtime.split_points % runtime.downsample_factor != 0:
        raise ValueError(
            "chunking.split_points must be divisible by chunking.downsample_factor: "
            f"{runtime.split_points} % {runtime.downsample_factor}"
        )
    return runtime


def to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas-style scalars to plain JSON values."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def build_meta_json(
    session_meta: dict[str, Any],
    sfreq: float,
    chunk_index: int,
    chunk_start_sample: int,
    chunk_end_sample: int,
) -> dict[str, Any]:
    n_samples = chunk_end_sample - chunk_start_sample
    meta = dict(session_meta)
    meta.update({
        "chunk_index": int(chunk_index),
        "chunk_start_sample": int(chunk_start_sample),
        "chunk_end_sample": int(chunk_end_sample),
        "chunk_start_time": float(chunk_start_sample) / float(sfreq),
        "chunk_end_time": float(chunk_end_sample) / float(sfreq),
        "n_samples": int(n_samples),
    })
    return meta


def _time_based_splits(signal: np.ndarray, sfreq: float, session_meta: dict, session: SourceSession, runtime: Stage1RuntimeConfig):
    """Yield chunks by fixed-length time windows (split_points samples each)."""
    n_points = int(signal.shape[1])
    points_per_split = int(runtime.split_points)
    n_full_splits = n_points // points_per_split
    tail_points = n_points - n_full_splits * points_per_split
    has_tail = tail_points > 0
    num_splits = n_full_splits + (1 if has_tail else 0)

    if num_splits == 0:
        logger.warning("文件 %s 长度为 0，跳过。", session.source_path)
        return

    tail_tokens_note = (
        f"（不足 {runtime.downsample_factor} 的整数倍，下游需 padding）"
        if has_tail and tail_points % runtime.downsample_factor != 0
        else ""
    )
    logger.info(
        "文件 %s time 切片: %d 点 (~%.4f s, %d tokens) × %d；尾段: %d 点 (~%.6f s)%s",
        session.source_path, points_per_split, points_per_split / sfreq,
        points_per_split // runtime.downsample_factor,
        n_full_splits, tail_points, tail_points / sfreq, tail_tokens_note,
    )

    subject = session_meta.get("Subject", {}) or {}
    sub_id = safe_filename(str(subject.get("ID", "")) or "unknown")
    dataset_name = str(session_meta.get("Dataset", runtime.dataset_name))

    for split_idx in range(num_splits):
        start_pt = split_idx * points_per_split
        is_tail = has_tail and split_idx == num_splits - 1
        cur_points = tail_points if is_tail else points_per_split
        end_pt = start_pt + cur_points

        signal_slice = signal[:, start_pt:end_pt]
        if not np.isfinite(signal_slice).all():
            logger.warning("文件 %s 第 %d 个切片包含 NaN/Inf，跳过。", session.source_path, split_idx)
            continue

        timestamps = np.arange(0, cur_points, dtype=np.float32) / np.float32(sfreq)
        metadata = build_meta_json(session_meta=session_meta, sfreq=sfreq,
                                   chunk_index=split_idx, chunk_start_sample=start_pt,
                                   chunk_end_sample=end_pt)
        metadata["chunking_strategy"] = "rest"
        sample_key = f"{dataset_name}-{sub_id}-{session.safe_stem}_{split_idx:0{CHUNK_INDEX_PAD}d}"

        yield {
            "__key__": sample_key,
            "metadata.json": json.dumps(metadata, ensure_ascii=False, default=to_jsonable, allow_nan=True),
            f"{session.modality}.npy": np.ascontiguousarray(signal_slice.astype(np.float32)),
            f"{session.modality}_timestamps.npy": np.ascontiguousarray(timestamps.astype(np.float32)),
        }


def _event_based_splits(signal: np.ndarray, sfreq: float, session_meta: dict, session: SourceSession, runtime: Stage1RuntimeConfig):
    """Use event boundaries as split points to divide the continuous signal.

    Every event contributes its ``start_time`` and ``end_time`` as a split
    point.  Together with the signal start (0) and end (n_samples / sfreq),
    they partition the recording into adjacent, non-overlapping segments.

    For example, events [(0.5, 1.3), (3.0, 4.5)] produce::

        [0, 0.5)  [0.5, 1.3)  [1.3, 3.0)  [3.0, 4.5)  [4.5, end)

    Each segment's metadata records whether it falls inside an event and, if
    so, which event.
    """
    events = session.events or []
    if not events:
        logger.warning("文件 %s 无事件标注，回退为 time 切分。", session.source_path)
        yield from _time_based_splits(signal, sfreq, session_meta, session, runtime)
        return

    # Collect all unique split-points (times in seconds)
    boundaries_sec: list[float] = []
    event_by_range: dict[tuple[float, float], tuple[int, dict]] = {}
    for event_index, ev in enumerate(events):
        t0 = float(ev.get("start_time", 0))
        t1 = float(ev.get("end_time", 0))
        if t1 <= t0:
            logger.warning("事件 start_time=%.4f >= end_time=%.4f，跳过。", t0, t1)
            continue
        if t0 not in boundaries_sec:
            boundaries_sec.append(t0)
        if t1 not in boundaries_sec:
            boundaries_sec.append(t1)
        key = (round(t0, 6), round(t1, 6))
        event_by_range.setdefault(key, (event_index, ev))

    if not boundaries_sec:
        logger.warning("文件 %s 所有事件时间无效，回退为 time 切分。", session.source_path)
        yield from _time_based_splits(signal, sfreq, session_meta, session, runtime)
        return

    boundaries_sec = sorted(set(boundaries_sec))

    max_sample = int(signal.shape[1])
    signal_end_sec = max_sample / sfreq

    # Build final boundary list:  0, b0, b1, ..., bn, end
    cut_points = [0.0] + boundaries_sec + [signal_end_sec]
    # Remove zero-length edge cases (duplicate boundary)
    cut_points = sorted(set(cut_points))

    segments = []
    for i in range(len(cut_points) - 1):
        seg_start = cut_points[i]
        seg_end = cut_points[i + 1]
        duration = seg_end - seg_start
        if duration <= 0:
            continue

        # Determine if this segment is inside an event
        seg_mid = (seg_start + seg_end) / 2.0
        inside_event = False
        matched_event_index = None
        matched_event = None
        for (et0, et1), (event_index, ev) in event_by_range.items():
            if et0 <= seg_mid < et1:
                inside_event = True
                matched_event_index = event_index
                matched_event = ev
                break

        segments.append((seg_start, seg_end, inside_event, matched_event_index, matched_event))

    if not segments:
        logger.warning("文件 %s 切分后无有效段，回退为 time 切分。", session.source_path)
        yield from _time_based_splits(signal, sfreq, session_meta, session, runtime)
        return

    logger.info("文件 %s event 切分: %d 个事件 → %d 个边界点 → %d 个段。",
                session.source_path, len(event_by_range), len(boundaries_sec), len(segments))

    subject = session_meta.get("Subject", {}) or {}
    sub_id = safe_filename(str(subject.get("ID", "")) or "unknown")
    dataset_name = str(session_meta.get("Dataset", runtime.dataset_name))

    for split_idx, (seg_start, seg_end, inside_event, matched_event_index, matched_event) in enumerate(segments):
        start_pt = int(round(seg_start * sfreq))
        end_pt = int(round(seg_end * sfreq))
        start_pt = max(0, min(start_pt, max_sample))
        end_pt = max(0, min(end_pt, max_sample))

        if end_pt <= start_pt:
            continue

        signal_slice = signal[:, start_pt:end_pt]
        if not np.isfinite(signal_slice).all():
            logger.warning("段 %d [%.4f, %.4f) 包含 NaN/Inf，跳过。", split_idx, seg_start, seg_end)
            continue

        cur_points = end_pt - start_pt
        timestamps = np.arange(start_pt, end_pt, dtype=np.float32) / np.float32(sfreq)

        metadata = build_meta_json(session_meta=session_meta, sfreq=sfreq,
                                   chunk_index=split_idx, chunk_start_sample=start_pt,
                                   chunk_end_sample=end_pt)
        metadata["chunking_strategy"] = "event" if inside_event else "rest"
        metadata["segment_type"] = "event" if inside_event else "inter_event"
        if matched_event is not None and matched_event_index is not None:
            metadata["event_index"] = int(matched_event_index)
            metadata["event_type"] = str(matched_event.get("type", ""))
            metadata["event_value"] = str(matched_event.get("value", ""))
            metadata["event_start_time"] = float(matched_event.get("start_time", 0))
            metadata["event_end_time"] = float(matched_event.get("end_time", 0))

        sample_key = f"{dataset_name}-{sub_id}-{session.safe_stem}_{split_idx:0{CHUNK_INDEX_PAD}d}"

        yield {
            "__key__": sample_key,
            "metadata.json": json.dumps(metadata, ensure_ascii=False, default=to_jsonable, allow_nan=True),
            f"{session.modality}.npy": np.ascontiguousarray(signal_slice.astype(np.float32)),
            f"{session.modality}_timestamps.npy": np.ascontiguousarray(timestamps.astype(np.float32)),
        }


def iter_samples_from_session(session: SourceSession, runtime: Stage1RuntimeConfig):
    """Yield chronological WebDataset chunk samples for one standard session.

    Two modes exist, controlled by ``runtime.chunking_strategy``:

    ``"rest"``
        Fixed-length chunks: every ``split_points`` samples become one chunk.
        An optional tail chunk captures the remainder.

    ``"event"``
        Event-boundary splitting: each event's ``start_time`` and ``end_time``
        becomes a split point. Together with the signal start (0) and end, the
        recording is partitioned into adjacent, non-overlapping segments.
        Event segments use ``chunking_strategy="event"`` and retain their event
        index; inter-event segments use ``chunking_strategy="rest"``. Falls back
        to rest splitting when no valid events are present.
    """
    signal = np.asarray(session.signal)
    sfreq = float(session.sfreq)
    session_meta = session.session_metadata

    if signal.ndim != 2:
        raise ValueError(f"{session.source_path} {session.modality} signal must be [C, T], got {signal.shape}")

    if runtime.chunking_strategy == "event":
        yield from _event_based_splits(signal, sfreq, session_meta, session, runtime)
    else:
        yield from _time_based_splits(signal, sfreq, session_meta, session, runtime)


def package_one_session(session: SourceSession, result_dir: str | Path, runtime: Stage1RuntimeConfig) -> int:
    """Write one standard SourceSession to ``<result_dir>/<safe_stem>/``."""
    output_dir = Path(result_dir) / session.safe_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / TAR_FILENAME

    logger.info(
        "文件 %s session 信息: Subject.ID=%s, Health_group=%s, n_samples_total=%d, #events=%d, #channels=%d",
        session.source_path,
        session.subject.get("ID", ""),
        session.subject.get("Health_group", ""),
        int(session.session_metadata.get("n_samples_total", 0)),
        len(session.events),
        len(session.channels),
    )

    count = 0
    with wds.TarWriter(str(tar_path)) as sink:
        for sample in iter_samples_from_session(session, runtime):
            sink.write(sample)
            count += 1
            if count % 1000 == 0:
                logger.info("文件 %s 已写入 %d 个样本。", session.source_path, count)

    (output_dir / SESSION_METADATA_FILENAME).write_text(
        json.dumps(session.session_metadata, ensure_ascii=False, default=to_jsonable, allow_nan=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{session.modality}_channels.json").write_text(
        json.dumps(session.channels, ensure_ascii=False, default=to_jsonable, allow_nan=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / "events_info.json").write_text(
        json.dumps(session.events, ensure_ascii=False, default=to_jsonable, allow_nan=True, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "文件 %s 打包完成，共生成 %d 个 chunk 样本。输出目录: %s"
        "（含 session_metadata.json / %s_channels.json / events_info.json sidecar）",
        session.source_path,
        count,
        output_dir,
        session.modality,
    )
    return count


def main(runtime: Stage1RuntimeConfig, config: dict) -> None:
    runtime.output_base.mkdir(parents=True, exist_ok=True)

    adapter = build_adapter(config)
    source_files = adapter.discover(runtime.input_root)

    logger.info("开始处理 %s 数据集", runtime.dataset_name)
    logger.info("输入目录: %s", runtime.input_root)
    logger.info("输出目录: %s", runtime.output_base)
    logger.info("源数据 adapter: %s", adapter.__class__.__name__)
    logger.info("每个 source h5 → 1 个 tar (%s)，不再按大小分 shard。", TAR_FILENAME)

    if not source_files:
        logger.warning("未找到任何 source 文件，退出。")
        return

    total_samples = 0
    for idx, source_path in enumerate(source_files, start=1):
        logger.info("[%d/%d] 处理文件: %s", idx, len(source_files), source_path)
        try:
            session = adapter.load_session(Path(source_path))
            total_samples += package_one_session(session=session, result_dir=runtime.output_base, runtime=runtime)
        except Exception as exc:
            logger.exception("处理失败 %s: %s", source_path, exc)

    logger.info("所有文件处理完成，共生成 %d 个样本。", total_samples)
    logger.info("打包完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: package source sessions into WebDataset tar files.")
    parser.add_argument("--config", type=Path, required=True, help="TOML pipeline config file.")
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="输入根目录（直接含 source 文件，不需要 train/val/test 子目录）。覆盖 config/脚本默认值。",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=None,
        help="输出根目录。覆盖 config/脚本默认值。",
    )
    args = parser.parse_args()

    pipeline_config = load_config(args.config)
    runtime_config = build_runtime_config(
        pipeline_config,
        input_root_override=args.input_root,
        output_base_override=args.output_base,
    )
    main(runtime=runtime_config, config=pipeline_config)
