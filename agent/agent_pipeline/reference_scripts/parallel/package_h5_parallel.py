#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel H5 → tar packaging for Stage 1 of the NeuroToken pipeline.

This is a **drop-in parallel replacement** for
``1_make_tar_packages4BrainToken_final.py``.  Every H5 source file is
processed by an independent worker process (``ProcessPoolExecutor``), so
multi-core machines can saturate I/O and CPU during packaging.

The packaging logic itself (adapter, chunking, tar writing) is imported
directly from the original Stage 1 module — this script only adds the
parallel orchestration layer.  The original script is **not modified**.

Usage::

    # same CLI as the original, plus -j
    python parallel/package_h5_parallel.py --config demo.toml -j4
    python parallel/package_h5_parallel.py --config demo.toml -j0   # auto
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import the original Stage 1 module (its file name starts with a digit so we
# use importlib).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PARENT_SCRIPT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_SCRIPT_DIR))
_STAGE1_PATH = _PARENT_SCRIPT_DIR / "1_make_tar_packages4BrainToken_final.py"
_SPEC = importlib.util.spec_from_file_location("_stage1_package", str(_STAGE1_PATH))
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load Stage 1 module from {_STAGE1_PATH}")
_stage1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _stage1
_SPEC.loader.exec_module(_stage1)

# Re-export the key symbols we depend on (no need to qualify them later).
build_runtime_config = _stage1.build_runtime_config
package_one_session = _stage1.package_one_session
TAR_FILENAME = _stage1.TAR_FILENAME

# These live in sibling packages — import normally.
from adapters import build_adapter                       # noqa: E402
from neurotoken_config import cfg_get, cfg_path, load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("parallel.package")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_worker_count(requested: int) -> int:
    """Translate the ``--workers`` flag into a concrete worker count.

    * ``0`` or negative → ``os.cpu_count()`` (auto-detect).
    * positive         → clamped to ``[1, os.cpu_count()]``.
    """
    cpu = os.cpu_count() or 1
    if requested <= 0:
        return max(1, cpu)
    return max(1, min(requested, cpu))


def _try_import_tqdm():
    """Best-effort tqdm import; returns the class or ``None``."""
    try:
        from tqdm.auto import tqdm  # type: ignore[assignment]
        return tqdm
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------


def _package_one_h5_file(
    source_path_str: str,
    config_dict: dict,
    output_base_str: str,
) -> dict:
    """Package a single H5 file → tar directory inside a worker process.

    The worker reconstructs its own adapter and runtime config from the
    serialisable ``config_dict`` so every process is fully self-contained.
    """
    source_path = Path(source_path_str)
    output_base = Path(output_base_str)
    try:
        runtime = build_runtime_config(
            config_dict,
            input_root_override=None,
            output_base_override=output_base_str,
        )
        adapter = build_adapter(config_dict)
        session = adapter.load_session(source_path)
        count = package_one_session(
            session=session, result_dir=output_base, runtime=runtime,
        )
        return {"path": str(source_path), "count": int(count), "error": None}
    except Exception as exc:
        return {"path": str(source_path), "count": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _main_sequential(adapter, source_files: list, runtime, strict: bool = False) -> int:
    """Run Stage 1 sequentially, optionally failing on any source error."""
    total = 0
    failed: list[str] = []
    for idx, source_path in enumerate(source_files, start=1):
        logger.info("[%d/%d] 处理文件: %s", idx, len(source_files), source_path)
        try:
            session = adapter.load_session(Path(source_path))
            total += package_one_session(
                session=session, result_dir=runtime.output_base, runtime=runtime,
            )
        except Exception:
            logger.exception("处理失败: %s", source_path)
            failed.append(str(source_path))
    if failed and strict:
        raise RuntimeError(f"{len(failed)}/{len(source_files)} source files failed: {failed}")
    return total


def _main_parallel(
    config: dict,
    source_files: list,
    output_base: str,
    max_workers: int,
    strict: bool = False,
) -> int:
    """Dispatch one H5 file per worker; collect results as they finish."""
    worker_args = [(str(p), config, output_base) for p in source_files]
    total = 0
    failed: list[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_package_one_h5_file, *args): args[0]
            for args in worker_args
        }

        tqdm = _try_import_tqdm()
        pbar = (
            tqdm(total=len(future_to_path), desc="打包 H5 文件", unit="file")
            if tqdm is not None
            else None
        )

        for future in as_completed(future_to_path):
            src = future_to_path[future]
            try:
                result = future.result()
                if result.get("error"):
                    logger.error("[失败] %s: %s", result["path"], result["error"])
                    failed.append(result["path"])
                else:
                    cnt = int(result.get("count", 0))
                    total += cnt
                    if pbar is not None:
                        stem = Path(result["path"]).name
                        pbar.set_postfix_str(f"{stem} ✓ {cnt} chunks")
            except Exception:
                logger.exception("Worker 异常: %s", src)
                failed.append(src)
            finally:
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

    if failed:
        logger.warning(
            "%d/%d 个文件处理失败: %s", len(failed), len(source_files), failed,
        )
        if strict:
            raise RuntimeError(
                f"{len(failed)}/{len(source_files)} source files failed: {failed}"
            )
    return total


def run(config_path: str | Path, workers: int = 1,
        input_root: str | None = None, output_base: str | None = None,
        strict: bool = False) -> int:
    """Entry point callable from other Python code.

    Returns the total number of chunk samples written across all H5 files.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)

    # Resolve I/O roots (CLI overrides take precedence over config file).
    runtime = build_runtime_config(
        config,
        input_root_override=input_root,
        output_base_override=output_base,
    )
    runtime.output_base.mkdir(parents=True, exist_ok=True)

    adapter = build_adapter(config)
    source_files = adapter.discover(runtime.input_root)

    logger.info("开始处理 %s 数据集", runtime.dataset_name)
    logger.info("输入目录: %s", runtime.input_root)
    logger.info("输出目录: %s", runtime.output_base)
    logger.info("源数据 adapter: %s", adapter.__class__.__name__)
    logger.info(
        "每个 source h5 → 1 个 tar (%s)，不按大小分 shard。", TAR_FILENAME,
    )

    if not source_files:
        logger.warning("未找到任何 source 文件，退出。")
        return 0

    actual_workers = _resolve_worker_count(workers)

    if actual_workers > 1 and len(source_files) > 1:
        logger.info(
            "并行模式: %d worker 进程 × %d H5 文件",
            actual_workers, len(source_files),
        )
        total = _main_parallel(
            config=config,
            source_files=source_files,
            output_base=str(runtime.output_base),
            max_workers=actual_workers,
            strict=strict,
        )
    else:
        if workers > 1:
            logger.info("仅 %d 个文件，回退为顺序处理。", len(source_files))
        total = _main_sequential(adapter, source_files, runtime, strict=strict)

    logger.info("所有文件处理完成，共生成 %d 个样本。", total)
    logger.info("打包完成！")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel H5 → tar packaging (drop-in for Stage 1).",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="TOML pipeline config file.",
    )
    parser.add_argument(
        "--input_root", type=str, default=None,
        help="输入根目录（直接含 source 文件）。覆盖 config/脚本默认值。",
    )
    parser.add_argument(
        "--output_base", type=str, default=None,
        help="输出根目录。覆盖 config/脚本默认值。",
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=1,
        help=(
            "并行 worker 进程数（每个进程处理一个 H5 文件）。"
            " 0 或负数 = 自动检测 CPU 核数；1 = 顺序处理（默认）。"
        ),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="任意 source H5 处理失败时返回失败。",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        config_path=args.config,
        workers=args.workers,
        input_root=args.input_root,
        output_base=args.output_base,
        strict=args.strict,
    )
