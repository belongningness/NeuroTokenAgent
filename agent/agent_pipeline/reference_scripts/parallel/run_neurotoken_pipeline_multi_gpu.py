#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the complete NeuroToken Stage 0–6 pipeline with parallel Stage 1/2.

Stage 1 delegates to the H5 process pool and Stage 2 delegates to the
one-worker-per-CUDA-device scheduler. All remaining stages use their canonical
single-process scripts. Every stage is separated by a strict success barrier.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

PARALLEL_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PARALLEL_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from neurotoken_config import cfg_path, load_config
from neurotoken_report import make_report, write_report
from parallel.resource_planner import (
    DEFAULT_MAX_AUTO_GPU_DEVICES,
    DEFAULT_TOKEN_WORKER_MEMORY_GB,
)
from run_neurotoken_pipeline import write_human_summary


STAGE_ORDER = ["preflight", "package", "tokenize", "readme", "docs", "validate", "reconstruct"]
STAGE_TITLES = {
    "preflight": "Stage 0 / preflight",
    "package": "Stage 1 / parallel H5 → tar",
    "tokenize": "Stage 2 / one worker per GPU",
    "readme": "Stage 3 / README",
    "docs": "Stage 4 / docs and schemas",
    "validate": "Stage 5 / validation",
    "reconstruct": "Stage 6 / reconstruction",
}
CHILD_REPORT_FILENAMES = {
    "preflight": "preflight_report.json",
    "package": "stage1_parallel_report.json",
    "tokenize": "stage2_multi_gpu_report.json",
    "validate": "validate_report.json",
    "reconstruct": "reconstruct_report.json",
}


def _load_json(path: Optional[Path]) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        return {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}


def _file_signature(path: Optional[Path]) -> Optional[tuple[int, int]]:
    if path is None:
        return None
    try:
        stat = path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return None


def _bounded_smoke_value(value: Optional[int], default: int, smoke_test: bool) -> Optional[int]:
    if value is not None:
        return value
    return default if smoke_test else None


def _directory_has_entries(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if not path.is_dir():
        return True
    try:
        return next(path.iterdir(), None) is not None
    except OSError:
        return True


def _first_symlink_component(path: Path) -> Optional[Path]:
    current = path.expanduser().absolute()
    while True:
        if current.is_symlink():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _remove_tar_root_safely(
    path: Path,
    pipeline_config: dict,
    config_path: Path,
    report_root: Path,
    *,
    perform_removal: bool = True,
) -> None:
    raw_path = path.expanduser().absolute()
    symlink_component = _first_symlink_component(raw_path)
    if symlink_component is not None:
        raise ValueError(
            f"Refusing tar_root cleanup through symlink component: {symlink_component}"
        )
    path = raw_path.resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        SCRIPT_DIR.resolve(),
        SCRIPT_DIR.parent.resolve(),
        PARALLEL_DIR.resolve(),
        config_path.expanduser().resolve(),
        config_path.expanduser().resolve().parent,
        report_root.expanduser().resolve(),
    }
    for key in (
        "paths.h5_root",
        "paths.token_root",
        "paths.report_root",
        "tokenizer.project_root",
        "tokenizer.checkpoint_root",
        "tokenizer.checkpoint_path",
        "tokenizer.train_config",
    ):
        value = cfg_path(pipeline_config, key, None)
        if value is not None:
            protected.add(Path(value).expanduser().resolve())
    if any(path == item or path in item.parents for item in protected) or len(path.parts) < 4:
        raise ValueError(f"Refusing unsafe tar_root cleanup path: {path}")
    if perform_removal:
        shutil.rmtree(path)


def _validate_token_root_cleanup(
    path: Path,
    pipeline_config: dict,
    config_path: Path,
    report_root: Path,
) -> None:
    raw_path = path.expanduser().absolute()
    symlink_component = _first_symlink_component(raw_path)
    if symlink_component is not None:
        raise ValueError(
            f"Refusing token_root cleanup through symlink component: {symlink_component}"
        )
    path = raw_path.resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        SCRIPT_DIR.resolve(),
        SCRIPT_DIR.parent.resolve(),
        PARALLEL_DIR.resolve(),
        config_path.expanduser().resolve(),
        config_path.expanduser().resolve().parent,
        report_root.expanduser().resolve(),
    }
    for key in (
        "paths.h5_root",
        "paths.tar_root",
        "paths.report_root",
        "tokenizer.project_root",
        "tokenizer.checkpoint_root",
        "tokenizer.checkpoint_path",
        "tokenizer.train_config",
    ):
        value = cfg_path(pipeline_config, key, None)
        if value is not None:
            protected.add(Path(value).expanduser().resolve())
    if any(path == item or path in item.parents for item in protected) or len(path.parts) < 4:
        raise ValueError(f"Refusing unsafe token_root cleanup path: {path}")


def prepare_stage_execution(
    args: argparse.Namespace,
    stage: str,
    pipeline_config: dict,
    config_path: Path,
    report_root: Path,
    child_report_path: Optional[Path],
    *,
    dry_run: bool = False,
) -> None:
    """Reject stale release roots and clear only known generated reports."""
    if not dry_run and child_report_path is not None and child_report_path.exists():
        child_report_path.unlink()

    if stage == "package":
        tar_root_value = cfg_path(pipeline_config, "paths.tar_root", None)
        if tar_root_value is None:
            raise KeyError("Missing required config field: paths.tar_root")
        tar_root = Path(tar_root_value).expanduser().absolute()
        tar_symlink = _first_symlink_component(tar_root)
        if tar_symlink is not None:
            raise ValueError(
                f"Refusing tar_root through symlink component: {tar_symlink}"
            )
        if tar_root.exists() and not tar_root.is_dir():
            raise NotADirectoryError(f"tar_root is not a directory: {tar_root}")
        tar_root_has_entries = _directory_has_entries(tar_root)
        if tar_root_has_entries and not args.overwrite_tar_output:
            raise FileExistsError(
                f"tar_root is not empty: {tar_root}. Use --overwrite-tar-output "
                "to prevent stale sessions from a previous run."
            )
        if args.overwrite_tar_output and (tar_root.exists() or tar_root.is_symlink()):
            _remove_tar_root_safely(
                tar_root,
                pipeline_config,
                config_path,
                report_root,
                perform_removal=not dry_run,
            )

    if stage == "tokenize":
        token_root_value = cfg_path(pipeline_config, "paths.token_root", None)
        if token_root_value is None:
            raise KeyError("Missing required config field: paths.token_root")
        token_root = Path(token_root_value).expanduser().absolute()
        token_symlink = _first_symlink_component(token_root)
        if token_symlink is not None:
            raise ValueError(
                f"Refusing token_root through symlink component: {token_symlink}"
            )
        if token_root.exists() and not token_root.is_dir():
            raise NotADirectoryError(f"token_root is not a directory: {token_root}")
        token_root_has_entries = _directory_has_entries(token_root)
        if token_root_has_entries and not args.overwrite_token_output:
            raise FileExistsError(
                f"token_root is not empty: {token_root}. Use --overwrite-token-output "
                "to prevent a mixed release tree."
            )
        if args.overwrite_token_output and (token_root.exists() or token_root.is_symlink()):
            _validate_token_root_cleanup(
                token_root,
                pipeline_config,
                config_path,
                report_root,
            )


def build_stage_command(
    args: argparse.Namespace,
    stage: str,
    config_path: Path,
    report_root: Path,
) -> tuple[list[str], Optional[Path]]:
    """Build one canonical stage command and its machine-readable report path."""
    python = str(args.python)
    config_args = ["--config", str(config_path)]
    report_path = (
        report_root / CHILD_REPORT_FILENAMES[stage]
        if stage in CHILD_REPORT_FILENAMES
        else None
    )

    if stage == "preflight":
        command = [python, "-B", str(SCRIPT_DIR / "0_preflight_check.py"), *config_args]
        command += ["--report-path", str(report_path)]
        if args.smoke_test:
            command += ["--max-files", "1"]
        return command, report_path

    if stage == "package":
        command = [
            python,
            "-B",
            str(PARALLEL_DIR / "run_single_dataset_multi_gpu.py"),
            *config_args,
            "--package-only",
            "--report-path",
            str(report_path),
        ]
        if args.h5_workers is not None:
            command += ["--h5-workers", str(args.h5_workers)]
        return command, report_path

    if stage == "tokenize":
        command = [
            python,
            "-B",
            str(PARALLEL_DIR / "run_single_dataset_multi_gpu.py"),
            *config_args,
            "--skip-h5-to-tar",
            "--max-gpus",
            str(args.max_gpus),
            "--report-path",
            str(report_path),
        ]
        if args.devices:
            command += ["--devices", *args.devices]
        if args.token_worker_memory_gb is not None:
            command += ["--token-worker-memory-gb", str(args.token_worker_memory_gb)]
        max_samples = _bounded_smoke_value(args.max_samples, 2, args.smoke_test)
        max_sessions = _bounded_smoke_value(args.max_sessions, 1, args.smoke_test)
        if max_samples is not None:
            command += ["--max-samples", str(max_samples)]
        if max_sessions is not None:
            command += ["--max-sessions", str(max_sessions)]
        if args.overwrite_token_output:
            command.append("--overwrite-token-output")
        return command, report_path

    if stage == "readme":
        command = [python, "-B", str(SCRIPT_DIR / "3_make_readme.py"), *config_args]
        if args.allow_placeholder_readme or args.smoke_test:
            command.append("--allow-placeholder")
        return command, None

    if stage == "docs":
        return [
            python,
            "-B",
            str(SCRIPT_DIR / "4_generate_release_docs_and_schemas.py"),
            *config_args,
        ], None

    if stage == "validate":
        command = [python, "-B", str(SCRIPT_DIR / "5_validate_neurotoken_dataset.py"), *config_args]
        command += ["--json-report", str(report_path)]
        if args.strict_validation:
            command.append("--strict")
        if args.check_event_duration_semantics:
            command.append("--check-event-duration-semantics")
        return command, report_path

    if stage == "reconstruct":
        command = [python, "-B", str(SCRIPT_DIR / "6_verify_token_reconstruction.py"), *config_args]
        command += ["--report-path", str(report_path)]
        max_samples = _bounded_smoke_value(
            args.max_reconstruction_samples,
            2,
            args.smoke_test,
        )
        max_sessions = _bounded_smoke_value(
            args.max_reconstruction_sessions,
            1,
            args.smoke_test,
        )
        if max_samples is not None:
            command += ["--max-samples", str(max_samples)]
        if max_sessions is not None:
            command += ["--max-sessions", str(max_sessions)]
        if args.reconstruction_device:
            command += ["--device", args.reconstruction_device]
        if args.max_reconstruction_chunk_batch_size is not None:
            command += [
                "--max-chunk-batch-size",
                str(args.max_reconstruction_chunk_batch_size),
            ]
        if args.min_mean_pearson is not None:
            command += ["--min-mean-pearson", str(args.min_mean_pearson)]
        return command, report_path

    raise ValueError(f"Unsupported pipeline stage: {stage}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    demo_root = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run NeuroToken Stage 0–6 with parallel H5 packaging and exactly one "
            "persistent tokenization worker per selected CUDA GPU."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=demo_root / "configs" / "nod_eeg_demo.toml",
        help="TOML pipeline config file.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=STAGE_ORDER,
        help="Canonical-order stage subset (default: complete Stage 0–6).",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for child stages.")
    parser.add_argument("--report-root", type=Path, default=None, help="Unified report directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print every command without executing it.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Limit preflight/tokenization/reconstruction and allow README placeholders; "
            "defaults to one session and two samples."
        ),
    )

    stage1 = parser.add_argument_group("Stage 1 parallel H5 packaging")
    stage1.add_argument(
        "-j",
        "--h5-workers",
        type=int,
        default=None,
        help="H5 workers; omit or set 0 for conservative automatic planning.",
    )
    stage1.add_argument(
        "--overwrite-tar-output",
        action="store_true",
        help=(
            "Remove the configured tar_root before Stage 1. Without this flag, "
            "a non-empty tar_root fails safely to prevent stale sessions."
        ),
    )

    stage2 = parser.add_argument_group("Stage 2 multi-GPU tokenization")
    stage2.add_argument(
        "--devices",
        nargs="+",
        default=None,
        metavar="CUDA_DEVICE",
        help="Explicit devices such as cuda:0 cuda:1; omit or pass auto for safe auto-selection.",
    )
    stage2.add_argument(
        "--max-gpus",
        type=int,
        default=DEFAULT_MAX_AUTO_GPU_DEVICES,
        help=f"Maximum auto-selected GPUs (default: {DEFAULT_MAX_AUTO_GPU_DEVICES}).",
    )
    stage2.add_argument(
        "--token-worker-memory-gb",
        type=float,
        default=None,
        help=(
            "Conservative memory estimate per Stage 2 GPU worker "
            f"(default: config or {DEFAULT_TOKEN_WORKER_MEMORY_GB:g} GiB)."
        ),
    )
    stage2.add_argument("--max-samples", type=int, default=None, help="Stage 2 samples per session.")
    stage2.add_argument("--max-sessions", type=int, default=None, help="Stage 2 session limit.")
    stage2.add_argument(
        "--overwrite-token-output",
        action="store_true",
        help="Replace token_root once before Stage 2 fan-out.",
    )

    release = parser.add_argument_group("Stage 3–5 release generation and validation")
    release.add_argument("--allow-placeholder-readme", action="store_true")
    release.add_argument("--strict-validation", action="store_true")
    release.add_argument("--check-event-duration-semantics", action="store_true")

    reconstruction = parser.add_argument_group("Stage 6 reconstruction")
    reconstruction.add_argument("--max-reconstruction-samples", type=int, default=None)
    reconstruction.add_argument("--max-reconstruction-sessions", type=int, default=None)
    reconstruction.add_argument(
        "--reconstruction-device",
        default=None,
        help="Single Stage 6 device; defaults to runtime.device from TOML.",
    )
    reconstruction.add_argument(
        "--max-reconstruction-chunk-batch-size",
        type=int,
        default=None,
        help="Stage 6 EEG/MEG reconstruction micro-batch size.",
    )
    reconstruction.add_argument("--min-mean-pearson", type=float, default=None)

    args = parser.parse_args(argv)
    stage_indices = [STAGE_ORDER.index(stage) for stage in args.stages]
    if len(stage_indices) != len(set(stage_indices)):
        parser.error("--stages cannot contain duplicates")
    if stage_indices != sorted(stage_indices):
        parser.error(
            "--stages must follow canonical order: " + " ".join(STAGE_ORDER)
        )
    if args.h5_workers is not None and args.h5_workers < 0:
        parser.error("--h5-workers must be >= 0")
    if args.max_gpus <= 0:
        parser.error("--max-gpus must be positive")
    if args.token_worker_memory_gb is not None and args.token_worker_memory_gb <= 0:
        parser.error("--token-worker-memory-gb must be positive")
    positive_fields = [
        "max_samples",
        "max_sessions",
        "max_reconstruction_samples",
        "max_reconstruction_sessions",
        "max_reconstruction_chunk_batch_size",
    ]
    for field in positive_fields:
        value = getattr(args, field)
        if value is not None and value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def _pipeline_report(
    *,
    status: str,
    config_path: Path,
    report_root: Path,
    args: argparse.Namespace,
    stages: list[dict],
    started: float,
) -> dict:
    return make_report(
        "pipeline_multi_gpu",
        status=status,
        config=str(config_path),
        report_root=str(report_root),
        strict_stage_barriers=True,
        stage2_one_worker_per_gpu=True,
        smoke_test=bool(args.smoke_test),
        overwrite_tar_output=bool(args.overwrite_tar_output),
        overwrite_token_output=bool(args.overwrite_token_output),
        requested_devices=list(args.devices or ["auto"]),
        requested_stages=list(args.stages),
        stages=stages,
        duration_seconds=round(time.time() - started, 3),
    )


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    pipeline_config = load_config(config_path)
    report_root = Path(
        args.report_root
        or cfg_path(pipeline_config, "paths.report_root", SCRIPT_DIR / "reports")
    ).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    stage_results: list[dict] = []
    for stage in args.stages:
        command, child_report_path = build_stage_command(
            args,
            stage,
            config_path,
            report_root,
        )
        print(f"\n=== {STAGE_TITLES[stage]} ===")
        print(shlex.join(command))
        stage_started = time.time()
        report_signature_before = None
        returncode = 0
        error = None
        if args.dry_run:
            try:
                prepare_stage_execution(
                    args,
                    stage,
                    pipeline_config,
                    config_path,
                    report_root,
                    child_report_path,
                    dry_run=True,
                )
            except Exception as exc:
                error = f"Dry-run blocker: {type(exc).__name__}: {exc}"
                print(f"WARNING: {error}", file=sys.stderr)
        else:
            try:
                prepare_stage_execution(
                    args,
                    stage,
                    pipeline_config,
                    config_path,
                    report_root,
                    child_report_path,
                )
            except Exception as exc:
                returncode = 1
                error = f"Stage preparation failed: {type(exc).__name__}: {exc}"
            if returncode == 0:
                try:
                    completed = subprocess.run(command, cwd=SCRIPT_DIR, check=True)
                    returncode = int(completed.returncode)
                except subprocess.CalledProcessError as exc:
                    returncode = int(exc.returncode)
                    error = f"Stage command exited with code {returncode}"
                except Exception as exc:
                    returncode = 1
                    error = f"{type(exc).__name__}: {exc}"

        report_signature_after = _file_signature(child_report_path)
        child_report = (
            _load_json(child_report_path)
            if not args.dry_run and report_signature_after != report_signature_before
            else {}
        )
        stage_result = {
            "stage": stage,
            "title": STAGE_TITLES[stage],
            "status": (
                "dry_run_blocked" if args.dry_run and error
                else "dry_run" if args.dry_run
                else "ok" if returncode == 0
                else "failed"
            ),
            "command": command,
            "returncode": returncode,
            "duration_seconds": round(time.time() - stage_started, 3),
        }
        if (
            not args.dry_run
            and child_report_path is not None
            and report_signature_after is not None
            and report_signature_after != report_signature_before
        ):
            stage_result["report_path"] = str(child_report_path)
        if child_report:
            stage_result["stage_report"] = child_report
        if error:
            stage_result["error"] = error
        stage_results.append(stage_result)

        if returncode != 0:
            return _pipeline_report(
                status="failed",
                config_path=config_path,
                report_root=report_root,
                args=args,
                stages=stage_results,
                started=started,
            )

    dry_run_blocked = any(
        item.get("status") == "dry_run_blocked" for item in stage_results
    )
    return _pipeline_report(
        status=(
            "dry_run_blocked" if args.dry_run and dry_run_blocked
            else "dry_run" if args.dry_run
            else "ok"
        ),
        config_path=config_path,
        report_root=report_root,
        args=args,
        stages=stage_results,
        started=started,
    )


def build_dry_run_summary(report: dict) -> str:
    lines = [
        "# NeuroToken Multi-GPU Pipeline Dry Run",
        "",
        f"- Status: **{str(report.get('status', '')).upper()}**",
        f"- Config: `{report.get('config', '')}`",
        "- Machine-readable plan: `pipeline_dry_run_report.json`",
        "- No child stage was executed and no child stage report was generated.",
        "",
        "## Planned stages",
        "",
        "| Stage | Status |",
        "| --- | --- |",
    ]
    blockers = []
    for stage in report.get("stages", []):
        lines.append(f"| {stage.get('stage', '')} | {stage.get('status', '')} |")
        if stage.get("error"):
            blockers.append((stage.get("stage", ""), stage["error"]))
    if blockers:
        lines += ["", "## Blockers", ""]
        lines.extend(f"- **{stage}**: {message}" for stage, message in blockers)
    lines += [
        "",
        "Run again with the required overwrite flags or corrected paths before execution.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = run(args)
    report_root = Path(report["report_root"])
    if args.dry_run:
        pipeline_report_path = report_root / "pipeline_dry_run_report.json"
        summary_path = report_root / "pipeline_dry_run_summary.md"
        write_report(pipeline_report_path, report)
        summary_path.write_text(
            build_dry_run_summary(report),
            encoding="utf-8",
        )
    else:
        pipeline_report_path = report_root / "pipeline_report.json"
        write_report(pipeline_report_path, report)
        summary_path = write_human_summary(report_root, report)
    print(f"\nPipeline status: {report['status']}")
    print(f"Pipeline report: {pipeline_report_path}")
    print(f"Human summary: {summary_path}")
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
