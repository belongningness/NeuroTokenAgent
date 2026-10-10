"""Conservative resource detection for the single-dataset parallel pipeline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


GIB = 1024 ** 3
DEFAULT_H5_MEMORY_MULTIPLIER = 3.0
DEFAULT_TOKEN_WORKER_MEMORY_GB = 16.0
DEFAULT_MAX_AUTO_H5_WORKERS = 16
DEFAULT_MAX_AUTO_GPU_DEVICES = 64


@dataclass(frozen=True)
class HostResources:
    visible_cpus: int
    available_memory_bytes: Optional[int]


@dataclass(frozen=True)
class CudaDeviceResources:
    index: int
    name: str
    free_memory_bytes: int
    total_memory_bytes: int

    @property
    def device(self) -> str:
        return f"cuda:{self.index}"


@dataclass
class CudaResourceDetection:
    devices: list[CudaDeviceResources]
    visible_device_indices: list[int]
    query_errors: list[dict] = field(default_factory=list)


@dataclass
class H5WorkerPlan:
    workers: int
    automatic: bool
    source_count: int
    visible_cpus: int
    available_memory_bytes: Optional[int]
    largest_source_bytes: int
    estimated_worker_memory_bytes: int
    max_auto_workers: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiGpuPlan:
    devices: list[str]
    automatic: bool
    session_count: int
    estimated_worker_memory_bytes: int
    max_auto_devices: int
    detected_cuda_devices: list[dict]
    cuda_query_errors: list[dict] = field(default_factory=list)
    skipped_devices: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def workers(self) -> int:
        return len(self.devices)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["workers"] = self.workers
        return result


def _detect_cgroup_available_memory() -> Optional[int]:
    candidates = [
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ]
    for limit_path, usage_path in candidates:
        try:
            limit_text = limit_path.read_text(encoding="utf-8").strip()
            if limit_text == "max":
                continue
            limit = int(limit_text)
            usage = int(usage_path.read_text(encoding="utf-8").strip())
            if limit > 0:
                return max(0, limit - usage)
        except (OSError, ValueError):
            continue
    return None


def detect_host_resources() -> HostResources:
    """Return CPU affinity and currently available host/cgroup memory."""
    try:
        visible_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        visible_cpus = os.cpu_count() or 1

    available_memory = None
    meminfo_path = Path("/proc/meminfo")
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_memory = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass

    if available_memory is None:
        try:
            available_memory = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            pass
    cgroup_available = _detect_cgroup_available_memory()
    if cgroup_available is not None:
        available_memory = (
            cgroup_available
            if available_memory is None
            else min(available_memory, cgroup_available)
        )
    return HostResources(max(1, int(visible_cpus)), available_memory)


def detect_cuda_resource_snapshot() -> CudaResourceDetection:
    """Inspect visible CUDA devices and retain per-device telemetry failures."""
    try:
        import torch
    except Exception as exc:
        return CudaResourceDetection([], [], [{
            "device": None,
            "phase": "torch_import",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }])
    try:
        if not torch.cuda.is_available():
            return CudaResourceDetection([], [])
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        return CudaResourceDetection([], [], [{
            "device": None,
            "phase": "cuda_enumeration",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }])

    visible_indices = list(range(device_count))
    devices = []
    query_errors = []
    for index in visible_indices:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            name = str(torch.cuda.get_device_name(index))
        except Exception as exc:
            query_errors.append({
                "device": f"cuda:{index}",
                "phase": "resource_query",
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            continue
        devices.append(CudaDeviceResources(
            index=index,
            name=name,
            free_memory_bytes=int(free_bytes),
            total_memory_bytes=int(total_bytes),
        ))
    return CudaResourceDetection(devices, visible_indices, query_errors)


def estimate_largest_h5_source_bytes(paths: Iterable[Path], signal_key: str) -> int:
    """Estimate the largest uncompressed source payload among the H5 files."""
    largest = 0
    for value in paths:
        path = Path(value)
        try:
            largest = max(largest, int(path.stat().st_size))
        except OSError:
            continue
        try:
            import h5py
            with h5py.File(path, "r") as handle:
                if signal_key in handle:
                    dataset = handle[signal_key]
                    largest = max(largest, int(dataset.size) * int(dataset.dtype.itemsize))
        except Exception:
            continue
    return largest


def plan_h5_workers(
    source_paths: Iterable[Path],
    requested_workers: Optional[int],
    *,
    signal_key: str = "eeg",
    host: Optional[HostResources] = None,
    memory_multiplier: float = DEFAULT_H5_MEMORY_MULTIPLIER,
    max_auto_workers: int = DEFAULT_MAX_AUTO_H5_WORKERS,
) -> H5WorkerPlan:
    """Choose an H5 worker count from CPU, RAM and source-file estimates."""
    paths = [Path(path) for path in source_paths]
    host = host or detect_host_resources()
    if memory_multiplier <= 0:
        raise ValueError("memory_multiplier must be positive")
    if max_auto_workers <= 0:
        raise ValueError("max_auto_workers must be positive")

    largest_source = estimate_largest_h5_source_bytes(paths, signal_key)
    estimated_worker_memory = max(GIB, int(largest_source * memory_multiplier))
    cpu_limit = max(1, host.visible_cpus - 2) if host.visible_cpus >= 4 else host.visible_cpus
    memory_limit = max_auto_workers
    if host.available_memory_bytes is not None:
        memory_budget = int(host.available_memory_bytes * 0.60)
        memory_limit = max(1, memory_budget // estimated_worker_memory)
    task_limit = max(1, len(paths))
    safe_limit = max(1, min(cpu_limit, memory_limit, task_limit, max_auto_workers))

    automatic = requested_workers in (None, 0)
    workers = safe_limit if automatic else int(requested_workers)
    if workers <= 0:
        raise ValueError("requested H5 workers must be positive or 0/None for auto")

    warnings = []
    if not automatic and workers > safe_limit:
        warnings.append(
            f"Explicit H5 worker count {workers} exceeds the conservative detected limit {safe_limit}."
        )
    return H5WorkerPlan(
        workers=workers,
        automatic=automatic,
        source_count=len(paths),
        visible_cpus=host.visible_cpus,
        available_memory_bytes=host.available_memory_bytes,
        largest_source_bytes=largest_source,
        estimated_worker_memory_bytes=estimated_worker_memory,
        max_auto_workers=max_auto_workers,
        warnings=warnings,
    )


def _cuda_index(device: str) -> Optional[int]:
    value = str(device).strip().lower()
    if value == "cuda":
        return 0
    if value.startswith("cuda:"):
        try:
            return int(value.split(":", 1)[1])
        except ValueError:
            return None
    return None


def plan_multi_gpu_devices(
    requested_devices: Optional[Iterable[str]],
    session_count: int,
    *,
    cuda_devices: Optional[list[CudaDeviceResources]] = None,
    estimated_worker_memory_gb: float = DEFAULT_TOKEN_WORKER_MEMORY_GB,
    max_auto_devices: int = DEFAULT_MAX_AUTO_GPU_DEVICES,
) -> MultiGpuPlan:
    """Select CUDA devices for exactly one persistent model worker per device."""
    if estimated_worker_memory_gb <= 0:
        raise ValueError("estimated_worker_memory_gb must be positive")
    if max_auto_devices <= 0:
        raise ValueError("max_auto_devices must be positive")
    if session_count < 0:
        raise ValueError("session_count must be non-negative")

    if cuda_devices is None:
        detection = detect_cuda_resource_snapshot()
        detected = detection.devices
        visible_indices = set(detection.visible_device_indices)
        query_errors = list(detection.query_errors)
    else:
        detected = list(cuda_devices)
        visible_indices = {item.index for item in detected}
        query_errors = []
    detected_by_index = {item.index: item for item in detected}
    detected_dicts = [asdict(item) | {"device": item.device} for item in detected]
    estimated_worker_memory = int(estimated_worker_memory_gb * GIB)
    values = [] if requested_devices is None else [
        str(value).strip().lower() for value in requested_devices if str(value).strip()
    ]
    automatic = not values or values == ["auto"]
    if "auto" in values and not automatic:
        raise ValueError("'auto' cannot be combined with explicit CUDA devices")

    warnings: list[str] = []
    skipped: list[dict] = []
    for error in query_errors:
        if error.get("device"):
            skipped.append({
                "device": error["device"],
                "reason": "resource_query_failed",
                "error_type": error.get("error_type"),
                "message": error.get("message"),
            })
    if query_errors:
        failed_labels = [
            str(error.get("device") or error.get("phase", "CUDA telemetry"))
            for error in query_errors
        ]
        warnings.append(
            "Could not read CUDA resource telemetry for: " + ", ".join(failed_labels) + "."
        )
    if session_count == 0:
        warnings.append("No session tar files are available, so no GPU workers were selected.")
        return MultiGpuPlan(
            devices=[],
            automatic=automatic,
            session_count=0,
            estimated_worker_memory_bytes=estimated_worker_memory,
            max_auto_devices=max_auto_devices,
            detected_cuda_devices=detected_dicts,
            cuda_query_errors=query_errors,
            skipped_devices=skipped,
            warnings=warnings,
        )

    if automatic:
        eligible: list[CudaDeviceResources] = []
        for item in detected:
            reserve = max(2 * GIB, int(item.total_memory_bytes * 0.10))
            usable = max(0, item.free_memory_bytes - reserve)
            if usable >= estimated_worker_memory:
                eligible.append(item)
            else:
                skipped.append({
                    "device": item.device,
                    "reason": "insufficient_free_memory",
                    "free_memory_bytes": item.free_memory_bytes,
                    "reserved_memory_bytes": reserve,
                    "required_worker_memory_bytes": estimated_worker_memory,
                })
        limit = min(int(session_count), max_auto_devices)
        selected = sorted(
            eligible,
            key=lambda item: (-item.free_memory_bytes, item.index),
        )[:limit]
        devices = [item.device for item in selected]
        if not visible_indices:
            warnings.append("No visible CUDA devices were detected; multi-GPU tokenization cannot start.")
        elif not detected:
            warnings.append(
                "CUDA devices are visible, but none has readable resource telemetry for safe auto-selection."
            )
        elif not devices:
            warnings.append(
                f"No visible CUDA device has the conservative {estimated_worker_memory_gb:g} GiB "
                "worker budget after accelerator-memory reservation."
            )
        elif len(eligible) > limit:
            warnings.append(
                f"Selected {limit} of {len(eligible)} eligible CUDA devices because the "
                "session count or automatic device cap is smaller."
            )
    else:
        devices = []
        seen: set[str] = set()
        for value in values:
            index = _cuda_index(value)
            if index is None:
                raise ValueError(f"Expected a CUDA device such as cuda:0, got {value!r}")
            canonical = f"cuda:{index}"
            if canonical in seen:
                continue
            seen.add(canonical)
            item = detected_by_index.get(index)
            if item is None:
                if index not in visible_indices:
                    raise ValueError(
                        f"Explicit device {canonical} is not among the visible CUDA devices"
                    )
                warnings.append(
                    f"Explicit device {canonical} is visible, but its free memory could not "
                    "be queried; retaining it without a memory-safety check."
                )
                devices.append(canonical)
                continue
            reserve = max(2 * GIB, int(item.total_memory_bytes * 0.10))
            usable = max(0, item.free_memory_bytes - reserve)
            if usable < estimated_worker_memory:
                warnings.append(
                    f"Explicit device {canonical} has less than the conservative "
                    f"{estimated_worker_memory_gb:g} GiB worker estimate after reservation."
                )
            devices.append(canonical)
        if len(devices) > session_count:
            warnings.append(
                f"Selected only {session_count} of {len(devices)} explicit CUDA devices "
                "because there are fewer sessions than devices."
            )
            devices = devices[:session_count]

    return MultiGpuPlan(
        devices=devices,
        automatic=automatic,
        session_count=int(session_count),
        estimated_worker_memory_bytes=estimated_worker_memory,
        max_auto_devices=max_auto_devices,
        detected_cuda_devices=detected_dicts,
        cuda_query_errors=query_errors,
        skipped_devices=skipped,
        warnings=warnings,
    )
