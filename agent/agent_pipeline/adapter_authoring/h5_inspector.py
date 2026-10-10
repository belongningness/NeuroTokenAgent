"""Read-only HDF5 structure inspection for adapter prompts."""

from __future__ import annotations

from pathlib import Path

from agent_pipeline.adapter_authoring.models import H5NodeInfo


def _attrs(obj) -> dict[str, str]:
    return {str(k): str(v) for k, v in obj.attrs.items()}


def inspect_h5(path: str | Path, max_nodes: int = 80) -> list[H5NodeInfo]:
    import h5py

    nodes: list[H5NodeInfo] = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        def visit(name: str, obj) -> None:
            if len(nodes) >= max_nodes:
                return
            full = "/" + name if name else "/"
            if isinstance(obj, h5py.Dataset):
                nodes.append(H5NodeInfo(full, "dataset", tuple(obj.shape), str(obj.dtype), _attrs(obj)))
            else:
                nodes.append(H5NodeInfo(full, "group", attrs=_attrs(obj)))
        visit("", h5)
        h5.visititems(visit)
    return nodes


def format_h5_summary(nodes: list[H5NodeInfo]) -> str:
    lines = []
    for node in nodes:
        if node.kind == "dataset":
            lines.append(f"- {node.path}: dataset shape={node.shape} dtype={node.dtype} attrs={node.attrs}")
        else:
            lines.append(f"- {node.path}: group attrs={node.attrs}")
    return "\n".join(lines)
