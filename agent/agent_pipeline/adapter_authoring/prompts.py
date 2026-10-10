"""Prompts for adapter authoring coding agents."""

from __future__ import annotations

from pathlib import Path

from agent_pipeline.adapter_authoring.models import AdapterSpec


REVIEW_CONTEXT_PATHS = (
    "agent_pipeline/adapters/base.py",
    "agent_pipeline/adapters/__init__.py",
    "agent_pipeline/adapters/meg_h5.py",
    "agent_pipeline/adapters/eeg_h5.py",
    "agent_pipeline/package.py",
    "agent_pipeline/tools.py",
    "agent_pipeline/dynamic_config.py",
)


KNOWLEDGE_BASE_PATH = Path(__file__).with_name("knowledge_base.md")
TEMPLATE_ADAPTERS = ("agent_pipeline/adapters/meg_h5.py", "agent_pipeline/adapters/eeg_h5.py")


def load_adapter_knowledge() -> str:
    try:
        return KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")[:20_000]
    except OSError:
        return "No adapter knowledge base is available. Rely on the existing adapter templates and explicit HDF5 evidence."


def build_adapter_prompt(spec: AdapterSpec, h5_summary: str) -> str:
    class_name = "".join(part.capitalize() for part in spec.name.replace("-", "_").split("_")) + "Adapter"
    module_name = spec.name.lower().replace("-", "_")
    return f"""
You are editing the NeuroToken agent codebase to add a concise source adapter.

Goal:
- Add adapter `{spec.name}` for `{spec.source_path}`.
- Create `agent_pipeline/adapters/{module_name}.py` with class `{class_name}`.
- Rely on the existing generated-adapter discovery; do not add static registration imports unless discovery cannot load the required class.
- Follow the closest existing adapter template (`agent_pipeline/adapters/meg_h5.py` or `agent_pipeline/adapters/eeg_h5.py`) and return `SourceSession`.
- Keep code short. Do not add broad defensive programming.
- Edit the adapter implementation under `agent_pipeline/adapters/` and create the required root-level `ADAPTER_DESIGN.md`.
- Do not silently select event/rest chunking in the adapter; preserve normalized events and explain the recommended pipeline strategy.
- `ADAPTER_DESIGN.md` is part of the candidate deliverable, not optional documentation; include an explicit template comparison.

Knowledge base file:
- Read `agent_pipeline/adapter_authoring/knowledge_base.md` from the current worktree before editing.
- Use it as repository guidance; do not reproduce the whole document in your response.

Source modality: {spec.modality}
User notes: {spec.notes or "none"}

HDF5 structure:
{h5_summary}
""".strip()


def build_refinement_prompt(spec: AdapterSpec, h5_summary: str, request: str) -> str:
    return f"""
Continue improving the existing candidate adapter for `{spec.name}`.
The current candidate patch has already been applied in this isolated worktree.

Human request:
{request}

Requirements:
- Inspect the current adapter implementation before editing.
- Answer the request by modifying code only when modification is needed.
- Keep the implementation concise and consistent with existing adapters.
- Edit the adapter implementation under `agent_pipeline/adapters/` and update the root-level `ADAPTER_DESIGN.md`.
- Do not revert correct parts of the current candidate.
- Update `ADAPTER_DESIGN.md` with the revised logic, event/rest recommendation, and template differences.

Knowledge base file:
- Read `agent_pipeline/adapter_authoring/knowledge_base.md` from the current worktree before editing.
- Use it as repository guidance; do not reproduce the whole document in your response.

Source: {spec.source_path}
Modality: {spec.modality}
Original notes: {spec.notes or "none"}

HDF5 structure:
{h5_summary}
""".strip()


def load_review_context(worktree: str | Path) -> str:
    root = Path(worktree)
    sections: list[str] = []
    total_chars = 0
    for relative_path in REVIEW_CONTEXT_PATHS:
        path = root / relative_path
        try:
            content = path.read_text(encoding="utf-8")[:20_000]
        except OSError:
            continue
        remaining = 100_000 - total_chars
        if remaining <= 0:
            break
        content = content[:remaining]
        sections.append(f"## {relative_path}\n{content}")
        total_chars += len(content)
    return "\n\n".join(sections)


def build_adapter_review_prompt(
    spec: AdapterSpec,
    h5_summary: str,
    instruction: str,
    design_document: str,
    diff: str,
    shared_context: str,
    *,
    original_user_request: str = "",
    requirements_snapshot: str = "{}",
    effective_config: str = "{}",
    preflight_evidence: str = "{}",
    allowed_paths: list[str] | None = None,
) -> str:
    return f"""
Independently review a NeuroToken adapter candidate for alignment with the human's intent.
This is a read-only semantic gate before human patch approval. Do not assume the adapter file alone controls behavior: trace normalized `SourceSession` values through shared packaging code when the request concerns segmentation or output records.

Verdicts:
- Return `pass` only when there are no error findings.
- Return `needs_revision` when the candidate can satisfy the requirement by changing files within the supplied allowed paths.
- Return `blocked_by_scope` when an explicit requirement is not satisfied and the required fix belongs outside the supplied allowed paths, such as shared packaging or config-generation code. Do not ask the adapter author to claim compliance in documentation or fabricate adapter-level behavior that the shared pipeline controls.

Each blocking finding must use severity `error`, cite evidence from the supplied candidate, effective config, or shared code, and state an actionable required change. Use severity `warning` for material residual risks that do not block the explicit request. Do not reject for style or speculative concerns.

Immutable original human request:
{original_user_request or "<unavailable>"}

Structured requirement snapshot captured when the task was created:
{requirements_snapshot}

Current coding instruction and recorded refinement context:
{instruction}

Candidate allowed paths:
{allowed_paths or []}

Proposed effective config for semantic review:
{effective_config}

Executable candidate preflight evidence generated by the server from the isolated worktree:
{preflight_evidence}

Review boundary rules:
- Treat effective output routing and executable preflight evidence as server-owned review inputs, not as adapter-candidate files.
- Do not require the adapter candidate to edit config-generation or invocation code when the effective config already reflects the captured output requirement.
- A preflight status of `ok` with `source_session_loaded=true` is executable evidence for adapter loading. For event chunking, use `valid_nonempty_clipped_event_intervals` for normalized source events and the `package_*` fields for executable shared-package evidence. `package_ordinary_segment_count` counts recording ranges without `event_*` payload; `package_full_recording_coverage` must be true, while `package_invented_gap_label_count` and `package_fabricated_ordinary_payload_count` must remain zero to preserve the reference metadata contract.
- If server-owned review evidence is missing or failed, report that as a review/infrastructure failure; do not misclassify it as an adapter revision unless the evidence identifies an adapter exception or invalid normalized data.

Adapter identity:
- name: {spec.name}
- source: {spec.source_path}
- modality: {spec.modality}
- original notes: {spec.notes or "none"}

Inspected HDF5 structure:
{h5_summary}

Repository adapter knowledge:
{load_adapter_knowledge()}

Candidate design document:
{design_document or "<missing>"}

Candidate diff:
{diff}

Relevant shared implementation context:
{shared_context}
""".strip()
