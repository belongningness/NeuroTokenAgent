"""Pipeline configuration selection and natural-language request parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent_pipeline.pipeline_models import PipelineRequest
from agent_pipeline.pipeline_stages import STAGE_ORDER


CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def available_configs() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    return sorted(str(path.relative_to(root)) for path in CONFIGS_DIR.glob("*.toml"))


def config_modality(path: str) -> str:
    try:
        from agent_pipeline.config import load_config

        return str(load_config(path).get("dataset", {}).get("modality", "")).strip().lower()
    except Exception:
        return ""


def select_config(user_request: str, default_config: str) -> str:
    text = user_request.lower()
    configs = available_configs()
    if "eeg" in text:
        for config in configs:
            if "eeg" in config.lower():
                return config
    if "meg" in text or "camcan" in text:
        for config in configs:
            if "meg" in config.lower():
                return config
    return default_config


_REMOVAL_TERMS = ("remove", "omit", "skip", "drop", "exclude", "delete", "去掉", "移除", "删除", "不要", "跳过", "取消")


def explicitly_removed_stages(instruction: str) -> list[str]:
    """Return stage names named in an explicit natural-language removal request."""
    text = instruction.strip().lower()
    if not text or not any(term in text for term in _REMOVAL_TERMS):
        return []
    removed: list[str] = []
    for stage in STAGE_ORDER:
        if re.search(rf"(?<![a-z0-9_]){re.escape(stage)}(?![a-z0-9_])", text):
            removed.append(stage)
    return removed


def refine_request(
    llm,
    current_plan: PipelineRequest,
    instruction: str,
) -> PipelineRequest:
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("A natural-language plan refinement instruction is required.")
    configs = available_configs()
    config_hints = "\n".join(
        f"- {config} (modality={config_modality(config)})" for config in configs
    )
    parser = llm.with_structured_output(PipelineRequest, method="function_calling")
    prompt = f"""
Revise an existing NeuroToken pipeline plan according to the human instruction.
Return the complete revised plan, not a patch.

Available stages, in order:
{list(STAGE_ORDER)}

Available config files:
{config_hints}

Rules:
- Preserve every existing value unless the instruction explicitly changes it or a stage dependency requires a change.
- Preserve exact input and output paths and all linked execution fields unless the instruction explicitly replaces them.
- Keep execution_strategy=linked when refining a linked plan unless the human explicitly asks to remove the validation-before-full flow.
- Keep stages in the available stage order and do not invent stages.
- Do not invent config paths; config_path must be one of the available config files.
- Apply the human instruction precisely, including stage selection, smoke-test limits, validation strictness, overwrite behavior, adapter selection, output location, and config overrides.
- Return a complete PipelineRequest suitable for another human approval round.

Current plan:
{json.dumps(current_plan.model_dump(mode="json"), ensure_ascii=False, indent=2)}

Human refinement instruction:
{instruction}
""".strip()
    refined = parser.invoke(prompt)
    if refined.config_path not in configs:
        refined.config_path = current_plan.config_path
    removed = set(explicitly_removed_stages(instruction))
    if removed:
        refined.stages = [stage for stage in refined.stages if stage not in removed]
    return refined


def parse_request(llm, user_request: str, default_config: str) -> PipelineRequest:
    configs = available_configs()
    config_hints = "\n".join(
        f"- {config} (modality={config_modality(config)})" for config in configs
    )
    selected_default = select_config(user_request, default_config)
    parser = llm.with_structured_output(PipelineRequest, method="function_calling")
    prompt = f"""
You convert a user's natural-language request into NeuroToken pipeline parameters.

Available stages, in order:
{list(STAGE_ORDER)}

Available config files:
{config_hints}

Rules:
- If the user asks to inspect/check inputs, use ["preflight"].
- If the user asks to convert/package H5 to tar, include "preflight" and "package".
- If the user asks full workflow, use all stages.
- If the user provides an exact .h5 path, set target_h5 to that path and leave input_root null.
- If the user provides an input directory rather than an H5 file, set input_root to that directory and leave target_h5 null.
- If the user asks to validate one H5 first and then run the whole input directory, set execution_strategy="linked", preserve the directory as full_input_root, preserve an explicit H5 as validation_target_h5, and use the linked phase defaults. Do not confuse this with an ordinary one-file smoke test.
- If linked execution is requested with only a directory, set full_input_root to that directory; the server will deterministically select and persist the first H5 during model validation.
- If the user says test/smoke/small/one file but does not ask for a later full run, set max_files=1 when appropriate and smoke_test=true for downstream tokenize/reconstruct stages.
- If the user mentions an output/result/save/report directory, set report_root to that exact directory. Treat report_root as the output root; tar, neurotoken, reports, and generated config directories are derived under it.
- If the user asks to overwrite existing token outputs or rerun tokenization from scratch, set overwrite_token_output=true.
- If the user asks for strict validation, set strict_validation=true.
- Map resting-state/rest data to config_overrides.chunking.strategy="rest".
- Map task/event-based data to config_overrides.chunking.strategy="event" only when the user explicitly requests event-based splitting; do not infer it solely from the presence of events.
- If the user says placeholder README is acceptable, set allow_placeholder_readme=true.
- Use dry_run=true only when the user explicitly asks not to execute.
- Select config_path by matching the user's data type to the available config files above. If the user says EEG or eeg, use the eeg config. If the user says MEG or CamCAN, use the meg config. When no data type is mentioned, use: {selected_default}
- Do not invent config_path values outside the available config list.
- Do not invent paths other than config_path, target_h5, or report_root unless the user explicitly provides one.

User request:
{user_request}
""".strip()
    request = parser.invoke(prompt)
    if request.config_path not in configs:
        request.config_path = selected_default
    return request
