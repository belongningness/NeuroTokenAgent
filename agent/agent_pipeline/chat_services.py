"""UI-independent conversational services shared by the Agent Server."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agent_pipeline.pipeline_models import PipelineRequest, TextResponse
from agent_pipeline.pipeline_planning import parse_request, select_config
from agent_pipeline.pipeline_stages import resolve_stage_dependencies
from agent_pipeline.report_summaries import compact_report, compact_stage_report


IntentName = Literal[
    "question",
    "plan",
    "execute",
    "report",
    "continue",
    "continue_coding",
    "edit_plan",
    "adapter_authoring",
    "sdk_cleanup",
    "cancel",
    "exit",
]


class UserIntent(BaseModel):
    intent: IntentName
    explanation: str = Field(description="Short reason for the selected intent.")
    adapter_name: str | None = None
    input_path: str | None = None
    output_path: str | None = None
    modality: str | None = None
    stages: list[str] | None = None
    execution_strategy: Literal["single", "linked"] | None = None
    validation_target_h5: str | None = None
    full_input_root: str | None = None
    reconstruction_gate: dict[str, Any] | None = None
    max_files: int | None = None
    smoke_test: bool | None = None
    dry_run: bool | None = None
    overwrite_token_output: bool | None = None
    allow_placeholder_readme: bool | None = None
    strict_validation: bool | None = None
    summary_language: Literal["zh", "en"] | None = None
    config_overrides: dict[str, dict[str, Any]] | None = None
    other_requirements: list[str] = Field(default_factory=list)


class AdapterAuthoringRequest(BaseModel):
    name: str
    source_path: str
    modality: str = "meg"
    notes: str = ""


PIPELINE_CONTEXT = """
NeuroToken stages: preflight checks inputs and configuration; package converts source H5
sessions to tar samples; tokenize creates neural tokens; readme/docs describe the release;
validate checks output structure; reconstruct measures token reconstruction quality.
""".strip()

H5_PATH_PATTERN = re.compile(r"(?P<path>(?:~|/)[^\s\"'<>]+?\.(?:h5|hdf5))", re.IGNORECASE)
ADAPTER_AUTHORING_TOKENS = (
    "编写代码", "写代码", "代码实现", "coding", "implement", "write code", "开发适配器",
    "编写 adapter", "写 adapter", "create adapter", "build adapter", "implement adapter",
)
EXISTING_ADAPTER_USE_PATTERN = re.compile(
    r"(?:使用|采用|use|using|with)\s+[`'\"]?(?P<name>[a-z0-9_]+_adapter(?:_\d+)?)",
    re.IGNORECASE,
)
OUTPUT_PATH_PATTERN = re.compile(
    r"(?:输出到|保存到|写入|output\s+to|save\s+to)\s*[`'\"]?(?P<path>(?:~|/)[^\s`'\"<>]+)",
    re.IGNORECASE,
)
ADAPTER_AUTHORING_PATTERN = re.compile(
    r"(?:编写|写|开发|实现|create|build|implement|write)(?:\s|[a-z]+|[\u4e00-\u9fff]){0,12}adapter",
    re.IGNORECASE,
)


def requests_adapter_authoring(user_input: str) -> bool:
    lower = user_input.lower()
    return any(token in lower for token in ADAPTER_AUTHORING_TOKENS) or bool(ADAPTER_AUTHORING_PATTERN.search(lower))


def explicit_h5_path(user_input: str) -> str | None:
    match = H5_PATH_PATTERN.search(user_input)
    return match.group("path").rstrip(".,;:!?，。；：！？、") if match else None


def explicit_modality(user_input: str) -> str | None:
    lower = user_input.lower()
    if re.search(r"(?<![a-z0-9])eeg(?![a-z0-9])|脑电", lower):
        return "eeg"
    if re.search(r"(?<![a-z0-9])meg(?![a-z0-9])|脑磁", lower):
        return "meg"
    return None


def explicit_adapter_request(user_input: str, existing: list[str]) -> AdapterAuthoringRequest | None:
    source_path = explicit_h5_path(user_input)
    text = user_input.strip()
    lower = text.lower()
    modality = explicit_modality(lower)
    if not source_path or not modality or not requests_adapter_authoring(lower):
        return None
    stem = Path(source_path).stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    modality_suffix = f"_{modality}"
    suffix = "_adapter" if name.endswith(modality_suffix) else f"{modality_suffix}_adapter"
    if not name.endswith("_adapter"):
        name = f"{name}{suffix}"
    base = name or f"custom_{modality}_adapter"
    counter = 2
    while name in existing:
        name = f"{base}_{counter}"
        counter += 1
    return AdapterAuthoringRequest(name=name, source_path=source_path, modality=modality, notes=text)


def classify_intent(llm, state: dict[str, Any], user_input: str) -> UserIntent:
    text = user_input.strip().lower()
    source_path = explicit_h5_path(user_input)
    existing_adapter = EXISTING_ADAPTER_USE_PATTERN.search(text)
    if source_path and existing_adapter:
        output = OUTPUT_PATH_PATTERN.search(user_input)
        return UserIntent(
            intent="execute",
            explanation="Run the pipeline with an explicitly selected existing adapter.",
            adapter_name=existing_adapter.group("name"),
            input_path=source_path,
            output_path=output.group("path").rstrip(".,;:!?，。；：！？、") if output else None,
            modality=explicit_modality(text),
        )
    coding_mode = state.get("mode") in {"ready_for_coding", "coding", "adapter_question", "adapter_revision_required"}
    if coding_mode and any(token in text for token in ("编写代码", "写代码", "继续编写", "继续 coding", "continue coding", "implement", "write code")):
        return UserIntent(intent="continue_coding", explanation="Continue the existing adapter coding task.")
    if source_path and requests_adapter_authoring(text):
        modality = explicit_modality(text)
        output = OUTPUT_PATH_PATTERN.search(user_input)
        return UserIntent(
            intent="adapter_authoring",
            explanation="Explicit HDF5 coding request.",
            input_path=source_path,
            output_path=output.group("path").rstrip(".,;:!?，。；：！？、") if output else None,
            modality=modality,
        )
    classifier = llm.with_structured_output(UserIntent, method="function_calling")
    prompt = f"""
Classify the request and extract explicit NeuroToken requirements.
Use null for unspecified values. Preserve paths exactly.
    Valid intents: question, plan, execute, report, continue, continue_coding, edit_plan,
  adapter_authoring, sdk_cleanup, cancel, exit. Use continue_coding when the user asks to write, implement, code, or continue coding in an existing adapter coding task. Use sdk_cleanup when the user asks to remove or rollback SDK-generated adapter code.
Valid stages: preflight, package, tokenize, readme, docs, validate, reconstruct.
When the user asks to validate one H5 and then automatically run all files, set execution_strategy=linked, validation_target_h5 to the explicit sample when provided, and full_input_root to the full directory. Keep a plain one-file smoke request as execution_strategy=single.
Map config values into config_overrides grouped by TOML section.

State:
mode={state.get('mode', 'idle')}
current_plan={json.dumps(state.get('current_plan'), ensure_ascii=False)}
pending_stages={state.get('pending_stages', [])}
completed_stages={state.get('completed_stages', [])}
last_adapter={json.dumps(state.get('last_adapter'), ensure_ascii=False)}
requirements={json.dumps(state.get('requirements', {}), ensure_ascii=False)}

User input:
{user_input}
""".strip()
    return classifier.invoke(prompt)


def merge_requirements(
    requirements: dict[str, Any],
    other_requirements: list[str],
    intent: UserIntent,
) -> tuple[dict[str, Any], list[str]]:
    merged = dict(requirements)
    values = intent.model_dump(exclude={"intent", "explanation", "other_requirements"})
    overrides = values.pop("config_overrides", None) or {}
    for key, value in values.items():
        if value is not None:
            merged[key] = value
    if overrides:
        target = merged.setdefault("config_overrides", {})
        for section, section_values in overrides.items():
            target.setdefault(section, {}).update(section_values)
    extra = list(other_requirements)
    for item in intent.other_requirements:
        if item not in extra:
            extra.append(item)
    return merged, extra


def build_pipeline_request(
    llm,
    user_input: str,
    default_config: str,
    requirements: dict[str, Any],
    other_requirements: list[str],
    last_adapter: dict[str, Any] | None = None,
) -> tuple[PipelineRequest, list[str]]:
    context = {
        "requirements": requirements,
        "other_requirements": other_requirements,
        "last_adapter": last_adapter,
    }
    request = parse_request(
        llm,
        f"{user_input}\n\nConversation context:\n{json.dumps(context, ensure_ascii=False)}",
        default_config,
    )
    input_path = requirements.get("input_path")
    if input_path:
        input_path = str(Path(input_path).expanduser())
        if input_path.lower().endswith((".h5", ".hdf5")):
            request.target_h5 = input_path
            request.input_root = None
        else:
            request.input_root = input_path
            request.target_h5 = None
    if request.target_h5:
        target = Path(request.target_h5).expanduser()
        if not target.exists() and "ses01" in target.name:
            corrected = target.with_name(target.name.replace("ses01", "ses-01"))
            if corrected.exists():
                request.target_h5 = str(corrected)
    field_map = {
        "adapter_name": "source_adapter",
        "output_path": "report_root",
        "execution_strategy": "execution_strategy",
        "validation_target_h5": "validation_target_h5",
        "full_input_root": "full_input_root",
        "reconstruction_gate": "reconstruction_gate",
        "max_files": "max_files",
        "smoke_test": "smoke_test",
        "dry_run": "dry_run",
        "overwrite_token_output": "overwrite_token_output",
        "allow_placeholder_readme": "allow_placeholder_readme",
        "strict_validation": "strict_validation",
        "config_overrides": "config_overrides",
        "summary_language": "summary_language",
    }
    for source, target in field_map.items():
        if requirements.get(source) is not None:
            setattr(request, target, requirements[source])
    if requirements.get("stages"):
        request.stages = requirements["stages"]
    if request.execution_strategy == "linked":
        request.full_input_root = request.full_input_root or request.input_root
        request.validation_target_h5 = request.validation_target_h5 or request.target_h5
        if request.full_input_root:
            request.input_root = None
            request.target_h5 = None
        request = PipelineRequest.model_validate(request.model_dump(mode="python"))
    if requirements.get("modality"):
        request.config_path = select_config(requirements["modality"], request.config_path)
    if last_adapter:
        request.source_adapter = request.source_adapter or last_adapter.get("name")
        request.target_h5 = request.target_h5 or last_adapter.get("source_path")
    if (request.target_h5 or request.input_root) and request.report_root is None:
        label = request.source_adapter or requirements.get("modality") or "pipeline"
        request.report_root = str(Path(__file__).resolve().parents[1] / "outputs" / f"{label}_run")
    added = resolve_stage_dependencies(request)
    return request, added


def answer_question(llm, state: dict[str, Any], user_input: str) -> str:
    responder = llm.with_structured_output(TextResponse)
    prompt = f"""
用中文简洁回答 NeuroToken pipeline 问题。必要时解释技术细节。

Context:
{PIPELINE_CONTEXT}

State:
mode={state.get('mode')}
current_plan={json.dumps(state.get('current_plan'), ensure_ascii=False)}
completed_stages={state.get('completed_stages', [])}
pending_stages={state.get('pending_stages', [])}
last_error={json.dumps(state.get('last_error'), ensure_ascii=False)}
latest_report={json.dumps(compact_report({'status': state.get('mode'), 'stages': state.get('stage_reports', [])}), ensure_ascii=False)}

Question:
{user_input}
""".strip()
    return responder.invoke(prompt).text


def answer_adapter_question(llm, pending: dict[str, Any], question: str) -> str:
    responder = llm.with_structured_output(TextResponse)
    prompt = f"""
用中文回答人类对候选 adapter 实现的技术问题。只解释当前实现和 diff；不要声称代码已经应用。
如果问题实际要求修改，说明可以使用“按要求修改”提交该要求。

Adapter context:
name={pending.get('name')}
source_path={pending.get('source_path')}
modality={pending.get('modality')}
notes={pending.get('notes')}
h5_summary={pending.get('h5_summary')}
coding_summary={pending.get('coding_summary')}
current_diff:
{pending.get('diff')}

Question:
{question}
""".strip()
    return responder.invoke(prompt).text


def _format_preview_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "大小未知"
    size = float(value)
    if size < 1_024:
        return f"{int(size)} B"
    if size < 1_048_576:
        return f"{size / 1_024:.1f} KiB"
    return f"{size / 1_048_576:.1f} MiB"


def _format_preview_number(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "未知"
    rendered = f"{float(value):.{digits}f}"
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def package_sample_summary(report: dict[str, Any]) -> str:
    if report.get("stage") != "package" or not isinstance(report.get("sessions"), list):
        return ""
    segmentation_sections: list[str] = []
    content_sections: list[str] = []
    for index, value in enumerate(report["sessions"][:3]):
        if not isinstance(value, dict):
            continue
        output_dir = value.get("output_dir")
        session_name = Path(output_dir).name if isinstance(output_dir, str) and output_dir else f"Session {index + 1}"
        review = value.get("segmentation_review")
        if isinstance(review, dict):
            lines = [f"### {session_name}"]
            strategy = str(review.get("strategy") or report.get("chunking_strategy") or "unknown")
            lines.append(
                f"- 策略：`{strategy}`；采样率 {_format_preview_number(review.get('sfreq'))} Hz；"
                f"原始记录 {review.get('total_samples', '未知')} samples / {_format_preview_number(review.get('total_duration_seconds'))} s。"
            )
            if strategy == "rest":
                lines.append(
                    f"- 规则：从 sample 0 开始，每 {_format_preview_number(review.get('split_points'), 0)} samples 切一段，"
                    "最后一段保留实际剩余长度；区间均为左闭右开 `[start, end)`。"
                )
            else:
                lines.append(
                    "- 规则：以 event 边界切分从 sample 0 到 recording 末尾的完整时间轴；"
                    "真实 event 段保留既有 `event_*`，其余普通时间段不增加 event 字段或类型标签。"
                )
            lines.append(
                f"- 结果：{review.get('segment_count', '未知')} 段；覆盖率 {_format_preview_number(review.get('coverage_percent'))}%；"
                f"长度 {_format_preview_number(review.get('min_segment_seconds'))}–{_format_preview_number(review.get('max_segment_seconds'))} s。"
            )
            if strategy == "event":
                lines.append(
                    f"- 组成：{review.get('event_segment_count', '未知')} 个 event 段；"
                    f"{review.get('ordinary_segment_count', '未知')} 个无 event metadata 的普通时间段。"
                )
            preview_segments = review.get("preview_segments")
            if isinstance(preview_segments, list):
                for segment in preview_segments:
                    if not isinstance(segment, dict):
                        continue
                    event_label = ""
                    if segment.get("event_type") or segment.get("event_value"):
                        event_label = f"；event={segment.get('event_type', '')}/{segment.get('event_value', '')}"
                    segment_label = "event" if segment.get("event_index") is not None else "segment"
                    lines.append(
                        f"- `#{segment.get('chunk_index', '?')}` {segment_label}："
                        f"samples `[{segment.get('start_sample')}, {segment.get('end_sample')})`；"
                        f"time `[{_format_preview_number(segment.get('start_time'))}, {_format_preview_number(segment.get('end_time'))}) s`；"
                        f"时长 {_format_preview_number(segment.get('duration_seconds'))} s{event_label}。"
                    )
            uncovered = review.get("uncovered_ranges")
            overlaps = review.get("overlaps")
            if isinstance(uncovered, list) and uncovered:
                lines.append(f"- 需人工确认：检测到 {len(uncovered)} 个未覆盖区间（可能是 event 模式预期的首尾范围，也可能是遗漏）。")
            if isinstance(overlaps, list) and overlaps:
                lines.append(f"- 需人工确认：检测到 {len(overlaps)} 个重叠区间。")
            if isinstance(value.get("segmentation_manifest_path"), str):
                lines.append(f"- 完整边界：`{value['segmentation_manifest_path']}`")
            segmentation_sections.append("\n".join(lines))

        entries = value.get("tar_sample")
        metadata = value.get("tar_sample_metadata")
        if not isinstance(entries, list) and not isinstance(metadata, dict):
            continue
        lines = [f"### {session_name}"]
        if isinstance(value.get("tar_path"), str):
            lines.append(f"- tar: `{value['tar_path']}`")
        if isinstance(entries, list):
            for entry in entries[:6]:
                if not isinstance(entry, dict):
                    continue
                lines.append(f"- `{entry.get('name', '未命名 member')}`（{_format_preview_bytes(entry.get('size'))}）")
        if isinstance(metadata, dict):
            member = value.get("tar_sample_metadata_member") or "metadata.json"
            rendered = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
            lines.extend([f"- 样例元数据 `{member}`：", "```json", rendered, "```"])
        content_sections.append("\n".join(lines))
    sections = []
    if segmentation_sections:
        sections.append("## tar 切分审核（边界均为 `[start, end)`）\n\n" + "\n\n".join(segmentation_sections))
    if content_sections:
        sections.append("## tar 内容样例（最多 3 个 session、每个 6 个 member）\n\n" + "\n\n".join(content_sections))
    return "\n\n".join(sections)


def append_package_sample_summary(summary: str, report: dict[str, Any]) -> str:
    preview = package_sample_summary(report)
    return f"{summary.rstrip()}\n\n{preview}" if preview else summary


def deterministic_stage_summary(report: dict[str, Any]) -> str:
    compact = compact_stage_report(report)
    stage = str(compact.get("stage") or "unknown")
    status = str(compact.get("status") or "unknown")
    parts = [f"NeuroToken 的 {stage} 阶段已完成，状态为 {status}。"]
    duration = compact.get("duration_seconds")
    if duration is not None:
        parts.append(f"耗时 {duration} 秒。")
    if compact.get("n_h5_files") is not None:
        parts.append(f"检查了 {compact['n_h5_files']} 个 HDF5 文件。")
    if compact.get("n_sessions") is not None:
        parts.append(f"处理了 {compact['n_sessions']} 个 session。")
    if compact.get("samples") is not None:
        parts.append(f"生成 {compact['samples']} 个样本。")
    if compact.get("report_path"):
        parts.append(f"报告：{compact['report_path']}。")
    if stage == "reconstruct":
        metrics = compact.get("metrics") or {}
        labels = {"pearson": "Pearson", "ssim": "SSIM", "mse": "MSE", "mae": "MAE", "relative_l2": "relative L2"}
        values = [f"{labels[key]}={float(metrics[key]):.6g}" for key in labels if metrics.get(key) is not None]
        if values:
            parts.append(f"实际重建指标：{', '.join(values)}。")
        else:
            parts.append("本次没有成功评估任何重建窗口，因此没有实际重建指标。")
        parts.append("Pearson、SSIM 越接近 1 越好；MSE、MAE、relative L2 越接近 0 越好。")
    return " ".join(parts)


def summarize_stage(llm, report: dict[str, Any]) -> str:
    responder = llm.with_structured_output(TextResponse)
    prompt = f"""
用中文简洁解释刚完成的 NeuroToken 阶段。说明是否正常、关键数字和输出。
reconstruct 必须解释 Pearson、SSIM、MSE、MAE、relative L2：前两者越接近 1 越好，误差越接近 0 越好。

{json.dumps(compact_stage_report(report), ensure_ascii=False, indent=2)}
""".strip()
    try:
        return responder.invoke(prompt).text
    except Exception:
        return deterministic_stage_summary(report)


def parse_adapter_request(llm, user_input: str, existing: list[str]) -> AdapterAuthoringRequest:
    explicit = explicit_adapter_request(user_input, existing)
    if explicit is not None:
        return explicit
    parser = llm.with_structured_output(AdapterAuthoringRequest, method="function_calling")
    prompt = f"""
Extract an adapter authoring request: unique snake_case name, example HDF5 source path,
modality and notes. Do not reuse: {existing}.

User input:
{user_input}
""".strip()
    return parser.invoke(prompt)
