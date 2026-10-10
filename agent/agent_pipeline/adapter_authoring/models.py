"""Small data models for adapter authoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ADAPTER_CODING_PROTOCOL_VERSION = "1.0"

AdapterCodingStatus = Literal[
    "ready_for_coding",
    "coding",
    "validating",
    "awaiting_patch_approval",
    "needs_revision",
    "applied",
    "rejected",
    "cancelled",
    "failed",
]


class AdapterCodingDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    modality: str
    notes: str = ""


class AdapterCodingHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = ADAPTER_CODING_PROTOCOL_VERSION
    task_id: str
    thread_id: str
    revision: int = Field(ge=1)
    status: AdapterCodingStatus
    adapter: AdapterCodingDescriptor
    allowed_paths: list[str]
    instruction: str
    codex_thread_id: str | None = None
    created_at: str


class AdapterCandidateSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = ADAPTER_CODING_PROTOCOL_VERSION
    action: Literal["submit_candidate"] = "submit_candidate"
    task_id: str
    thread_id: str
    codex_thread_id: str
    replaces_codex_thread_id: str | None = None
    revision: int = Field(ge=1)


class AdapterPatchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = ADAPTER_CODING_PROTOCOL_VERSION
    action: Literal["apply", "apply_and_commit", "ask", "reject", "refine"]
    task_id: str
    revision: int = Field(ge=1)
    message: str | None = None
    commit_message: str | None = None


class AdapterIntentReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["error", "warning"]
    title: str
    evidence: str
    required_change: str


class AdapterIntentReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["pass", "needs_revision", "blocked_by_scope"]
    summary: str
    findings: list[AdapterIntentReviewFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def findings_match_verdict(self) -> "AdapterIntentReview":
        has_errors = any(finding.severity == "error" for finding in self.findings)
        if self.verdict == "pass" and has_errors:
            raise ValueError("A passing adapter intent review cannot contain error findings.")
        if self.verdict in {"needs_revision", "blocked_by_scope"} and not has_errors:
            raise ValueError(f"A {self.verdict} adapter intent review requires at least one error finding.")
        return self


@dataclass
class AdapterSpec:
    name: str
    source_path: Path
    modality: str = "meg"
    notes: str = ""


@dataclass
class H5NodeInfo:
    path: str
    kind: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class CodingTask:
    instruction: str
    worktree: Path
    allowed_paths: list[str]


@dataclass
class CodingAgentResult:
    summary: str
    diff: str


@dataclass
class AdapterAuthoringResult:
    spec: AdapterSpec
    h5_summary: str
    coding_result: CodingAgentResult
