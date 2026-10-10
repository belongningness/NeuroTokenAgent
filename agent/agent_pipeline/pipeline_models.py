"""Shared request and structured response models for NeuroToken workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


StageName = Literal["preflight", "package", "tokenize", "readme", "docs", "validate", "reconstruct"]
ExecutionStrategy = Literal["single", "linked"]


class ReconstructionGateThresholds(BaseModel):
    min_pearson: float | None = Field(default=None, ge=-1.0, le=1.0)
    min_ssim: float | None = Field(default=None, ge=-1.0, le=1.0)
    max_mse: float | None = Field(default=None, ge=0.0)
    max_mae: float | None = Field(default=None, ge=0.0)
    max_relative_l2: float | None = Field(default=None, ge=0.0)
    min_evaluated_sessions: int = Field(default=1, ge=1)
    min_evaluated_windows: int = Field(default=1, ge=1)


class PipelineRequest(BaseModel):
    config_path: str = Field(description="TOML config path for the pipeline.")
    stages: list[StageName]
    execution_strategy: ExecutionStrategy = "single"
    validation_target_h5: str | None = None
    full_input_root: str | None = None
    validation_stages: list[StageName] = Field(
        default_factory=lambda: ["preflight", "package", "tokenize", "reconstruct"]
    )
    full_stages: list[StageName] = Field(
        default_factory=lambda: ["preflight", "package", "tokenize"]
    )
    reconstruction_gate: ReconstructionGateThresholds = Field(
        default_factory=ReconstructionGateThresholds
    )
    reconstruction_gate_min_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Compatibility shorthand for reconstruction_gate.min_pearson.",
    )
    max_files: int | None = Field(
        default=None,
        description="Maximum number of H5 files to process when target_h5 is not provided.",
    )
    target_h5: str | None = Field(default=None, description="Exact H5 file path to process.")
    input_root: str | None = Field(
        default=None,
        description="Input directory to scan when no exact H5 file is selected.",
    )
    smoke_test: bool = False
    dry_run: bool = False
    overwrite_token_output: bool = False
    allow_placeholder_readme: bool = False
    strict_validation: bool = False
    report_root: str | None = None
    source_adapter: str | None = Field(
        default=None,
        description="Adapter name overriding source.adapter in the runtime config.",
    )
    config_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional runtime TOML section/key overrides.",
    )
    summary_language: Literal["zh", "en"] = "zh"

    @model_validator(mode="after")
    def validate_linked_strategy(self) -> "PipelineRequest":
        if self.reconstruction_gate_min_threshold is not None:
            self.reconstruction_gate.min_pearson = self.reconstruction_gate_min_threshold
        if self.execution_strategy == "single":
            return self
        if not self.full_input_root:
            raise ValueError("linked execution requires full_input_root")
        root = Path(self.full_input_root).expanduser().resolve()
        if self.validation_target_h5:
            target = Path(self.validation_target_h5).expanduser().resolve()
            if target != root and root not in target.parents:
                raise ValueError("validation_target_h5 must be within full_input_root")
        if "reconstruct" not in self.validation_stages:
            raise ValueError("linked validation_stages must include reconstruct")
        if not {"package", "tokenize"}.issubset(self.full_stages):
            raise ValueError("linked full_stages must include package and tokenize")
        return self

    def request_for_phase(self, phase: Literal["single", "validation", "full"]) -> "PipelineRequest":
        if phase == "single" or self.execution_strategy == "single":
            return self
        if phase == "validation":
            return self.model_copy(update={
                "stages": list(self.validation_stages),
                "target_h5": self.validation_target_h5,
                "input_root": None,
                "max_files": None,
                "smoke_test": True,
            })
        return self.model_copy(update={
            "stages": list(self.full_stages),
            "target_h5": None,
            "input_root": self.full_input_root,
            "max_files": None,
            "smoke_test": False,
        })

    def with_updates(self, **updates: Any) -> "PipelineRequest":
        """Return a fully revalidated copy after user or orchestration updates."""
        return type(self).model_validate({**self.model_dump(mode="python"), **updates})


class TextResponse(BaseModel):
    text: str
