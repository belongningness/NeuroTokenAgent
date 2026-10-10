"""Stage ordering, dependencies, and internal worker mappings."""

from __future__ import annotations

from agent_pipeline.pipeline_models import PipelineRequest, StageName


STAGE_ORDER: tuple[StageName, ...] = (
    "preflight",
    "package",
    "tokenize",
    "readme",
    "docs",
    "validate",
    "reconstruct",
)

REFERENCE_STAGE_SCRIPTS: dict[str, str] = {
    "tokenize": "2_neural2token_human_new.py",
    "readme": "3_make_readme.py",
    "docs": "4_generate_release_docs_and_schemas.py",
    "validate": "5_validate_neurotoken_dataset.py",
    "reconstruct": "6_verify_token_reconstruction.py",
}

DEFAULT_REFERENCE_SCRIPTS = "agent_pipeline/reference_scripts"

STAGE_DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    "preflight": (),
    "package": ("preflight",),
    "tokenize": ("package",),
    "readme": (),
    "docs": (),
    "validate": ("readme", "docs"),
    "reconstruct": ("tokenize",),
}


def dependency_closure(stages: list[StageName]) -> list[StageName]:
    resolved = set(stages)
    changed = True
    while changed:
        changed = False
        for stage in tuple(resolved):
            for dependency in STAGE_DEPENDENCIES[stage]:
                if dependency not in resolved:
                    resolved.add(dependency)
                    changed = True
    return [stage for stage in STAGE_ORDER if stage in resolved]


def resolve_stage_dependencies(request: PipelineRequest) -> list[str]:
    requested = set(request.stages)
    request.stages = dependency_closure(list(request.stages))
    return [stage for stage in request.stages if stage not in requested]
