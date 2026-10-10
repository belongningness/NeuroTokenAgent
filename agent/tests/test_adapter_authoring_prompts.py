from agent_pipeline.adapter_authoring.models import AdapterSpec
from agent_pipeline.adapter_authoring.prompts import (
    build_adapter_prompt,
    build_adapter_review_prompt,
    load_adapter_knowledge,
    load_review_context,
)


def test_adapter_prompt_includes_knowledge_base_and_design_deliverable(tmp_path):
    prompt = build_adapter_prompt(
        AdapterSpec("example_adapter", tmp_path / "source.h5", "meg"),
        "- /meg: dataset shape=(2, 10) dtype=float32",
    )
    assert "Knowledge base file:" in prompt
    assert "event" in prompt and "rest" in prompt
    assert "ADAPTER_DESIGN.md" in prompt
    assert "template comparison" in prompt


def test_adapter_knowledge_preserves_reference_event_metadata_contract():
    knowledge = load_adapter_knowledge()

    assert 'Do not append synthetic gap records to `SourceSession.events`' in knowledge
    assert 'no `event_index`, no `event_*`' in knowledge
    assert 'Do not write `segment_kind="gap"`' in knowledge
    assert "event_segment_count" in knowledge
    assert "ordinary_segment_count" in knowledge
    assert "complete recording from sample 0" in knowledge
    assert "ordinary `[0, 1)`" in knowledge
    assert "long semantic intervals remaining logically intact" in knowledge


def test_review_prompt_checks_shared_packaging_against_human_intent(tmp_path):
    prompt = build_adapter_review_prompt(
        AdapterSpec(
            "example_adapter",
            tmp_path / "source.h5",
            "meg",
            "Each event and each gap between adjacent events must be a separate segment.",
        ),
        "events_pkl_u8: 2 records",
        "Implement event and inter-event gap segmentation.",
        "# Design\nEvents are parsed by the adapter.",
        "diff --git a/agent_pipeline/adapters/example.py b/agent_pipeline/adapters/example.py",
        "## agent_pipeline/package.py\ndef sample_ranges(): pass",
        original_user_request="Each event and adjacent gap is one segment.",
        requirements_snapshot='{"requirements":{"config_overrides":{"chunking":{"strategy":"event"}}}}',
        effective_config='{"source":{"adapter":"example_adapter"},"chunking":{"strategy":"event"}}',
        allowed_paths=["agent_pipeline/adapters", "ADAPTER_DESIGN.md"],
    )

    assert "trace normalized `SourceSession` values through shared packaging code" in prompt
    assert "each gap between adjacent events" in prompt
    assert "Immutable original human request" in prompt
    assert "blocked_by_scope" in prompt
    assert "agent_pipeline/adapters" in prompt
    assert "agent_pipeline/package.py" in prompt
    assert "Return `needs_revision`" in prompt
    assert "package_ordinary_segment_count" in prompt
    assert "package_full_recording_coverage" in prompt
    assert "package_invented_gap_label_count" in prompt


def test_review_context_includes_shared_packaging(tmp_path):
    package = tmp_path / "agent_pipeline" / "package.py"
    package.parent.mkdir(parents=True)
    package.write_text("def sample_ranges():\n    pass\n", encoding="utf-8")

    context = load_review_context(tmp_path)

    assert "## agent_pipeline/package.py" in context
    assert "def sample_ranges" in context
