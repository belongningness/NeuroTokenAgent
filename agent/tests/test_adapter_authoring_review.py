from pathlib import Path
from unittest.mock import Mock

import pytest

from agent_pipeline.adapter_authoring.models import AdapterIntentReview, AdapterSpec
from agent_pipeline.adapter_authoring.review import review_adapter_candidate, review_feedback, run_candidate_preflight


def test_adapter_intent_review_requires_error_for_revision():
    with pytest.raises(ValueError, match="requires at least one error"):
        AdapterIntentReview(
            verdict="needs_revision",
            summary="Gap segments are missing.",
            findings=[{
                "severity": "warning",
                "title": "Missing gap coverage",
                "evidence": "Only event ranges are emitted.",
                "required_change": "Emit inter-event gaps.",
            }],
        )


def test_adapter_intent_review_requires_error_for_scope_block():
    with pytest.raises(ValueError, match="requires at least one error"):
        AdapterIntentReview(
            verdict="blocked_by_scope",
            summary="Shared packaging must change.",
            findings=[],
        )


def test_adapter_intent_review_rejects_error_in_pass():
    with pytest.raises(ValueError, match="cannot contain error"):
        AdapterIntentReview(
            verdict="pass",
            summary="Looks good.",
            findings=[{
                "severity": "error",
                "title": "Missing gap coverage",
                "evidence": "Only event ranges are emitted.",
                "required_change": "Emit inter-event gaps.",
            }],
        )


def test_review_adapter_candidate_uses_structured_read_only_context(tmp_path: Path):
    package = tmp_path / "agent_pipeline" / "package.py"
    package.parent.mkdir(parents=True)
    package.write_text("def sample_ranges():\n    pass\n", encoding="utf-8")
    structured = Mock()
    structured.invoke.return_value = AdapterIntentReview(
        verdict="needs_revision",
        summary="Inter-event time ranges are not emitted.",
        findings=[{
            "severity": "error",
            "title": "Retain inter-event time ranges",
            "evidence": "sample_ranges only yields events.",
            "required_change": "Yield each non-empty interval between adjacent events without fabricated event metadata.",
        }],
    )
    llm = Mock()
    llm.with_structured_output.return_value = structured

    review = review_adapter_candidate(
        llm,
        spec=AdapterSpec("example", tmp_path / "source.h5", "meg", "Include gaps"),
        h5_summary="two events",
        instruction="Split events and inter-event gaps.",
        design_document="# Design",
        diff="adapter diff",
        worktree=tmp_path,
        original_user_request="Each event and adjacent gap must be a separate segment.",
        requirements_snapshot={"requirements": {"config_overrides": {"chunking": {"strategy": "event"}}}},
        effective_config={"source": {"adapter": "example"}, "chunking": {"strategy": "event"}},
        preflight_evidence={"status": "ok", "source_session_loaded": True, "valid_nonempty_clipped_event_intervals": 2},
        allowed_paths=["agent_pipeline/adapters", "ADAPTER_DESIGN.md"],
    )

    assert review.verdict == "needs_revision"
    prompt = structured.invoke.call_args.args[0]
    assert "sample_ranges" in prompt
    assert "Immutable original human request" in prompt
    assert "Each event and adjacent gap" in prompt
    assert '"strategy": "event"' in prompt
    assert '"valid_nonempty_clipped_event_intervals": 2' in prompt
    assert "server-owned review inputs" in prompt
    assert "blocked_by_scope" in prompt
    assert "Required change: Yield each non-empty interval" in review_feedback(review)
    llm.with_structured_output.assert_called_once_with(AdapterIntentReview, method="function_calling")


def test_candidate_preflight_runs_adapter_from_isolated_worktree(tmp_path: Path):
    package = tmp_path / "agent_pipeline" / "adapters"
    package.mkdir(parents=True)
    (tmp_path / "agent_pipeline" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text(
        "from .candidate import CandidateAdapter\n"
        "def build_adapter(config):\n"
        "    return CandidateAdapter(config)\n",
        encoding="utf-8",
    )
    (package / "candidate.py").write_text(
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "class CandidateAdapter:\n"
        "    def __init__(self, config): self.config = config\n"
        "    def discover(self, root): return [Path(root)]\n"
        "    def load_session(self, path):\n"
        "        return SimpleNamespace(signal=SimpleNamespace(shape=(2, 100)), sfreq=10.0, events=[\n"
        "            {'start_time': 0.0, 'end_time': 1.0, 'type': 'stim', 'file_path': str(path)},\n"
        "            {'start_time': 2.0, 'end_time': 3.0, 'type': 'stim', 'file_path': str(path)},\n"
        "        ])\n",
        encoding="utf-8",
    )
    (tmp_path / "agent_pipeline" / "package.py").write_text(
        "def event_sample_bounds(event, sfreq, n_points):\n"
        "    start = max(0, min(int(event['start_time'] * sfreq), n_points))\n"
        "    end = max(0, min(int(event['end_time'] * sfreq), n_points))\n"
        "    if end <= start: raise ValueError('empty interval')\n"
        "    return start, end\n"
        "def event_ranges(session):\n"
        "    normalized = []\n"
        "    for source_index, source_event in enumerate(session.events):\n"
        "        event = dict(source_event)\n"
        "        event['event_index'] = source_index\n"
        "        start, end = event_sample_bounds(event, session.sfreq, session.signal.shape[1])\n"
        "        normalized.append((start, end, source_index, event))\n"
        "    normalized.sort(key=lambda item: (item[0], item[1], item[2]))\n"
        "    ranges = []\n"
        "    covered_until = 0\n"
        "    for start, end, _, event in normalized:\n"
        "        if covered_until < start:\n"
        "            ranges.append((len(ranges), covered_until, start, None))\n"
        "        ranges.append((len(ranges), start, end, event))\n"
        "        covered_until = max(covered_until, end)\n"
        "    if covered_until < session.signal.shape[1]:\n"
        "        ranges.append((len(ranges), covered_until, session.signal.shape[1], None))\n"
        "    return ranges\n"
        "def segment_record(chunk_index, start, end, event, sfreq, strategy):\n"
        "    record = {'chunk_index': chunk_index, 'start_sample': start, 'end_sample': end, 'chunking_strategy': strategy}\n"
        "    if event is not None:\n"
        "        record['event_index'] = event['event_index']\n"
        "    return record\n",
        encoding="utf-8",
    )
    target = tmp_path / "sample.h5"
    target.touch()

    evidence = run_candidate_preflight(
        worktree=tmp_path,
        effective_config={"source": {"adapter": "candidate"}, "chunking": {"strategy": "event"}},
        target_h5=target,
    )

    assert evidence["status"] == "ok"
    assert evidence["source_session_loaded"] is True
    assert evidence["event_count"] == 2
    assert evidence["valid_nonempty_clipped_event_intervals"] == 2
    assert evidence["events_with_semantic_labels"] == 2
    assert evidence["events_with_source_file_paths"] == 2
    assert evidence["chunking_strategy"] == "event"
    assert evidence["package_segment_count"] == 4
    assert evidence["package_event_segment_count"] == 2
    assert evidence["package_ordinary_segment_count"] == 2
    assert evidence["package_full_recording_coverage"] is True
    assert evidence["package_invented_gap_label_count"] == 0
    assert evidence["package_fabricated_ordinary_payload_count"] == 0
    assert [item["has_source_event"] for item in evidence["package_boundary_preview"]] == [True, False, True, False]


def test_candidate_preflight_rejects_synthetic_gap_source_events(tmp_path: Path):
    adapters = tmp_path / "agent_pipeline" / "adapters"
    adapters.mkdir(parents=True)
    (tmp_path / "agent_pipeline" / "__init__.py").write_text("", encoding="utf-8")
    (adapters / "__init__.py").write_text(
        "from .candidate import CandidateAdapter\n"
        "def build_adapter(config): return CandidateAdapter(config)\n",
        encoding="utf-8",
    )
    (adapters / "candidate.py").write_text(
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "class CandidateAdapter:\n"
        "    def __init__(self, config): self.config = config\n"
        "    def discover(self, root): return [Path(root)]\n"
        "    def load_session(self, path):\n"
        "        return SimpleNamespace(signal=SimpleNamespace(shape=(2, 100)), sfreq=10.0, events=[\n"
        "            {'start_time': 0.0, 'end_time': 1.0, 'type': 'stim'},\n"
        "            {'start_time': 1.0, 'end_time': 2.0, 'type': 'gap'},\n"
        "        ])\n",
        encoding="utf-8",
    )
    (tmp_path / "agent_pipeline" / "package.py").write_text(
        "def event_sample_bounds(event, sfreq, n_points): return 0, 1\n"
        "def event_ranges(session): return []\n"
        "def segment_record(*args): return {}\n",
        encoding="utf-8",
    )
    target = tmp_path / "sample.h5"
    target.touch()

    evidence = run_candidate_preflight(
        worktree=tmp_path,
        effective_config={"source": {"adapter": "candidate"}, "chunking": {"strategy": "event"}},
        target_h5=target,
    )

    assert evidence["status"] == "failed"
    assert "synthetic gap record" in evidence["stderr"]


def test_candidate_preflight_rejects_event_ranges_without_full_recording_coverage(tmp_path: Path):
    adapters = tmp_path / "agent_pipeline" / "adapters"
    adapters.mkdir(parents=True)
    (tmp_path / "agent_pipeline" / "__init__.py").write_text("", encoding="utf-8")
    (adapters / "__init__.py").write_text(
        "from .candidate import CandidateAdapter\n"
        "def build_adapter(config): return CandidateAdapter(config)\n",
        encoding="utf-8",
    )
    (adapters / "candidate.py").write_text(
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "class CandidateAdapter:\n"
        "    def __init__(self, config): self.config = config\n"
        "    def discover(self, root): return [Path(root)]\n"
        "    def load_session(self, path):\n"
        "        return SimpleNamespace(signal=SimpleNamespace(shape=(2, 100)), sfreq=10.0, events=[\n"
        "            {'start_time': 1.0, 'end_time': 3.0, 'type': 'stim'},\n"
        "        ])\n",
        encoding="utf-8",
    )
    (tmp_path / "agent_pipeline" / "package.py").write_text(
        "def event_sample_bounds(event, sfreq, n_points): return 10, 30\n"
        "def event_ranges(session): return [(0, 10, 30, {'event_index': 0})]\n"
        "def segment_record(*args): return {'event_index': 0}\n",
        encoding="utf-8",
    )
    target = tmp_path / "sample.h5"
    target.touch()

    evidence = run_candidate_preflight(
        worktree=tmp_path,
        effective_config={"source": {"adapter": "candidate"}, "chunking": {"strategy": "event"}},
        target_h5=target,
    )

    assert evidence["status"] == "failed"
    assert "do not cover the complete recording" in evidence["stderr"]
