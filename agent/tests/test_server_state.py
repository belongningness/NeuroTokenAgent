from agent_pipeline.server_state import ArtifactEnvelope, WorkspaceSnapshot


def test_artifact_envelope_is_versioned_and_owned():
    artifact = ArtifactEnvelope(
        artifact_id="artifact_1",
        kind="reconstruct_quality",
        owner="user_1",
        thread_id="thread_1",
        workflow_id="wf_1",
        run_id="run_1",
        stage="reconstruct",
        summary="quality report",
        created_at="2026-07-21T00:00:00+00:00",
        payload={"pearson": 0.9},
    )
    assert artifact.schema_version == "1.0"
    assert artifact.owner == "user_1"
    assert artifact.payload["pearson"] == 0.9


def test_workspace_snapshot_serializes_versioned_defaults():
    snapshot = WorkspaceSnapshot()
    payload = snapshot.model_dump(mode="json")
    assert payload["schema_version"] == "1.0"
    assert payload["mode"] == "idle"
    assert payload["run"] == {
        "workflow_id": None,
        "run_id": None,
        "report_root": None,
        "config_path": None,
        "active_phase": "single",
        "linked_run_ids": {},
    }
    assert payload["stages"] == []
    assert payload["artifacts"] == []
