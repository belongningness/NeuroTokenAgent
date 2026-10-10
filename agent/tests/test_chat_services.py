from unittest.mock import Mock

from agent_pipeline.chat_services import append_package_sample_summary, summarize_stage


def test_summarize_stage_falls_back_when_llm_is_unavailable():
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("auth unavailable")

    summary = summarize_stage(
        llm,
        {
            "stage": "docs",
            "status": "ok",
            "duration_seconds": 0.297,
            "report_path": "/tmp/docs_report.json",
        },
    )

    assert "docs 阶段已完成" in summary
    assert "状态为 ok" in summary
    assert "0.297 秒" in summary
    assert "/tmp/docs_report.json" in summary


def test_summarize_reconstruct_stage_fallback_includes_obtained_metrics():
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("auth unavailable")

    summary = summarize_stage(
        llm,
        {
            "stage": "reconstruct",
            "status": "ok",
            "evaluated_sessions": 1,
            "evaluated_windows": 2,
            "metrics": {"pearson": 0.91, "ssim": 0.84, "mse": 0.001, "mae": 0.02, "relative_l2": 0.1},
            "sessions": [{"session_id": "session-1", "evaluated_windows": 2}],
        },
    )

    assert "实际重建指标" in summary
    assert "Pearson=0.91" in summary
    assert "MSE=0.001" in summary


def test_append_package_sample_summary_lists_bounded_members_and_metadata():
    summary = append_package_sample_summary(
        "Package 正常完成。",
        {
            "stage": "package",
            "sessions": [{
                "output_dir": "/tmp/tar/session-01",
                "tar_path": "/tmp/tar/session-01/data.tar",
                "tar_sample": [
                    {"name": "sample.meg.npy", "size": 4096, "type": "file"},
                    {"name": "sample.metadata.json", "size": 256, "type": "file"},
                ],
                "segmentation_manifest_path": "/tmp/tar/session-01/segmentation_manifest.json",
                "segmentation_review": {
                    "strategy": "rest",
                    "sfreq": 100.0,
                    "total_samples": 2500,
                    "total_duration_seconds": 25.0,
                    "segment_count": 3,
                    "coverage_percent": 100.0,
                    "min_segment_seconds": 5.0,
                    "max_segment_seconds": 10.0,
                    "split_points": 1000,
                    "uncovered_ranges": [],
                    "overlaps": [],
                    "preview_segments": [
                        {"chunk_index": 0, "start_sample": 0, "end_sample": 1000, "start_time": 0.0, "end_time": 10.0, "duration_seconds": 10.0},
                        {"chunk_index": 2, "start_sample": 2000, "end_sample": 2500, "start_time": 20.0, "end_time": 25.0, "duration_seconds": 5.0},
                    ],
                },
                "tar_sample_metadata_member": "sample.metadata.json",
                "tar_sample_metadata": {"chunk_index": 0, "n_samples": 1000},
            }],
        },
    )

    assert summary.startswith("Package 正常完成。")
    assert "## tar 切分审核" in summary
    assert "每 1000 samples 切一段" in summary
    assert "samples `[0, 1000)`" in summary
    assert "time `[20, 25) s`" in summary
    assert "覆盖率 100%" in summary
    assert "## tar 内容样例" in summary
    assert "### session-01" in summary
    assert "`sample.meg.npy`（4.0 KiB）" in summary
    assert '"chunk_index": 0' in summary
