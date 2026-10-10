/** @vitest-environment jsdom */
import { useState } from "react";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { NeuroTokenInterrupt, WorkspaceStage } from "./contracts";
import type { ApplyMode } from "./handoffState";
import type { LangGraphDecision } from "./runtime";
import { WorkflowDecisionCard } from "./WorkflowDecisionCard";

const stages: WorkspaceStage[] = [
  { name: "inspect", status: "completed" },
  { name: "tokenize", status: "running" },
];

const baseInterrupt = {
  schema_version: "1.0" as const,
  owner: "neurotoken",
  thread_id: "thread-1",
};

function ControlledCard({
  interrupt,
  onDecision = vi.fn(),
  stages: cardStages = stages,
}: {
  interrupt: NeuroTokenInterrupt;
  onDecision?: (decision: LangGraphDecision) => void | Promise<void>;
  stages?: WorkspaceStage[];
}) {
  const [applyMode, setApplyMode] = useState<ApplyMode>("apply");
  const [commitMessage, setCommitMessage] = useState("");
  const [refine, setRefine] = useState("");
  const [question, setQuestion] = useState("");
  return (
    <WorkflowDecisionCard
      interrupt={interrupt}
      stages={cardStages}
      applyMode={applyMode}
      commitMessage={commitMessage}
      refine={refine}
      question={question}
      onApplyModeChange={setApplyMode}
      onCommitMessageChange={setCommitMessage}
      onRefineChange={setRefine}
      onQuestionChange={setQuestion}
      onDecision={onDecision}
    />
  );
}

afterEach(cleanup);

describe("WorkflowDecisionCard", () => {
  it("asks whether to process data after the adapter is ready", async () => {
    const onDecision = vi.fn();
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "adapter_execution_confirmation",
      adapter_name: "custom_eeg_adapter",
      input_path: "/input/example.h5",
      output_path: "/output/run",
      allowed_actions: ["continue", "pause"],
    };
    render(<ControlledCard interrupt={interrupt} onDecision={onDecision} stages={[]} />);

    expect(screen.getByText("Adapter ready")).toBeTruthy();
    expect(screen.getByText(/custom_eeg_adapter/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Start processing" }));

    await waitFor(() => expect(onDecision).toHaveBeenCalledWith({
      schema_version: "1.0",
      action: "continue",
      plan: null,
      message: null,
    }));
  });

  it("renders plan approval and submits a typed plan decision", async () => {
    const onDecision = vi.fn();
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "plan_approval",
      plan: { stages: ["inspect", "tokenize"] },
      allowed_actions: ["approve", "refine", "reject"],
    };
    render(<ControlledCard interrupt={interrupt} onDecision={onDecision} />);

    expect(screen.getByText("Plan approval")).toBeTruthy();
    expect(screen.getByText("Current proposed plan")).toBeTruthy();
    expect(screen.getByText("2 stages")).toBeTruthy();
    expect(screen.getByText("tokenize")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Request revision" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Approve plan" }));

    await waitFor(() => expect(onDecision).toHaveBeenCalledWith({
      schema_version: "1.0",
      action: "approve",
      plan: interrupt.plan,
      message: null,
    }));
    expect(await screen.findByText("Submitted: Approve plan")).toBeTruthy();
  });

  it("uses the refined interrupt plan instead of stale workspace stages", () => {
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "plan_approval",
      plan: { stages: ["preflight", "package"] },
      allowed_actions: ["approve", "refine", "reject"],
    };
    render(
      <ControlledCard
        interrupt={interrupt}
        stages={[
          { name: "preflight", status: "pending" },
          { name: "package", status: "pending" },
          { name: "readme", status: "pending" },
          { name: "docs", status: "pending" },
        ]}
      />,
    );

    const progress = screen.getByLabelText("Workspace stages");
    expect(progress.textContent).toContain("preflight");
    expect(progress.textContent).toContain("package");
    expect(progress.textContent).not.toContain("readme");
    expect(progress.textContent).not.toContain("docs");
    expect(screen.getByText("0 / 2")).toBeTruthy();
  });

  it("keeps refinement controlled and includes it in the decision", async () => {
    const onDecision = vi.fn();
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "plan_approval",
      plan: {},
      allowed_actions: ["refine"],
    };
    render(<ControlledCard interrupt={interrupt} onDecision={onDecision} />);

    fireEvent.change(screen.getByLabelText("Plan refinement instructions"), {
      target: { value: "Validate every output" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Request revision" }));

    await waitFor(() => expect(onDecision).toHaveBeenCalledWith(expect.objectContaining({
      action: "refine",
      message: "Validate every output",
    })));
  });

  it("clears stale submission feedback when a revised plan interrupt arrives", async () => {
    const onDecision = vi.fn();
    const first: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "plan_approval",
      plan: { stages: ["inspect"] },
      allowed_actions: ["approve", "refine", "reject"],
    };
    const revised: NeuroTokenInterrupt = {
      ...first,
      plan_revision: 2,
    };
    const rendered = render(<ControlledCard interrupt={first} onDecision={onDecision} />);
    fireEvent.click(screen.getByRole("button", { name: "Approve plan" }));
    expect(await screen.findByText("Submitted: Approve plan")).toBeTruthy();

    rendered.rerender(<ControlledCard interrupt={revised} onDecision={onDecision} />);
    await waitFor(() => expect(screen.queryByText("Submitted: Approve plan")).toBeNull());
    expect(screen.getByRole("button", { name: "Approve plan" })).toBeTruthy();
  });

  it("clearly renders stage review actions and remaining stages", async () => {
    const onDecision = vi.fn();
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "stage_review",
      workflow_id: "workflow-1",
      stage: "inspect",
      summary: "Inspection found two compatible datasets.",
      report: { datasetCount: 2 },
      remaining_stages: ["tokenize", "package"],
      allowed_actions: ["continue", "pause", "cancel"],
    };
    render(<ControlledCard interrupt={interrupt} onDecision={onDecision} />);

    expect(screen.getByText("Stage review")).toBeTruthy();
    expect(screen.getByText("Inspection found two compatible datasets.")).toBeTruthy();
    expect(screen.getByText("Up next")).toBeTruthy();
    expect(screen.getByText("tokenize → package")).toBeTruthy();
    expect(screen.getByLabelText("Workspace stages").querySelectorAll("li")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "Continue workflow" }));

    await waitFor(() => expect(onDecision).toHaveBeenCalledWith(expect.objectContaining({
      action: "continue",
    })));
  });

  it("shows single-file evidence and requires a gate-bound full-run decision", async () => {
    const onDecision = vi.fn();
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "smoke_validation_review",
      workflow_id: "workflow-validation-1",
      gate_id: "gate-1",
      validation_target_h5: "/input/session-01.h5",
      full_input_root: "/input/all-sessions",
      status: "awaiting_decision",
      verdict: "pass",
      report: {
        metrics: { valid_chunks: 12, pearson: 0.934567 },
      },
      allowed_actions: ["start_full_run", "rerun_smoke", "pause", "cancel"],
    };
    render(<ControlledCard interrupt={interrupt} onDecision={onDecision} stages={[]} />);

    const result = screen.getByLabelText("Single-file validation result");
    expect(within(result).getByText("/input/session-01.h5")).toBeTruthy();
    expect(within(result).getByText("pass")).toBeTruthy();
    expect(within(result).getByText("valid chunks")).toBeTruthy();
    expect(within(result).getByText("12")).toBeTruthy();
    expect(within(result).getByText("0.934567")).toBeTruthy();
    expect(within(result).getByText(/will not run until/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Start full run" }));
    await waitFor(() => expect(onDecision).toHaveBeenCalledWith({
      schema_version: "1.0",
      action: "start_full_run",
      plan: null,
      message: null,
      gate_id: "gate-1",
    }));
  });

  it("shows a bounded tar member and metadata sample during package review", () => {
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "stage_review",
      workflow_id: "workflow-1",
      stage: "package",
      summary: "Package completed.",
      report: {
        sessions: [{
          output_dir: "/tmp/tar/session-01",
          tar_path: "/tmp/tar/session-01/data.tar",
          tar_sample: [
            { name: "sample.meg.npy", size: 4096, type: "file" },
            { name: "sample.metadata.json", size: 256, type: "file" },
          ],
          segmentation_manifest_path: "/tmp/tar/session-01/segmentation_manifest.json",
          segmentation_review: {
            strategy: "rest",
            sfreq: 100,
            total_samples: 2500,
            total_duration_seconds: 25,
            segment_count: 3,
            coverage_percent: 100,
            min_segment_seconds: 5,
            max_segment_seconds: 10,
            split_points: 1000,
            uncovered_ranges: [],
            overlaps: [],
            preview_segments: [
              { chunk_index: 0, start_sample: 0, end_sample: 1000, start_time: 0, end_time: 10, duration_seconds: 10 },
              { chunk_index: 2, start_sample: 2000, end_sample: 2500, start_time: 20, end_time: 25, duration_seconds: 5 },
            ],
          },
          tar_sample_metadata_member: "sample.metadata.json",
          tar_sample_metadata: { chunk_index: 0, n_samples: 1000 },
        }],
      },
      remaining_stages: ["tokenize"],
      allowed_actions: ["continue"],
    };

    render(<ControlledCard interrupt={interrupt} />);

    const segmentation = screen.getByLabelText("Tar segmentation review");
    expect(segmentation).toBeTruthy();
    expect(within(segmentation).getByText("Fixed sample windows")).toBeTruthy();
    expect(within(segmentation).getByText(/cut every/).textContent).toContain("1000");
    expect(within(segmentation).getByText("[0, 1000)")).toBeTruthy();
    expect(within(segmentation).getByText("[20, 25) s")).toBeTruthy();
    expect(within(segmentation).getByText("100%")).toBeTruthy();
    expect(screen.getByLabelText("Tar content sample")).toBeTruthy();
    expect(screen.getAllByText("session-01").length).toBeGreaterThan(0);
    expect(screen.getByText("sample.meg.npy")).toBeTruthy();
    expect(screen.getByText("4.0 KiB")).toBeTruthy();
    fireEvent.click(screen.getByText(/Sample metadata/));
    expect(screen.getByText(/"chunk_index": 0/)).toBeTruthy();
  });

  it("shows complete event-boundary coverage without a gap kind", () => {
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "stage_review",
      workflow_id: "workflow-1",
      stage: "package",
      summary: "Package completed.",
      report: {
        sessions: [{
          output_dir: "/tmp/tar/session-01",
          segmentation_review: {
            strategy: "event",
            sfreq: 10,
            total_samples: 210,
            total_duration_seconds: 21,
            segment_count: 5,
            event_segment_count: 2,
            ordinary_segment_count: 3,
            coverage_percent: 100,
            min_segment_seconds: 1,
            max_segment_seconds: 10,
            uncovered_ranges: [],
            overlaps: [],
            preview_segments: [
              { chunk_index: 0, start_sample: 0, end_sample: 10, start_time: 0, end_time: 1, duration_seconds: 1 },
              { chunk_index: 1, start_sample: 10, end_sample: 90, start_time: 1, end_time: 9, duration_seconds: 8, event_index: 0, event_type: "first" },
              { chunk_index: 2, start_sample: 90, end_sample: 100, start_time: 9, end_time: 10, duration_seconds: 1 },
              { chunk_index: 3, start_sample: 100, end_sample: 200, start_time: 10, end_time: 20, duration_seconds: 10, event_index: 1, event_type: "second" },
              { chunk_index: 4, start_sample: 200, end_sample: 210, start_time: 20, end_time: 21, duration_seconds: 1 },
            ],
          },
        }],
      },
      remaining_stages: ["tokenize"],
      allowed_actions: ["continue"],
    };

    render(<ControlledCard interrupt={interrupt} />);

    const segmentation = screen.getByLabelText("Tar segmentation review");
    expect(within(segmentation).getByText("Semantic event intervals")).toBeTruthy();
    expect(within(segmentation).getByText(/partition the complete recording/)).toBeTruthy();
    expect(within(segmentation).getByText("2 event · 3 ordinary")).toBeTruthy();
    expect(within(segmentation).getByText("[0, 1) s")).toBeTruthy();
    expect(within(segmentation).getByText("[9, 10) s")).toBeTruthy();
    expect(within(segmentation).getByText("[20, 21) s")).toBeTruthy();
    expect(within(segmentation).queryByText(/segment_kind|gap/i)).toBeNull();
  });

  it("shows obtained aggregate and per-session metrics during reconstruction review", () => {
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "stage_review",
      workflow_id: "workflow-1",
      stage: "reconstruct",
      summary: "Reconstruction completed.",
      report: {
        evaluated_sessions: 1,
        evaluated_windows: 3,
        metrics: { pearson: 0.91234567, ssim: 0.8342, mse: 0.0012, mae: 0.023, relative_l2: 0.11 },
        sessions: [{
          session_id: "subject-01",
          modality: "meg",
          evaluated_windows: 3,
          metrics_mean: { pearson: 0.91234567, ssim: 0.8342, mse: 0.0012, mae: 0.023, relative_l2: 0.11 },
        }],
      },
      remaining_stages: [],
      allowed_actions: ["continue"],
    };

    render(<ControlledCard interrupt={interrupt} />);

    const preview = screen.getByLabelText("Reconstruction metrics");
    expect(within(preview).getByText("Obtained reconstruction metrics")).toBeTruthy();
    expect(within(preview).getByText("All evaluated windows")).toBeTruthy();
    expect(within(preview).getByText("subject-01")).toBeTruthy();
    expect(within(preview).getAllByText("0.912346")).toHaveLength(2);
    expect(within(preview).getAllByText("0.0012")).toHaveLength(2);
    expect(within(preview).getAllByText("Higher is better · ideal 1").length).toBeGreaterThan(0);
  });

  it("states when reconstruction produced no actual metrics", () => {
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "stage_review",
      workflow_id: "workflow-1",
      stage: "reconstruct",
      summary: "No sessions were evaluated.",
      report: { sessions: [], findings: [] },
      remaining_stages: [],
      allowed_actions: ["continue"],
    };

    render(<ControlledCard interrupt={interrupt} />);

    expect(screen.getByLabelText("Reconstruction metrics").textContent).toContain("did not produce Pearson");
  });

  it("shows the generated README body during readme review", () => {
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "stage_review",
      workflow_id: "workflow-1",
      stage: "readme",
      summary: "README generated.",
      report: {
        readme_path: "/tmp/neurotoken/README.md",
        readme_content: "# Example dataset\n\nGenerated README body.\n\n## Tokenizer\n\nv1",
      },
      remaining_stages: ["docs"],
      allowed_actions: ["continue"],
    };

    render(<ControlledCard interrupt={interrupt} />);

    expect(screen.getByLabelText("README content preview")).toBeTruthy();
    expect(screen.getByText("/tmp/neurotoken/README.md")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Example dataset" })).toBeTruthy();
    expect(screen.getByText("Generated README body.")).toBeTruthy();
  });

  it("renders stage summaries as markdown and labels final continuation as completion", async () => {
    const onDecision = vi.fn();
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "stage_review",
      workflow_id: "workflow-1",
      stage: "inspect",
      summary: "**Status**: `ok` • **Findings**: 0",
      report: {},
      remaining_stages: [],
      allowed_actions: ["continue"],
    };
    render(<ControlledCard interrupt={interrupt} onDecision={onDecision} />);
    expect(screen.getByText("Status")).toBeTruthy();
    expect(screen.getByText("ok")).toBeTruthy();
    expect(screen.getByText("Findings")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Continue workflow" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Complete workflow" }));
    await waitFor(() => expect(onDecision).toHaveBeenCalledWith(expect.objectContaining({ action: "continue" })));
  });

  it("supports controlled patch modes, messages, and all patch decisions", async () => {
    const onDecision = vi.fn();
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "adapter_patch_approval",
      task_id: "task-1",
      revision: 3,
      adapter: {
        name: "eeg",
        diff: "diff --git a/eeg.py b/eeg.py\n--- a/eeg.py\n+++ b/eeg.py\n@@ -1 +1 @@\n-old\n+new",
      },
      allowed_actions: ["apply", "apply_and_commit", "ask", "refine", "reject"],
    };
    render(<ControlledCard interrupt={interrupt} onDecision={onDecision} />);

    expect(screen.getByText("Adapter patch approval")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Question about this patch"), {
      target: { value: "Why change the registry?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(onDecision).toHaveBeenLastCalledWith(expect.objectContaining({
      action: "ask",
      task_id: "task-1",
      revision: 3,
      message: "Why change the registry?",
    })));
    expect(await screen.findByText("Submitted: Ask")).toBeTruthy();

    fireEvent.click(screen.getByLabelText("Apply and commit"));
    expect((screen.getByRole("button", { name: "Apply and commit" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Commit message"), { target: { value: "Add EEG adapter" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply and commit" }));
    await waitFor(() => expect(onDecision).toHaveBeenLastCalledWith(expect.objectContaining({
      action: "apply_and_commit",
      commit_message: "Add EEG adapter",
    })));
  });

  it("disables patch application when the approval has no frozen diff", () => {
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "adapter_patch_approval",
      task_id: "task-1",
      revision: 3,
      adapter: {
        name: "eeg",
        diff: "",
        design_document: "# Adapter Design\n\nReview only.",
      },
      allowed_actions: ["apply", "refine", "reject"],
    };

    render(<ControlledCard interrupt={interrupt} />);

    expect(screen.getByRole("alert").textContent).toContain("no frozen candidate diff");
    expect((screen.getByRole("button", { name: "Apply only" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole("heading", { name: "Adapter Design" })).toBeTruthy();
  });

  it("shows the independent adapter intent review", () => {
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "adapter_patch_approval",
      task_id: "task-1",
      revision: 3,
      adapter: {
        name: "meg",
        diff: "diff --git a/meg.py b/meg.py\n--- a/meg.py\n+++ b/meg.py\n@@ -1 +1 @@\n-old\n+new",
        design_document: "# Adapter Design\n\nEvent based.",
        original_user_request: "Keep each event and each adjacent gap as separate segments.",
        effective_config: {
          dataset: { name: "SMN_MEG", modality: "meg" },
          source: { adapter: "smn_meg" },
          source_h5: { signal_key: "meg", events_key: "events_pkl_u8" },
          chunking: { strategy: "event", split_points: 30000 },
          paths: { h5_root: "/private/source" },
        },
        intent_review: {
          verdict: "pass",
          summary: "Event boundaries partition the complete recording without inventing segment labels.",
          findings: [{
            severity: "warning",
            title: "Verify complete recording coverage",
            evidence: "Leading, internal, and trailing ordinary ranges must be retained.",
            required_change: "Confirm package_full_recording_coverage before approval.",
          }],
        },
      },
      allowed_actions: ["apply", "refine", "reject"],
    };

    render(<ControlledCard interrupt={interrupt} />);

    expect(screen.getByRole("heading", { name: "Original human request" })).toBeTruthy();
    expect(screen.getByText(/Keep each event and each adjacent gap/)).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Proposed effective config" })).toBeTruthy();
    expect(screen.getByText(/"strategy": "event"/)).toBeTruthy();
    expect(screen.queryByText(/private\/source/)).toBeNull();
    expect(screen.getByRole("heading", { name: "Independent intent review · pass" })).toBeTruthy();
    expect(screen.getByText(/Event boundaries partition the complete recording/)).toBeTruthy();
    expect(screen.getByText(/WARNING: Verify complete recording coverage/)).toBeTruthy();
  });

  it("disables every action immediately and reports asynchronous submission", async () => {
    let resolveDecision: (() => void) | undefined;
    const onDecision = vi.fn(() => new Promise<void>((resolve) => {
      resolveDecision = resolve;
    }));
    const interrupt: NeuroTokenInterrupt = {
      ...baseInterrupt,
      kind: "stage_review",
      workflow_id: "workflow-1",
      stage: "inspect",
      summary: "Ready for review.",
      report: {},
      remaining_stages: ["tokenize"],
      allowed_actions: ["continue", "pause"],
    };
    render(<ControlledCard interrupt={interrupt} onDecision={onDecision} />);

    fireEvent.click(screen.getByRole("button", { name: "Continue workflow" }));

    expect(screen.getByText("Submitting: Continue workflow…")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Submitting…" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Pause workflow" }) as HTMLButtonElement).disabled).toBe(true);
    resolveDecision?.();
    expect(await screen.findByText("Submitted: Continue workflow")).toBeTruthy();
  });
});
