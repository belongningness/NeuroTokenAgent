/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StageOutputPanel } from "./StageOutputPanel";
import type { WorkspaceSnapshot } from "./contracts";

afterEach(cleanup);

const workspace: WorkspaceSnapshot = {
  schema_version: "1.0",
  mode: "failed",
  intent: "execute",
  is_working: false,
  plan: null,
  run: {},
  stages: [
    { name: "preflight", status: "completed", duration_seconds: 0.03 },
    { name: "validate", status: "failed" },
  ],
  artifacts: [],
  approval: null,
  error: {
    code: "stage_validation_failed",
    stage: "validate",
    message: "Missing required file: DATA_DICTIONARY.md; Missing required directory: schemas",
    retryable: true,
    details: {
      summary: { errors: 2, warnings: 8 },
      findings: [
        { severity: "error", category: "format_error", message: "Missing required file: DATA_DICTIONARY.md" },
        { severity: "warning", category: "quality_warning", message: "Schema missing for manifest.json" },
      ],
    },
  },
};

function renderPanel(overrides: { collapsed?: boolean; onCollapsedChange?: (collapsed: boolean) => void } = {}) {
  render(<StageOutputPanel workspace={workspace} error="validate failed" artifactPreview={null} artifactError={null} retrySubmitting={false} editRetryOpen={false} editRetryRequest="" editRetrySubmitting={false} editRetryError={null} onOpenArtifact={vi.fn()} onDownloadArtifact={vi.fn()} onRetry={vi.fn()} onBeginEditRetry={vi.fn()} onCancelEditRetry={vi.fn()} onEditRetryRequestChange={vi.fn()} onEditRetry={vi.fn()} {...overrides} />);
}

describe("StageOutputPanel", () => {
  it("groups validation and full stages without colliding on repeated names", () => {
    render(<StageOutputPanel workspace={{
      ...workspace,
      mode: "running",
      error: null,
      stages: [
        { name: "tokenize", status: "completed", stage_id: "validation:tokenize", scope: "validation", run_id: "validation-1" },
        { name: "tokenize", status: "running", stage_id: "full:tokenize", scope: "full", run_id: "full-1" },
      ],
    }} error={null} artifactPreview={null} artifactError={null} retrySubmitting={false} editRetryOpen={false} editRetryRequest="" editRetrySubmitting={false} editRetryError={null} onOpenArtifact={vi.fn()} onDownloadArtifact={vi.fn()} onRetry={vi.fn()} onBeginEditRetry={vi.fn()} onCancelEditRetry={vi.fn()} onEditRetryRequestChange={vi.fn()} onEditRetry={vi.fn()} />);

    const validation = screen.getByRole("region", { name: "Single-file validation" });
    const full = screen.getByRole("region", { name: "Full dataset run" });
    expect(within(validation).getByText("tokenize")).toBeTruthy();
    expect(within(validation).getByText(/validation-1/)).toBeTruthy();
    expect(within(full).getByText("tokenize")).toBeTruthy();
    expect(within(full).getByText(/full-1/)).toBeTruthy();
  });

  it("keeps legacy unscoped stages in one compatible section", () => {
    renderPanel();
    const stages = screen.getByRole("region", { name: "Pipeline stages" });
    expect(within(stages).getByText("preflight")).toBeTruthy();
    expect(within(stages).getByText("validate")).toBeTruthy();
  });

  it("shows durable attempt identity and reconnect log tail", () => {
    render(<StageOutputPanel workspace={{ ...workspace, mode: "detached", external_stage_job: { stage_execution_id: "stage_tokenize_123", stage: "tokenize", status: "detached", phase: "validation", run_id: "validation-1", log_tail: "encoding batch 4/20", updated_at: "2026-08-17T07:00:00Z", reconnectable: true } }} error={null} artifactPreview={null} artifactError={null} retrySubmitting={false} editRetryOpen={false} editRetryRequest="" editRetrySubmitting={false} editRetryError={null} onOpenArtifact={vi.fn()} onDownloadArtifact={vi.fn()} onRetry={vi.fn()} onBeginEditRetry={vi.fn()} onCancelEditRetry={vi.fn()} onEditRetryRequestChange={vi.fn()} onEditRetry={vi.fn()} />);
    expect(screen.getByRole("region", { name: "Durable stage attempt" })).toBeTruthy();
    expect(screen.getByText("stage_tokenize_123")).toBeTruthy();
    expect(screen.getByText("validation · validation-1")).toBeTruthy();
    expect(screen.getByText("encoding batch 4/20")).toBeTruthy();
  });

  it("shows concrete validation findings separately from conversation", () => {
    renderPanel();
    const panel = screen.getByRole("complementary", { name: "Stage output" });
    expect(within(panel).getByText("Missing required file: DATA_DICTIONARY.md")).toBeTruthy();
    expect(within(panel).getByText("Schema missing for manifest.json")).toBeTruthy();
    expect(within(panel).getByText("errors").querySelector("b")?.textContent).toBe("2");
    expect(within(panel).getByText("warnings").querySelector("b")?.textContent).toBe("8");
  });

  it("lists file names for clickable stage artifacts", () => {
    const artifact = {
      artifact_id: "docs-readme",
      kind: "stage_document",
      stage: "docs",
      summary: "docs output: README.md",
      file_name: "README.md",
      media_type: "text/markdown",
    };
    render(<StageOutputPanel workspace={{ ...workspace, mode: "reviewing_stage", error: null, artifacts: [artifact] }} error={null} artifactPreview={null} artifactError={null} retrySubmitting={false} editRetryOpen={false} editRetryRequest="" editRetrySubmitting={false} editRetryError={null} onOpenArtifact={vi.fn()} onDownloadArtifact={vi.fn()} onRetry={vi.fn()} onBeginEditRetry={vi.fn()} onCancelEditRetry={vi.fn()} onEditRetryRequestChange={vi.fn()} onEditRetry={vi.fn()} />);
    expect(screen.getByText("README.md")).toBeTruthy();
    expect(screen.getByText("Click a report, README, or document to inspect its contents.")).toBeTruthy();
  });

  it("does not offer pipeline retry for an adapter authoring error", () => {
    render(<StageOutputPanel workspace={{ ...workspace, mode: "adapter_revision_required", error: { ...workspace.error!, retryable: false } }} error="candidate submit failed" artifactPreview={null} artifactError={null} retrySubmitting={false} editRetryOpen={false} editRetryRequest="" editRetrySubmitting={false} editRetryError={null} onOpenArtifact={vi.fn()} onDownloadArtifact={vi.fn()} onRetry={vi.fn()} onBeginEditRetry={vi.fn()} onCancelEditRetry={vi.fn()} onEditRetryRequestChange={vi.fn()} onEditRetry={vi.fn()} />);
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Edit & retry" })).toBeNull();
  });

  it("notifies the parent so the layout column collapses with the panel", () => {
    const onCollapsedChange = vi.fn();
    renderPanel({ onCollapsedChange });
    fireEvent.click(screen.getByRole("button", { name: "Collapse stage output" }));
    expect(onCollapsedChange).toHaveBeenCalledWith(true);
  });

  it("hides panel contents when controlled as collapsed", () => {
    renderPanel({ collapsed: true });
    expect(screen.getByRole("button", { name: "Expand stage output" })).toBeTruthy();
    expect(screen.queryByText("Schema missing for manifest.json")).toBeNull();
  });
});
