import { Check, Circle, CircleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { Markdown } from "../messages/components/Markdown";
import type { NeuroTokenInterrupt, WorkspaceStage } from "./contracts";
import type { ApplyMode } from "./handoffState";
import type { LangGraphDecision } from "./runtime";
import "./WorkflowDecisionCard.css";

export type WorkflowDecisionCardProps = {
  interrupt: NeuroTokenInterrupt;
  stages: WorkspaceStage[];
  applyMode: ApplyMode;
  commitMessage: string;
  refine: string;
  question: string;
  onApplyModeChange: (mode: ApplyMode) => void;
  onCommitMessageChange: (message: string) => void;
  onRefineChange: (message: string) => void;
  onQuestionChange: (question: string) => void;
  onDecision: (decision: LangGraphDecision) => void | Promise<void>;
};

type InterruptAction = NeuroTokenInterrupt["allowed_actions"][number];

const ACTION_LABELS: Record<InterruptAction, string> = {
  approve: "Approve plan",
  refine: "Request revision",
  reject: "Reject",
  continue: "Continue workflow",
  pause: "Pause workflow",
  cancel: "Cancel workflow",
  apply: "Apply only",
  apply_and_commit: "Apply and commit",
  ask: "Ask",
  start_full_run: "Start full run",
  rerun_smoke: "Rerun single-file validation",
};

function titleFor(interrupt: NeuroTokenInterrupt): string {
  if (interrupt.kind === "plan_approval") return "Approve workflow plan";
  if (interrupt.kind === "stage_review") return `Review stage: ${interrupt.stage}`;
  if (interrupt.kind === "smoke_validation_review") return "Single-file validation complete";
  if (interrupt.kind === "adapter_execution_confirmation") return "Process data with this adapter?";
  return "Review and apply adapter";
}

function stageSummaryMarkdown(value: string): string {
  const normalized = value
    .replace(/\\n/g, "\n")
    .replace(/\s+(?=#{1,6}\s)/g, "\n\n")
    .replace(/\s+-\s+(?=\*\*[^*]+:\*\*)/g, "\n- ")
    .replace(/\s+(?=\*\*结论：\*\*|\*\*输出：\*\*)/g, "\n\n")
    .trim();
  return normalized.startsWith("- ") || normalized.startsWith("#") ? normalized : normalized.replace(/^/, "");
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function packagePreviewSessions(interrupt: NeuroTokenInterrupt) {
  if (interrupt.kind !== "stage_review" || interrupt.stage !== "package") return [];
  const sessions = interrupt.report.sessions;
  if (!Array.isArray(sessions)) return [];
  return sessions.flatMap((value) => {
    const session = asRecord(value);
    if (!session) return [];
    const entries = Array.isArray(session.tar_sample)
      ? session.tar_sample.flatMap((entry) => {
        const record = asRecord(entry);
        return record ? [record] : [];
      })
      : [];
    if (entries.length === 0 && !asRecord(session.tar_sample_metadata)) return [];
    return [{ session, entries }];
  });
}

function segmentationPreviewSessions(interrupt: NeuroTokenInterrupt) {
  if (interrupt.kind !== "stage_review" || interrupt.stage !== "package") return [];
  const sessions = interrupt.report.sessions;
  if (!Array.isArray(sessions)) return [];
  return sessions.flatMap((value, index) => {
    const session = asRecord(value);
    const review = asRecord(session?.segmentation_review);
    if (!session || !review) return [];
    const segments = Array.isArray(review.preview_segments)
      ? review.preview_segments.flatMap((segment) => {
        const record = asRecord(segment);
        return record ? [record] : [];
      })
      : [];
    return [{ session, review, segments, index }];
  });
}

function readmePreview(interrupt: NeuroTokenInterrupt) {
  if (interrupt.kind !== "stage_review" || interrupt.stage !== "readme") return null;
  return typeof interrupt.report.readme_content === "string" && interrupt.report.readme_content.trim()
    ? {
      content: interrupt.report.readme_content,
      path: typeof interrupt.report.readme_path === "string" ? interrupt.report.readme_path : "README.md",
    }
    : null;
}

const RECONSTRUCTION_METRICS = [
  { key: "pearson", label: "Pearson", target: "Higher is better · ideal 1" },
  { key: "ssim", label: "SSIM", target: "Higher is better · ideal 1" },
  { key: "mse", label: "MSE", target: "Lower is better · ideal 0" },
  { key: "mae", label: "MAE", target: "Lower is better · ideal 0" },
  { key: "relative_l2", label: "relative L2", target: "Lower is better · ideal 0" },
] as const;

function smokeValidationPreview(interrupt: NeuroTokenInterrupt) {
  if (interrupt.kind !== "smoke_validation_review") return null;
  const reportGate = asRecord(interrupt.report.validation_gate);
  const gateReport = asRecord(reportGate?.report);
  const metrics = asRecord(interrupt.report.metrics)
    ?? asRecord(interrupt.report.summary)
    ?? asRecord(gateReport?.metrics);
  const target = interrupt.validation_target_h5
    ?? (typeof interrupt.report.validation_target_h5 === "string" ? interrupt.report.validation_target_h5 : null)
    ?? (typeof interrupt.report.input_path === "string" ? interrupt.report.input_path : null);
  const verdict = interrupt.verdict
    ?? (typeof interrupt.report.verdict === "string" ? interrupt.report.verdict : null)
    ?? (typeof reportGate?.verdict === "string" ? reportGate.verdict : null)
    ?? "unknown";
  const status = interrupt.status
    ?? (typeof interrupt.report.status === "string" ? interrupt.report.status : null)
    ?? (typeof reportGate?.status === "string" ? reportGate.status : null);
  return { target, verdict, status, metrics };
}

function reconstructionMetricsPreview(interrupt: NeuroTokenInterrupt) {
  if (interrupt.kind !== "stage_review" || interrupt.stage !== "reconstruct") return null;
  const sessions = Array.isArray(interrupt.report.sessions)
    ? interrupt.report.sessions.flatMap((value) => {
      const session = asRecord(value);
      const metrics = asRecord(session?.metrics_mean);
      if (!session || !metrics) return [];
      return [{ session, metrics }];
    })
    : [];
  const summary = asRecord(interrupt.report.metrics);
  const evaluatedWindows = interrupt.report.evaluated_windows;
  return { sessions, summary, evaluatedWindows };
}

function formatMetric(value: unknown): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (value === 0) return "0";
  const absolute = Math.abs(value);
  if (absolute < 0.0001 || absolute >= 10_000) return value.toExponential(4);
  return value.toFixed(6).replace(/\.?0+$/, "");
}

function formatBytes(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "unknown size";
  if (value < 1_024) return `${value} B`;
  if (value < 1_048_576) return `${(value / 1_024).toFixed(1)} KiB`;
  return `${(value / 1_048_576).toFixed(1)} MiB`;
}

function formatPreviewNumber(value: unknown, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "unknown";
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

function sessionDisplayName(session: Record<string, unknown>, fallbackIndex: number): string {
  return typeof session.output_dir === "string"
    ? session.output_dir.split(/[\\/]/).filter(Boolean).pop() || `Session ${fallbackIndex + 1}`
    : `Session ${fallbackIndex + 1}`;
}

function adapterReviewContextMarkdown(adapter: Record<string, unknown>): string {
  const originalRequest = typeof adapter.original_user_request === "string"
    ? adapter.original_user_request.trim()
    : "";
  const effectiveConfig = asRecord(adapter.effective_config);
  const selectedConfig = effectiveConfig
    ? {
      dataset: asRecord(effectiveConfig.dataset),
      source: asRecord(effectiveConfig.source),
      chunking: asRecord(effectiveConfig.chunking),
      source_h5: asRecord(effectiveConfig.source_h5),
    }
    : null;
  return [
    originalRequest ? `## Original human request\n\n${originalRequest}` : "",
    selectedConfig
      ? `## Proposed effective config\n\n\`\`\`json\n${JSON.stringify(selectedConfig, null, 2)}\n\`\`\``
      : "",
  ].filter(Boolean).join("\n\n");
}

function intentReviewMarkdown(adapter: Record<string, unknown>): string {
  const review = asRecord(adapter.intent_review);
  if (!review) return "";
  const verdict = typeof review.verdict === "string" ? review.verdict : "unknown";
  const summary = typeof review.summary === "string" ? review.summary.trim() : "No review summary was returned.";
  const findings = Array.isArray(review.findings)
    ? review.findings.flatMap((value) => {
      const finding = asRecord(value);
      if (!finding) return [];
      const severity = typeof finding.severity === "string" ? finding.severity.toUpperCase() : "INFO";
      const title = typeof finding.title === "string" ? finding.title : "Untitled finding";
      const evidence = typeof finding.evidence === "string" ? finding.evidence : "";
      const requiredChange = typeof finding.required_change === "string" ? finding.required_change : "";
      return [`- **${severity}: ${title}**${evidence ? ` — ${evidence}` : ""}${requiredChange ? ` Required change: ${requiredChange}` : ""}`];
    })
    : [];
  return [
    `## Independent intent review · ${verdict.replace(/_/g, " ")}`,
    summary,
    ...findings,
  ].filter(Boolean).join("\n\n");
}

function descriptionFor(interrupt: NeuroTokenInterrupt): string {
  if (interrupt.kind === "plan_approval") {
    return "Review the proposed workflow before execution starts.";
  }
  if (interrupt.kind === "stage_review") return interrupt.summary;
  if (interrupt.kind === "smoke_validation_review") {
    return "Review the validation result for the selected H5 file before explicitly starting the full dataset run.";
  }
  if (interrupt.kind === "adapter_execution_confirmation") {
    const destination = interrupt.output_path ? ` and write results to ${interrupt.output_path}` : "";
    return `Adapter ${interrupt.adapter_name ?? "the new adapter"} is ready. Start processing ${interrupt.input_path ?? "the source data"}${destination}?`;
  }
  const adapter = interrupt.adapter;
  const design = typeof adapter.design_document === "string" ? adapter.design_document : "";
  const strategy = typeof adapter.recommended_chunking_strategy === "string" ? adapter.recommended_chunking_strategy : null;
  const strategyNote = strategy ? `\n\n**Recommended chunking:** \`${strategy}\`` : "";
  const context = adapterReviewContextMarkdown(adapter);
  const review = intentReviewMarkdown(adapter);
  return [
    `${design || `Review adapter revision ${interrupt.revision} before applying it.`}${strategyNote}`,
    context,
    review,
  ].filter(Boolean).join("\n\n");
}

function StageStatusIcon({ status }: { status: WorkspaceStage["status"] }) {
  if (status === "completed") return <Check aria-hidden="true" />;
  if (status === "failed") return <CircleAlert aria-hidden="true" />;
  return <Circle aria-hidden="true" />;
}

export function WorkflowDecisionCard({
  interrupt,
  stages,
  applyMode,
  commitMessage,
  refine,
  question,
  onApplyModeChange,
  onCommitMessageChange,
  onRefineChange,
  onQuestionChange,
  onDecision,
}: WorkflowDecisionCardProps) {
  const [pendingAction, setPendingAction] = useState<InterruptAction | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);

  const submitting = pendingAction !== null;
  const interruptIdentity = interrupt.kind === "smoke_validation_review"
    ? `${interrupt.thread_id}:smoke_validation_review:${interrupt.gate_id}`
    : interrupt.kind === "adapter_patch_approval"
      ? `${interrupt.thread_id}:adapter_patch_approval:${interrupt.task_id}:${interrupt.revision}`
      : JSON.stringify(interrupt);
  const allowedActions = interrupt.allowed_actions as InterruptAction[];
  const displayedStages = interrupt.kind === "plan_approval" && Array.isArray(interrupt.plan.stages)
    ? interrupt.plan.stages
      .filter((stage): stage is string => typeof stage === "string")
      .map((name) => stages.find((stage) => stage.name === name) ?? { name, status: "pending" as const })
    : stages;

  useEffect(() => {
    setPendingAction(null);
    setFeedback(null);
  }, [interruptIdentity]);

  function decisionFor(action: InterruptAction): LangGraphDecision {
    if (interrupt.kind === "adapter_patch_approval") {
      const message = action === "ask" ? question.trim() : action === "refine" ? refine.trim() : null;
      return {
        schema_version: interrupt.schema_version,
        action: action as "apply" | "apply_and_commit" | "ask" | "reject" | "refine",
        task_id: interrupt.task_id,
        revision: interrupt.revision,
        message,
        commit_message: action === "apply_and_commit" ? commitMessage.trim() : null,
      };
    }

    if (interrupt.kind === "smoke_validation_review") {
      return {
        schema_version: interrupt.schema_version,
        action: action as "start_full_run" | "rerun_smoke" | "pause" | "cancel",
        gate_id: interrupt.gate_id,
        plan: null,
        message: null,
      };
    }

    return {
      schema_version: interrupt.schema_version,
      action: action as "approve" | "refine" | "reject" | "continue" | "pause" | "cancel",
      plan: interrupt.kind === "plan_approval" && action === "approve" ? interrupt.plan : null,
      message: interrupt.kind === "plan_approval" && action === "refine" ? refine.trim() : null,
    };
  }

  function actionDisabled(action: InterruptAction): boolean {
    if (submitting) return true;
    if ((action === "apply" || action === "apply_and_commit") && !patchDiffAvailable) return true;
    if (action === "refine") return !refine.trim();
    if (action === "ask") return !question.trim();
    if (action === "apply_and_commit") return !commitMessage.trim();
    return false;
  }

  async function submit(action: InterruptAction) {
    if (actionDisabled(action)) return;
    setPendingAction(action);
    setFeedback(`Submitting: ${ACTION_LABELS[action]}…`);
    try {
      await onDecision(decisionFor(action));
      setFeedback(`Submitted: ${ACTION_LABELS[action]}`);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Unable to submit decision.");
    } finally {
      setPendingAction(null);
    }
  }

  const patchApplyAction = applyMode;
  const isFinalStageReview = interrupt.kind === "stage_review" && interrupt.remaining_stages.length === 0;
  const packageSessions = packagePreviewSessions(interrupt);
  const segmentationSessions = segmentationPreviewSessions(interrupt);
  const readme = readmePreview(interrupt);
  const smokeValidation = smokeValidationPreview(interrupt);
  const reconstruction = reconstructionMetricsPreview(interrupt);
  const gateAllowsFullRun = interrupt.kind !== "smoke_validation_review" || interrupt.verdict === "pass";
  const reconstructionHasMetrics = Boolean(reconstruction && (
    RECONSTRUCTION_METRICS.some(({ key }) => formatMetric(reconstruction.summary?.[key]) !== null) ||
    reconstruction.sessions.some(({ metrics }) => RECONSTRUCTION_METRICS.some(({ key }) => formatMetric(metrics[key]) !== null))
  ));
  const patchDiffAvailable = interrupt.kind !== "adapter_patch_approval"
    || (typeof interrupt.adapter.diff === "string" && interrupt.adapter.diff.trim().length > 0);
  const patchSecondaryActions = allowedActions.filter(
    (action) => action !== "apply" && action !== "apply_and_commit",
  );

  return (
    <section
      className={`workflow-decision-card workflow-decision-card--${interrupt.kind}`}
      aria-labelledby="workflow-decision-title"
      data-testid="workflow-decision-card"
    >
      <header className="workflow-decision-card__header">
        <span className="workflow-decision-card__kind">
          {interrupt.kind === "plan_approval"
            ? "Plan approval"
            : interrupt.kind === "stage_review"
              ? "Stage review"
              : interrupt.kind === "smoke_validation_review"
                ? "Validation gate"
                : interrupt.kind === "adapter_execution_confirmation"
                  ? "Adapter ready"
                  : "Adapter patch approval"}
        </span>
        <h2 id="workflow-decision-title">{titleFor(interrupt)}</h2>
        {interrupt.kind === "stage_review" || interrupt.kind === "adapter_patch_approval" ? (
          <Markdown
            value={stageSummaryMarkdown(descriptionFor(interrupt))}
            className="workflow-decision-card__summary markdown"
            showFilePath={false}
          />
        ) : (
          <p>{descriptionFor(interrupt)}</p>
        )}
      </header>

      {displayedStages.length > 0 && (
        <div className="workflow-decision-card__progress">
          <div className="workflow-decision-card__progress-heading">
            <span>Workflow progress</span>
            <strong>{displayedStages.filter((stage) => stage.status === "completed").length} / {displayedStages.length}</strong>
          </div>
          <ol className="workflow-decision-card__stages" aria-label="Workspace stages">
            {displayedStages.map((stage) => {
              const current = interrupt.kind === "stage_review" && stage.name === interrupt.stage;
              return (
                <li key={stage.name} data-status={stage.status} data-current={current || undefined}>
                  <i><StageStatusIcon status={stage.status} /></i>
                  <span>{stage.name}</span>
                </li>
              );
            })}
          </ol>
        </div>
      )}

      {interrupt.kind === "plan_approval" && (
        <div className="workflow-decision-card__plan">
          <div className="workflow-decision-card__plan-heading">
            <span>Current proposed plan</span>
            <strong>{Array.isArray(interrupt.plan.stages) ? `${interrupt.plan.stages.length} stages` : "Updated plan"}</strong>
          </div>
          <pre>{JSON.stringify(interrupt.plan, null, 2)}</pre>
        </div>
      )}

      {smokeValidation && (
        <section className="workflow-decision-card__validation" aria-label="Single-file validation result">
          <div className="workflow-decision-card__validation-heading">
            <span>Validation H5</span>
            <strong data-verdict={smokeValidation.verdict.toLowerCase()}>{smokeValidation.verdict.replace(/_/g, " ")}</strong>
          </div>
          <code>{smokeValidation.target ?? "Validation target not reported"}</code>
          {smokeValidation.status && <p>Status: <strong>{smokeValidation.status.replace(/_/g, " ")}</strong></p>}
          {smokeValidation.metrics && Object.keys(smokeValidation.metrics).length > 0 && (
            <dl aria-label="Validation metrics">
              {Object.entries(smokeValidation.metrics).map(([key, value]) => (
                <div key={key}><dt>{key.replace(/_/g, " ")}</dt><dd>{formatMetric(value) ?? String(value)}</dd></div>
              ))}
            </dl>
          )}
          {interrupt.kind === "smoke_validation_review" && interrupt.full_input_root && (
            <p>Full input root: <code>{interrupt.full_input_root}</code></p>
          )}
          <p className="workflow-decision-card__validation-note">
            The full dataset will not run until you choose Start full run.
          </p>
        </section>
      )}

      {interrupt.kind === "plan_approval" && (
        <label className="workflow-decision-card__field">
          <span>Plan refinement instructions</span>
          <textarea
            value={refine}
            disabled={submitting}
            onChange={(event) => onRefineChange(event.target.value)}
            placeholder="Describe how the workflow plan should change"
          />
        </label>
      )}

      {segmentationSessions.length > 0 && (
        <section className="workflow-decision-card__preview workflow-decision-card__segmentation" aria-label="Tar segmentation review">
          <div className="workflow-decision-card__preview-heading">
            <span>Tar segmentation review</span>
            <strong>{segmentationSessions.length} session{segmentationSessions.length === 1 ? "" : "s"}</strong>
          </div>
          {segmentationSessions.map(({ session, review, segments, index }) => {
            const strategy = String(review.strategy ?? (interrupt.kind === "stage_review" ? interrupt.report.chunking_strategy ?? "unknown" : "unknown"));
            const uncovered = Array.isArray(review.uncovered_ranges) ? review.uncovered_ranges : [];
            const overlaps = Array.isArray(review.overlaps) ? review.overlaps : [];
            const sessionName = sessionDisplayName(session, index);
            return (
              <article key={`${sessionName}-segmentation-${index}`}>
                <header>
                  <strong>{sessionName}</strong>
                  {typeof session.segmentation_manifest_path === "string" && <code>{session.segmentation_manifest_path}</code>}
                </header>
                <div className="workflow-decision-card__segmentation-rule">
                  <strong>{strategy === "rest" ? "Fixed sample windows" : "Semantic event intervals"}</strong>
                  <p>
                    {strategy === "rest"
                      ? <>Starting at sample 0, cut every <code>{String(review.split_points ?? "unknown")}</code> samples; keep the actual remainder as the final segment{review.downsample_factor ? <>; configured downsample factor: <code>{String(review.downsample_factor)}</code></> : null}.</>
                      : <>Use event boundaries to partition the complete recording from sample 0 to its end. Real event segments retain existing event metadata; all leading, internal, and trailing ranges remain ordinary segments without fabricated event metadata or type labels.</>}
                  </p>
                  <p>All displayed boundaries are half-open: the start is included and the end is excluded.</p>
                </div>
                <dl className="workflow-decision-card__segmentation-stats">
                  <div><dt>Sampling rate</dt><dd>{formatPreviewNumber(review.sfreq)} Hz</dd></div>
                  <div><dt>Recording</dt><dd>{String(review.total_samples ?? "unknown")} samples · {formatPreviewNumber(review.total_duration_seconds)} s</dd></div>
                  <div><dt>Segments</dt><dd>{String(review.segment_count ?? segments.length)}</dd></div>
                  <div><dt>Coverage</dt><dd>{formatPreviewNumber(review.coverage_percent)}%</dd></div>
                  <div><dt>Segment length</dt><dd>{formatPreviewNumber(review.min_segment_seconds)}–{formatPreviewNumber(review.max_segment_seconds)} s</dd></div>
                  {strategy === "event" && <div><dt>Composition</dt><dd>{String(review.event_segment_count ?? "unknown")} event · {String(review.ordinary_segment_count ?? "unknown")} ordinary</dd></div>}
                </dl>
                {(uncovered.length > 0 || overlaps.length > 0) && (
                  <div className="workflow-decision-card__segmentation-alert" role="note">
                    Manual check: {uncovered.length} uncovered range{uncovered.length === 1 ? "" : "s"} and {overlaps.length} overlap{overlaps.length === 1 ? "" : "s"} detected.
                  </div>
                )}
                {segments.length > 0 && (
                  <div className="workflow-decision-card__segmentation-table-wrap">
                    <table>
                      <thead>
                        <tr><th>Chunk</th><th>Sample interval</th><th>Time interval</th><th>Duration</th><th>Event</th></tr>
                      </thead>
                      <tbody>
                        {segments.map((segment, segmentIndex) => (
                          <tr key={`${String(segment.chunk_index ?? segmentIndex)}-${segmentIndex}`}>
                            <td>#{String(segment.chunk_index ?? segmentIndex)}</td>
                            <td><code>[{String(segment.start_sample)}, {String(segment.end_sample)})</code></td>
                            <td><code>[{formatPreviewNumber(segment.start_time)}, {formatPreviewNumber(segment.end_time)}) s</code></td>
                            <td>{formatPreviewNumber(segment.duration_seconds)} s</td>
                            <td>{[segment.event_type, segment.event_value].filter((value) => typeof value === "string" && value).join(" / ") || "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {typeof review.segment_count === "number" && review.segment_count > segments.length && (
                  <p className="workflow-decision-card__segmentation-note">
                    Showing {segments.length} representative boundaries across the start, middle, and end. The manifest contains all {review.segment_count} segments.
                  </p>
                )}
              </article>
            );
          })}
        </section>
      )}

      {packageSessions.length > 0 && (
        <section className="workflow-decision-card__preview" aria-label="Tar content sample">
          <div className="workflow-decision-card__preview-heading">
            <span>Tar content sample</span>
            <strong>{packageSessions.length} session{packageSessions.length === 1 ? "" : "s"}</strong>
          </div>
          {packageSessions.map(({ session, entries }, sessionIndex) => {
            const metadata = asRecord(session.tar_sample_metadata);
            const sessionName = sessionDisplayName(session, sessionIndex);
            return (
              <article key={`${sessionName}-${sessionIndex}`}>
                <header>
                  <strong>{sessionName || `Session ${sessionIndex + 1}`}</strong>
                  <code>{String(session.tar_path ?? "data.tar")}</code>
                </header>
                <ul>
                  {entries.map((entry, entryIndex) => (
                    <li key={`${String(entry.name ?? "entry")}-${entryIndex}`}>
                      <code>{String(entry.name ?? "Unnamed member")}</code>
                      <span>{formatBytes(entry.size)}</span>
                    </li>
                  ))}
                </ul>
                {metadata && (
                  <details>
                    <summary>Sample metadata · {String(session.tar_sample_metadata_member ?? "metadata.json")}</summary>
                    <pre>{JSON.stringify(metadata, null, 2)}</pre>
                  </details>
                )}
              </article>
            );
          })}
        </section>
      )}

      {readme && (
        <section className="workflow-decision-card__preview workflow-decision-card__readme" aria-label="README content preview">
          <div className="workflow-decision-card__preview-heading">
            <span>README preview</span>
            <code>{readme.path}</code>
          </div>
          <Markdown
            value={readme.content}
            className="workflow-decision-card__readme-content markdown"
            codeBlockStyle="message"
            showFilePath={false}
          />
        </section>
      )}

      {reconstruction && (
        <section className="workflow-decision-card__preview workflow-decision-card__metrics" aria-label="Reconstruction metrics">
          <div className="workflow-decision-card__preview-heading">
            <span>Obtained reconstruction metrics</span>
            <strong>{reconstruction.sessions.length} evaluated session{reconstruction.sessions.length === 1 ? "" : "s"}</strong>
          </div>
          {reconstructionHasMetrics ? (
            <>
              {RECONSTRUCTION_METRICS.some(({ key }) => formatMetric(reconstruction.summary?.[key]) !== null) && (
                <article className="workflow-decision-card__metric-group">
                  <header>
                    <strong>All evaluated windows</strong>
                    <code>{String(reconstruction.evaluatedWindows ?? "—")} window(s)</code>
                  </header>
                  <dl>
                    {RECONSTRUCTION_METRICS.map(({ key, label, target }) => {
                      const value = formatMetric(reconstruction.summary?.[key]);
                      return value === null ? null : (
                        <div key={key}>
                          <dt>{label}</dt>
                          <dd><strong>{value}</strong><span>{target}</span></dd>
                        </div>
                      );
                    })}
                  </dl>
                </article>
              )}
              {reconstruction.sessions.map(({ session, metrics }, index) => (
                <article className="workflow-decision-card__metric-group" key={`${String(session.session_id ?? "session")}-${index}`}>
                  <header>
                    <strong>{String(session.session_id ?? `Session ${index + 1}`)}</strong>
                    <code>{String(session.evaluated_windows ?? "—")} window(s) · {String(session.modality ?? "unknown")}</code>
                  </header>
                  <dl>
                    {RECONSTRUCTION_METRICS.map(({ key, label, target }) => {
                      const value = formatMetric(metrics[key]);
                      return value === null ? null : (
                        <div key={key}>
                          <dt>{label}</dt>
                          <dd><strong>{value}</strong><span>{target}</span></dd>
                        </div>
                      );
                    })}
                  </dl>
                </article>
              ))}
            </>
          ) : (
            <p className="workflow-decision-card__metrics-empty" role="status">
              No reconstruction window was evaluated, so this run did not produce Pearson, SSIM, MSE, MAE, or relative L2 values.
            </p>
          )}
        </section>
      )}

      {interrupt.kind === "stage_review" && interrupt.remaining_stages.length > 0 && (
        <div className="workflow-decision-card__remaining">
          <span>Up next</span>
          <p>{interrupt.remaining_stages.join(" → ")}</p>
        </div>
      )}

      {interrupt.kind === "adapter_patch_approval" && (
        <div className="workflow-decision-card__patch-controls">
          {!patchDiffAvailable && (
            <p className="workflow-decision-card__warning" role="alert">
              This checkpoint has no frozen candidate diff. Applying is disabled; request a revision and submit the candidate again.
            </p>
          )}
          <fieldset disabled={submitting}>
            <legend>Apply mode</legend>
            <label>
              <input
                type="radio"
                name={`patch-apply-mode-${interrupt.task_id}`}
                checked={applyMode === "apply"}
                onChange={() => onApplyModeChange("apply")}
              />
              Apply only
            </label>
            <label>
              <input
                type="radio"
                name={`patch-apply-mode-${interrupt.task_id}`}
                checked={applyMode === "apply_and_commit"}
                onChange={() => onApplyModeChange("apply_and_commit")}
              />
              Apply and commit
            </label>
          </fieldset>
          {applyMode === "apply_and_commit" && (
            <label className="workflow-decision-card__field">
              <span>Commit message</span>
              <input
                value={commitMessage}
                disabled={submitting}
                onChange={(event) => onCommitMessageChange(event.target.value)}
              />
            </label>
          )}
          <label className="workflow-decision-card__field">
            <span>Question about this patch</span>
            <textarea
              value={question}
              disabled={submitting}
              onChange={(event) => onQuestionChange(event.target.value)}
              placeholder="Ask why a patch change is needed"
            />
          </label>
          <label className="workflow-decision-card__field">
            <span>Patch refinement instructions</span>
            <textarea
              value={refine}
              disabled={submitting}
              onChange={(event) => onRefineChange(event.target.value)}
              placeholder="Describe the revision you want"
            />
          </label>
        </div>
      )}

      {feedback && (
        <p className="workflow-decision-card__feedback" role="status" aria-live="polite">
          {feedback}
        </p>
      )}

      <footer className="workflow-decision-card__actions">
        {(interrupt.kind === "adapter_patch_approval" ? patchSecondaryActions : allowedActions).map((action) => (
          <button
            type="button"
            key={action}
            className={action === "start_full_run" ? "workflow-decision-card__primary" : undefined}
            disabled={actionDisabled(action) || (action === "start_full_run" && !gateAllowsFullRun)}
            aria-busy={pendingAction === action}
            onClick={() => void submit(action)}
          >
            {pendingAction === action
              ? "Submitting…"
              : interrupt.kind === "adapter_execution_confirmation" && action === "continue"
                ? "Start processing"
                : interrupt.kind === "adapter_execution_confirmation" && action === "pause"
                  ? "Not now"
                  : isFinalStageReview && action === "continue"
                    ? "Complete workflow"
                    : ACTION_LABELS[action]}
          </button>
        ))}
        {interrupt.kind === "adapter_patch_approval" && allowedActions.includes(patchApplyAction) && (
          <button
            type="button"
            className="workflow-decision-card__primary"
            disabled={actionDisabled(patchApplyAction)}
            aria-busy={pendingAction === patchApplyAction}
            onClick={() => void submit(patchApplyAction)}
          >
            {pendingAction === patchApplyAction ? "Submitting…" : ACTION_LABELS[patchApplyAction]}
          </button>
        )}
      </footer>
    </section>
  );
}
