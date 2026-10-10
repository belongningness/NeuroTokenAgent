import ChevronDown from "lucide-react/dist/esm/icons/chevron-down";
import ChevronRight from "lucide-react/dist/esm/icons/chevron-right";
import type { WorkspaceSnapshot } from "./contracts";
import type { LangGraphArtifact } from "./runtime";

type StageOutputPanelProps = {
  workspace: WorkspaceSnapshot | null;
  error: string | null;
  artifactPreview: LangGraphArtifact | null;
  artifactError: string | null;
  retrySubmitting: boolean;
  editRetryOpen: boolean;
  editRetryRequest: string;
  editRetrySubmitting: boolean;
  editRetryError: string | null;
  onOpenArtifact: (artifactId: string) => void;
  onDownloadArtifact: (artifact: LangGraphArtifact) => void;
  onRetry: () => void;
  onBeginEditRetry: () => void;
  onCancelEditRetry: () => void;
  onEditRetryRequestChange: (value: string) => void;
  onEditRetry: () => void;
  collapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
};

function stageKey(stage: WorkspaceSnapshot["stages"][number], index: number): string {
  const scopedIdentity = [stage.scope, stage.run_id, stage.name].filter(Boolean).join(":");
  return stage.stage_id ?? (scopedIdentity || `${stage.name}:${index}`);
}

type StageGroup = {
  scope: "validation" | "full" | "legacy";
  label: string;
  stages: WorkspaceSnapshot["stages"];
};

function stageGroups(workspace: WorkspaceSnapshot | null): StageGroup[] {
  const stages = workspace?.stages ?? [];
  const groups: StageGroup[] = ([
    { scope: "validation", label: "Single-file validation", stages: stages.filter((stage) => stage.scope === "validation") },
    { scope: "full", label: "Full dataset run", stages: stages.filter((stage) => stage.scope === "full") },
  ] satisfies StageGroup[]).filter((group) => group.stages.length > 0);
  const legacy = stages.filter((stage) => stage.scope !== "validation" && stage.scope !== "full");
  if (legacy.length > 0) groups.push({ scope: "legacy", label: groups.length ? "Other stages" : "Pipeline stages", stages: legacy });
  return groups;
}

function findingList(workspace: WorkspaceSnapshot | null) {
  const findings = workspace?.error?.details?.findings;
  return Array.isArray(findings)
    ? findings.filter((finding): finding is Record<string, unknown> => Boolean(finding && typeof finding === "object"))
    : [];
}

export function StageOutputPanel({
  workspace,
  error,
  artifactPreview,
  artifactError,
  retrySubmitting,
  editRetryOpen,
  editRetryRequest,
  editRetrySubmitting,
  editRetryError,
  onOpenArtifact,
  onDownloadArtifact,
  onRetry,
  onBeginEditRetry,
  onCancelEditRetry,
  onEditRetryRequestChange,
  onEditRetry,
  collapsed = false,
  onCollapsedChange,
}: StageOutputPanelProps) {
  const findings = findingList(workspace);
  const groupedStages = stageGroups(workspace);
  const rawSummary = workspace?.error?.details?.summary;
  const summary = rawSummary && typeof rawSummary === "object" && !Array.isArray(rawSummary)
    ? rawSummary as Record<string, unknown>
    : null;

  return (
    <aside className={`neurotoken-stage-output${collapsed ? " is-collapsed" : ""}`} aria-label="Stage output">
      <header>
        <button type="button" aria-label={collapsed ? "Expand stage output" : "Collapse stage output"} onClick={() => onCollapsedChange?.(!collapsed)}>
          {collapsed ? <ChevronRight /> : <ChevronDown />}
        </button>
        {!collapsed && <><strong>Stage output</strong><span>{workspace?.mode.replace(/_/g, " ") ?? "No run"}</span></>}
      </header>
      {!collapsed && (
        <div className="neurotoken-stage-output__body">
          {groupedStages.length ? (
            <div className="neurotoken-stage-groups">
              {groupedStages.map((group) => (
                <section className="neurotoken-stage-list" aria-label={group.label} key={group.scope} data-scope={group.scope}>
                  <header><strong>{group.label}</strong><span>{group.stages.length} stage{group.stages.length === 1 ? "" : "s"}</span></header>
                  {group.stages.map((stage, index) => (
                    <article key={stageKey(stage, index)} className={`status-${stage.status}`}>
                      <i>{index + 1}</i>
                      <div>
                        <strong>{stage.name}</strong>
                        <span>{stage.status}{stage.run_id ? ` · ${stage.run_id}` : ""}</span>
                      </div>
                      {stage.duration_seconds != null && <small>{stage.duration_seconds.toFixed(2)}s</small>}
                    </article>
                  ))}
                </section>
              ))}
            </div>
          ) : <p className="neurotoken-stage-empty">Stage reports will appear here when a pipeline starts.</p>}

          {workspace?.external_stage_job && (
            <section className="neurotoken-stage-attempt" aria-label="Durable stage attempt">
              <span>{workspace.external_stage_job.status.replace(/_/g, " ")}</span>
              <strong>{workspace.external_stage_job.stage}</strong>
              {(workspace.external_stage_job.phase || workspace.external_stage_job.run_id) && (
                <small>{[workspace.external_stage_job.phase, workspace.external_stage_job.run_id].filter(Boolean).join(" · ")}</small>
              )}
              <code>{workspace.external_stage_job.stage_execution_id}</code>
              {workspace.external_stage_job.updated_at && <small>Updated {workspace.external_stage_job.updated_at}</small>}
              {workspace.external_stage_job.log_tail && <pre>{workspace.external_stage_job.log_tail}</pre>}
            </section>
          )}

          {error && (
            <section className="neurotoken-stage-failure" role="alert">
              <span>Run failed{workspace?.error?.stage ? ` · ${workspace.error.stage}` : ""}</span>
              <strong>{workspace?.error?.message ?? error}</strong>
              {summary && (
                <div className="neurotoken-stage-summary">
                  {Object.entries(summary).map(([key, value]) => <span key={key}><b>{String(value)}</b>{key}</span>)}
                </div>
              )}
              {findings.length > 0 && (
                <ul>
                  {findings.map((finding, index) => (
                    <li key={`${String(finding.category ?? "finding")}-${index}`} className={`severity-${String(finding.severity ?? "info")}`}>
                      <span>{String(finding.severity ?? "info")}</span>
                      <p>{String(finding.message ?? "Unknown validation finding")}</p>
                    </li>
                  ))}
                </ul>
              )}
              {editRetryOpen && (
                <textarea aria-label="Edit request before retry" value={editRetryRequest} disabled={editRetrySubmitting} onChange={(event) => onEditRetryRequestChange(event.target.value)} />
              )}
              {editRetryError && editRetryOpen && <p className="is-error">{editRetryError}</p>}
              {!new Set(["authoring_adapter", "ready_for_coding", "adapter_revision_required", "awaiting_patch_approval"]).has(workspace?.mode ?? "") && (
                <footer>
                  {editRetryOpen ? (
                    <><button type="button" onClick={onCancelEditRetry}>Cancel edit</button><button type="button" disabled={!editRetryRequest.trim() || editRetrySubmitting} onClick={onEditRetry}>{editRetrySubmitting ? "Starting…" : "Run edited request"}</button></>
                  ) : (
                    <><button type="button" disabled={retrySubmitting} onClick={onRetry}>{retrySubmitting ? "Retrying…" : "Retry"}</button><button type="button" onClick={onBeginEditRetry}>Edit & retry</button></>
                  )}
                </footer>
              )}
            </section>
          )}

          {Boolean(workspace?.artifacts.length) && (
            <section className="neurotoken-stage-artifacts" aria-label="Task artifacts">
              <h3>Stage artifacts</h3>
              <p className="neurotoken-stage-artifacts__hint">Click a report, README, or document to inspect its contents.</p>
              {workspace?.artifacts.map((artifact) => (
                <button type="button" key={artifact.artifact_id} onClick={() => onOpenArtifact(artifact.artifact_id)}>
                  <span>{artifact.stage ?? artifact.kind}</span><strong>{artifact.file_name ?? artifact.summary}</strong><small>{artifact.media_type}</small>
                </button>
              ))}
              {artifactError && <p role="alert">{artifactError}</p>}
              {artifactPreview && (
                <div className="neurotoken-artifact-preview">
                  <header><strong>{artifactPreview.fileName}</strong><button type="button" onClick={() => onDownloadArtifact(artifactPreview)}>Download</button></header>
                  <pre>{artifactPreview.content}</pre>
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </aside>
  );
}
