export const NEUROTOKEN_PROTOCOL_VERSION = "1.0" as const;

export type NeuroTokenProtocolVersion = typeof NEUROTOKEN_PROTOCOL_VERSION;

export type AdapterCodingStatus =
  | "ready_for_coding"
  | "coding"
  | "validating"
  | "awaiting_patch_approval"
  | "needs_revision"
  | "applied"
  | "rejected"
  | "cancelled"
  | "failed";

export type AdapterCodingDescriptor = {
  name: string;
  modality: string;
  notes: string;
};

export type AdapterCodingHandoff = {
  schema_version: NeuroTokenProtocolVersion;
  task_id: string;
  thread_id: string;
  revision: number;
  status: AdapterCodingStatus;
  adapter: AdapterCodingDescriptor;
  allowed_paths: string[];
  instruction: string;
  codex_thread_id?: string | null;
  created_at: string;
};

export type AdapterCandidateSubmission = {
  schema_version: NeuroTokenProtocolVersion;
  action: "submit_candidate";
  task_id: string;
  thread_id: string;
  codex_thread_id: string;
  replaces_codex_thread_id?: string | null;
  revision: number;
};

export type AdapterPatchDecision = {
  schema_version: NeuroTokenProtocolVersion;
  action: "apply" | "apply_and_commit" | "ask" | "reject" | "refine";
  task_id: string;
  revision: number;
  message: string | null;
  commit_message: string | null;
};

export type WorkspaceStageStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed";

export type WorkspaceRunScope = "single" | "validation" | "full";
export type WorkspaceActivePhase = WorkspaceRunScope | "gate";

export type WorkspaceStage = {
  name: string;
  status: WorkspaceStageStatus;
  duration_seconds?: number | null;
  stage_id?: string | null;
  scope?: WorkspaceRunScope | null;
  run_id?: string | null;
};

export type WorkspaceValidationGate = {
  gate_id: string;
  status?: string | null;
  verdict?: string | null;
  report?: Record<string, unknown> | null;
};

export type WorkspaceLinkedRun = Record<string, unknown> & {
  run_id?: string | null;
  scope?: WorkspaceRunScope | null;
  workflow_id?: string | null;
  status?: string | null;
};

export type WorkspacePlan = {
  stages: string[];
  input_path?: string | null;
  output_path?: string | null;
  adapter?: string | null;
  execution_strategy?: "single" | "linked" | null;
  active_phase?: WorkspaceActivePhase | null;
  validation_target_h5?: string | null;
  full_input_root?: string | null;
  validation_stages?: string[] | null;
  full_stages?: string[] | null;
  linked_runs?: WorkspaceLinkedRun[] | null;
  validation_gate?: WorkspaceValidationGate | null;
  [key: string]: unknown;
};

export type WorkspaceExternalStageJob = {
  stage_execution_id: string;
  stage: string;
  workflow_id?: string | null;
  run_id?: string | null;
  phase?: WorkspaceRunScope | null;
  status: "submitted" | "running" | "detached" | "reconnecting" | "completed" | "failed" | "cancelling" | "cancelled" | "orphaned";
  tmux_session?: string | null;
  stdout_log?: string | null;
  stderr_log?: string | null;
  log_tail?: string;
  updated_at?: string | null;
  reconnectable?: boolean;
};

export type WorkspaceArtifactRef = {
  artifact_id: string;
  kind: string;
  stage?: string | null;
  summary: string;
  file_name?: string | null;
  storage_uri?: string | null;
  media_type: string;
  created_at?: string | null;
};

export type WorkspaceSnapshot = {
  schema_version: NeuroTokenProtocolVersion;
  mode: string;
  intent?: string | null;
  is_working: boolean;
  plan?: WorkspacePlan | null;
  active_phase?: WorkspaceActivePhase | null;
  validation_target_h5?: string | null;
  full_input_root?: string | null;
  linked_runs?: WorkspaceLinkedRun[] | null;
  validation_gate?: WorkspaceValidationGate | null;
  run: {
    workflow_id?: string | null;
    run_id?: string | null;
    report_root?: string | null;
    config_path?: string | null;
    active_phase?: WorkspaceActivePhase | null;
    linked_run_ids?: Record<string, string> | null;
  };
  stages: WorkspaceStage[];
  artifacts: WorkspaceArtifactRef[];
  external_stage_job?: WorkspaceExternalStageJob | null;
  approval?: {
    kind: string;
    title: string;
    stage?: string | null;
    request_id?: string | null;
  } | null;
  error?: {
    code: string;
    message: string;
    retryable: boolean;
    stage?: string | null;
    details?: Record<string, unknown>;
  } | null;
};

export type NeuroTokenInterrupt =
  | {
      schema_version: NeuroTokenProtocolVersion;
      kind: "plan_approval";
      owner: string;
      thread_id: string;
      plan: Record<string, unknown>;
      plan_revision?: number;
      allowed_actions: Array<"approve" | "refine" | "reject">;
    }
  | {
      schema_version: NeuroTokenProtocolVersion;
      kind: "stage_review";
      owner: string;
      thread_id: string;
      workflow_id: string;
      stage: string;
      summary: string;
      report: Record<string, unknown>;
      remaining_stages: string[];
      allowed_actions: Array<"continue" | "pause" | "cancel">;
    }
  | {
      schema_version: NeuroTokenProtocolVersion;
      kind: "smoke_validation_review";
      owner: string;
      thread_id: string;
      workflow_id?: string | null;
      gate_id: string;
      validation_target_h5?: string | null;
      full_input_root?: string | null;
      status?: string | null;
      verdict?: string | null;
      report: Record<string, unknown>;
      allowed_actions: Array<"start_full_run" | "rerun_smoke" | "pause" | "cancel">;
    }
  | {
      schema_version: NeuroTokenProtocolVersion;
      kind: "adapter_execution_confirmation";
      owner: string;
      thread_id: string;
      adapter_name?: string | null;
      input_path?: string | null;
      output_path?: string | null;
      allowed_actions: Array<"continue" | "pause">;
    }
  | {
      schema_version: NeuroTokenProtocolVersion;
      kind: "adapter_patch_approval";
      owner: string;
      thread_id: string;
      task_id: string;
      revision: number;
      adapter: Record<string, unknown>;
      allowed_actions: Array<"apply" | "apply_and_commit" | "ask" | "refine" | "reject">;
    };
