import type {
  AdapterCandidateSubmission,
  AdapterCodingHandoff,
  NeuroTokenInterrupt,
  WorkspaceSnapshot,
} from "./contracts";
import type {
  LangGraphClient,
  LangGraphDecision,
  LangGraphEvent,
  LangGraphThread,
  Unsubscribe,
} from "./runtime";

export type LangGraphMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type MockLangGraphView = {
  workspace: WorkspaceSnapshot | null;
  interrupt: NeuroTokenInterrupt | null;
  handoff: AdapterCodingHandoff | null;
  messages: LangGraphMessage[];
  error: string | null;
};

export const EMPTY_LANGGRAPH_VIEW: MockLangGraphView = {
  workspace: null,
  interrupt: null,
  handoff: null,
  messages: [],
  error: null,
};

function messageFingerprint(message: Pick<LangGraphMessage, "role" | "content">): string {
  return `${message.role}:${message.content.replace(/\s+/g, " ").trim()}`;
}

function isOptimisticMessage(message: LangGraphMessage): boolean {
  return message.id.startsWith("user:") || message.id.startsWith("user:retry:");
}

function mergeSnapshotMessages(
  current: MockLangGraphView["messages"],
  incoming: LangGraphEvent & { type: "snapshot" },
): MockLangGraphView["messages"] {
  const merged = current.map((message) => ({ ...message }));
  const indexById = new Map(merged.map((message, index) => [message.id, index]));
  const optimisticIndexByFingerprint = new Map(
    merged.flatMap((message, index) => isOptimisticMessage(message)
      ? [[messageFingerprint(message), index] as const]
      : []),
  );
  for (const message of incoming.snapshot.messages) {
    const normalized = {
      id: message.id ?? `${message.role}:${message.content}`,
      role: message.role,
      content: message.content,
    };
    const idIndex = indexById.get(normalized.id);
    const optimisticIndex = optimisticIndexByFingerprint.get(messageFingerprint(normalized));
    const index = idIndex ?? optimisticIndex;
    if (index === undefined) {
      indexById.set(normalized.id, merged.length);
      merged.push(normalized);
    } else {
      const replaced = merged[index];
      indexById.delete(replaced.id);
      optimisticIndexByFingerprint.delete(messageFingerprint(replaced));
      merged[index] = normalized;
      indexById.set(normalized.id, index);
    }
  }
  return merged;
}

export function reduceLangGraphEvent(
  state: MockLangGraphView,
  event: LangGraphEvent,
): MockLangGraphView {
  switch (event.type) {
    case "snapshot": {
      const supersededPatchApproval = event.snapshot.interrupt?.kind === "adapter_patch_approval" &&
        event.snapshot.handoff?.task_id === event.snapshot.interrupt.task_id &&
        event.snapshot.handoff.revision > event.snapshot.interrupt.revision;
      const normalizedInterrupt = supersededPatchApproval ? null : event.snapshot.interrupt;
      const normalizedWorkspace = supersededPatchApproval && event.snapshot.workspace
        ? { ...event.snapshot.workspace, mode: "ready_for_coding", is_working: false, approval: null }
        : event.snapshot.workspace;
      const currentTaskId = state.handoff?.task_id ??
        (state.interrupt?.kind === "adapter_patch_approval" ? state.interrupt.task_id : null);
      const currentRevision = state.handoff?.revision ??
        (state.interrupt?.kind === "adapter_patch_approval" ? state.interrupt.revision : null);
      const incomingTaskId = event.snapshot.handoff?.task_id ??
        (normalizedInterrupt?.kind === "adapter_patch_approval" ? normalizedInterrupt.task_id : null);
      const incomingRevision = event.snapshot.handoff?.revision ??
        (normalizedInterrupt?.kind === "adapter_patch_approval" ? normalizedInterrupt.revision : null);
      if (
        currentTaskId && incomingTaskId === currentTaskId &&
        currentRevision != null && incomingRevision != null && incomingRevision < currentRevision
      ) {
        return { ...state, messages: mergeSnapshotMessages(state.messages, event) };
      }
      return {
        workspace: normalizedWorkspace,
        interrupt: normalizedInterrupt,
        handoff: event.snapshot.handoff,
        messages: mergeSnapshotMessages(state.messages, event),
        error: event.snapshot.workspace?.mode === "completed" || event.snapshot.workspace?.mode === "adapter_ready"
          ? null
          : event.snapshot.error,
      };
    }
    case "workspace":
      return { ...state, workspace: event.snapshot };
    case "interrupt":
      if (
        event.interrupt.kind === "adapter_patch_approval" &&
        state.handoff?.task_id === event.interrupt.task_id &&
        state.handoff.revision > event.interrupt.revision
      ) return state;
      return { ...state, interrupt: event.interrupt };
    case "message": {
      const id = `${event.role}:${event.content}`;
      if (state.messages.some((message) => message.id === id)) return state;
      return {
        ...state,
        messages: [...state.messages, { id, role: event.role, content: event.content }],
      };
    }
    case "error":
      return { ...state, error: event.message };
    case "handoff": {
      const supersedesPatchApproval = state.interrupt?.kind === "adapter_patch_approval" &&
        state.interrupt.task_id === event.handoff.task_id &&
        state.interrupt.revision < event.handoff.revision;
      return {
        ...state,
        handoff: event.handoff,
        interrupt: supersedesPatchApproval ? null : state.interrupt,
        workspace: supersedesPatchApproval && state.workspace
          ? { ...state.workspace, mode: "ready_for_coding", is_working: false, approval: null }
          : state.workspace,
      };
    }
  }
}

const BASE_RUN = {
  workflow_id: "workflow-demo-001",
  run_id: "run-demo-001",
  report_root: null,
  config_path: null,
};

const PLAN_INTERRUPT: NeuroTokenInterrupt = {
  schema_version: "1.0",
  kind: "plan_approval",
  owner: "local-user",
  thread_id: "thread-pipeline-001",
  plan: {
    stages: ["preflight", "tokenize", "package"],
    target_h5: "/datasets/example.h5",
  },
  allowed_actions: ["approve", "refine", "reject"],
};

const STAGE_INTERRUPT: NeuroTokenInterrupt = {
  schema_version: "1.0",
  kind: "stage_review",
  owner: "local-user",
  thread_id: "thread-run-001",
  workflow_id: "workflow-demo-001",
  stage: "preflight",
  summary: "Input structure and channel metadata passed preflight checks.",
  report: { status: "ok", datasets: 14 },
  remaining_stages: ["tokenize", "package"],
  allowed_actions: ["continue", "pause", "cancel"],
};

type PatchInterrupt = Extract<NeuroTokenInterrupt, { kind: "adapter_patch_approval" }>;
const PATCH_INTERRUPT: PatchInterrupt = {
  schema_version: "1.0",
  kind: "adapter_patch_approval",
  owner: "local-user",
  thread_id: "thread-adapter-001",
  task_id: "task-adapter-001",
  revision: 2,
  adapter: {
    task_id: "task-adapter-001",
    name: "example_eeg",
    revision: 2,
    coding_summary: "Read EEG channels and sampling metadata from HDF5 attributes.",
    design_document: "# Adapter Design\n\nReview the EEG source mapping before applying.",
    diff: "diff --git a/agent_pipeline/adapters/example_eeg.py b/agent_pipeline/adapters/example_eeg.py\n--- a/agent_pipeline/adapters/example_eeg.py\n+++ b/agent_pipeline/adapters/example_eeg.py\n@@ -1 +1 @@\n-ADAPTER = 'old'\n+ADAPTER = 'example_eeg'",
  },
  allowed_actions: ["apply", "apply_and_commit", "ask", "refine", "reject"],
};

export const MOCK_THREAD_EVENTS: Record<string, LangGraphEvent[]> = {
  "thread-pipeline-001": [
    {
      type: "message",
      role: "user",
      content: "Prepare NeuroTokens from the example EEG recording.",
    },
    {
      type: "message",
      role: "assistant",
      content: "I inspected the request and prepared a three-stage pipeline plan.",
    },
    {
      type: "workspace",
      snapshot: {
        schema_version: "1.0",
        mode: "awaiting_plan_approval",
        intent: "execute",
        is_working: false,
        plan: {
          stages: ["preflight", "tokenize", "package"],
          input_path: "/datasets/example.h5",
          output_path: "/outputs/example",
          adapter: "example_eeg",
        },
        run: {},
        stages: [
          { name: "preflight", status: "pending" },
          { name: "tokenize", status: "pending" },
          { name: "package", status: "pending" },
        ],
        artifacts: [],
        approval: { kind: "plan_approval", title: "Confirm pipeline plan" },
        error: null,
      },
    },
    { type: "interrupt", interrupt: PLAN_INTERRUPT },
  ],
  "thread-run-001": [
    {
      type: "message",
      role: "assistant",
      content: "Preflight completed. Review the report before tokenization starts.",
    },
    {
      type: "workspace",
      snapshot: {
        schema_version: "1.0",
        mode: "reviewing_stage",
        intent: "execute",
        is_working: false,
        plan: {
          stages: ["preflight", "tokenize", "package"],
          input_path: "/datasets/subject-0001.h5",
          output_path: "/outputs/subject-0001",
          adapter: "example_eeg",
        },
        run: BASE_RUN,
        stages: [
          { name: "preflight", status: "completed", duration_seconds: 4.8 },
          { name: "tokenize", status: "pending" },
          { name: "package", status: "pending" },
        ],
        artifacts: [
          {
            artifact_id: "artifact-preflight-001",
            kind: "stage_report",
            stage: "preflight",
            summary: "HDF5 structure and EEG metadata validated",
            storage_uri: "artifact://preflight/report.json",
            media_type: "application/json",
            created_at: "2026-08-04T07:34:00Z",
          },
        ],
        approval: {
          kind: "stage_review",
          title: "Review preflight",
          stage: "preflight",
        },
        error: null,
      },
    },
    { type: "interrupt", interrupt: STAGE_INTERRUPT },
  ],
  "thread-adapter-001": [
    {
      type: "message",
      role: "assistant",
      content: "Adapter revision 2 is ready for final review.",
    },
    {
      type: "workspace",
      snapshot: {
        schema_version: "1.0",
        mode: "awaiting_patch_approval",
        intent: "author_adapter",
        is_working: false,
        plan: null,
        run: {},
        stages: [],
        artifacts: [],
        approval: {
          kind: "adapter_patch_approval",
          title: "Review example EEG adapter",
          request_id: "task-adapter-001",
        },
        error: null,
      },
    },
    { type: "interrupt", interrupt: PATCH_INTERRUPT },
  ],
};

export function viewForThread(threadId: string): MockLangGraphView {
  return (MOCK_THREAD_EVENTS[threadId] ?? []).reduce(
    reduceLangGraphEvent,
    EMPTY_LANGGRAPH_VIEW,
  );
}

export class MockLangGraphClient implements LangGraphClient {
  private listeners = new Map<string, Set<(event: LangGraphEvent) => void>>();

  async listThreads(): Promise<LangGraphThread[]> {
    return Object.keys(MOCK_THREAD_EVENTS).map((threadId) => ({
      threadId,
      title: threadId.replace("thread-", ""),
      updatedAt: "2026-08-04T07:42:00Z",
      workspaceMode: viewForThread(threadId).workspace?.mode,
    }));
  }

  async createThread(): Promise<LangGraphThread> {
    return {
      threadId: "thread-new-001",
      title: "New task",
      updatedAt: "2026-08-04T07:42:00Z",
    };
  }

  async deleteThread(threadId: string): Promise<void> {
    delete MOCK_THREAD_EVENTS[threadId];
    this.listeners.delete(threadId);
  }

  async sendMessage(threadId: string, content: string): Promise<void> {
    this.emit(threadId, { type: "message", role: "user", content });
  }

  async retry(threadId: string): Promise<void> {
    this.emit(threadId, { type: "message", role: "assistant", content: "Retry started" });
  }

  async submitCandidate(submission: AdapterCandidateSubmission): Promise<void> {
    this.emit(submission.thread_id, {
      type: "interrupt",
      interrupt: {
        ...PATCH_INTERRUPT,
        thread_id: submission.thread_id,
        task_id: submission.task_id,
        revision: submission.revision,
        adapter: {
          ...PATCH_INTERRUPT.adapter,
          task_id: submission.task_id,
          revision: submission.revision,
        },
      },
    });
  }

  async resume(threadId: string, decision: LangGraphDecision): Promise<void> {
    const decisionMessage = {
      id: `decision:${decision.action}`,
      role: "assistant" as const,
      content: `Decision recorded: ${decision.action}`,
    };
    this.emit(threadId, {
      type: "message",
      role: decisionMessage.role,
      content: decisionMessage.content,
    });
    if (
      threadId === PATCH_INTERRUPT.thread_id &&
      (decision.action === "apply" || decision.action === "apply_and_commit")
    ) {
      this.emit(threadId, {
        type: "snapshot",
        snapshot: {
          workspace: {
            schema_version: "1.0",
            mode: "adapter_ready",
            intent: "author_adapter",
            is_working: false,
            plan: null,
            run: {},
            stages: [],
            artifacts: [],
            approval: null,
            error: null,
          },
          interrupt: null,
          handoff: null,
          messages: [decisionMessage],
          error: null,
        },
      });
    }
    if (threadId === PATCH_INTERRUPT.thread_id && decision.action === "refine") {
      this.emit(threadId, {
        type: "handoff",
        handoff: {
          schema_version: "1.0",
          task_id: PATCH_INTERRUPT.task_id,
          thread_id: threadId,
          revision: PATCH_INTERRUPT.revision + 1,
          status: "ready_for_coding",
          adapter: {
            name: String(PATCH_INTERRUPT.adapter.name ?? "example_eeg"),
            modality: "eeg",
            notes: "",
          },
          allowed_paths: ["agent_pipeline/adapters"],
          instruction: decision.message ?? "Continue refining the adapter.",
          created_at: "2026-08-04T07:42:00Z",
        },
      });
    }
  }

  async getArtifact(threadId: string, artifactId: string) {
    const artifact = viewForThread(threadId).workspace?.artifacts.find(
      (candidate) => candidate.artifact_id === artifactId,
    );
    if (!artifact) throw new Error(`Unknown artifact: ${artifactId}`);
    return {
      artifactId,
      mediaType: artifact.media_type,
      fileName: `${artifactId}.json`,
      content: JSON.stringify({ artifact_id: artifactId, summary: artifact.summary }, null, 2),
    };
  }

  subscribe(threadId: string, onEvent: (event: LangGraphEvent) => void): Unsubscribe {
    const listeners = this.listeners.get(threadId) ?? new Set();
    listeners.add(onEvent);
    this.listeners.set(threadId, listeners);
    for (const event of MOCK_THREAD_EVENTS[threadId] ?? []) onEvent(event);
    return () => {
      listeners.delete(onEvent);
      if (listeners.size === 0) this.listeners.delete(threadId);
    };
  }

  private emit(threadId: string, event: LangGraphEvent) {
    for (const listener of this.listeners.get(threadId) ?? []) listener(event);
  }
}
