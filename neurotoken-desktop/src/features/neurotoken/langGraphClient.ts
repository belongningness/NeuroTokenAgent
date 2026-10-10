import type {
  AdapterCandidateSubmission,
  AdapterCodingHandoff,
  NeuroTokenInterrupt,
  WorkspaceSnapshot,
} from "./contracts";
import type {
  LangGraphArtifact,
  LangGraphClient,
  LangGraphDecision,
  LangGraphEvent,
  LangGraphThread,
  Unsubscribe,
} from "./runtime";

export type HttpLangGraphClientOptions = {
  apiUrl: string;
  assistantId?: string;
  apiKey?: string;
  authScheme?: string;
  fetch?: typeof fetch;
  reconnectDelayMs?: number;
  statePollIntervalMs?: number;
};

type JsonRecord = Record<string, unknown>;
type StateSnapshot = { values?: JsonRecord; tasks?: unknown[] };
type ArtifactEnvelope = JsonRecord & {
  artifact_id: string;
  media_type?: string;
  storage_uri?: string | null;
  payload?: unknown;
};

const DEFAULT_ASSISTANT_ID = "neurotoken_agent";
const DEFAULT_RECONNECT_DELAY_MS = 1_000;
const DEFAULT_STATE_POLL_INTERVAL_MS = 2_000;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function workflowContext(workspace: JsonRecord): JsonRecord {
  const plan = isRecord(workspace.plan) ? workspace.plan : null;
  const context: JsonRecord = {};
  const run = isRecord(workspace.run) ? workspace.run : null;
  for (const key of [
    "active_phase",
    "validation_target_h5",
    "full_input_root",
  ] as const) {
    const value = workspace[key] ?? plan?.[key] ?? (key === "active_phase" ? run?.active_phase : undefined);
    if (value !== undefined && value !== null) context[key] = value;
  }
  return context;
}

function decisionIdentity(threadId: string, decision: LangGraphDecision): string {
  const gateId = "gate_id" in decision ? decision.gate_id : undefined;
  const taskId = "task_id" in decision ? decision.task_id : undefined;
  const revision = "revision" in decision ? decision.revision : undefined;
  if (gateId) return [threadId, "gate", gateId].join(":");
  return [threadId, taskId ?? "workflow", revision ?? "", decision.action].join(":");
}

function messageContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) return "";
  return value
    .map((block) => {
      if (typeof block === "string") return block;
      return isRecord(block) && typeof block.text === "string" ? block.text : "";
    })
    .filter(Boolean)
    .join("\n");
}

function messageRole(message: JsonRecord): "user" | "assistant" | null {
  const raw = stringValue(message.type) ?? stringValue(message.role);
  if (raw === "human" || raw === "user") return "user";
  if (raw === "ai" || raw === "assistant") return "assistant";
  return null;
}

function interruptValues(state: StateSnapshot): unknown[] {
  const values = isRecord(state.values) ? state.values : state as JsonRecord;
  const direct = values.__interrupt__;
  const result = Array.isArray(direct) ? [...direct] : [];
  for (const task of state.tasks ?? []) {
    if (!isRecord(task) || !Array.isArray(task.interrupts)) continue;
    result.push(...task.interrupts);
  }
  return result.map((entry) => (isRecord(entry) && "value" in entry ? entry.value : entry));
}

function stateSnapshot(state: StateSnapshot): LangGraphEvent {
  const values = isRecord(state.values) ? state.values : state as JsonRecord;
  let workspace = isRecord(values.workspace) && values.workspace.schema_version === "1.0"
    ? values.workspace as WorkspaceSnapshot
    : null;
  const messages: Array<{ id?: string; role: "user" | "assistant"; content: string }> = [];
  if (Array.isArray(values.messages)) {
    for (const raw of values.messages) {
      if (!isRecord(raw)) continue;
      const role = messageRole(raw);
      const content = messageContent(raw.content);
      if (role && content) messages.push({ id: stringValue(raw.id) ?? undefined, role, content });
    }
  }

  let handoff: AdapterCodingHandoff | null = null;
  let interrupt: NeuroTokenInterrupt | null = null;
  const pendingAdapter = isRecord(values.pending_adapter) ? values.pending_adapter : null;
  if (pendingAdapter && stringValue(pendingAdapter.task_id)) {
    const adapterName = stringValue(pendingAdapter.name) ?? "adapter";
    const modality = stringValue(pendingAdapter.modality) ?? "unknown";
    handoff = {
      schema_version: "1.0",
      task_id: pendingAdapter.task_id as string,
      thread_id: stringValue(values.thread_id) ?? "",
      revision: typeof pendingAdapter.revision === "number" ? pendingAdapter.revision : 1,
      status: workspace?.mode === "adapter_revision_required" ? "needs_revision" : "ready_for_coding",
      adapter: { name: adapterName, modality, notes: stringValue(pendingAdapter.notes) ?? "" },
      allowed_paths: ["agent_pipeline/adapters", "ADAPTER_DESIGN.md"],
      instruction: stringValue(pendingAdapter.instruction)
        ?? (stringValue(pendingAdapter.coding_summary)
          ? `Correct the adapter candidate. Validation feedback: ${stringValue(pendingAdapter.coding_summary)}`
          : `Continue implementing ${adapterName} for the inspected ${modality} HDF5 source.`),
      codex_thread_id: stringValue(pendingAdapter.codex_thread_id),
      created_at: stringValue(pendingAdapter.created_at)
        ?? stringValue(values.updated_at)
        ?? "1970-01-01T00:00:00.000Z",
    };
  }
  for (const raw of interruptValues(state)) {
    if (!isRecord(raw)) continue;
    if (raw.kind === "adapter_coding_handoff" && isRecord(raw.handoff)) {
      handoff = raw.handoff as AdapterCodingHandoff;
    } else if (
      raw.kind === "plan_approval" ||
      raw.kind === "stage_review" ||
      raw.kind === "smoke_validation_review" ||
      raw.kind === "adapter_execution_confirmation" ||
      raw.kind === "adapter_patch_approval"
    ) {
      interrupt = raw as NeuroTokenInterrupt;
    }
  }
  if (interrupt?.kind === "adapter_patch_approval") {
    const supersededByHandoff = handoff?.task_id === interrupt.task_id && handoff.revision > interrupt.revision;
    if (supersededByHandoff && workspace) {
      workspace = { ...workspace, mode: "ready_for_coding", is_working: false, approval: null };
    }
    if (workspace?.mode !== "awaiting_patch_approval" || supersededByHandoff) interrupt = null;
  }
  if (interrupt?.kind === "smoke_validation_review") {
    const activeGate = workspace?.validation_gate?.gate_id;
    if (workspace?.mode !== "awaiting_validation_gate" || !activeGate || interrupt.gate_id !== activeGate) {
      interrupt = null;
    }
  }

  let error: string | null = null;
  if (workspace?.error) {
    const stage = workspace.error.stage ? `${workspace.error.stage}: ` : "";
    error = `${stage}${workspace.error.message}`;
  } else if (isRecord(values.last_error)) {
    const message = stringValue(values.last_error.message);
    const stage = stringValue(values.last_error.stage);
    if (message) error = stage ? `${stage}: ${message}` : message;
  }
  if (!error) for (const task of state.tasks ?? []) {
    if (!isRecord(task) || task.error == null) continue;
    const detail = typeof task.error === "string" ? task.error : JSON.stringify(task.error);
    const node = stringValue(task.name);
    error = node ? `${node}: ${detail}` : detail;
    break;
  }
  return { type: "snapshot", snapshot: { workspace, interrupt, handoff, messages, error } };
}

export async function consumeSse(
  response: Response,
  onEvent: (event: { id?: string; event?: string; data: unknown }) => void,
): Promise<void> {
  if (!response.body) throw new Error("LangGraph stream returned no response body.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentId: string | undefined;
  let currentEvent: string | undefined;
  let data: string[] = [];

  const dispatch = () => {
    if (data.length === 0) return;
    const raw = data.join("\n");
    let parsed: unknown = raw;
    try {
      parsed = JSON.parse(raw);
    } catch {
      // SSE data is allowed to be plain text.
    }
    onEvent({ id: currentId, event: currentEvent, data: parsed });
    currentId = undefined;
    currentEvent = undefined;
    data = [];
  };

  let done = false;
  while (!done) {
    const chunk = await reader.read();
    const { value } = chunk;
    done = chunk.done;
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split(/\r?\n/);
    buffer = done ? "" : (lines.pop() ?? "");
    for (const line of lines) {
      if (line === "") dispatch();
      else if (line.startsWith("id:")) currentId = line.slice(3).trim();
      else if (line.startsWith("event:")) currentEvent = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (done) dispatch();
  }
}

export class HttpLangGraphClient implements LangGraphClient {
  private readonly apiUrl: string;
  private readonly assistantId: string;
  private readonly request: typeof fetch;
  private readonly headers: Headers;
  private readonly reconnectDelayMs: number;
  private readonly statePollIntervalMs: number;
  private artifactCache = new Map<string, Map<string, ArtifactEnvelope>>();
  private listeners = new Map<string, Set<(event: LangGraphEvent) => void>>();
  private lastReconcileAt = new Map<string, number>();
  private reconcileInFlight = new Set<string>();
  private decisionInFlight = new Map<string, Promise<void>>();

  constructor(options: HttpLangGraphClientOptions) {
    this.apiUrl = options.apiUrl.replace(/\/$/, "");
    this.assistantId = options.assistantId ?? DEFAULT_ASSISTANT_ID;
    this.request = options.fetch ?? fetch;
    this.reconnectDelayMs = options.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS;
    this.statePollIntervalMs = options.statePollIntervalMs ?? DEFAULT_STATE_POLL_INTERVAL_MS;
    this.headers = new Headers({ "Content-Type": "application/json" });
    if (options.apiKey) this.headers.set("X-Api-Key", options.apiKey);
    if (options.authScheme) this.headers.set("X-Auth-Scheme", options.authScheme);
  }

  async listThreads(): Promise<LangGraphThread[]> {
    const threads = await this.json("/threads/search", {
      method: "POST",
      body: JSON.stringify({ metadata: { graph_id: this.assistantId }, limit: 100 }),
    });
    if (!Array.isArray(threads)) return [];
    const uniqueThreads = new Map<string, JsonRecord>();
    for (const raw of threads) {
      if (isRecord(raw) && stringValue(raw.thread_id)) {
        uniqueThreads.set(raw.thread_id as string, raw);
      }
    }
    const summaries = [...uniqueThreads.values()].flatMap((thread) => {
      if (!isRecord(thread) || !stringValue(thread.thread_id)) return [];
      const values = isRecord(thread.values) ? thread.values : {};
      const messages = Array.isArray(values.messages) ? values.messages : [];
      const first = messages.find(isRecord);
      return [{
        threadId: thread.thread_id as string,
        title: first ? messageContent(first.content) || "New task" : "New task",
        updatedAt: stringValue(thread.updated_at) ?? new Date(0).toISOString(),
        workspaceMode: stringValue(values.workspace && isRecord(values.workspace) ? values.workspace.mode : values.mode) ?? undefined,
      }];
    });
    return Promise.all(summaries.map(async (summary) => {
      if (summary.workspaceMode) return summary;
      try {
        const state = await this.getState(summary.threadId);
        const values = isRecord(state.values) ? state.values : {};
        const workspace = isRecord(values.workspace) ? values.workspace : null;
        return {
          ...summary,
          workspaceMode: stringValue(workspace?.mode ?? values.mode) ?? undefined,
        };
      } catch {
        return summary;
      }
    }));
  }

  async createThread(): Promise<LangGraphThread> {
    const value = await this.json("/threads", {
      method: "POST",
      body: JSON.stringify({ metadata: { graph_id: this.assistantId } }),
    });
    if (!isRecord(value) || !stringValue(value.thread_id)) {
      throw new Error("LangGraph did not return a thread ID.");
    }
    return {
      threadId: value.thread_id as string,
      title: "New task",
      updatedAt: stringValue(value.updated_at) ?? new Date().toISOString(),
    };
  }

  async deleteThread(threadId: string): Promise<void> {
    const response = await this.request(`${this.apiUrl}/threads/${encodeURIComponent(threadId)}`, {
      method: "DELETE",
      headers: this.headers,
    });
    if (!response.ok) throw await this.httpError(response);
    this.artifactCache.delete(threadId);
  }

  async sendMessage(threadId: string, content: string): Promise<void> {
    await this.run(threadId, {
      input: {
        messages: [{ type: "human", content }],
        owner: "local-user",
        thread_id: threadId,
      },
    });
  }

  async retry(threadId: string): Promise<void> {
    await this.startRun(threadId, {
      input: {
        retry_requested: true,
        owner: "local-user",
        thread_id: threadId,
      },
    });
  }

  async submitCandidate(submission: AdapterCandidateSubmission): Promise<void> {
    await this.run(submission.thread_id, { command: { resume: submission } });
  }

  async resume(threadId: string, decision: LangGraphDecision): Promise<void> {
    const identity = decisionIdentity(threadId, decision);
    const existing = this.decisionInFlight.get(identity);
    if (existing) return existing;
    const request = this.run(threadId, { command: { resume: decision } })
      .finally(() => this.decisionInFlight.delete(identity));
    this.decisionInFlight.set(identity, request);
    return request;
  }

  async getArtifact(threadId: string, artifactId: string): Promise<LangGraphArtifact> {
    if (!this.artifactCache.get(threadId)?.has(artifactId)) await this.getState(threadId);
    const artifact = this.artifactCache.get(threadId)?.get(artifactId);
    if (!artifact) throw new Error(`Artifact not found: ${artifactId}`);
    const mediaType = stringValue(artifact.media_type) ?? "application/json";
    const extension = mediaType.includes("json") ? "json" : mediaType.startsWith("text/") ? "txt" : "json";
    return {
      artifactId,
      mediaType,
      fileName: stringValue(artifact.file_name) ?? `${artifactId}.${extension}`,
      content: typeof artifact.payload === "string"
        ? artifact.payload
        : JSON.stringify(artifact.payload ?? artifact, null, 2),
    };
  }

  subscribe(threadId: string, onEvent: (event: LangGraphEvent) => void): Unsubscribe {
    const listeners = this.listeners.get(threadId) ?? new Set();
    listeners.add(onEvent);
    this.listeners.set(threadId, listeners);
    const controller = new AbortController();
    void this.watch(threadId, onEvent, controller.signal);
    return () => {
      controller.abort();
      listeners.delete(onEvent);
      if (listeners.size === 0) this.listeners.delete(threadId);
    };
  }

  private maybeReconcile(threadId: string, workspace: JsonRecord | null): void {
    const externalJob = workspace && isRecord(workspace.external_stage_job) ? workspace.external_stage_job : null;
    const status = externalJob ? stringValue(externalJob.status) : null;
    if (!status || !["submitted", "running", "detached", "reconnecting"].includes(status)) return;
    const now = Date.now();
    const last = this.lastReconcileAt.get(threadId) ?? 0;
    if (this.reconcileInFlight.has(threadId) || now - last < Math.max(this.statePollIntervalMs * 3, 5_000)) return;
    this.lastReconcileAt.set(threadId, now);
    this.reconcileInFlight.add(threadId);
    void this.startRun(threadId, {
      input: {
        reconcile_requested: true,
        owner: "local-user",
        thread_id: threadId,
        ...(workspace ? workflowContext(workspace) : {}),
      },
    }).catch(() => undefined).finally(() => this.reconcileInFlight.delete(threadId));
  }

  private async watch(
    threadId: string,
    emit: (event: LangGraphEvent) => void,
    signal: AbortSignal,
  ): Promise<void> {
    let lastSnapshotSignature: string | undefined;
    while (!signal.aborted) {
      try {
        const checkpoint = await this.getState(threadId);
        if (signal.aborted) return;
        const event = stateSnapshot(checkpoint);
        const values = isRecord(checkpoint.values) ? checkpoint.values : {};
        const workspace = isRecord(values.workspace) ? values.workspace : null;
        this.maybeReconcile(threadId, workspace);
        if (event.type === "snapshot") {
          const signature = JSON.stringify(event.snapshot);
          if (signature !== lastSnapshotSignature) {
            lastSnapshotSignature = signature;
            emit(event);
          }
        }
        await this.waitForNextPoll(signal, this.statePollIntervalMs);
      } catch (error) {
        if (signal.aborted) return;
        emit({ type: "error", message: error instanceof Error ? error.message : String(error), retryable: true });
        await this.waitForNextPoll(signal, this.reconnectDelayMs);
      }
    }
  }

  private async waitForNextPoll(signal: AbortSignal, delayMs: number): Promise<void> {
    if (signal.aborted) return;
    await new Promise<void>((resolve) => {
      const timer = setTimeout(finish, delayMs);
      function finish() {
        signal.removeEventListener("abort", finish);
        clearTimeout(timer);
        resolve();
      }
      signal.addEventListener("abort", finish, { once: true });
    });
  }

  private async getState(threadId: string): Promise<StateSnapshot> {
    const state = await this.json(`/threads/${encodeURIComponent(threadId)}/state`);
    const snapshot = isRecord(state) ? state as StateSnapshot : {};
    this.cacheArtifacts(threadId, snapshot.values);
    return snapshot;
  }

  private cacheArtifacts(threadId: string, values: JsonRecord | undefined) {
    const artifacts = Array.isArray(values?.artifacts) ? values.artifacts : [];
    const cache = new Map<string, ArtifactEnvelope>();
    for (const artifact of artifacts) {
      if (isRecord(artifact) && stringValue(artifact.artifact_id)) {
        cache.set(artifact.artifact_id as string, artifact as ArtifactEnvelope);
      }
    }
    this.artifactCache.set(threadId, cache);
  }

  private emit(threadId: string, event: LangGraphEvent) {
    for (const listener of this.listeners.get(threadId) ?? []) listener(event);
  }

  private async startRun(threadId: string, body: JsonRecord): Promise<void> {
    const response = await this.request(
      `${this.apiUrl}/threads/${encodeURIComponent(threadId)}/runs`,
      {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({
          assistant_id: this.assistantId,
          ...body,
        }),
      },
    );
    if (!response.ok) throw await this.httpError(response);
  }

  private async run(threadId: string, body: JsonRecord): Promise<void> {
    const response = await this.request(
      `${this.apiUrl}/threads/${encodeURIComponent(threadId)}/runs/wait`,
      {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({
          assistant_id: this.assistantId,
          ...body,
        }),
      },
    );
    if (!response.ok) throw await this.httpError(response);
    const state = await response.json().catch(() => null);
    if (isRecord(state)) {
      const checkpoint = state as StateSnapshot;
      const values = isRecord(checkpoint.values) ? checkpoint.values : state;
      this.cacheArtifacts(threadId, values);
      this.emit(threadId, stateSnapshot(checkpoint));
    }
  }

  private async json(path: string, init: RequestInit = {}): Promise<unknown> {
    const response = await this.request(`${this.apiUrl}${path}`, { ...init, headers: this.headers });
    if (!response.ok) throw await this.httpError(response);
    return response.json();
  }

  private async httpError(response: Response): Promise<Error> {
    const details = await response.text().catch(() => "");
    return new Error(`LangGraph request failed (${response.status})${details ? `: ${details}` : ""}`);
  }
}
