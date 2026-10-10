import { invoke } from "@tauri-apps/api/core";
import type { AppServerEvent } from "../../types";
import { subscribeAppServerEvents } from "../../services/events";
import type { AdapterCodingHandoff } from "./contracts";
import type {
  CodingDiagnosticEntry,
  CodingDiagnostics,
  CodingEvent,
  CodingRawAppServerEvent,
  CodingRuntime,
  CodingSession,
  RemoteFile,
  RemoteFileWrite,
  RemoteFileWriteResult,
  Unsubscribe,
} from "./runtime";

type Invoke = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;
type EventSubscriber = (
  listener: (event: AppServerEvent) => void,
  options?: { onError?: (error: unknown) => void },
) => Unsubscribe;

type BridgeSession = {
  taskId: string;
  threadId: string;
  revision: number;
  status: string;
  codexThreadId?: string | null;
  turnStarted?: boolean;
};

type OpenResponse = {
  session: BridgeSession;
  codexThreadId: string;
  recovered?: boolean;
  reusedProcess?: boolean;
};

type PersistedAppServerMessage = Record<string, unknown>;

function sessionStatus(status: string): CodingSession["status"] {
  if (status === "running" || status === "starting" || status === "recovering") return "running";
  if (status === "awaiting_approval") return "awaiting_approval";
  if (status === "error" || status === "failed") return "failed";
  return "idle";
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

export function readableRuntimeError(value: unknown): string {
  if (value instanceof Error && value.message) return value.message;
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "Codex turn failed";
  if (typeof value !== "object") return String(value);

  const record = asRecord(value);
  for (const key of ["message", "error", "detail", "reason"]) {
    if (record[key] !== undefined && record[key] !== value) {
      return readableRuntimeError(record[key]);
    }
  }
  try {
    return JSON.stringify(value);
  } catch {
    return "Codex turn failed with an unreadable error response";
  }
}

function normalizeAppServerMessage(event: AppServerEvent): CodingEvent | null {
  const message = asRecord(event.message);
  const raw = { rawMessage: message, workspaceId: event.workspace_id };
  const method = typeof message.method === "string" ? message.method : "";
  const params = asRecord(message.params);
  const requestId = typeof message.id === "string" || typeof message.id === "number"
    ? message.id
    : typeof params.requestId === "string" || typeof params.requestId === "number"
      ? params.requestId
      : typeof params.request_id === "string" || typeof params.request_id === "number"
        ? params.request_id
        : undefined;

  if (method === "item/agentMessage/delta") {
    return {
      type: "app_server",
      message: { kind: "assistant_message", text: String(params.delta ?? params.text ?? "") },
      ...raw,
    };
  }
  if (method === "item/commandExecution/outputDelta") {
    return {
      type: "app_server",
      message: {
        kind: "command_output",
        title: String(params.command ?? "Command output"),
        output: String(params.delta ?? ""),
        status: "running",
      },
      ...raw,
    };
  }
  if (method.endsWith("requestApproval") || method === "item/tool/requestUserInput") {
    const command = Array.isArray(params.command)
      ? params.command.map(String).join(" ")
      : String(params.command ?? params.reason ?? "Codex tool request");
    return {
      type: "app_server",
      requestId,
      message: {
        kind: "tool_call",
        request_id: requestId,
        title: String(params.title ?? params.reason ?? "Codex requests approval"),
        command,
        status: "awaiting_approval",
      },
      ...raw,
    };
  }
  if (method === "turn/completed") return { type: "turn_completed" };
  if (method === "error" || method === "turn/failed" || method === "turn/cancelled") {
    return {
      type: "error",
      message: readableRuntimeError(params.message ?? params.error ?? message.error),
      retryable: true,
    };
  }
  return { type: "app_server", requestId, message, ...raw };
}

const HISTORY_STORAGE_KEY = "neurotoken.codex.history.v2";

function historyKey(taskId: string, revision: number): string {
  return `${taskId}:${revision}`;
}

function loadStoredHistory(taskId: string, revision: number): CodingEvent[] {
  if (typeof localStorage === "undefined") return [];
  try {
    const stored = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) ?? "{}") as Record<string, unknown>;
    const events = stored[historyKey(taskId, revision)];
    return Array.isArray(events) ? events as CodingEvent[] : [];
  } catch {
    return [];
  }
}

function storeHistory(history: Map<string, CodingEvent[]>) {
  if (typeof localStorage === "undefined") return;
  const value = Object.fromEntries(history.entries());
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(value));
}

export class RemoteCodingRuntime implements CodingRuntime {
  private readonly invoke: Invoke;
  private readonly subscribeEvents: EventSubscriber;
  private readonly sessions = new Map<string, CodingSession>();
  private readonly openingRevisions = new Map<string, number>();
  private readonly rawListeners = new Map<string, Set<(event: CodingRawAppServerEvent) => void>>();
  private readonly history = new Map<string, CodingEvent[]>();
  private readonly rawHistory = new Map<string, CodingRawAppServerEvent[]>();
  private readonly diagnostics: CodingDiagnosticEntry[] = [];

  private recordDiagnostic(
    taskId: string,
    revision: number | null,
    event: string,
    details: Record<string, unknown> = {},
  ) {
    this.diagnostics.push({ timestamp: new Date().toISOString(), taskId, revision, event, details });
    if (this.diagnostics.length > 500) this.diagnostics.splice(0, this.diagnostics.length - 500);
  }

  private ensureHistory(taskId: string, revision: number): CodingEvent[] {
    const key = historyKey(taskId, revision);
    const existing = this.history.get(key);
    if (existing) return existing;
    const restored = loadStoredHistory(taskId, revision).slice(-500);
    this.history.set(key, restored);
    return restored;
  }

  constructor(options: { invoke?: Invoke; subscribeEvents?: EventSubscriber } = {}) {
    this.invoke = options.invoke ?? invoke;
    this.subscribeEvents = options.subscribeEvents ?? subscribeAppServerEvents;
  }

  async openTask(handoff: AdapterCodingHandoff): Promise<CodingSession> {
    this.openingRevisions.set(handoff.task_id, handoff.revision);
    this.recordDiagnostic(handoff.task_id, handoff.revision, "open.request", {
      handoffThreadId: handoff.thread_id,
      instructionLength: handoff.instruction.length,
      allowedPathCount: handoff.allowed_paths.length,
    });
    let response: OpenResponse;
    try {
      response = await this.invoke<OpenResponse>("neurotoken_coding_open", {
        taskId: handoff.task_id,
        revision: handoff.revision,
      });
    } catch (error) {
      const message = readableRuntimeError(error);
      this.recordDiagnostic(handoff.task_id, handoff.revision, "open.error", { message });
      throw new Error(message);
    }
    this.recordDiagnostic(handoff.task_id, handoff.revision, "open.response", {
      sessionStatus: response.session.status,
      sessionRevision: response.session.revision,
      codexThreadId: response.codexThreadId,
      recovered: response.recovered ?? false,
      reusedProcess: response.reusedProcess ?? false,
    });
    const restoredHistory = this.ensureHistory(handoff.task_id, handoff.revision);
    const remoteMessages = await this.invoke<PersistedAppServerMessage[]>("neurotoken_coding_history", {
      taskId: handoff.task_id,
      revision: handoff.revision,
    });
    this.recordDiagnostic(handoff.task_id, handoff.revision, "history.initial", {
      count: remoteMessages.length,
      methods: remoteMessages.slice(-50).map((message) => String(message.method ?? "<missing>")),
    });
    const workspaceId = `neurotoken:${handoff.task_id}`;
    const status = sessionStatus(response.session.status);
    const turnStarted = Boolean(response.session.turnStarted) || status !== "idle";
    const currentRevisionMessages = turnStarted ? remoteMessages : [];
    const restoredRemoteEvents = currentRevisionMessages
      .map((message) => normalizeAppServerMessage({ workspace_id: workspaceId, message }))
      .filter((event): event is CodingEvent => event !== null);
    restoredHistory.splice(0, restoredHistory.length, ...restoredRemoteEvents.slice(-500));
    this.rawHistory.set(
      historyKey(handoff.task_id, handoff.revision),
      currentRevisionMessages.slice(-500).map((message) => ({ workspace_id: workspaceId, message })),
    );
    storeHistory(this.history);
    const hasConversation = restoredHistory.some((event) =>
      event.type === "turn_completed" ||
      (event.type === "app_server" && ["assistant_message", "command_output", "tool_call"].includes(String(event.message.kind))),
    );
    const session = {
      taskId: handoff.task_id,
      codexThreadId: response.codexThreadId,
      revision: response.session.revision,
      status,
      turnStarted,
      shouldStartInstruction: !turnStarted && status === "idle",
    } satisfies CodingSession;
    this.sessions.set(handoff.task_id, session);
    this.recordDiagnostic(handoff.task_id, handoff.revision, "open.ready", {
      normalizedHistoryCount: restoredHistory.length,
      hasConversation,
      turnStarted: session.turnStarted ?? false,
      shouldStartInstruction: session.shouldStartInstruction ?? false,
      normalizedKinds: restoredHistory.slice(-50).map((event) =>
        event.type === "app_server" ? String(event.message.kind ?? event.rawMessage?.method ?? "app_server") : event.type
      ),
    });
    return session;
  }

  getHistory(taskId: string, revision: number): CodingEvent[] {
    return [...this.ensureHistory(taskId, revision)];
  }

  getRawHistory(taskId: string, revision: number): CodingRawAppServerEvent[] {
    return (this.rawHistory.get(historyKey(taskId, revision)) ?? []).map((event) => ({
      workspace_id: event.workspace_id,
      message: { ...event.message },
    }));
  }

  async refreshRawHistory(taskId: string, revision: number): Promise<CodingRawAppServerEvent[]> {
    const remoteMessages = await this.invoke<PersistedAppServerMessage[]>("neurotoken_coding_history", {
      taskId,
      revision,
    });
    this.recordDiagnostic(taskId, revision, "history.refresh", {
      count: remoteMessages.length,
      methods: remoteMessages.slice(-50).map((message) => String(message.method ?? "<missing>")),
    });
    const workspaceId = `neurotoken:${taskId}`;
    const rawEvents = remoteMessages.slice(-500).map((message) => ({
      workspace_id: workspaceId,
      message: { ...message },
    }));
    const normalized = rawEvents
      .map((event) => normalizeAppServerMessage(event))
      .filter((event): event is CodingEvent => event !== null);
    const key = historyKey(taskId, revision);
    this.rawHistory.set(key, rawEvents);
    this.history.set(key, normalized);
    storeHistory(this.history);
    return this.getRawHistory(taskId, revision);
  }

  subscribeRaw(
    taskId: string,
    onEvent: (event: CodingRawAppServerEvent) => void,
  ): Unsubscribe {
    const listeners = this.rawListeners.get(taskId) ?? new Set();
    listeners.add(onEvent);
    this.rawListeners.set(taskId, listeners);
    return () => listeners.delete(onEvent);
  }

  async sendInstruction(taskId: string, text: string): Promise<void> {
    const session = this.requireSession(taskId);
    this.recordDiagnostic(taskId, session.revision, "turn.request", {
      codexThreadId: session.codexThreadId,
      textLength: text.length,
      textPreview: text.replace(/\s+/g, " ").slice(0, 160),
    });
    try {
      session.turnStarted = true;
      session.shouldStartInstruction = false;
      const response = await this.invoke<Record<string, unknown>>("neurotoken_coding_turn_start", {
        taskId,
        revision: session.revision,
        text,
        model: null,
        effort: null,
      });
      this.recordDiagnostic(taskId, session.revision, "turn.response", {
        responseKeys: response && typeof response === "object" ? Object.keys(response) : [],
        hasError: Boolean(response && typeof response === "object" && response.error),
      });
    } catch (error) {
      const message = readableRuntimeError(error);
      this.recordDiagnostic(taskId, session.revision, "turn.error", { message });
      throw new Error(message);
    }
  }

  async listFiles(taskId: string): Promise<string[]> {
    return this.invoke<string[]>("neurotoken_coding_list_files", { taskId });
  }

  async readFile(taskId: string, path: string): Promise<RemoteFile> {
    return this.invoke<RemoteFile>("neurotoken_coding_read_file", { taskId, path });
  }

  async writeFile(input: RemoteFileWrite): Promise<RemoteFileWriteResult> {
    return this.invoke<RemoteFileWriteResult>("neurotoken_coding_write_file", {
      taskId: input.taskId,
      path: input.path,
      content: input.content,
      expectedContentHash: input.expectedContentHash,
    });
  }

  async getDiff(taskId: string): Promise<string> {
    const response = await this.invoke<{ diff: string }>("neurotoken_coding_diff", { taskId });
    return response.diff;
  }

  async approveTool(
    taskId: string,
    requestId: string | number,
    decision: "allow" | "deny",
  ): Promise<void> {
    const session = this.requireSession(taskId);
    await this.invoke("neurotoken_coding_approve", {
      taskId,
      revision: session.revision,
      requestId,
      decision,
    });
  }

  subscribe(taskId: string, onEvent: (event: CodingEvent) => void): Unsubscribe {
    const workspaceId = `neurotoken:${taskId}`;
    const revision = this.sessions.get(taskId)?.revision ?? this.openingRevisions.get(taskId) ?? null;
    this.recordDiagnostic(taskId, revision, "events.subscribe", { workspaceId });
    return this.subscribeEvents(
      (event) => {
        if (event.workspace_id !== workspaceId) return;
        const activeRevision = this.sessions.get(taskId)?.revision ?? this.openingRevisions.get(taskId) ?? null;
        this.recordDiagnostic(taskId, activeRevision, "event.received", {
          workspaceId: event.workspace_id,
          method: String(event.message.method ?? "<missing>"),
          hasParams: Boolean(event.message.params),
        });
        const normalized = normalizeAppServerMessage(event);
        if (normalized) {
          const revision = this.sessions.get(taskId)?.revision ?? this.openingRevisions.get(taskId);
          if (revision === undefined) return;
          const key = historyKey(taskId, revision);
          const history = this.ensureHistory(taskId, revision);
          history.push(normalized);
          this.history.set(key, history.slice(-500));
          const rawEvent = { workspace_id: event.workspace_id, message: { ...event.message } };
          const rawHistory = this.rawHistory.get(key) ?? [];
          rawHistory.push(rawEvent);
          this.rawHistory.set(key, rawHistory.slice(-500));
          for (const listener of this.rawListeners.get(taskId) ?? []) listener(rawEvent);
          storeHistory(this.history);
          onEvent(normalized);
        }
      },
      {
        onError: (error) => {
          const activeRevision = this.sessions.get(taskId)?.revision ?? this.openingRevisions.get(taskId) ?? null;
          const message = readableRuntimeError(error);
          this.recordDiagnostic(taskId, activeRevision, "events.error", { message });
          onEvent({ type: "error", message, retryable: true });
        },
      },
    );
  }

  async getDiagnostics(taskId: string, revision: number): Promise<CodingDiagnostics> {
    let refreshError: string | null = null;
    try {
      await this.refreshRawHistory(taskId, revision);
    } catch (error) {
      refreshError = readableRuntimeError(error);
      this.recordDiagnostic(taskId, revision, "diagnostics.refresh_error", { message: refreshError });
    }
    const rawHistory = this.getRawHistory(taskId, revision);
    const normalizedHistory = this.getHistory(taskId, revision);
    const session = this.sessions.get(taskId) ?? null;
    const timeline = this.diagnostics.filter((entry) => entry.taskId === taskId);
    if (refreshError) {
      timeline.push({
        timestamp: new Date().toISOString(),
        taskId,
        revision,
        event: "diagnostics.warning",
        details: { refreshError },
      });
    }
    return {
      generatedAt: new Date().toISOString(),
      taskId,
      revision,
      session,
      normalizedHistoryCount: normalizedHistory.length,
      rawHistoryCount: rawHistory.length,
      rawEventMethods: rawHistory.map((event) => String(event.message.method ?? "<missing>")),
      timeline: timeline.slice(-300),
    };
  }

  private requireSession(taskId: string): CodingSession {
    const session = this.sessions.get(taskId);
    if (!session) throw new Error(`NeuroToken coding task is not open: ${taskId}`);
    return session;
  }
}
