import type { AdapterCodingHandoff } from "./contracts";
import type {
  CodingEvent,
  CodingRuntime,
  CodingSession,
  RemoteFile,
  RemoteFileWrite,
  RemoteFileWriteResult,
  Unsubscribe,
} from "./runtime";

const TASK_ID = "task-adapter-001";
const INITIAL_CONTENT = `from __future__ import annotations


def read_session(source_path: str) -> dict[str, object]:
    return {"source_path": source_path, "modality": "eeg"}
`;

function hashContent(content: string): string {
  let hash = 2166136261;
  for (let index = 0; index < content.length; index += 1) {
    hash ^= content.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function normalizeAllowedPath(path: string): string {
  const normalized = path.replace(/\\/g, "/").replace(/^\.\//, "");
  if (
    normalized.startsWith("/") ||
    /^[A-Za-z]:\//.test(normalized) ||
    normalized.split("/").includes("..")
  ) {
    throw new Error("Path escapes the allowed task scope.");
  }
  return normalized;
}

export const MOCK_CODING_HANDOFF: AdapterCodingHandoff = {
  schema_version: "1.0",
  task_id: TASK_ID,
  thread_id: "thread-adapter-001",
  revision: 2,
  status: "coding",
  adapter: {
    name: "example_eeg",
    modality: "eeg",
    notes: "Read EEG metadata from HDF5 attributes.",
  },
  allowed_paths: ["agent_pipeline/adapters/"],
  instruction: "Complete and validate the example EEG adapter.",
  created_at: "2026-08-04T07:38:00Z",
};

export const MOCK_CODING_EVENTS: CodingEvent[] = [
  {
    type: "session",
    session: {
      taskId: TASK_ID,
      codexThreadId: "codex-thread-001",
      revision: 2,
      status: "awaiting_approval",
    },
  },
  {
    type: "app_server",
    message: {
      kind: "assistant_message",
      text: "I found the EEG adapter skeleton and inspected the allowed file.",
    },
  },
  {
    type: "app_server",
    message: {
      kind: "tool_call",
      request_id: "tool-request-001",
      title: "Run adapter import validation",
      command: "python -c 'from agent_pipeline.adapters import build_adapter'",
      status: "awaiting_approval",
    },
  },
  {
    type: "app_server",
    message: {
      kind: "command_output",
      request_id: "tool-request-completed",
      title: "Inspect HDF5 datasets",
      output: "datasets: signal, channels, sampling_rate\nshape: (64, 120000)",
      status: "completed",
    },
  },
];

export class MockCodingRuntime implements CodingRuntime {
  private files = new Map<string, RemoteFile>([
    [
      "agent_pipeline/adapters/example_eeg.py",
      {
        path: "agent_pipeline/adapters/example_eeg.py",
        content: INITIAL_CONTENT,
        contentHash: hashContent(INITIAL_CONTENT),
      },
    ],
  ]);
  private listeners = new Map<string, Set<(event: CodingEvent) => void>>();
  private decisions = new Map<string | number, "allow" | "deny">();

  async openTask(handoff: AdapterCodingHandoff): Promise<CodingSession> {
    if (handoff.task_id !== TASK_ID) throw new Error("Unknown coding task.");
    return {
      taskId: handoff.task_id,
      codexThreadId: "codex-thread-001",
      revision: handoff.revision,
      status: "awaiting_approval",
      turnStarted: true,
      shouldStartInstruction: false,
    };
  }

  getHistory(_taskId: string, _revision: number): CodingEvent[] {
    return [];
  }

  async sendInstruction(taskId: string, text: string): Promise<void> {
    this.requireTask(taskId);
    this.emit(taskId, {
      type: "app_server",
      message: { kind: "assistant_message", text: `Working on: ${text}` },
    });
  }

  async listFiles(taskId: string): Promise<string[]> {
    this.requireTask(taskId);
    return [...this.files.keys()];
  }

  async readFile(taskId: string, path: string): Promise<RemoteFile> {
    this.requireTask(taskId);
    const safePath = this.requireAllowedPath(path);
    const file = this.files.get(safePath);
    if (!file) throw new Error("File is outside the allowed task files.");
    return { ...file };
  }

  async writeFile(input: RemoteFileWrite): Promise<RemoteFileWriteResult> {
    this.requireTask(input.taskId);
    const safePath = this.requireAllowedPath(input.path);
    const current = this.files.get(safePath);
    if (!current) throw new Error("File is outside the allowed task files.");
    if (current.contentHash !== input.expectedContentHash) {
      return { status: "conflict", current: { ...current } };
    }
    const file = {
      path: safePath,
      content: input.content,
      contentHash: hashContent(input.content),
    };
    this.files.set(safePath, file);
    return { status: "saved", file: { ...file } };
  }

  async getDiff(taskId: string): Promise<string> {
    this.requireTask(taskId);
    const current = this.files.get("agent_pipeline/adapters/example_eeg.py");
    if (!current) return "";
    return `diff --git a/agent_pipeline/adapters/example_eeg.py b/agent_pipeline/adapters/example_eeg.py\n--- a/agent_pipeline/adapters/example_eeg.py\n+++ b/agent_pipeline/adapters/example_eeg.py\n@@ mock @@\n-${INITIAL_CONTENT.trim()}\n+${current.content.trim()}`;
  }

  async approveTool(
    taskId: string,
    requestId: string | number,
    decision: "allow" | "deny",
  ): Promise<void> {
    this.requireTask(taskId);
    this.decisions.set(requestId, decision);
    this.emit(taskId, {
      type: "app_server",
      message: { kind: "tool_decision", request_id: requestId, decision },
    });
  }

  subscribe(taskId: string, onEvent: (event: CodingEvent) => void): Unsubscribe {
    this.requireTask(taskId);
    const listeners = this.listeners.get(taskId) ?? new Set();
    listeners.add(onEvent);
    this.listeners.set(taskId, listeners);
    for (const event of MOCK_CODING_EVENTS) onEvent(event);
    return () => listeners.delete(onEvent);
  }

  decisionFor(requestId: string | number): "allow" | "deny" | undefined {
    return this.decisions.get(requestId);
  }

  private requireTask(taskId: string) {
    if (taskId !== TASK_ID) throw new Error("Unknown coding task.");
  }

  private requireAllowedPath(path: string): string {
    const normalized = normalizeAllowedPath(path);
    const allowed = MOCK_CODING_HANDOFF.allowed_paths.some((prefix) =>
      normalized.startsWith(prefix),
    );
    if (!allowed) throw new Error("Path escapes the allowed task scope.");
    return normalized;
  }

  private emit(taskId: string, event: CodingEvent) {
    for (const listener of this.listeners.get(taskId) ?? []) listener(event);
  }
}
