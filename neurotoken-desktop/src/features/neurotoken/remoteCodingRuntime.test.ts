import { describe, expect, it, vi } from "vitest";
import type { AppServerEvent } from "../../types";
import { MOCK_CODING_HANDOFF } from "./mockCodingRuntime";
import { readableRuntimeError, RemoteCodingRuntime } from "./remoteCodingRuntime";

function createHarness(options: {
  recovered?: boolean;
  status?: string;
  turnStarted?: boolean;
  history?: Array<Record<string, unknown>>;
  turnError?: unknown;
} = {}) {
  const listeners = new Set<(event: AppServerEvent) => void>();
  const storage = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
    removeItem: (key: string) => storage.delete(key),
  });
  const invoke = vi.fn(async (command: string) => {
    if (command === "neurotoken_coding_open") {
      return {
        session: {
          taskId: MOCK_CODING_HANDOFF.task_id,
          threadId: MOCK_CODING_HANDOFF.thread_id,
          revision: MOCK_CODING_HANDOFF.revision,
          status: options.status ?? "idle",
          codexThreadId: "codex-real-1",
          turnStarted: options.turnStarted ?? Boolean(options.history?.length),
        },
        codexThreadId: "codex-real-1",
        recovered: options.recovered ?? false,
      };
    }
    if (command === "neurotoken_coding_history") {
      return options.history ?? [];
    }
    if (command === "neurotoken_coding_list_files") {
      return ["agent_pipeline/adapters/example_eeg.py"];
    }
    if (command === "neurotoken_coding_read_file") {
      return {
        path: "agent_pipeline/adapters/example_eeg.py",
        content: "initial\n",
        contentHash: "fnv1a-d46e15cb",
      };
    }
    if (command === "neurotoken_coding_write_file") {
      return {
        status: "saved",
        file: {
          path: "agent_pipeline/adapters/example_eeg.py",
          content: "changed\n",
          contentHash: "fnv1a-12345678",
        },
      };
    }
    if (command === "neurotoken_coding_turn_start" && options.turnError !== undefined) {
      throw options.turnError;
    }
    if (command === "neurotoken_coding_diff") return { diff: "+changed" };
    return { ok: true };
  });
  const runtime = new RemoteCodingRuntime({
    invoke: invoke as <T>(command: string, args?: Record<string, unknown>) => Promise<T>,
    subscribeEvents: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  });
  return {
    runtime,
    invoke,
    emit: (event: AppServerEvent) => listeners.forEach((listener) => listener(event)),
  };
}

describe("RemoteCodingRuntime", () => {
  it("extracts readable messages from nested object errors", () => {
    expect(readableRuntimeError({ error: { message: "The quota has been exceeded." } })).toBe(
      "The quota has been exceeded.",
    );
    expect(readableRuntimeError({ code: "transport_failed", data: { retryable: true } })).toBe(
      '{"code":"transport_failed","data":{"retryable":true}}',
    );
  });

  it("rethrows structured turn-start failures as readable errors", async () => {
    const { runtime } = createHarness({
      turnError: { error: { message: "Codex authentication is unavailable" } },
    });
    await runtime.openTask(MOCK_CODING_HANDOFF);

    await expect(runtime.sendInstruction(MOCK_CODING_HANDOFF.task_id, "Implement the adapter"))
      .rejects.toThrow("Codex authentication is unavailable");
  });

  it("opens a task and routes coding operations through narrow Tauri commands", async () => {
    const { runtime, invoke } = createHarness();
    const session = await runtime.openTask(MOCK_CODING_HANDOFF);

    expect(session.codexThreadId).toBe("codex-real-1");
    expect(session.status).toBe("idle");
    expect(session.shouldStartInstruction).toBe(true);
    await runtime.sendInstruction(MOCK_CODING_HANDOFF.task_id, "Implement the adapter");
    expect(invoke).toHaveBeenCalledWith("neurotoken_coding_turn_start", {
      taskId: MOCK_CODING_HANDOFF.task_id,
      revision: MOCK_CODING_HANDOFF.revision,
      text: "Implement the adapter",
      model: null,
      effort: null,
    });

    expect(await runtime.listFiles(MOCK_CODING_HANDOFF.task_id)).toHaveLength(1);
    const file = await runtime.readFile(
      MOCK_CODING_HANDOFF.task_id,
      "agent_pipeline/adapters/example_eeg.py",
    );
    const result = await runtime.writeFile({
      taskId: MOCK_CODING_HANDOFF.task_id,
      path: file.path,
      content: "changed\n",
      expectedContentHash: file.contentHash,
    });
    expect(result.status).toBe("saved");
    expect(await runtime.getDiff(MOCK_CODING_HANDOFF.task_id)).toBe("+changed");
  });

  it("filters events by task and normalizes messages, output, and approvals", async () => {
    const { runtime, emit } = createHarness();
    await runtime.openTask(MOCK_CODING_HANDOFF);
    const listener = vi.fn();
    const rawListener = vi.fn();
    runtime.subscribeRaw(MOCK_CODING_HANDOFF.task_id, rawListener);
    runtime.subscribe(MOCK_CODING_HANDOFF.task_id, listener);

    emit({
      workspace_id: "neurotoken:other-task",
      message: { method: "item/agentMessage/delta", params: { delta: "ignore" } },
    });
    emit({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: { method: "item/agentMessage/delta", params: { delta: "Adapter updated" } },
    });
    emit({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: {
        id: 42,
        method: "item/commandExecution/requestApproval",
        params: { command: ["python", "-m", "pytest"], reason: "Validate import" },
      },
    });

    expect(listener).toHaveBeenCalledTimes(2);
    expect(rawListener).toHaveBeenCalledTimes(2);
    expect(rawListener.mock.calls[0][0]).toEqual({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: { method: "item/agentMessage/delta", params: { delta: "Adapter updated" } },
    });
    expect(listener.mock.calls[0][0]).toMatchObject({
      message: {
        kind: "assistant_message",
        text: "Adapter updated",
      },
      rawMessage: {
        method: "item/agentMessage/delta",
        params: { delta: "Adapter updated" },
      },
      workspaceId: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
    });
    expect(listener.mock.calls[1][0]).toMatchObject({
      requestId: 42,
      message: { kind: "tool_call", command: "python -m pytest" },
    });
  });

  it("normalizes nested raw turn errors without object coercion", async () => {
    const { runtime, emit } = createHarness();
    await runtime.openTask(MOCK_CODING_HANDOFF);
    const listener = vi.fn();
    runtime.subscribe(MOCK_CODING_HANDOFF.task_id, listener);

    emit({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: {
        method: "turn/failed",
        params: { error: { message: "The quota has been exceeded.", code: "quota_exceeded" } },
      },
    });

    expect(listener).toHaveBeenCalledWith({
      type: "error",
      message: "The quota has been exceeded.",
      retryable: true,
    });
  });

  it("refreshes persisted daemon history to recover missed live events", async () => {
    const history: Array<Record<string, unknown>> = [
      { method: "item/agentMessage/delta", params: { delta: "Recovered output" } },
    ];
    const { runtime, invoke } = createHarness({ history });
    await runtime.openTask(MOCK_CODING_HANDOFF);

    history.push({ method: "turn/completed", params: {} });
    const refreshed = await runtime.refreshRawHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision);

    expect(invoke).toHaveBeenLastCalledWith("neurotoken_coding_history", {
      taskId: MOCK_CODING_HANDOFF.task_id,
      revision: MOCK_CODING_HANDOFF.revision,
    });
    expect(refreshed.map((event) => event.message.method)).toEqual([
      "item/agentMessage/delta",
      "turn/completed",
    ]);
    expect(runtime.getHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).toEqual([
      expect.objectContaining({ type: "app_server" }),
      { type: "turn_completed" },
    ]);
  });

  it("exports task-scoped Codex lifecycle diagnostics", async () => {
    const history: Array<Record<string, unknown>> = [
      { method: "turn/started", params: { turn: { id: "turn-1" } } },
      { method: "item/agentMessage/delta", params: { delta: "Working" } },
    ];
    const { runtime } = createHarness({ history });
    await runtime.openTask(MOCK_CODING_HANDOFF);
    await runtime.sendInstruction(MOCK_CODING_HANDOFF.task_id, "Implement the adapter");

    await expect(runtime.getDiagnostics(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).resolves.toMatchObject({
      taskId: MOCK_CODING_HANDOFF.task_id,
      revision: MOCK_CODING_HANDOFF.revision,
      rawHistoryCount: 2,
      rawEventMethods: ["turn/started", "item/agentMessage/delta"],
      timeline: expect.arrayContaining([
        expect.objectContaining({ event: "open.request" }),
        expect.objectContaining({ event: "open.response" }),
        expect.objectContaining({ event: "turn.request" }),
        expect.objectContaining({ event: "turn.response" }),
      ]),
    });
  });

  it("retains task history and emits a completion signal", async () => {
    const { runtime, emit } = createHarness();
    await runtime.openTask(MOCK_CODING_HANDOFF);
    const listener = vi.fn();
    const unsubscribe = runtime.subscribe(MOCK_CODING_HANDOFF.task_id, listener);

    emit({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: { method: "item/agentMessage/delta", params: { delta: "Finished" } },
    });
    emit({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: { method: "turn/completed", params: {} },
    });
    unsubscribe();

    expect(runtime.getHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).toEqual([
      expect.objectContaining({ type: "app_server" }),
      { type: "turn_completed" },
    ]);
    expect(listener).toHaveBeenLastCalledWith({ type: "turn_completed" });
  });

  it("does not treat a connection notification as existing conversation", async () => {
    const { runtime, emit } = createHarness();
    runtime.subscribe(MOCK_CODING_HANDOFF.task_id, vi.fn());
    emit({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: { method: "codex/connected", params: {} },
    });

    await expect(runtime.openTask(MOCK_CODING_HANDOFF)).resolves.toMatchObject({
      shouldStartInstruction: true,
    });
  });

  it("restores a pending approval from daemon history", async () => {
    const { runtime } = createHarness({
      recovered: true,
      status: "awaiting_approval",
      history: [{
        id: 42,
        method: "item/commandExecution/requestApproval",
        params: { command: ["python", "-m", "pytest"], reason: "Validate import" },
      }],
    });

    await expect(runtime.openTask(MOCK_CODING_HANDOFF)).resolves.toMatchObject({
      status: "awaiting_approval",
      shouldStartInstruction: false,
    });
    expect(runtime.getHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).toEqual([
      expect.objectContaining({
        requestId: 42,
        message: expect.objectContaining({ kind: "tool_call", status: "awaiting_approval" }),
      }),
    ]);
    expect(runtime.getRawHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).toEqual([{
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: expect.objectContaining({
        id: 42,
        method: "item/commandExecution/requestApproval",
      }),
    }]);
  });

  it("treats an empty daemon history as authoritative over cached UI history", async () => {
    const { runtime, emit } = createHarness();
    runtime.subscribe(MOCK_CODING_HANDOFF.task_id, vi.fn());
    emit({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: { method: "turn/completed", params: { turn: { id: "old-turn" } } },
    });

    await expect(runtime.openTask(MOCK_CODING_HANDOFF)).resolves.toMatchObject({
      shouldStartInstruction: true,
    });
    expect(runtime.getHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).toEqual([]);
    expect(runtime.getRawHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).toEqual([]);
  });

  it("starts the instruction for an idle recovered process without revision history", async () => {
    const { runtime } = createHarness({ recovered: true });

    await expect(runtime.openTask(MOCK_CODING_HANDOFF)).resolves.toMatchObject({
      shouldStartInstruction: true,
    });
  });

  it("ignores resumed-thread replay before the revision turn starts", async () => {
    const { runtime } = createHarness({
      recovered: true,
      turnStarted: false,
      history: [
        { method: "item/agentMessage/delta", params: { delta: "Old revision output" } },
        { method: "turn/completed", params: { turn: { id: "old-turn" } } },
      ],
    });

    await expect(runtime.openTask(MOCK_CODING_HANDOFF)).resolves.toMatchObject({
      turnStarted: false,
      shouldStartInstruction: true,
    });
    expect(runtime.getHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).toEqual([]);
    expect(runtime.getRawHistory(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.revision)).toEqual([]);
  });

  it("does not start another instruction while the recovered process is running", async () => {
    const { runtime } = createHarness({ recovered: true, status: "running" });

    await expect(runtime.openTask(MOCK_CODING_HANDOFF)).resolves.toMatchObject({
      shouldStartInstruction: false,
    });
  });

  it("keeps conversation history isolated by revision", async () => {
    const { runtime, emit } = createHarness();
    await runtime.openTask(MOCK_CODING_HANDOFF);
    runtime.subscribe(MOCK_CODING_HANDOFF.task_id, vi.fn());
    emit({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: { method: "turn/completed", params: {} },
    });
    const nextRevision = { ...MOCK_CODING_HANDOFF, revision: MOCK_CODING_HANDOFF.revision + 1 };

    await expect(runtime.openTask(nextRevision)).resolves.toMatchObject({
      shouldStartInstruction: true,
    });
    expect(runtime.getHistory(nextRevision.task_id, nextRevision.revision)).toEqual([]);
  });

  it("sends approval decisions with the open task revision", async () => {
    const { runtime, invoke } = createHarness();
    await runtime.openTask(MOCK_CODING_HANDOFF);
    await runtime.approveTool(MOCK_CODING_HANDOFF.task_id, 42, "allow");

    expect(invoke).toHaveBeenCalledWith("neurotoken_coding_approve", {
      taskId: MOCK_CODING_HANDOFF.task_id,
      revision: MOCK_CODING_HANDOFF.revision,
      requestId: 42,
      decision: "allow",
    });
  });
});
