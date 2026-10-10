import { describe, expect, it, vi } from "vitest";
import { consumeSse, HttpLangGraphClient } from "./langGraphClient";

function jsonResponse(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function sseResponse(chunks: string[]) {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  }), { headers: { "Content-Type": "text/event-stream" } });
}

describe("HTTP LangGraph client", () => {
  it("creates threads with explicit graph metadata", async () => {
    const request = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(JSON.parse(String(init?.body))).toEqual({ metadata: { graph_id: "neurotoken_agent" } });
      return jsonResponse({ thread_id: "thread-new", updated_at: "2026-08-05T07:00:00Z" });
    });
    const client = new HttpLangGraphClient({ apiUrl: "http://localhost:2024", fetch: request as typeof fetch });

    await expect(client.createThread()).resolves.toMatchObject({ threadId: "thread-new" });
  });

  it("deduplicates threads and restores their workspace mode", async () => {
    const request = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/threads/search")) {
        return jsonResponse([
          { thread_id: "thread-1", updated_at: "2026-08-05T07:00:00Z", values: { messages: [] } },
          { thread_id: "thread-1", updated_at: "2026-08-05T07:01:00Z", values: { messages: [] } },
        ]);
      }
      if (url.endsWith("/threads/thread-1/state")) {
        return jsonResponse({ values: { workspace: { mode: "awaiting_patch_approval" } } });
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    const client = new HttpLangGraphClient({ apiUrl: "http://localhost:2024", fetch: request as typeof fetch });

    await expect(client.listThreads()).resolves.toEqual([expect.objectContaining({
      threadId: "thread-1",
      workspaceMode: "awaiting_patch_approval",
    })]);
  });

  it("parses SSE frames split across network chunks", async () => {
    const events: unknown[] = [];
    await consumeSse(
      sseResponse([
        "id: 4\nevent: values\ndata: {\"workspace\":",
        "{\"schema_version\":\"1.0\"}}\n\n",
      ]),
      (event) => events.push(event),
    );
    expect(events).toEqual([{
      id: "4",
      event: "values",
      data: { workspace: { schema_version: "1.0" } },
    }]);
  });

  it("rehydrates workspace, messages, interrupts, and artifacts from checkpoint state", async () => {
    const request = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/state")) {
        return jsonResponse({
          values: {
            workspace: {
              schema_version: "1.0",
              mode: "reviewing_stage",
              is_working: false,
              run: {},
              stages: [],
              artifacts: [{
                artifact_id: "artifact-1",
                kind: "stage_report",
                summary: "Ready",
                media_type: "application/json",
              }],
            },
            messages: [{ id: "m1", type: "ai", content: "Review report" }],
            artifacts: [{
              artifact_id: "artifact-1",
              media_type: "application/json",
              payload: { status: "ok" },
            }],
            __interrupt__: [{
              value: {
                schema_version: "1.0",
                kind: "stage_review",
                owner: "local-user",
                thread_id: "thread-1",
                workflow_id: "workflow-1",
                stage: "preflight",
                summary: "Ready",
                report: { status: "ok" },
                remaining_stages: [],
                allowed_actions: ["continue"],
              },
            }],
          },
        });
      }
      if (url.includes("/stream?")) return sseResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      reconnectDelayMs: 100_000,
    });
    const events: unknown[] = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event));
    await vi.waitFor(() => expect(events).toHaveLength(1));
    unsubscribe();

    expect(events).toEqual([
      expect.objectContaining({
        type: "snapshot",
        snapshot: expect.objectContaining({
          workspace: expect.objectContaining({ mode: "reviewing_stage" }),
          messages: [expect.objectContaining({ id: "m1", content: "Review report" })],
          interrupt: expect.objectContaining({ kind: "stage_review" }),
        }),
      }),
    ]);
    await expect(client.getArtifact("thread-1", "artifact-1")).resolves.toMatchObject({
      mediaType: "application/json",
      content: '{\n  "status": "ok"\n}',
    });
  });

  it("preserves a revision instruction when reconstructing a coding handoff", async () => {
    const request = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/state")) {
        return jsonResponse({
          values: {
            mode: "ready_for_coding",
            thread_id: "thread-1",
            pending_adapter: {
              task_id: "task-1",
              name: "custom_meg",
              modality: "meg",
              revision: 2,
              instruction: "Split every event and every gap between adjacent events into separate chunks.",
              coding_summary: "",
            },
            workspace: {
              schema_version: "1.0",
              mode: "ready_for_coding",
              is_working: false,
              run: {},
              stages: [],
              artifacts: [],
            },
          },
        });
      }
      throw new Error(`Unexpected request: ${String(input)}`);
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      reconnectDelayMs: 100_000,
    });
    const events: Array<{ type: string; snapshot?: { handoff: { instruction?: string } | null } }> = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event as typeof events[number]));
    await vi.waitFor(() => expect(events).toHaveLength(1));
    unsubscribe();

    expect(events[0]?.snapshot?.handoff?.instruction).toBe(
      "Split every event and every gap between adjacent events into separate chunks.",
    );
  });

  it("drops a stale patch approval when the same checkpoint contains a newer handoff", async () => {
    const request = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/state")) {
        return jsonResponse({
          values: {
            mode: "ready_for_coding",
            thread_id: "thread-1",
            pending_adapter: {
              task_id: "task-1",
              name: "custom_meg",
              modality: "meg",
              revision: 3,
              instruction: "Continue revision three.",
            },
            workspace: {
              schema_version: "1.0",
              mode: "awaiting_patch_approval",
              is_working: false,
              run: {},
              stages: [],
              artifacts: [],
            },
            __interrupt__: [{
              value: {
                schema_version: "1.0",
                kind: "adapter_patch_approval",
                owner: "local-user",
                thread_id: "thread-1",
                task_id: "task-1",
                revision: 2,
                adapter: { diff: "old candidate" },
                allowed_actions: ["apply", "apply_and_commit", "ask", "refine", "reject"],
              },
            }],
          },
        });
      }
      throw new Error(`Unexpected request: ${String(input)}`);
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      reconnectDelayMs: 100_000,
    });
    const events: Array<{ type: string; snapshot?: { workspace: { mode?: string } | null; handoff: { revision?: number } | null; interrupt: unknown } }> = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event as typeof events[number]));
    await vi.waitFor(() => expect(events).toHaveLength(1));
    unsubscribe();

    expect(events[0]?.snapshot?.handoff?.revision).toBe(3);
    expect(events[0]?.snapshot?.workspace?.mode).toBe("ready_for_coding");
    expect(events[0]?.snapshot?.interrupt).toBeNull();
  });

  it("surfaces workspace stage failures from durable state", async () => {
    const request = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/state")) {
        return jsonResponse({
          values: {
            workspace: {
              schema_version: "1.0",
              mode: "failed",
              is_working: false,
              run: {},
              stages: [{ name: "preflight", status: "failed" }],
              artifacts: [],
              error: {
                code: "stage_failed",
                message: "FileNotFoundError: SMN_MEG-sub-02_ses-01_RDR_57.h5",
                retryable: false,
                stage: "preflight",
              },
            },
          },
          tasks: [],
        });
      }
      return sseResponse([]);
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      reconnectDelayMs: 100_000,
    });
    const events: unknown[] = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event));
    await vi.waitFor(() => expect(events).toContainEqual(expect.objectContaining({
      type: "snapshot",
      snapshot: expect.objectContaining({
        error: "preflight: FileNotFoundError: SMN_MEG-sub-02_ses-01_RDR_57.h5",
      }),
    })));
    unsubscribe();
  });

  it("surfaces checkpoint task failures as retryable errors", async () => {
    const request = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/state")) {
        return jsonResponse({
          values: {},
          tasks: [{ name: "understand", error: "APIConnectionError('Connection error.')" }],
        });
      }
      return sseResponse([]);
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      reconnectDelayMs: 100_000,
    });
    const events: unknown[] = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event));
    await vi.waitFor(() => expect(events).toContainEqual(expect.objectContaining({
      type: "snapshot",
      snapshot: expect.objectContaining({
        error: "understand: APIConnectionError('Connection error.')",
      }),
    })));
    unsubscribe();
  });

  it("submits retries asynchronously and maps other actions to bounded wait runs", async () => {
    const bodies: unknown[] = [];
    const urls: string[] = [];
    const request = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      urls.push(String(input));
      bodies.push(JSON.parse(String(init?.body)));
      return jsonResponse({});
    });
    const client = new HttpLangGraphClient({ apiUrl: "http://localhost:2024", fetch: request as typeof fetch });

    await client.sendMessage("thread-1", "Prepare EEG");
    await client.retry("thread-1");
    await client.resume("thread-1", { schema_version: "1.0", action: "continue" });
    await client.resume("thread-1", {
      schema_version: "1.0",
      action: "apply_and_commit",
      task_id: "task-1",
      revision: 2,
      message: null,
      commit_message: "Add EEG adapter",
    });
    await client.submitCandidate({
      schema_version: "1.0",
      action: "submit_candidate",
      task_id: "task-1",
      thread_id: "thread-1",
      codex_thread_id: "codex-1",
      revision: 2,
    });

    expect(bodies).toEqual([
      expect.objectContaining({ input: expect.objectContaining({ messages: [{ type: "human", content: "Prepare EEG" }] }) }),
      expect.objectContaining({
        input: expect.objectContaining({ retry_requested: true, thread_id: "thread-1" }),
      }),
      expect.objectContaining({ command: { resume: expect.objectContaining({ action: "continue" }) } }),
      expect.objectContaining({ command: { resume: expect.objectContaining({ action: "apply_and_commit" }) } }),
      expect.objectContaining({ command: { resume: expect.objectContaining({ action: "submit_candidate" }) } }),
    ]);
    expect(urls[1]).toMatch(/\/threads\/thread-1\/runs$/);
    expect(urls.filter((_, index) => index !== 1).every((url) => url.endsWith("/runs/wait"))).toBe(true);
    expect(urls.every((url) => !url.includes("/stream"))).toBe(true);
  });

  it("emits the bounded wait checkpoint immediately after an approval", async () => {
    const request = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/runs/wait")) {
        return jsonResponse({
          values: {
            mode: "awaiting_patch_approval",
            workspace: { schema_version: "1.0", mode: "awaiting_patch_approval", is_working: false, run: {}, stages: [], artifacts: [] },
            __interrupt__: [{ value: { kind: "adapter_patch_approval", thread_id: "thread-1", task_id: "task-1", revision: 1 } }],
          },
        });
      }
      return jsonResponse({ values: {} });
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      statePollIntervalMs: 100_000,
    });
    const events: Array<{ type: string; snapshot?: { interrupt: unknown } }> = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event as typeof events[number]));

    await client.resume("thread-1", { schema_version: "1.0", action: "approve" });

    expect(events.some((event) =>
      (event.snapshot?.interrupt as { kind?: string } | null)?.kind === "adapter_patch_approval"
    )).toBe(true);
    unsubscribe();
  });

  it("rehydrates structured validation fields and smoke review interrupts", async () => {
    const request = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith("/state")) return jsonResponse({
        values: {
          workspace: {
            schema_version: "1.0",
            mode: "awaiting_validation_gate",
            is_working: false,
            active_phase: "validation",
            validation_target_h5: "/input/session-01.h5",
            full_input_root: "/input",
            linked_runs: [{ run_id: "validation-1", scope: "validation" }],
            validation_gate: { gate_id: "gate-1", status: "awaiting_decision", verdict: "pass", report: { metrics: { chunks: 12 } } },
            plan: { stages: ["inspect"], active_phase: "validation", validation_target_h5: "/input/session-01.h5" },
            run: {},
            stages: [{ name: "inspect", status: "completed", stage_id: "validation:inspect", scope: "validation", run_id: "validation-1" }],
            artifacts: [],
          },
          __interrupt__: [{ value: {
            schema_version: "1.0",
            kind: "smoke_validation_review",
            owner: "local-user",
            thread_id: "thread-1",
            gate_id: "gate-1",
            validation_target_h5: "/input/session-01.h5",
            verdict: "pass",
            report: { metrics: { chunks: 12 } },
            allowed_actions: ["start_full_run", "rerun_smoke", "pause", "cancel"],
          } }],
        },
      });
      throw new Error(`Unexpected request: ${String(input)}`);
    });
    const client = new HttpLangGraphClient({ apiUrl: "http://localhost:2024", fetch: request as typeof fetch, statePollIntervalMs: 100_000 });
    const events: Array<{ type: string; snapshot?: { workspace: Record<string, unknown> | null; interrupt: Record<string, unknown> | null } }> = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event as typeof events[number]));
    await vi.waitFor(() => expect(events).toHaveLength(1));
    unsubscribe();

    expect(events[0]?.snapshot?.workspace).toMatchObject({
      active_phase: "validation",
      validation_target_h5: "/input/session-01.h5",
      validation_gate: { gate_id: "gate-1", verdict: "pass" },
    });
    expect(events[0]?.snapshot?.interrupt).toMatchObject({ kind: "smoke_validation_review", gate_id: "gate-1" });
  });

  it("drops a stale smoke review when its gate does not match the workspace", async () => {
    const request = vi.fn(async () => jsonResponse({
      values: {
        workspace: {
          schema_version: "1.0", mode: "running", is_working: true,
          active_phase: "full", validation_gate: { gate_id: "gate-current", status: "started" },
          run: {}, stages: [], artifacts: [],
        },
        __interrupt__: [{ value: {
          schema_version: "1.0", kind: "smoke_validation_review", owner: "local-user",
          thread_id: "thread-1", gate_id: "gate-stale", report: {},
          allowed_actions: ["start_full_run", "rerun_smoke", "pause", "cancel"],
        } }],
      },
    }));
    const client = new HttpLangGraphClient({ apiUrl: "http://localhost:2024", fetch: request as typeof fetch, statePollIntervalMs: 100_000 });
    const events: Array<{ type: string; snapshot?: { interrupt: unknown } }> = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event as typeof events[number]));
    await vi.waitFor(() => expect(events).toHaveLength(1));
    unsubscribe();
    expect(events[0]?.snapshot?.interrupt).toBeNull();
  });

  it("deduplicates conflicting in-flight actions for the same gate identity", async () => {
    let release!: () => void;
    const pending = new Promise<void>((resolve) => { release = resolve; });
    const request = vi.fn(async () => {
      await pending;
      return jsonResponse({});
    });
    const client = new HttpLangGraphClient({ apiUrl: "http://localhost:2024", fetch: request as typeof fetch });
    const decision = { schema_version: "1.0" as const, action: "start_full_run" as const, gate_id: "gate-1" };
    const conflicting = { schema_version: "1.0" as const, action: "rerun_smoke" as const, gate_id: "gate-1" };

    const first = client.resume("thread-1", decision);
    const duplicate = client.resume("thread-1", decision);
    const conflict = client.resume("thread-1", conflicting);
    expect(request).toHaveBeenCalledTimes(1);
    release();
    await Promise.all([first, duplicate, conflict]);
  });

  it("requests bounded asynchronous reconciliation for detached durable stages", async () => {
    const urls: string[] = [];
    const bodies: unknown[] = [];
    const request = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith("/state")) return jsonResponse({ values: { workspace: { schema_version: "1.0", mode: "detached", is_working: true, active_phase: "validation", validation_target_h5: "/input/session-01.h5", full_input_root: "/input", linked_runs: [{ run_id: "validation-1", scope: "validation" }], validation_gate: { gate_id: "gate-1", status: "running" }, run: {}, stages: [], artifacts: [], external_stage_job: { stage_execution_id: "stage_tokenize_123", stage: "tokenize", status: "detached", reconnectable: true } } } });
      if (url.endsWith("/runs")) {
        bodies.push(JSON.parse(String(init?.body)));
        return jsonResponse({});
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    const client = new HttpLangGraphClient({ apiUrl: "http://localhost:2024", fetch: request as typeof fetch, statePollIntervalMs: 100_000 });
    const unsubscribe = client.subscribe("thread-1", () => undefined);
    await vi.waitFor(() => expect(urls.some((url) => url.endsWith("/runs"))).toBe(true));
    unsubscribe();
    expect(bodies).toContainEqual(expect.objectContaining({ input: expect.objectContaining({
      reconcile_requested: true,
      thread_id: "thread-1",
      active_phase: "validation",
      validation_target_h5: "/input/session-01.h5",
      full_input_root: "/input",
    }) }));
    const reconcileInput = (bodies[0] as { input: Record<string, unknown> }).input;
    expect(reconcileInput).not.toHaveProperty("linked_runs");
    expect(reconcileInput).not.toHaveProperty("validation_gate");
  });

  it("deduplicates unchanged checkpoint polls without opening a permanent stream", async () => {
    const state = {
      values: {
        workspace: { schema_version: "1.0", mode: "running", is_working: true, run: {}, stages: [], artifacts: [] },
        messages: [{ id: "m1", type: "ai", content: "Running" }],
      },
    };
    const request = vi.fn(async (input: RequestInfo | URL) => {
      expect(String(input)).toMatch(/\/threads\/thread-1\/state$/);
      return jsonResponse(state);
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      reconnectDelayMs: 100_000,
      statePollIntervalMs: 5,
    });
    const events: unknown[] = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event));
    await vi.waitFor(() => expect(request.mock.calls.length).toBeGreaterThanOrEqual(2));
    unsubscribe();
    expect(events).toHaveLength(1);
    expect(events.every((event) => (event as { type?: string }).type === "snapshot")).toBe(true);
    expect(request.mock.calls.every(([input]) => !String(input).includes("/stream"))).toBe(true);
  });

  it("reconciles a changed checkpoint through polling", async () => {
    const coding = {
      values: {
        mode: "ready_for_coding",
        thread_id: "thread-1",
        pending_adapter: { task_id: "task-1", name: "eeg", modality: "eeg", revision: 1 },
        workspace: { schema_version: "1.0", mode: "ready_for_coding", is_working: false, run: {}, stages: [], artifacts: [] },
      },
    };
    const approval = {
      values: {
        mode: "awaiting_patch_approval",
        workspace: { schema_version: "1.0", mode: "awaiting_patch_approval", is_working: false, run: {}, stages: [], artifacts: [] },
        __interrupt__: [{ value: { kind: "adapter_patch_approval", thread_id: "thread-1", task_id: "task-1", revision: 1 } }],
      },
    };
    let stateReads = 0;
    const request = vi.fn(async (input: RequestInfo | URL) => {
      expect(String(input)).toMatch(/\/state$/);
      return jsonResponse(stateReads++ === 0 ? coding : approval);
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      reconnectDelayMs: 100_000,
      statePollIntervalMs: 5,
    });
    const events: Array<{ type: string; snapshot?: { handoff: unknown; interrupt: unknown } }> = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event as typeof events[number]));
    await vi.waitFor(() => expect(events[events.length - 1]?.snapshot?.interrupt).toMatchObject({ kind: "adapter_patch_approval" }));
    unsubscribe();

    expect(events[0]?.snapshot?.handoff).not.toBeNull();
    expect(events[events.length - 1]?.snapshot?.handoff).toBeNull();
  });

  it("clears stale handoffs and interrupts when a newer snapshot leaves coding", async () => {
    const coding = {
      values: {
        mode: "ready_for_coding",
        thread_id: "thread-1",
        pending_adapter: { task_id: "task-1", name: "eeg", modality: "eeg", revision: 1 },
        workspace: { schema_version: "1.0", mode: "ready_for_coding", is_working: false, run: {}, stages: [], artifacts: [] },
      },
    };
    const approval = {
      values: {
        mode: "awaiting_patch_approval",
        workspace: { schema_version: "1.0", mode: "awaiting_patch_approval", is_working: false, run: {}, stages: [], artifacts: [] },
        __interrupt__: [{ value: { kind: "adapter_patch_approval", thread_id: "thread-1", task_id: "task-1", revision: 1 } }],
      },
    };
    let stateReads = 0;
    const request = vi.fn(async (input: RequestInfo | URL) => {
      expect(String(input)).toMatch(/\/state$/);
      return jsonResponse(stateReads++ === 0 ? coding : approval);
    });
    const client = new HttpLangGraphClient({
      apiUrl: "http://localhost:2024",
      fetch: request as typeof fetch,
      reconnectDelayMs: 100_000,
      statePollIntervalMs: 5,
    });
    const events: Array<{ type: string; snapshot?: { handoff: unknown; interrupt: unknown } }> = [];
    const unsubscribe = client.subscribe("thread-1", (event) => events.push(event as typeof events[number]));
    await vi.waitFor(() => expect(events).toHaveLength(2));
    unsubscribe();

    expect(events[0]?.snapshot?.handoff).not.toBeNull();
    expect(events[1]?.snapshot?.handoff).toBeNull();
    expect(events[1]?.snapshot?.interrupt).toMatchObject({ kind: "adapter_patch_approval" });
  });
});
