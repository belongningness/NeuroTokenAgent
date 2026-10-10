import { describe, expect, it, vi } from "vitest";
import {
  EMPTY_LANGGRAPH_VIEW,
  MockLangGraphClient,
  reduceLangGraphEvent,
  viewForThread,
} from "./mockLangGraph";

describe("mock LangGraph runtime", () => {
  it("reduces out-of-order workspace and message events", () => {
    const withWorkspace = reduceLangGraphEvent(EMPTY_LANGGRAPH_VIEW, {
      type: "workspace",
      snapshot: {
        schema_version: "1.0",
        mode: "running",
        is_working: true,
        run: {},
        stages: [],
        artifacts: [],
      },
    });
    const withMessage = reduceLangGraphEvent(withWorkspace, {
      type: "message",
      role: "assistant",
      content: "Running",
    });

    expect(withMessage.workspace?.mode).toBe("running");
    expect(withMessage.messages[0]?.content).toBe("Running");
  });

  it("deduplicates repeated message events", () => {
    const event = { type: "message", role: "assistant", content: "Ready" } as const;
    const once = reduceLangGraphEvent(EMPTY_LANGGRAPH_VIEW, event);
    const twice = reduceLangGraphEvent(once, event);

    expect(twice.messages).toHaveLength(1);
  });

  it("replaces optimistic user messages with their durable checkpoint identity", () => {
    const optimistic = {
      ...EMPTY_LANGGRAPH_VIEW,
      messages: [{
        id: "user:process-data",
        role: "user" as const,
        content: "Process   the dataset",
      }],
    };
    const durable = reduceLangGraphEvent(optimistic, {
      type: "snapshot",
      snapshot: {
        workspace: null,
        interrupt: null,
        handoff: null,
        messages: [{ id: "human-uuid", role: "user", content: "Process the dataset" }],
        error: null,
      },
    });

    expect(durable.messages).toEqual([{
      id: "human-uuid",
      role: "user",
      content: "Process the dataset",
    }]);
  });

  it("preserves history when a decision-only snapshot arrives", () => {
    const initial = reduceLangGraphEvent(EMPTY_LANGGRAPH_VIEW, {
      type: "snapshot",
      snapshot: {
        workspace: null,
        interrupt: null,
        handoff: null,
        messages: [
          { id: "user-1", role: "user", content: "Build the adapter" },
          { id: "assistant-1", role: "assistant", content: "Candidate ready" },
        ],
        error: null,
      },
    });
    const decided = reduceLangGraphEvent(initial, {
      type: "snapshot",
      snapshot: {
        workspace: null,
        interrupt: null,
        handoff: null,
        messages: [{ id: "assistant-2", role: "assistant", content: "Decision submitted" }],
        error: null,
      },
    });

    expect(decided.messages.map((message) => message.content)).toEqual([
      "Build the adapter",
      "Candidate ready",
      "Decision submitted",
    ]);
  });

  it("clears stale errors when a durable snapshot marks the workflow completed", () => {
    const failed = {
      ...EMPTY_LANGGRAPH_VIEW,
      error: "Failed to resolve task worktree: No such file or directory",
    };
    const completed = reduceLangGraphEvent(failed, {
      type: "snapshot",
      snapshot: {
        workspace: {
          schema_version: "1.0",
          mode: "completed",
          is_working: false,
          run: {},
          stages: [],
          artifacts: [],
        },
        interrupt: null,
        handoff: null,
        messages: [],
        error: "Failed to resolve task worktree: No such file or directory",
      },
    });

    expect(completed.error).toBeNull();
  });

  it("clears the last coding handoff when an authoritative snapshot omits active coding state", () => {
    const handoff = {
      schema_version: "1.0" as const,
      task_id: "task-1",
      thread_id: "thread-1",
      revision: 1,
      status: "ready_for_coding" as const,
      adapter: { name: "eeg", modality: "eeg", notes: "" },
      allowed_paths: ["agent_pipeline/adapters/"],
      instruction: "Implement adapter",
      created_at: "2026-08-06T00:00:00Z",
    };
    const coding = reduceLangGraphEvent(EMPTY_LANGGRAPH_VIEW, { type: "handoff", handoff });
    const applied = reduceLangGraphEvent(coding, {
      type: "snapshot",
      snapshot: {
        workspace: {
          schema_version: "1.0",
          mode: "adapter_ready",
          is_working: false,
          run: {},
          stages: [],
          artifacts: [],
        },
        interrupt: null,
        handoff: null,
        messages: [{ id: "done", role: "assistant", content: "Adapter applied" }],
        error: null,
      },
    });

    expect(applied.handoff).toBeNull();
    expect(applied.messages[applied.messages.length - 1]?.content).toBe("Adapter applied");
  });

  it("stores coding handoffs for the real runtime boundary", () => {
    const state = reduceLangGraphEvent(EMPTY_LANGGRAPH_VIEW, {
      type: "handoff",
      handoff: {
        schema_version: "1.0",
        task_id: "task-1",
        thread_id: "thread-1",
        revision: 1,
        status: "ready_for_coding",
        adapter: { name: "eeg", modality: "eeg", notes: "" },
        allowed_paths: ["agent_pipeline/adapters/"],
        instruction: "Implement adapter",
        created_at: "2026-08-06T00:00:00Z",
      },
    });

    expect(state.handoff?.task_id).toBe("task-1");
  });

  it("keeps a newer coding revision when a stale approval snapshot arrives", () => {
    const revisionTwo = {
      schema_version: "1.0" as const,
      task_id: "task-1",
      thread_id: "thread-1",
      revision: 2,
      status: "ready_for_coding" as const,
      adapter: { name: "eeg", modality: "eeg", notes: "" },
      allowed_paths: ["agent_pipeline/adapters/"],
      instruction: "Initial implementation",
      created_at: "2026-08-06T00:00:00Z",
    };
    const awaitingApproval = reduceLangGraphEvent(EMPTY_LANGGRAPH_VIEW, {
      type: "snapshot",
      snapshot: {
        workspace: {
          schema_version: "1.0",
          mode: "awaiting_patch_approval",
          is_working: false,
          run: {},
          stages: [],
          artifacts: [],
        },
        interrupt: {
          schema_version: "1.0",
          kind: "adapter_patch_approval",
          owner: "local-user",
          thread_id: "thread-1",
          task_id: "task-1",
          revision: 2,
          adapter: { diff: "revision two" },
          allowed_actions: ["apply", "apply_and_commit", "ask", "refine", "reject"],
        },
        handoff: revisionTwo,
        messages: [],
        error: null,
      },
    });
    const revisionThree = reduceLangGraphEvent(awaitingApproval, {
      type: "handoff",
      handoff: { ...revisionTwo, revision: 3, instruction: "Requested revision" },
    });
    const afterStalePoll = reduceLangGraphEvent(revisionThree, {
      type: "snapshot",
      snapshot: {
        workspace: awaitingApproval.workspace,
        interrupt: awaitingApproval.interrupt,
        handoff: revisionTwo,
        messages: [{ id: "old", role: "assistant", content: "Old approval checkpoint" }],
        error: null,
      },
    });

    expect(revisionThree.workspace?.mode).toBe("ready_for_coding");
    expect(revisionThree.interrupt).toBeNull();
    const afterStaleInterrupt = reduceLangGraphEvent(afterStalePoll, {
      type: "interrupt",
      interrupt: awaitingApproval.interrupt!,
    });

    expect(afterStalePoll.handoff?.revision).toBe(3);
    expect(afterStalePoll.workspace?.mode).toBe("ready_for_coding");
    expect(afterStalePoll.interrupt).toBeNull();
    expect(afterStaleInterrupt.interrupt).toBeNull();
  });

  it("builds all three interrupt scenarios from fixtures", () => {
    expect(viewForThread("thread-pipeline-001").interrupt?.kind).toBe("plan_approval");
    expect(viewForThread("thread-run-001").interrupt?.kind).toBe("stage_review");
    expect(viewForThread("thread-adapter-001").interrupt?.kind).toBe(
      "adapter_patch_approval",
    );
  });

  it("rebuilds current state when a client reconnects", () => {
    const client = new MockLangGraphClient();
    const firstListener = vi.fn();
    const secondListener = vi.fn();

    const unsubscribe = client.subscribe("thread-run-001", firstListener);
    unsubscribe();
    client.subscribe("thread-run-001", secondListener);

    expect(firstListener).toHaveBeenCalledTimes(3);
    expect(secondListener).toHaveBeenCalledTimes(3);
  });
});
