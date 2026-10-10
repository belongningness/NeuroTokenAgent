import { describe, expect, it } from "vitest";
import { NativeCodingUiAdapter } from "./nativeUiAdapter";
import type { CodingEvent, CodingRawAppServerEvent } from "./runtime";

const taskId = "task-1";
const revision = 2;
const workspaceId = `neurotoken:${taskId}`;
const threadId = "thread-1";

function raw(message: Record<string, unknown>): CodingRawAppServerEvent {
  return { workspace_id: workspaceId, message };
}

function coding(message: Record<string, unknown>): CodingEvent {
  return {
    type: "app_server",
    message,
    rawMessage: message,
    workspaceId,
  };
}

describe("NativeCodingUiAdapter", () => {
  it("converts native items and merges streaming deltas by item identity", () => {
    const adapter = new NativeCodingUiAdapter(taskId, revision, threadId);

    adapter.applyRawEvent(raw({
      method: "item/started",
      params: {
        threadId,
        item: { id: "command-1", type: "commandExecution", command: "npm test", status: "inProgress" },
      },
    }));
    adapter.applyRawEvent(raw({
      method: "item/commandExecution/outputDelta",
      params: { threadId, itemId: "command-1", delta: "first\n" },
    }));
    adapter.applyRawEvent(raw({
      method: "item/completed",
      params: {
        threadId,
        item: {
          id: "command-1",
          type: "commandExecution",
          command: "npm test",
          status: "completed",
          aggregatedOutput: "first\nsecond\n",
        },
      },
    }));

    expect(adapter.getState()).toMatchObject({
      isProcessing: true,
      conversationItems: [{
        id: "command-1",
        kind: "tool",
        status: "completed",
        output: "first\nsecond\n",
      }],
    });
  });

  it("builds assistant and reasoning items from raw deltas", () => {
    const adapter = new NativeCodingUiAdapter(taskId, revision, threadId);
    adapter.applyRawEvent(raw({
      method: "item/agentMessage/delta",
      params: { threadId, itemId: "answer-1", delta: "Hello " },
    }));
    adapter.applyRawEvent(raw({
      method: "item/agentMessage/delta",
      params: { threadId, itemId: "answer-1", delta: "world" },
    }));
    adapter.applyRawEvent(raw({
      method: "item/reasoning/summaryTextDelta",
      params: { threadId, itemId: "reason-1", delta: "Inspecting" },
    }));

    expect(adapter.getState().conversationItems).toEqual([
      { id: "answer-1", kind: "message", role: "assistant", text: "Hello world" },
      { id: "reason-1", kind: "reasoning", summary: "Inspecting", content: "" },
    ]);
  });

  it("keeps assistant output when Codex deltas omit item identity", () => {
    const adapter = new NativeCodingUiAdapter(taskId, revision, threadId);
    adapter.applyRawEvent(raw({
      method: "item/agentMessage/delta",
      params: { threadId, delta: "Visible " },
    }));
    adapter.applyRawEvent(raw({
      method: "item/agentMessage/delta",
      params: { threadId, delta: "output" },
    }));

    expect(adapter.getState().conversationItems).toEqual([
      { id: "streaming-message", kind: "message", role: "assistant", text: "Visible output" },
    ]);
  });

  it("collects approval requests and derives processing state", () => {
    const adapter = new NativeCodingUiAdapter(taskId, revision, threadId);
    adapter.applyRawEvent(raw({
      method: "turn/started",
      params: { threadId, turn: { id: "turn-1" } },
    }));
    adapter.applyRawEvent(raw({
      id: 42,
      method: "item/commandExecution/requestApproval",
      params: { threadId, command: ["npm", "test"] },
    }));

    expect(adapter.getState()).toMatchObject({
      isProcessing: true,
      approvalRequests: [{
        workspace_id: workspaceId,
        request_id: 42,
        method: "item/commandExecution/requestApproval",
      }],
    });

    adapter.resolveApproval(42);
    expect(adapter.getState().approvalRequests).toEqual([]);

    adapter.applyRawEvent(raw({
      method: "turn/completed",
      params: { threadId, turn: { id: "turn-1" } },
    }));
    expect(adapter.getState().isProcessing).toBe(false);
  });

  it("uses replacement daemon history as authoritative and deduplicates identities", () => {
    const adapter = new NativeCodingUiAdapter(taskId, revision, threadId);
    const completion = coding({
      method: "item/completed",
      params: {
        threadId,
        item: { id: "answer-1", type: "agentMessage", text: "Authoritative answer" },
      },
    });

    adapter.applyCodingEvent(coding({
      method: "item/agentMessage/delta",
      params: { threadId, itemId: "stale", delta: "stale" },
    }));
    adapter.replaceHistory([completion, completion]);

    expect(adapter.getState().conversationItems).toEqual([
      { id: "answer-1", kind: "message", role: "assistant", text: "Authoritative answer" },
    ]);
  });

  it("isolates events by task workspace and thread", () => {
    const adapter = new NativeCodingUiAdapter(taskId, revision, threadId);
    adapter.applyRawEvent({
      workspace_id: "neurotoken:other-task",
      message: { method: "item/agentMessage/delta", params: { threadId, itemId: "other", delta: "no" } },
    });
    adapter.applyRawEvent(raw({
      method: "item/agentMessage/delta",
      params: { threadId: "other-thread", itemId: "other", delta: "no" },
    }));

    expect(adapter.getState().conversationItems).toEqual([]);
  });
});
