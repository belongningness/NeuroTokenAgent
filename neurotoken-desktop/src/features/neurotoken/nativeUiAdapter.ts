import type { ApprovalRequest, ConversationItem } from "../../types";
import {
  buildConversationItem,
  buildConversationItemFromThreadItem,
  buildItemsFromThread,
  mergeThreadItems,
  upsertItem,
} from "../../utils/threadItems";
import type { CodingEvent, CodingRawAppServerEvent } from "./runtime";

export type NativeCodingUiState = {
  taskId: string;
  revision: number;
  conversationItems: ConversationItem[];
  approvalRequests: ApprovalRequest[];
  isProcessing: boolean;
};

type Identity = string | number | undefined;

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function asIdentity(value: unknown): Identity {
  return typeof value === "string" || typeof value === "number" ? value : undefined;
}

function field(record: Record<string, unknown>, camel: string, snake: string): Identity {
  return asIdentity(record[camel] ?? record[snake]);
}

function rawEventFromCodingEvent(event: CodingEvent): CodingRawAppServerEvent | null {
  if (event.type !== "app_server" || !event.rawMessage) return null;
  return {
    workspace_id: event.workspaceId ?? "",
    message: event.rawMessage,
  };
}

function appendToItem(
  items: ConversationItem[],
  itemId: Identity,
  delta: string,
  target: "message" | "output" | "summary" | "content",
): ConversationItem[] {
  if (!delta) return items;
  const id = String(itemId ?? `streaming-${target}`);
  const existing = items.find((item) => item.id === id);
  if (!existing) {
    if (target === "message") {
      return upsertItem(items, { id, kind: "message", role: "assistant", text: delta });
    }
    if (target === "summary" || target === "content") {
      return upsertItem(items, {
        id,
        kind: "reasoning",
        summary: target === "summary" ? delta : "",
        content: target === "content" ? delta : "",
      });
    }
    return items;
  }
  if (target === "message" && existing.kind === "message") {
    return items.map((item) => item.id === id ? { ...existing, text: existing.text + delta } : item);
  }
  if (target === "output" && existing.kind === "tool") {
    return items.map((item) => item.id === id
      ? { ...existing, output: `${existing.output ?? ""}${delta}` }
      : item);
  }
  if ((target === "summary" || target === "content") && existing.kind === "reasoning") {
    return items.map((item) => item.id === id
      ? {
          ...existing,
          summary: target === "summary" ? existing.summary + delta : existing.summary,
          content: target === "content" ? existing.content + delta : existing.content,
        }
      : item);
  }
  return items;
}

export class NativeCodingUiAdapter {
  private readonly workspaceId: string;
  private readonly threadId: string | null;
  private state: NativeCodingUiState;
  private seenEvents = new Set<string>();

  constructor(taskId: string, revision: number, threadId: string | null = null) {
    this.workspaceId = `neurotoken:${taskId}`;
    this.threadId = threadId;
    this.state = {
      taskId,
      revision,
      conversationItems: [],
      approvalRequests: [],
      isProcessing: false,
    };
  }

  getState(): NativeCodingUiState {
    return {
      ...this.state,
      conversationItems: [...this.state.conversationItems],
      approvalRequests: [...this.state.approvalRequests],
    };
  }

  replaceHistory(events: CodingEvent[]): NativeCodingUiState {
    this.reset();
    events.forEach((event) => this.applyCodingEvent(event));
    return this.getState();
  }

  replaceRawHistory(events: CodingRawAppServerEvent[]): NativeCodingUiState {
    this.reset();
    events.forEach((event) => this.applyRawEvent(event));
    return this.getState();
  }

  resolveApproval(requestId: string | number): NativeCodingUiState {
    this.state.approvalRequests = this.state.approvalRequests.filter(
      (request) => request.request_id !== requestId,
    );
    return this.getState();
  }

  applyCodingEvent(event: CodingEvent): NativeCodingUiState {
    if (event.type === "session") {
      if (event.session.taskId !== this.state.taskId || event.session.revision !== this.state.revision) {
        return this.getState();
      }
      this.state.isProcessing = event.session.status === "running";
      return this.getState();
    }
    if (event.type === "turn_completed" || event.type === "error") {
      this.state.isProcessing = false;
      return this.getState();
    }
    const raw = rawEventFromCodingEvent(event);
    if (raw) {
      this.applyRawEvent(raw);
      return this.getState();
    }
    if (event.type === "app_server") {
      const message = asRecord(event.message);
      const kind = String(message.kind ?? "");
      const id = String(message.item_id ?? message.request_id ?? `${kind}:${this.state.conversationItems.length}`);
      if (kind === "assistant_message") {
        this.state.conversationItems = upsertItem(this.state.conversationItems, {
          id,
          kind: "message",
          role: "assistant",
          text: String(message.text ?? ""),
        });
      } else if (kind === "command_output") {
        this.state.conversationItems = upsertItem(this.state.conversationItems, {
          id,
          kind: "tool",
          toolType: "commandExecution",
          title: String(message.title ?? "Command"),
          detail: "",
          output: String(message.output ?? ""),
          status: String(message.status ?? "completed"),
        });
      } else if (kind === "tool_call" && (event.requestId ?? message.request_id) !== undefined) {
        const requestId = event.requestId ?? message.request_id as string | number;
        this.state.approvalRequests = [...this.state.approvalRequests.filter(
          (request) => String(request.request_id) !== String(requestId),
        ), {
          workspace_id: this.workspaceId,
          request_id: requestId,
          method: String(message.method ?? "codex/requestApproval/tool"),
          params: message,
        }];
      }
    }
    return this.getState();
  }

  applyRawEvent(event: CodingRawAppServerEvent): NativeCodingUiState {
    if (event.workspace_id && event.workspace_id !== this.workspaceId) return this.getState();
    const message = asRecord(event.message);
    const method = String(message.method ?? "");
    const params = asRecord(message.params);
    const eventThreadId = field(params, "threadId", "thread_id")
      ?? field(asRecord(params.turn), "threadId", "thread_id");
    if (this.threadId && eventThreadId !== undefined && String(eventThreadId) !== this.threadId) {
      return this.getState();
    }
    const identity = this.eventIdentity(method, message, params);
    if (identity && this.seenEvents.has(identity)) return this.getState();
    if (identity) this.seenEvents.add(identity);

    if (method === "turn/started") this.state.isProcessing = true;
    if (method === "turn/completed" || method === "turn/failed" || method === "turn/cancelled" || method === "error") {
      this.state.isProcessing = false;
    }

    if (method.endsWith("requestApproval")) {
      const requestId = asIdentity(message.id) ?? field(params, "requestId", "request_id");
      if (requestId !== undefined) {
        const request: ApprovalRequest = {
          workspace_id: this.workspaceId,
          request_id: requestId,
          method,
          params,
        };
        const withoutRequest = this.state.approvalRequests.filter(
          (entry) => entry.request_id !== requestId,
        );
        this.state.approvalRequests = [...withoutRequest, request];
      }
      return this.getState();
    }

    if (method === "item/started" || method === "item/completed") {
      const item = asRecord(params.item);
      const itemForDisplay = {
        ...item,
        ...((item.type === "contextCompaction" || item.type === "webSearch") && !item.status
          ? { status: method === "item/started" ? "inProgress" : "completed" }
          : {}),
      };
      const converted = method === "item/completed"
        ? buildConversationItemFromThreadItem(itemForDisplay)
        : buildConversationItem(itemForDisplay);
      if (converted) this.state.conversationItems = upsertItem(this.state.conversationItems, converted);
      if (method === "item/started") this.state.isProcessing = true;
      return this.getState();
    }

    if (method === "item/agentMessage/delta") {
      this.state.conversationItems = appendToItem(
        this.state.conversationItems,
        field(params, "itemId", "item_id"),
        String(params.delta ?? ""),
        "message",
      );
      this.state.isProcessing = true;
      return this.getState();
    }
    if (method === "item/commandExecution/outputDelta" || method === "item/fileChange/outputDelta" || method === "item/plan/delta") {
      this.state.conversationItems = appendToItem(
        this.state.conversationItems,
        field(params, "itemId", "item_id"),
        String(params.delta ?? ""),
        "output",
      );
      return this.getState();
    }
    if (method === "item/reasoning/summaryTextDelta") {
      this.state.conversationItems = appendToItem(
        this.state.conversationItems,
        field(params, "itemId", "item_id"),
        String(params.delta ?? ""),
        "summary",
      );
      this.state.isProcessing = true;
      return this.getState();
    }
    if (method === "item/reasoning/summaryPartAdded") {
      this.state.conversationItems = appendToItem(
        this.state.conversationItems,
        field(params, "itemId", "item_id"),
        "\n",
        "summary",
      );
      this.state.isProcessing = true;
      return this.getState();
    }
    if (method === "item/reasoning/textDelta") {
      this.state.conversationItems = appendToItem(
        this.state.conversationItems,
        field(params, "itemId", "item_id"),
        String(params.delta ?? ""),
        "content",
      );
      this.state.isProcessing = true;
      return this.getState();
    }

    const thread = asRecord(params.thread ?? message.thread);
    if (Array.isArray(thread.turns)) {
      const daemonItems = buildItemsFromThread(thread);
      this.state.conversationItems = mergeThreadItems(daemonItems, this.state.conversationItems);
    }
    return this.getState();
  }

  private reset() {
    this.state = {
      ...this.state,
      conversationItems: [],
      approvalRequests: [],
      isProcessing: false,
    };
    this.seenEvents = new Set();
  }

  private eventIdentity(
    method: string,
    message: Record<string, unknown>,
    params: Record<string, unknown>,
  ): string | null {
    const item = asRecord(params.item);
    const itemId = field(params, "itemId", "item_id") ?? asIdentity(item.id);
    const turn = asRecord(params.turn);
    const turnId = field(params, "turnId", "turn_id") ?? asIdentity(turn.id);
    const requestId = asIdentity(message.id) ?? field(params, "requestId", "request_id");
    const eventId = field(params, "eventId", "event_id") ?? asIdentity(message.event_id);
    if (eventId !== undefined) return `${method}:event:${String(eventId)}`;
    if ((method === "item/started" || method === "item/completed") && itemId !== undefined) {
      return `${method}:item:${String(itemId)}`;
    }
    if ((method === "turn/started" || method === "turn/completed") && turnId !== undefined) {
      return `${method}:turn:${String(turnId)}`;
    }
    if (requestId !== undefined && method.endsWith("requestApproval")) {
      return `${method}:request:${String(requestId)}`;
    }
    return null;
  }
}
