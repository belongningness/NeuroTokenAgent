import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  Bot,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Moon,
  Plus,
  ScrollText,
  Server,
  Sun,
  Trash2,
} from "lucide-react";
import type { ConversationItem, ThreadSummary } from "../../types";
import type { AdapterCodingHandoff } from "./contracts";
import type { ThreadStatusById } from "../../utils/threadStatus";
import { ThreadList } from "../app/components/ThreadList";
import { ModalShell } from "../design-system/components/modal/ModalShell";
import { CodingWorkspace } from "./CodingWorkspace";
import { NeuroTokenConversation } from "./NeuroTokenConversation";
import { StageOutputPanel } from "./StageOutputPanel";
import { WorkflowDecisionCard } from "./WorkflowDecisionCard";
import { INITIAL_HANDOFF_STATE, handoffReducer, type HandoffTask } from "./handoffState";
import { HttpLangGraphClient } from "./langGraphClient";
import { MOCK_CODING_HANDOFF, MockCodingRuntime } from "./mockCodingRuntime";
import { RemoteCodingRuntime } from "./remoteCodingRuntime";
import { SystemSshManager } from "./systemSshManager";
import {
  EMPTY_LANGGRAPH_VIEW,
  MockLangGraphClient,
  reduceLangGraphEvent,
  type MockLangGraphView,
} from "./mockLangGraph";
import type {
  CodingRuntime,
  CodingSession,
  LangGraphArtifact,
  LangGraphClient,
  LangGraphDecision,
  SshConnectionState,
  SshManager,
} from "./runtime";
import {
  EMPTY_NEUROTOKEN_SHELL_STATE,
  INITIAL_NEUROTOKEN_SHELL_STATE,
  neuroTokenShellReducer,
  type CenterWorkspace,
  type NeuroTokenTask,
} from "./shellState";
import "./neurotoken-shell.css";

type PersistedCodingContext = {
  handoff: AdapterCodingHandoff;
};

type PendingPatchRevision = {
  threadId: string;
  taskId: string;
  revision: number;
};

const CODING_CONTEXT_STORAGE_KEY = "neurotoken.coding-contexts.v1";

function loadCodingContexts(): Record<string, PersistedCodingContext> {
  try {
    const value = window.localStorage.getItem(CODING_CONTEXT_STORAGE_KEY);
    return value ? JSON.parse(value) as Record<string, PersistedCodingContext> : {};
  } catch {
    return {};
  }
}

const workspaceLabels: Record<CenterWorkspace, string> = {
  conversation: "Conversation",
  coding: "Coding",
  editor: "Editor",
};

function createLangGraphClient(): LangGraphClient {
  const apiUrl = import.meta.env.VITE_NEUROTOKEN_LANGGRAPH_URL;
  if (!apiUrl) return new MockLangGraphClient();
  return new HttpLangGraphClient({
    apiUrl,
    assistantId: import.meta.env.VITE_NEUROTOKEN_LANGGRAPH_ASSISTANT ?? "neurotoken_agent",
    apiKey: import.meta.env.VITE_NEUROTOKEN_LANGGRAPH_API_KEY,
  });
}

function createCodingRuntime(): CodingRuntime {
  return import.meta.env.VITE_NEUROTOKEN_CODING_RUNTIME === "remote"
    ? new RemoteCodingRuntime()
    : new MockCodingRuntime();
}

function createSshManager(): SshManager | null {
  return import.meta.env.VITE_NEUROTOKEN_CODING_RUNTIME === "remote"
    ? new SystemSshManager()
    : null;
}

const DEFAULT_SSH_STATE: SshConnectionState = { status: "disconnected" };
const IS_REMOTE_MODE = import.meta.env.VITE_NEUROTOKEN_CODING_RUNTIME === "remote";
const OPERATIONAL_CODING_MODES = new Set(["ready_for_coding", "authoring_adapter", "awaiting_patch_approval"]);
const TERMINAL_WORKFLOW_MODES = new Set(["adapter_ready", "completed", "cancelled", "failed"]);
const TERMINAL_HANDOFF_STATUSES = new Set(["applied", "rejected", "cancelled", "failed"]);

function taskFromThread(thread: { threadId: string; title: string; updatedAt: string; workspaceMode?: string }): NeuroTokenTask {
  const status = thread.workspaceMode ? taskStatusFromWorkspaceMode(thread.workspaceMode) : "idle";
  return {
    id: thread.threadId,
    threadId: thread.threadId,
    title: taskTitle(thread.title),
    subtitle: thread.workspaceMode ? taskSubtitleFromWorkspaceMode(thread.workspaceMode) : "Remote LangGraph task",
    status,
    updatedAt: thread.updatedAt,
  };
}

function taskTitle(request: string): string {
  const firstLine = request.split(/\r?\n/, 1)[0]?.trim() || "New task";
  return firstLine.length > 54 ? `${firstLine.slice(0, 51)}…` : firstLine;
}

function taskStatusFromWorkspaceMode(mode: string): NeuroTokenTask["status"] {
  if (mode === "completed" || mode === "adapter_ready") return "completed";
  if (mode === "failed") return "failed";
  if (mode === "awaiting_plan_approval" || mode === "reviewing_stage" || mode === "awaiting_patch_approval" || mode === "adapter_revision_required" || mode === "paused") return "awaiting_input";
  if (mode === "ready_for_coding" || mode === "authoring_adapter") return "coding";
  if (mode === "approved" || mode === "running") return "running";
  return "idle";
}

function taskSubtitleFromWorkspaceMode(mode: string): string {
  if (mode === "adapter_ready") return "Adapter ready";
  if (mode === "completed") return "Pipeline completed";
  if (mode === "failed") return "Pipeline failed";
  if (mode === "awaiting_patch_approval") return "Review and apply adapter";
  if (mode === "adapter_revision_required") return "Adapter validation stopped — retry explicitly";
  if (mode === "awaiting_plan_approval") return "Plan ready for review";
  if (mode === "reviewing_stage") return "Stage ready for review";
  if (mode === "ready_for_coding" || mode === "authoring_adapter") return "Coding workspace active";
  if (mode === "paused") return "Pipeline paused";
  if (mode === "running" || mode === "approved") return "Pipeline running";
  return "Remote LangGraph task";
}

function downloadArtifact(artifact: LangGraphArtifact) {
  const url = URL.createObjectURL(new Blob([artifact.content], { type: artifact.mediaType }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = artifact.fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

type NeuroTokenDesktopShellProps = {
  langGraphClient?: LangGraphClient;
  codingRuntime?: CodingRuntime;
};

export function NeuroTokenDesktopShell({ langGraphClient, codingRuntime: injectedCodingRuntime }: NeuroTokenDesktopShellProps = {}) {
  const [state, dispatch] = useReducer(
    neuroTokenShellReducer,
    IS_REMOTE_MODE ? EMPTY_NEUROTOKEN_SHELL_STATE : INITIAL_NEUROTOKEN_SHELL_STATE,
  );
  const [langGraphView, setLangGraphView] = useState(EMPTY_LANGGRAPH_VIEW);
  const [codingContexts, setCodingContexts] = useState<Record<string, PersistedCodingContext>>(loadCodingContexts);
  const [invalidCodingSessions, setInvalidCodingSessions] = useState<Set<string>>(() => new Set());
  const [client] = useState(() => langGraphClient ?? createLangGraphClient());
  const [tasksLoading, setTasksLoading] = useState(IS_REMOTE_MODE);
  const [taskListError, setTaskListError] = useState<string | null>(null);
  const [newTaskOpen, setNewTaskOpen] = useState(false);
  const [newTaskRequest, setNewTaskRequest] = useState("");
  const [newTaskError, setNewTaskError] = useState<string | null>(null);
  const [newTaskSubmitting, setNewTaskSubmitting] = useState(false);
  const [deleteSubmitting, setDeleteSubmitting] = useState(false);
  const [taskPendingDeletion, setTaskPendingDeletion] = useState<NeuroTokenTask | null>(null);
  const [retrySubmitting, setRetrySubmitting] = useState(false);
  const [editRetryOpen, setEditRetryOpen] = useState(false);
  const [editRetryRequest, setEditRetryRequest] = useState("");
  const [artifactPreview, setArtifactPreview] = useState<LangGraphArtifact | null>(null);
  const [artifactError, setArtifactError] = useState<string | null>(null);
  const [planRefinement, setPlanRefinement] = useState("");
  const [patchQuestion, setPatchQuestion] = useState("");
  const [patchRefinement, setPatchRefinement] = useState("");
  const [followUpMessage, setFollowUpMessage] = useState("");
  const [followUpSubmitting, setFollowUpSubmitting] = useState(false);
  const [patchReviewOpen, setPatchReviewOpen] = useState(false);
  const [pendingPatchRevision, setPendingPatchRevision] = useState<PendingPatchRevision | null>(null);
  const [stageOutputCollapsed, setStageOutputCollapsed] = useState(false);
  const [codingRuntime] = useState(() => injectedCodingRuntime ?? createCodingRuntime());
  const [sshManager] = useState(createSshManager);
  const [sshState, setSshState] = useState<SshConnectionState>(DEFAULT_SSH_STATE);
  const [sshDiagnosticsFeedback, setSshDiagnosticsFeedback] = useState<string | null>(null);
  const [handoff, dispatchHandoff] = useReducer(
    handoffReducer,
    IS_REMOTE_MODE ? { tasks: {} } : INITIAL_HANDOFF_STATE,
  );
  const submittingCandidates = useRef(new Set<string>());
  const submittedCandidates = useRef(new Set<string>());
  const submittingDecisions = useRef(new Map<string, Promise<void>>());
  const langGraphViewThreadRef = useRef<string | null>(null);
  const selectedThreadRef = useRef<string | null>(null);
  const threadViewsRef = useRef(new Map<string, MockLangGraphView>());
  const [leftPanelWidth, setLeftPanelWidth] = useState(292);
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  useEffect(() => {
    const root = document.documentElement;
    const previousTheme = root.dataset.theme;
    root.dataset.theme = theme;
    return () => {
      if (previousTheme) root.dataset.theme = previousTheme;
      else delete root.dataset.theme;
    };
  }, [theme]);
  const selectedTask = useMemo(
    () => state.tasks.find((task) => task.id === state.selectedTaskId) ?? null,
    [state.selectedTaskId, state.tasks],
  );
  const selectedThreadId = selectedTask?.threadId ?? null;
  selectedThreadRef.current = selectedThreadId;
  const visibleLangGraphView = langGraphViewThreadRef.current === selectedThreadId
    ? langGraphView
    : EMPTY_LANGGRAPH_VIEW;
  const conversationItems = useMemo<ConversationItem[]>(
    () => visibleLangGraphView.messages.map((message) => ({
      id: message.id,
      kind: "message",
      role: message.role,
      text: message.content,
    })),
    [visibleLangGraphView.messages],
  );
  const visibleAdapterTaskId = visibleLangGraphView.handoff?.task_id ??
    (visibleLangGraphView.interrupt?.kind === "adapter_patch_approval"
      ? visibleLangGraphView.interrupt.task_id
      : null);
  const visibleAdapterRevision = visibleLangGraphView.handoff?.revision ??
    (visibleLangGraphView.interrupt?.kind === "adapter_patch_approval"
      ? visibleLangGraphView.interrupt.revision
      : null);
  const isPreparingPatchRevision = Boolean(
    pendingPatchRevision &&
    pendingPatchRevision.threadId === selectedThreadId &&
    pendingPatchRevision.taskId === visibleAdapterTaskId &&
    visibleAdapterRevision != null &&
    visibleAdapterRevision <= pendingPatchRevision.revision,
  );
  const activeInterrupt = isPreparingPatchRevision ? null : visibleLangGraphView.interrupt;
  const activeWorkspace = isPreparingPatchRevision && visibleLangGraphView.workspace
    ? {
      ...visibleLangGraphView.workspace,
      mode: "authoring_adapter",
      is_working: true,
      approval: null,
    }
    : visibleLangGraphView.workspace;
  const decisionCardKey = useMemo(() => {
    const interrupt = activeInterrupt;
    if (!interrupt) return "none";
    const lastAssistantMessage = [...visibleLangGraphView.messages].reverse().find((message) => message.role === "assistant");
    const identity = interrupt.kind === "plan_approval"
      ? { plan: interrupt.plan, revision: interrupt.plan_revision ?? 0 }
      : interrupt.kind === "stage_review"
        ? { workflow_id: interrupt.workflow_id, stage: interrupt.stage, summary: interrupt.summary }
        : interrupt.kind === "smoke_validation_review"
          ? { workflow_id: interrupt.workflow_id, gate_id: interrupt.gate_id }
          : interrupt.kind === "adapter_execution_confirmation"
            ? { adapter_name: interrupt.adapter_name, input_path: interrupt.input_path, output_path: interrupt.output_path }
            : { task_id: interrupt.task_id, revision: interrupt.revision };
    return JSON.stringify([interrupt.kind, identity, lastAssistantMessage?.id ?? lastAssistantMessage?.content ?? ""]);
  }, [activeInterrupt, visibleLangGraphView.messages]);
  const persistedCodingContext = selectedThreadId ? codingContexts[selectedThreadId] : undefined;
  const workspaceMode = activeWorkspace?.mode ?? null;
  const handoffSignature = visibleLangGraphView.handoff
    ? JSON.stringify(visibleLangGraphView.handoff)
    : `${workspaceMode ?? ""}:${visibleLangGraphView.workspace?.approval?.request_id ?? ""}`;
  const recoveredHandoff = useMemo(() => {
    if (visibleLangGraphView.handoff) return visibleLangGraphView.handoff;
    const interrupt = activeInterrupt;
    if (interrupt?.kind === "adapter_patch_approval") {
      return {
        schema_version: "1.0" as const,
        task_id: interrupt.task_id,
        thread_id: interrupt.thread_id,
        revision: interrupt.revision,
        status: "ready_for_coding" as const,
        adapter: {
          name: String(interrupt.adapter.name ?? "adapter"),
          modality: String(interrupt.adapter.modality ?? "unknown"),
          notes: String(interrupt.adapter.notes ?? ""),
        },
        allowed_paths: ["agent_pipeline/adapters"],
        instruction: "Continue refining the adapter candidate.",
        created_at: "1970-01-01T00:00:00.000Z",
      };
    }
    if (workspaceMode !== "ready_for_coding" || !visibleLangGraphView.workspace?.approval?.request_id) return null;
    return {
      schema_version: "1.0" as const,
      task_id: visibleLangGraphView.workspace.approval.request_id,
      thread_id: selectedTask?.threadId ?? "",
      revision: 1,
      status: "ready_for_coding" as const,
      adapter: { name: "adapter", modality: "unknown", notes: "" },
      allowed_paths: ["agent_pipeline/adapters"],
      instruction: "Continue implementing the adapter for the inspected HDF5 source.",
      created_at: "1970-01-01T00:00:00.000Z",
    };
  }, [activeInterrupt, handoffSignature, selectedTask?.threadId, workspaceMode]);
  const activeCodingHandoff = useMemo(() => {
    if (isPreparingPatchRevision || !workspaceMode || !OPERATIONAL_CODING_MODES.has(workspaceMode)) return null;
    const mockFallback = !IS_REMOTE_MODE && MOCK_CODING_HANDOFF.thread_id === selectedThreadId
      ? persistedCodingContext?.handoff ?? MOCK_CODING_HANDOFF
      : null;
    const candidate = recoveredHandoff ?? mockFallback;
    if (
      !candidate ||
      TERMINAL_HANDOFF_STATUSES.has(candidate.status) ||
      invalidCodingSessions.has(`${candidate.task_id}:${candidate.revision}`)
    ) return null;
    return candidate;
  }, [invalidCodingSessions, isPreparingPatchRevision, persistedCodingContext?.handoff, recoveredHandoff, selectedThreadId, workspaceMode]);
  const patchApprovalInterrupt = activeInterrupt?.kind === "adapter_patch_approval" &&
    recoveredHandoff?.task_id === activeInterrupt.task_id &&
    recoveredHandoff.revision === activeInterrupt.revision
    ? activeInterrupt
    : null;
  useEffect(() => {
    if (!selectedThreadId || !recoveredHandoff) return;
    setCodingContexts((current) => {
      const previous = current[selectedThreadId];
      const next = { handoff: recoveredHandoff };
      if (JSON.stringify(previous) === JSON.stringify(next)) return current;
      const updated = { ...current, [selectedThreadId]: next };
      window.localStorage.setItem(CODING_CONTEXT_STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
  }, [patchApprovalInterrupt, recoveredHandoff, selectedThreadId]);
  const archivedCodingHandoff = !activeCodingHandoff
    ? persistedCodingContext?.handoff ?? null
    : null;
  const codingWorkspaceArchived = !activeCodingHandoff && Boolean(archivedCodingHandoff);
  const codingHandoff = activeCodingHandoff ?? archivedCodingHandoff;
  const registeredCodingTask = activeCodingHandoff ? handoff.tasks[activeCodingHandoff.task_id] : undefined;
  const codingTask: HandoffTask | undefined = registeredCodingTask ?? (archivedCodingHandoff ? {
    taskId: archivedCodingHandoff.task_id,
    threadId: archivedCodingHandoff.thread_id,
    codexThreadId: "",
    revision: archivedCodingHandoff.revision,
    status: workspaceMode === "failed" ? "failed" : "applied",
    validationError: null,
    commitMessage: "",
    applyMode: "apply",
  } : undefined);
  const availableWorkspaces: CenterWorkspace[] = codingHandoff && codingTask
    ? ["conversation", "coding", "editor"]
    : ["conversation"];
  const patchTaskId = activeInterrupt?.kind === "adapter_patch_approval"
    ? activeInterrupt.task_id
    : undefined;
  const patchTask = patchTaskId ? handoff.tasks[patchTaskId] : undefined;
  const handleCodingSessionOpened = useCallback((session: CodingSession) => {
    dispatchHandoff({
      type: "session_opened",
      taskId: session.taskId,
      revision: session.revision,
      codexThreadId: session.codexThreadId,
    });
  }, []);
  const handleCodingSessionInvalidated = useCallback((taskId: string, revision: number) => {
    setInvalidCodingSessions((current) => new Set(current).add(`${taskId}:${revision}`));
    dispatchHandoff({ type: "invalidate", taskId });
    if (selectedThreadRef.current) {
      const threadId = selectedThreadRef.current;
      setCodingContexts((current) => {
        if (!current[threadId]) return current;
        const updated = { ...current };
        delete updated[threadId];
        window.localStorage.setItem(CODING_CONTEXT_STORAGE_KEY, JSON.stringify(updated));
        return updated;
      });
    }
    dispatch({ type: "open_workspace", workspace: "conversation" });
  }, []);
  const activeTaskCount = state.tasks.filter(
    (task) => task.status !== "completed" && task.status !== "failed",
  ).length;
  const threadRows = useMemo(
    () => state.tasks.map((task) => ({
      thread: {
        id: task.id,
        name: task.title,
        updatedAt: Date.parse(task.updatedAt),
      } satisfies ThreadSummary,
      depth: 0,
    })),
    [state.tasks],
  );
  const threadStatusById = useMemo<ThreadStatusById>(
    () => Object.fromEntries(state.tasks.map((task) => [task.id, {
      isProcessing: task.status === "running" || task.status === "coding",
      hasUnread: task.status === "awaiting_input",
    }])),
    [state.tasks],
  );

  useEffect(() => {
    if (!IS_REMOTE_MODE || sshState.status !== "connected") return;
    let cancelled = false;
    setTasksLoading(true);
    setTaskListError(null);
    void client.listThreads().then((threads) => {
      if (!cancelled) dispatch({ type: "rehydrate", tasks: threads.map(taskFromThread) });
    }).catch((error) => {
      if (!cancelled) setTaskListError(error instanceof Error ? error.message : String(error));
    }).finally(() => {
      if (!cancelled) setTasksLoading(false);
    });
    return () => { cancelled = true; };
  }, [client, sshState.status]);

  useEffect(() => {
    if (!sshManager) {
      setSshState({ status: "connected" });
      return;
    }
    const unsubscribe = sshManager.subscribe(setSshState);
    void sshManager.checkAvailable().then((available) => {
      if (!available) {
        setSshState({ status: "error", message: "System SSH is not installed." });
        return;
      }
      const host = import.meta.env.VITE_NEUROTOKEN_SSH_HOST;
      if (!host) {
        setSshState({ status: "error", message: "Set VITE_NEUROTOKEN_SSH_HOST to connect." });
        return;
      }
      void sshManager.connect({
        host,
        langGraphLocalPort: Number(import.meta.env.VITE_NEUROTOKEN_LANGGRAPH_LOCAL_PORT ?? 2024),
        langGraphRemotePort: Number(import.meta.env.VITE_NEUROTOKEN_LANGGRAPH_REMOTE_PORT ?? 2024),
        codingLocalPort: Number(import.meta.env.VITE_NEUROTOKEN_CODING_LOCAL_PORT ?? 4732),
        codingRemotePort: Number(import.meta.env.VITE_NEUROTOKEN_CODING_REMOTE_PORT ?? 4732),
        remoteStartCommand: import.meta.env.VITE_NEUROTOKEN_REMOTE_START_COMMAND,
      }).catch((error) => {
        setSshState({
          status: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      });
    }).catch((error) => {
      setSshState({ status: "error", message: error instanceof Error ? error.message : String(error) });
    });
    return unsubscribe;
  }, [sshManager]);

  const selectedPhase = useMemo(() => {
    const mode = workspaceMode;
    if (!selectedTask || !mode) return null;
    const interrupt = activeInterrupt;
    const patchApproval = Boolean(
      interrupt?.kind === "adapter_patch_approval" && interrupt.thread_id === selectedTask.threadId,
    );
    const coding = Boolean(
      codingHandoff?.thread_id === selectedTask.threadId &&
      !patchApproval &&
      (mode === "ready_for_coding" || mode === "authoring_adapter"),
    );
    return {
      status: patchApproval ? "awaiting_input" as const : coding ? "coding" as const : taskStatusFromWorkspaceMode(mode),
      subtitle: patchApproval
        ? "Review and apply adapter"
        : coding
          ? "Coding workspace active"
          : taskSubtitleFromWorkspaceMode(mode),
    };
  }, [activeInterrupt, codingHandoff?.thread_id, selectedTask?.threadId, workspaceMode]);

  useEffect(() => {
    if (!selectedTask || !selectedPhase) return;
    if (selectedTask.status === selectedPhase.status && selectedTask.subtitle === selectedPhase.subtitle) return;
    dispatch({
      type: "upsert_task",
      task: {
        ...selectedTask,
        status: selectedPhase.status,
        subtitle: selectedPhase.subtitle,
        updatedAt: new Date().toISOString(),
      },
    });
  }, [selectedPhase, selectedTask]);

  useEffect(() => {
    if (!activeCodingHandoff || !selectedTask || activeCodingHandoff.thread_id !== selectedTask.threadId) return;
    dispatchHandoff({
      type: "register",
      taskId: activeCodingHandoff.task_id,
      threadId: activeCodingHandoff.thread_id,
      revision: activeCodingHandoff.revision,
    });
    if (workspaceMode === "ready_for_coding" || workspaceMode === "authoring_adapter") {
      dispatch({ type: "open_workspace", workspace: "coding" });
    }
  }, [
    activeCodingHandoff?.task_id,
    activeCodingHandoff?.thread_id,
    activeCodingHandoff?.revision,
    selectedTask?.threadId,
    workspaceMode,
  ]);

  useEffect(() => {
    if (!workspaceMode || !TERMINAL_WORKFLOW_MODES.has(workspaceMode)) return;
    const terminalTaskId = archivedCodingHandoff?.task_id ?? persistedCodingContext?.handoff.task_id;
    if (terminalTaskId) dispatchHandoff({ type: "invalidate", taskId: terminalTaskId });
    dispatch({ type: "open_workspace", workspace: "conversation" });
  }, [archivedCodingHandoff?.task_id, persistedCodingContext?.handoff.task_id, workspaceMode]);

  useEffect(() => {
    const interrupt = activeInterrupt;
    if (!interrupt) return;
    if (interrupt.kind === "adapter_patch_approval" && interrupt.thread_id === selectedTask?.threadId) {
      dispatchHandoff({
        type: "register",
        taskId: interrupt.task_id,
        threadId: interrupt.thread_id,
        revision: interrupt.revision,
      });
      dispatch({ type: "open_workspace", workspace: "coding" });
      setPatchReviewOpen(true);
      return;
    }
    dispatch({ type: "open_workspace", workspace: "conversation" });
  }, [activeInterrupt, selectedTask?.threadId]);

  useEffect(() => {
    if (
      pendingPatchRevision &&
      pendingPatchRevision.threadId === selectedThreadId &&
      pendingPatchRevision.taskId === visibleAdapterTaskId &&
      visibleAdapterRevision != null &&
      visibleAdapterRevision > pendingPatchRevision.revision
    ) {
      setPendingPatchRevision(null);
    }
  }, [pendingPatchRevision, selectedThreadId, visibleAdapterRevision, visibleAdapterTaskId]);

  useEffect(() => {
    if (availableWorkspaces.includes(state.centerWorkspace)) return;
    dispatch({ type: "open_workspace", workspace: "conversation" });
  }, [availableWorkspaces.join(","), state.centerWorkspace]);

  useEffect(() => {
    langGraphViewThreadRef.current = selectedThreadId;
    setPatchReviewOpen(false);
    setLangGraphView(selectedThreadId ? threadViewsRef.current.get(selectedThreadId) ?? EMPTY_LANGGRAPH_VIEW : EMPTY_LANGGRAPH_VIEW);
    setArtifactPreview(null);
    setArtifactError(null);
    setPlanRefinement("");
    setEditRetryOpen(false);
    setEditRetryRequest("");
    if (!selectedThreadId) return;
    const subscribedThreadId = selectedThreadId;
    return client.subscribe(subscribedThreadId, (event) => {
      if (subscribedThreadId !== selectedThreadRef.current) return;
      setLangGraphView((current) => {
        const next = reduceLangGraphEvent(current, event);
        threadViewsRef.current.set(subscribedThreadId, next);
        return next;
      });
    });
  }, [client, selectedThreadId]);

  async function createTask() {
    const request = newTaskRequest.trim();
    if (!request || newTaskSubmitting) return;
    setNewTaskSubmitting(true);
    setNewTaskError(null);
    let createdTask: NeuroTokenTask | null = null;
    try {
      const thread = await client.createThread();
      createdTask = {
        ...taskFromThread(thread),
        title: taskTitle(request),
        subtitle: "Starting LangGraph run…",
        status: "running",
        updatedAt: new Date().toISOString(),
      };
      dispatch({ type: "upsert_task", task: createdTask });
      dispatch({ type: "select_task", taskId: createdTask.id });
      setNewTaskRequest("");
      setNewTaskOpen(false);
      setNewTaskSubmitting(false);
      void client.sendMessage(thread.threadId, request).then(() => {
        dispatch({
          type: "upsert_task",
          task: { ...createdTask!, subtitle: "LangGraph run active", updatedAt: new Date().toISOString() },
        });
      }).catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        setNewTaskError(message);
        dispatch({
          type: "upsert_task",
          task: { ...createdTask!, subtitle: message, status: "failed", updatedAt: new Date().toISOString() },
        });
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNewTaskError(message);
      if (createdTask) {
        dispatch({
          type: "upsert_task",
          task: { ...createdTask, subtitle: message, status: "failed", updatedAt: new Date().toISOString() },
        });
      }
    } finally {
      setNewTaskSubmitting(false);
    }
  }

  function requestTaskDeletion(task: NeuroTokenTask) {
    if (!deleteSubmitting) setTaskPendingDeletion(task);
  }

  async function confirmTaskDeletion() {
    const task = taskPendingDeletion;
    if (!task || deleteSubmitting) return;
    setDeleteSubmitting(true);
    try {
      await client.deleteThread(task.threadId);
      dispatch({ type: "remove_task", taskId: task.id });
      threadViewsRef.current.delete(task.threadId);
      setCodingContexts((current) => {
        const updated = { ...current };
        delete updated[task.threadId];
        window.localStorage.setItem(CODING_CONTEXT_STORAGE_KEY, JSON.stringify(updated));
        return updated;
      });
      if (task.id === selectedTask?.id) setLangGraphView(EMPTY_LANGGRAPH_VIEW);
      setTaskPendingDeletion(null);
    } catch (error) {
      setTaskListError(error instanceof Error ? error.message : String(error));
    } finally {
      setDeleteSubmitting(false);
    }
  }

  async function retryTask() {
    if (!selectedTask || retrySubmitting) return;
    setRetrySubmitting(true);
    setLangGraphView((current) => ({ ...current, error: null }));
    dispatch({
      type: "upsert_task",
      task: { ...selectedTask, status: "running", subtitle: "Retrying from failed step…", updatedAt: new Date().toISOString() },
    });
    try {
      await client.retry(selectedTask.threadId);
      dispatch({
        type: "upsert_task",
        task: { ...selectedTask, status: "running", subtitle: "LangGraph run active", updatedAt: new Date().toISOString() },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLangGraphView((current) => ({ ...current, error: message }));
      dispatch({
        type: "upsert_task",
        task: { ...selectedTask, status: "failed", subtitle: message, updatedAt: new Date().toISOString() },
      });
    } finally {
      setRetrySubmitting(false);
    }
  }

  function beginEditRetry() {
    const lastUserMessage = [...visibleLangGraphView.messages].reverse().find((message) => message.role === "user");
    setEditRetryRequest(lastUserMessage?.content ?? selectedTask?.title ?? "");
    setNewTaskError(null);
    setEditRetryOpen(true);
  }

  async function editAndRetry() {
    const request = editRetryRequest.trim();
    if (!selectedTask || !request || newTaskSubmitting) return;
    setNewTaskSubmitting(true);
    setNewTaskError(null);
    setLangGraphView((current) => ({
      ...current,
      error: null,
      messages: [
        ...current.messages,
        { id: `user:retry:${Date.now()}`, role: "user", content: request },
      ],
    }));
    dispatch({
      type: "upsert_task",
      task: {
        ...selectedTask,
        title: taskTitle(request),
        subtitle: "Retrying edited request in this conversation…",
        status: "running",
        updatedAt: new Date().toISOString(),
      },
    });
    try {
      await client.sendMessage(selectedTask.threadId, request);
      setEditRetryOpen(false);
      setEditRetryRequest("");
      dispatch({
        type: "upsert_task",
        task: {
          ...selectedTask,
          title: taskTitle(request),
          subtitle: "LangGraph run active",
          status: "running",
          updatedAt: new Date().toISOString(),
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNewTaskError(message);
      setEditRetryOpen(true);
      setLangGraphView((current) => ({ ...current, error: message }));
      dispatch({
        type: "upsert_task",
        task: { ...selectedTask, subtitle: message, status: "failed", updatedAt: new Date().toISOString() },
      });
    } finally {
      setNewTaskSubmitting(false);
    }
  }

  async function sendFollowUp(content = followUpMessage) {
    const message = content.trim();
    if (!selectedTask || !message || followUpSubmitting) return;
    setFollowUpSubmitting(true);
    setLangGraphView((current) => ({
      ...current,
      error: null,
      messages: [...current.messages, { id: `user:${message}`, role: "user", content: message }],
    }));
    dispatch({
      type: "upsert_task",
      task: { ...selectedTask, status: "running", subtitle: "Sending request…", updatedAt: new Date().toISOString() },
    });
    try {
      await client.sendMessage(selectedTask.threadId, message);
      setFollowUpMessage("");
    } catch (error) {
      setLangGraphView((current) => ({
        ...current,
        error: error instanceof Error ? error.message : String(error),
      }));
    } finally {
      setFollowUpSubmitting(false);
    }
  }

  async function openArtifact(artifactId: string) {
    if (!selectedTask) return;
    setArtifactError(null);
    try {
      setArtifactPreview(await client.getArtifact(selectedTask.threadId, artifactId));
    } catch (error) {
      setArtifactError(error instanceof Error ? error.message : String(error));
    }
  }

  const submitCodingCandidate = useCallback(async (task: typeof codingTask) => {
    if (!task || !task.codexThreadId) return;
    const submissionKey = `${task.taskId}:${task.revision}`;
    if (submittingCandidates.current.has(submissionKey) || submittedCandidates.current.has(submissionKey)) return;
    submittingCandidates.current.add(submissionKey);
    dispatchHandoff({ type: "submit", taskId: task.taskId, revision: task.revision });
    try {
      const diff = await codingRuntime.getDiff(task.taskId);
      if (!diff.trim()) throw new Error("Codex completed without producing an adapter diff.");
      await client.submitCandidate({
        schema_version: "1.0",
        action: "submit_candidate",
        task_id: task.taskId,
        thread_id: task.threadId,
        codex_thread_id: task.codexThreadId,
        replaces_codex_thread_id: codingHandoff?.codex_thread_id ?? null,
        revision: task.revision,
      });
      submittedCandidates.current.add(submissionKey);
      dispatchHandoff({ type: "validated", taskId: task.taskId, revision: task.revision });
    } catch (error) {
      dispatchHandoff({
        type: "validated",
        taskId: task.taskId,
        revision: task.revision,
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      submittingCandidates.current.delete(submissionKey);
    }
  }, [client, codingHandoff?.codex_thread_id, codingRuntime]);

  const handleCodingTurnCompleted = useCallback((session: CodingSession | null) => {
    if (!codingTask || !session || session.taskId !== codingTask.taskId || session.revision !== codingTask.revision) return;
    const completedTask = codingTask.codexThreadId
      ? codingTask
      : { ...codingTask, codexThreadId: session.codexThreadId };
    if (completedTask.status === "coding" || completedTask.status === "needs_revision") {
      void submitCodingCandidate(completedTask);
    }
  }, [codingTask, submitCodingCandidate]);

  async function submitWorkflowDecision(decision: LangGraphDecision) {
    if (!selectedTask) return;
    const gateId = "gate_id" in decision ? decision.gate_id : undefined;
    const taskId = "task_id" in decision ? decision.task_id : undefined;
    const revision = "revision" in decision ? decision.revision : undefined;
    const identity = [selectedTask.threadId, gateId ?? taskId ?? "workflow", revision ?? "", decision.action].join(":");
    const existing = submittingDecisions.current.get(identity);
    if (existing) return existing;
    const submission = performWorkflowDecision(decision, selectedTask.threadId, identity);
    submittingDecisions.current.set(identity, submission);
    return submission;
  }

  async function performWorkflowDecision(decision: LangGraphDecision, threadId: string, identity: string) {
    const task = selectedTask;
    const patchRevision = task && decision.action === "refine" && activeInterrupt?.kind === "adapter_patch_approval"
      ? {
        threadId: task.threadId,
        taskId: activeInterrupt.task_id,
        revision: activeInterrupt.revision,
      }
      : null;
    if (patchRevision) {
      setPendingPatchRevision(patchRevision);
      setPatchReviewOpen(false);
      dispatch({ type: "open_workspace", workspace: "conversation" });
    }
    try {
      await client.resume(threadId, decision);
    } catch (error) {
      if (patchRevision) {
        setPendingPatchRevision(null);
        setPatchReviewOpen(true);
        dispatch({ type: "open_workspace", workspace: "coding" });
      }
      throw error;
    } finally {
      submittingDecisions.current.delete(identity);
    }
    if (task && decision.action === "reject" && activeInterrupt?.kind === "adapter_patch_approval") {
      dispatchHandoff({ type: "reject", taskId: activeInterrupt.task_id });
      submittedCandidates.current.add(`${activeInterrupt.task_id}:${activeInterrupt.revision}`);
      dispatch({
        type: "upsert_task",
        task: {
          ...task,
          status: "completed",
          subtitle: "Adapter candidate rejected — send a new request to continue",
          updatedAt: new Date().toISOString(),
        },
      });
    }
    if (decision.action === "ask") setPatchQuestion("");
    if (decision.action === "refine") {
      setPlanRefinement("");
      setPatchRefinement("");
    }
  }

  async function copySshDiagnostics() {
    if (!sshManager?.getDiagnostics) return;
    setSshDiagnosticsFeedback("Collecting SSH diagnostics…");
    try {
      const diagnostics = await sshManager.getDiagnostics();
      const report = [
        "# NeuroToken SSH diagnostics",
        `log_path: ${diagnostics.logPath}`,
        `desktop_pid: ${diagnostics.desktopPid}`,
        `watchdog_pid: ${diagnostics.watchdogPid ?? "none"}`,
        `generation: ${diagnostics.generation}`,
        `state: ${JSON.stringify(diagnostics.state)}`,
        `settings: ${JSON.stringify(diagnostics.settings)}`,
        "",
        "## Recent lifecycle log",
        diagnostics.recentLog,
      ].join("\n");
      await navigator.clipboard.writeText(report);
      setSshDiagnosticsFeedback(`SSH diagnostics copied. Log: ${diagnostics.logPath}`);
    } catch (error) {
      setSshDiagnosticsFeedback(`Unable to copy SSH diagnostics: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  function startPanelResize(
    event: ReactPointerEvent<HTMLDivElement>,
  ) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = leftPanelWidth;
    const divider = event.currentTarget;
    divider.setPointerCapture(event.pointerId);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      const delta = moveEvent.clientX - startX;
      const nextWidth = startWidth + delta;
      const clampedWidth = Math.max(200, Math.min(nextWidth, 520));
      setLeftPanelWidth(clampedWidth);
    };
    const handlePointerUp = (upEvent: PointerEvent) => {
      divider.releasePointerCapture(upEvent.pointerId);
      divider.removeEventListener("pointermove", handlePointerMove);
      divider.removeEventListener("pointerup", handlePointerUp);
      divider.removeEventListener("pointercancel", handlePointerUp);
    };

    divider.addEventListener("pointermove", handlePointerMove);
    divider.addEventListener("pointerup", handlePointerUp);
    divider.addEventListener("pointercancel", handlePointerUp);
  }

  const shellColumns = `${leftPanelCollapsed ? 44 : leftPanelWidth}px 3px minmax(420px, 1fr)`;
  const patchDecisionCard = patchApprovalInterrupt ? (
    <WorkflowDecisionCard
      key={decisionCardKey}
      interrupt={patchApprovalInterrupt}
      stages={activeWorkspace?.stages ?? []}
      applyMode={(patchTask ?? codingTask)?.applyMode ?? "apply"}
      commitMessage={(patchTask ?? codingTask)?.commitMessage ?? ""}
      refine={patchRefinement}
      question={patchQuestion}
      onApplyModeChange={(mode) => {
        const task = patchTask ?? codingTask;
        if (task) dispatchHandoff({ type: "set_apply", taskId: task.taskId, mode });
      }}
      onCommitMessageChange={(commitMessage) => {
        const task = patchTask ?? codingTask;
        if (task) dispatchHandoff({ type: "set_apply", taskId: task.taskId, mode: "apply_and_commit", commitMessage });
      }}
      onRefineChange={setPatchRefinement}
      onQuestionChange={setPatchQuestion}
      onDecision={submitWorkflowDecision}
    />
  ) : null;

  return (
    <div
      className={`neurotoken-shell theme-${theme}`}
      data-testid="neurotoken-shell"
      data-theme={theme}
      style={{ gridTemplateColumns: shellColumns }}
    >
      <aside className={`neurotoken-sidebar${leftPanelCollapsed ? " is-collapsed" : ""}`}>
        <header className="neurotoken-brand" data-tauri-drag-region>
          <div className="neurotoken-brand-mark"><Bot /></div>
          {!leftPanelCollapsed && <div className="neurotoken-brand-copy">
            <strong>NeuroToken</strong>
            <span>Remote workspace</span>
          </div>}
          <button
            type="button"
            className="neurotoken-panel-toggle"
            aria-label={leftPanelCollapsed ? "Expand task panel" : "Collapse task panel"}
            onClick={() => setLeftPanelCollapsed((collapsed) => !collapsed)}
          >
            {leftPanelCollapsed ? <ChevronRight /> : <ChevronLeft />}
          </button>
        </header>
        <div className="neurotoken-connection">
          <Server />
          <div>
            <strong>Linux runtime</strong>
            <span title={sshState.status === "error" ? sshState.message : undefined}>
              {import.meta.env.VITE_NEUROTOKEN_CODING_RUNTIME !== "remote"
                ? "Mock connection"
                : sshState.status === "connected"
                  ? "Remote connected"
                  : sshState.status === "connecting"
                    ? "Connecting…"
                    : sshState.status === "error"
                      ? "Connection error"
                      : "Disconnected"}
            </span>
          </div>
          <i aria-label={sshState.status} className={`status-${sshState.status}`} />
        </div>
        <div className="neurotoken-section-label neurotoken-task-heading">
          <span>Tasks</span>
          <div className="neurotoken-task-heading-actions">
            <button
              type="button"
              aria-label="Delete selected conversation"
              title={selectedTask ? `Delete “${selectedTask.title}”` : "Select a conversation to delete"}
              disabled={!selectedTask || deleteSubmitting}
              onClick={() => selectedTask && requestTaskDeletion(selectedTask)}
            >
              <Trash2 />
            </button>
            <button
              type="button"
              aria-label="New task"
              title="Create a new task"
              onClick={() => { setNewTaskOpen((open) => !open); setNewTaskError(null); }}
            >
              <Plus />
            </button>
          </div>
        </div>
        {newTaskOpen && (
          <form className="neurotoken-new-task" onSubmit={(event) => { event.preventDefault(); void createTask(); }}>
            <label htmlFor="new-task-request">What should NeuroToken do?</label>
            <textarea
              id="new-task-request"
              aria-label="New task request"
              placeholder="例如：分析这个 H5 文件并生成 EEG NeuroTokens"
              value={newTaskRequest}
              disabled={newTaskSubmitting}
              onChange={(event) => setNewTaskRequest(event.target.value)}
            />
            {newTaskError && <p role="alert">{newTaskError}</p>}
            <div>
              <button type="button" onClick={() => { setNewTaskOpen(false); setNewTaskError(null); }}>Cancel</button>
              <button type="submit" disabled={!newTaskRequest.trim() || newTaskSubmitting}>
                {newTaskSubmitting ? "Creating…" : "Create task"}
              </button>
            </div>
          </form>
        )}
        <nav className="neurotoken-task-list" aria-label="NeuroToken tasks">
          {tasksLoading && <span className="neurotoken-task-empty">Loading remote tasks…</span>}
          {!tasksLoading && taskListError && <span className="neurotoken-task-empty is-error">{taskListError}</span>}
          {!tasksLoading && !taskListError && state.tasks.length === 0 && (
            <span className="neurotoken-task-empty">No tasks yet. Create one to start.</span>
          )}
          <ThreadList
            workspaceId="neurotoken"
            pinnedRows={[]}
            unpinnedRows={threadRows}
            totalThreadRoots={threadRows.length}
            isExpanded
            showExpandToggle={false}
            nextCursor={null}
            isPaging={false}
            showLoadOlder={false}
            activeWorkspaceId="neurotoken"
            activeThreadId={state.selectedTaskId}
            threadStatusById={threadStatusById}
            pendingUserInputKeys={new Set(state.tasks.filter((task) => task.status === "awaiting_input").map((task) => `neurotoken:${task.id}`))}
            getThreadTime={(thread) => new Date(thread.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            isThreadPinned={() => false}
            onToggleExpanded={() => undefined}
            onLoadOlderThreads={() => undefined}
            onSelectThread={(_workspaceId, taskId) => dispatch({ type: "select_task", taskId })}
            onShowThreadMenu={(event, _workspaceId, taskId) => {
              event.preventDefault();
              const task = state.tasks.find((item) => item.id === taskId);
              if (task) requestTaskDeletion(task);
            }}
          />
        </nav>
        <footer className="neurotoken-sidebar-footer">
          <span>{activeTaskCount} active {activeTaskCount === 1 ? "task" : "tasks"}</span>
          <div className="neurotoken-ssh-diagnostics">
            <span>{sshState.status === "connected" ? "SSH ready" : `SSH ${sshState.status}`}</span>
            {sshManager?.getDiagnostics && (
              <button
                type="button"
                aria-label="Copy SSH diagnostics"
                title="Copy SSH lifecycle diagnostics for troubleshooting"
                onClick={() => void copySshDiagnostics()}
              >
                <ScrollText />
              </button>
            )}
          </div>
          {sshDiagnosticsFeedback && <span className="neurotoken-ssh-diagnostics-feedback" role="status">{sshDiagnosticsFeedback}</span>}
        </footer>
      </aside>

      <div
        className={`neurotoken-shell-resizer${leftPanelCollapsed ? " is-disabled" : ""}`}
        role="separator"
        aria-label="Resize task panel"
        aria-orientation="vertical"
        aria-valuemin={200}
        aria-valuemax={520}
        aria-valuenow={leftPanelCollapsed ? 44 : leftPanelWidth}
        tabIndex={leftPanelCollapsed ? -1 : 0}
        onPointerDown={(event) => {
          if (!leftPanelCollapsed) startPanelResize(event);
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setLeftPanelWidth((width) => Math.max(200, width - 24));
          if (event.key === "ArrowRight") setLeftPanelWidth((width) => Math.min(520, width + 24));
        }}
      />

      <main className="neurotoken-main">
        <header className="neurotoken-topbar" data-tauri-drag-region>
          <div className="neurotoken-breadcrumb">
            <span>Tasks</span>
            <ChevronRight />
            <strong>{selectedTask?.title ?? "No task selected"}</strong>
            <ChevronRight />
            <span>{workspaceLabels[state.centerWorkspace]}</span>
          </div>
          <div className="neurotoken-topbar-actions">
            {availableWorkspaces.map(
              (workspace) => (
                <button
                  type="button"
                  key={workspace}
                  className={state.centerWorkspace === workspace ? "is-active" : undefined}
                  onClick={() => dispatch({ type: "open_workspace", workspace })}
                >
                  {workspaceLabels[workspace]}
                </button>
              ),
            )}
            <span className="neurotoken-action-divider" />
            <button
              type="button"
              className="neurotoken-theme-toggle"
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              title={theme === "dark" ? "Use light theme" : "Use dark theme"}
              onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Sun /> : <Moon />}
              <span>{theme === "dark" ? "Light" : "Dark"}</span>
            </button>
          </div>
        </header>

        <section className={`neurotoken-workspace${state.centerWorkspace === "coding" ? " is-coding" : ""}`}>
          {state.centerWorkspace !== "conversation" ? (
            codingHandoff && codingTask ? (
              <>
                <div className={`neurotoken-coding-stage${patchDecisionCard ? " has-patch-review" : ""}${patchDecisionCard && !patchReviewOpen ? " is-patch-review-collapsed" : ""}`}>
                <CodingWorkspace
                  runtime={codingRuntime}
                  handoff={codingHandoff}
                  onSessionOpened={handleCodingSessionOpened}
                  onTurnCompleted={handleCodingTurnCompleted}
                  onSessionInvalidated={handleCodingSessionInvalidated}
                  theme={theme}
                  view={state.centerWorkspace === "editor" ? "editor" : "conversation"}
                  readOnly={workspaceMode === "awaiting_patch_approval"}
                  archived={codingWorkspaceArchived}
                />
                {patchDecisionCard && (
                  <aside className={`neurotoken-stage-output neurotoken-patch-review${patchReviewOpen ? "" : " is-collapsed"}`} aria-label="Patch review panel">
                    <header>
                      <button type="button" aria-label={patchReviewOpen ? "Collapse patch review" : "Expand patch review"} onClick={() => setPatchReviewOpen((open) => !open)}>
                        {patchReviewOpen ? <ChevronDown /> : <ChevronRight />}
                      </button>
                      {patchReviewOpen && <><strong>Review adapter</strong><span>Awaiting approval</span></>}
                    </header>
                    {patchReviewOpen && <div className="neurotoken-stage-output__body">{patchDecisionCard}</div>}
                  </aside>
                )}
                </div>
                <div className="neurotoken-submit-bar">
                  <span>Candidate revision {codingTask.revision}</span>
                  <strong>{codingTask.status.replace(/_/g, " ")}</strong>
                  <nav aria-label="Coding result views">
                    <button type="button" disabled={workspaceMode === "awaiting_patch_approval" || codingWorkspaceArchived || isPreparingPatchRevision} title={workspaceMode === "awaiting_patch_approval" ? "Use Refine in Review patch to continue coding" : isPreparingPatchRevision ? "Preparing the next Codex revision" : codingWorkspaceArchived ? "This workflow's coding session is archived" : undefined} onClick={() => dispatch({ type: "open_workspace", workspace: "coding" })}>{isPreparingPatchRevision ? "Preparing revision…" : workspaceMode === "awaiting_patch_approval" ? "Codex complete" : codingWorkspaceArchived ? "Codex archived" : "Continue with Codex"}</button>
                    <button type="button" onClick={() => dispatch({ type: "open_workspace", workspace: "editor" })}>View code</button>
                    {patchApprovalInterrupt && (
                      <button type="button" aria-expanded={patchReviewOpen} onClick={() => {
                        if (state.centerWorkspace !== "coding") dispatch({ type: "open_workspace", workspace: "coding" });
                        setPatchReviewOpen(true);
                      }}>Review adapter</button>
                    )}
                  </nav>
                  {codingTask.validationError && <span role="alert">{codingTask.validationError}</span>}
                  <button type="button" disabled={codingTask.status === "validating" || codingTask.status === "awaiting_approval"} onClick={() => void submitCodingCandidate(codingTask)}>Submit candidate</button>
                </div>
              </>
            ) : (
              <div className="neurotoken-placeholder-card">
                <span>CODING WORKSPACE</span>
                <strong>No coding handoff is available yet</strong>
                <p>The LangGraph task will open this workspace when adapter authoring is required.</p>
              </div>
            )
          ) : (
            <div className={`neurotoken-conversation-layout${stageOutputCollapsed ? " is-stage-output-collapsed" : ""}`}>
              <NeuroTokenConversation
                items={conversationItems}
                threadId={selectedTask?.threadId ?? null}
                isLoading={Boolean(selectedTask && !visibleLangGraphView.workspace && conversationItems.length === 0 && !visibleLangGraphView.error)}
                isProcessing={Boolean(selectedTask?.status === "running" || followUpSubmitting)}
                draft={followUpMessage}
                onDraftChange={setFollowUpMessage}
                onSend={(text) => {
                  setFollowUpMessage(text);
                  void sendFollowUp(text);
                }}
              >
          {activeInterrupt && activeInterrupt.kind !== "adapter_patch_approval" && (
            <WorkflowDecisionCard
              key={decisionCardKey}
              interrupt={activeInterrupt}
              stages={activeWorkspace?.stages ?? []}
              applyMode={(patchTask ?? codingTask)?.applyMode ?? "apply"}
              commitMessage={(patchTask ?? codingTask)?.commitMessage ?? ""}
              refine={activeInterrupt.kind === "plan_approval" ? planRefinement : patchRefinement}
              question={patchQuestion}
              onApplyModeChange={(mode) => {
                const task = patchTask ?? codingTask;
                if (task) dispatchHandoff({ type: "set_apply", taskId: task.taskId, mode });
              }}
              onCommitMessageChange={(commitMessage) => {
                const task = patchTask ?? codingTask;
                if (task) dispatchHandoff({ type: "set_apply", taskId: task.taskId, mode: "apply_and_commit", commitMessage });
              }}
              onRefineChange={(value) => {
                if (activeInterrupt?.kind === "plan_approval") setPlanRefinement(value);
                else setPatchRefinement(value);
              }}
              onQuestionChange={setPatchQuestion}
              onDecision={submitWorkflowDecision}
            />
          )}
              </NeuroTokenConversation>
              <StageOutputPanel
                workspace={activeWorkspace}
                error={visibleLangGraphView.error}
                artifactPreview={artifactPreview}
                artifactError={artifactError}
                retrySubmitting={retrySubmitting}
                editRetryOpen={editRetryOpen}
                editRetryRequest={editRetryRequest}
                editRetrySubmitting={newTaskSubmitting}
                editRetryError={newTaskError}
                onOpenArtifact={(artifactId) => void openArtifact(artifactId)}
                onDownloadArtifact={downloadArtifact}
                onRetry={() => void retryTask()}
                onBeginEditRetry={beginEditRetry}
                onCancelEditRetry={() => setEditRetryOpen(false)}
                onEditRetryRequestChange={setEditRetryRequest}
                onEditRetry={() => void editAndRetry()}
                collapsed={stageOutputCollapsed}
                onCollapsedChange={setStageOutputCollapsed}
              />
            </div>
          )}
        </section>
      </main>

      {taskPendingDeletion && (
        <ModalShell
          className="neurotoken-delete-modal"
          ariaLabel="Confirm conversation deletion"
          onBackdropClick={() => !deleteSubmitting && setTaskPendingDeletion(null)}
        >
          <div className="ds-modal-title">Delete conversation?</div>
          <div className="ds-modal-subtitle">
            This permanently deletes the selected LangGraph conversation and its saved history.
          </div>
          <div className="neurotoken-delete-target">
            <span>Selected conversation</span>
            <strong>{taskPendingDeletion.title}</strong>
            <small>{taskPendingDeletion.threadId}</small>
          </div>
          <div className="ds-modal-actions">
            <button type="button" className="ghost ds-modal-button" disabled={deleteSubmitting} onClick={() => setTaskPendingDeletion(null)}>
              Keep conversation
            </button>
            <button type="button" className="danger ds-modal-button" disabled={deleteSubmitting} onClick={() => void confirmTaskDeletion()}>
              {deleteSubmitting ? "Deleting…" : "Delete conversation"}
            </button>
          </div>
        </ModalShell>
      )}
    </div>
  );
}
