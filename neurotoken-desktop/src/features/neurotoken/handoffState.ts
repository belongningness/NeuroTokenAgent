export type ApplyMode = "apply" | "apply_and_commit";
export type HandoffStatus = "coding" | "validating" | "awaiting_approval" | "needs_revision" | "applied" | "rejected" | "failed";

export type HandoffTask = {
  taskId: string;
  threadId: string;
  codexThreadId: string;
  revision: number;
  status: HandoffStatus;
  validationError: string | null;
  commitMessage: string;
  applyMode: ApplyMode;
};

export type HandoffState = { tasks: Record<string, HandoffTask> };
export type HandoffAction =
  | { type: "register"; taskId: string; threadId: string; revision: number }
  | { type: "session_opened"; taskId: string; revision: number; codexThreadId: string }
  | { type: "submit"; taskId: string; revision: number }
  | { type: "validated"; taskId: string; revision: number; error?: string }
  | { type: "refine"; taskId: string; message: string }
  | { type: "reject"; taskId: string }
  | { type: "invalidate"; taskId: string }
  | { type: "set_apply"; taskId: string; mode: ApplyMode; commitMessage?: string }
  | { type: "applied"; taskId: string; revision: number };

export const INITIAL_HANDOFF_STATE: HandoffState = {
  tasks: {
    "task-adapter-001": {
      taskId: "task-adapter-001",
      threadId: "thread-adapter-001",
      codexThreadId: "codex-thread-001",
      revision: 2,
      status: "coding",
      validationError: null,
      commitMessage: "Add example EEG adapter",
      applyMode: "apply",
    },
  },
};

export function handoffReducer(state: HandoffState, action: HandoffAction): HandoffState {
  if (action.type === "invalidate") {
    if (!state.tasks[action.taskId]) return state;
    const tasks = { ...state.tasks };
    delete tasks[action.taskId];
    return { ...state, tasks };
  }
  if (action.type === "register") {
    const existing = state.tasks[action.taskId];
    if (existing?.revision === action.revision && existing.threadId === action.threadId) {
      return state;
    }
    const task: HandoffTask = {
      taskId: action.taskId,
      threadId: action.threadId,
      codexThreadId: existing?.revision === action.revision ? existing.codexThreadId : "",
      revision: action.revision,
      status: "coding",
      validationError: null,
      commitMessage: existing?.commitMessage ?? "Add generated adapter",
      applyMode: existing?.applyMode ?? "apply",
    };
    return { ...state, tasks: { ...state.tasks, [action.taskId]: task } };
  }
  const task = state.tasks[action.taskId];
  if (!task) return state;
  if ("revision" in action && action.revision !== task.revision) return state;
  let next = task;
  switch (action.type) {
    case "session_opened": next = { ...task, codexThreadId: action.codexThreadId }; break;
    case "submit": next = { ...task, status: "validating", validationError: null }; break;
    case "validated": next = action.error ? { ...task, status: "needs_revision", validationError: action.error } : { ...task, status: "awaiting_approval", validationError: null }; break;
    case "refine": next = { ...task, revision: task.revision + 1, status: "coding", validationError: null }; break;
    case "reject": next = { ...task, status: "rejected" }; break;
    case "set_apply": next = { ...task, applyMode: action.mode, commitMessage: action.commitMessage ?? task.commitMessage }; break;
    case "applied": next = { ...task, status: "applied" }; break;
  }
  return { ...state, tasks: { ...state.tasks, [task.taskId]: next } };
}
