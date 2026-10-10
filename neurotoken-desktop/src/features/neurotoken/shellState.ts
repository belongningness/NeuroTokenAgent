export type NeuroTokenTaskStatus =
  | "idle"
  | "running"
  | "awaiting_input"
  | "coding"
  | "completed"
  | "failed";

export type NeuroTokenTask = {
  id: string;
  threadId: string;
  title: string;
  subtitle: string;
  status: NeuroTokenTaskStatus;
  updatedAt: string;
};

export type CenterWorkspace = "conversation" | "coding" | "editor";

export type NeuroTokenShellState = {
  tasks: NeuroTokenTask[];
  selectedTaskId: string | null;
  centerWorkspace: CenterWorkspace;
};

export type NeuroTokenShellAction =
  | { type: "select_task"; taskId: string }
  | { type: "open_workspace"; workspace: CenterWorkspace }
  | { type: "rehydrate"; tasks: NeuroTokenTask[]; selectedTaskId?: string | null }
  | { type: "upsert_task"; task: NeuroTokenTask }
  | { type: "remove_task"; taskId: string };

export const MOCK_TASKS: NeuroTokenTask[] = [
  {
    id: "task-pipeline-001",
    threadId: "thread-pipeline-001",
    title: "Prepare EEG NeuroTokens",
    subtitle: "Plan ready for review",
    status: "awaiting_input",
    updatedAt: "2026-08-04T07:42:00Z",
  },
  {
    id: "task-adapter-001",
    threadId: "thread-adapter-001",
    title: "Author example EEG adapter",
    subtitle: "Coding workspace active",
    status: "coding",
    updatedAt: "2026-08-04T07:38:00Z",
  },
  {
    id: "task-run-001",
    threadId: "thread-run-001",
    title: "Package subject 0001",
    subtitle: "Tokenize stage running",
    status: "running",
    updatedAt: "2026-08-04T07:34:00Z",
  },
];

export const INITIAL_NEUROTOKEN_SHELL_STATE: NeuroTokenShellState = {
  tasks: MOCK_TASKS,
  selectedTaskId: MOCK_TASKS[0]?.id ?? null,
  centerWorkspace: "conversation",
};

export const EMPTY_NEUROTOKEN_SHELL_STATE: NeuroTokenShellState = {
  tasks: [],
  selectedTaskId: null,
  centerWorkspace: "conversation",
};

function taskDefaultWorkspace(task: NeuroTokenTask): CenterWorkspace {
  return task.status === "coding" ? "coding" : "conversation";
}

function sortTasks(tasks: NeuroTokenTask[]): NeuroTokenTask[] {
  return [...tasks].sort((left, right) =>
    right.updatedAt.localeCompare(left.updatedAt),
  );
}

export function neuroTokenShellReducer(
  state: NeuroTokenShellState,
  action: NeuroTokenShellAction,
): NeuroTokenShellState {
  switch (action.type) {
    case "select_task": {
      const task = state.tasks.find((item) => item.id === action.taskId);
      if (!task) return state;
      return {
        ...state,
        selectedTaskId: task.id,
        centerWorkspace: taskDefaultWorkspace(task),
      };
    }
    case "open_workspace":
      return { ...state, centerWorkspace: action.workspace };
    case "rehydrate": {
      const tasks = sortTasks(action.tasks);
      const requestedId = action.selectedTaskId ?? state.selectedTaskId;
      const selectedTaskId = tasks.some((task) => task.id === requestedId)
        ? requestedId
        : (tasks[0]?.id ?? null);
      const selectedTask = tasks.find((task) => task.id === selectedTaskId);
      return {
        ...state,
        tasks,
        selectedTaskId,
        centerWorkspace: selectedTask
          ? taskDefaultWorkspace(selectedTask)
          : "conversation",
      };
    }
    case "upsert_task": {
      const tasks = sortTasks([
        ...state.tasks.filter((task) => task.id !== action.task.id),
        action.task,
      ]);
      return { ...state, tasks };
    }
    case "remove_task": {
      const tasks = state.tasks.filter((task) => task.id !== action.taskId);
      const selectedTaskId = state.selectedTaskId === action.taskId
        ? (tasks[0]?.id ?? null)
        : state.selectedTaskId;
      const selectedTask = tasks.find((task) => task.id === selectedTaskId);
      return {
        ...state,
        tasks,
        selectedTaskId,
        centerWorkspace: selectedTask ? taskDefaultWorkspace(selectedTask) : "conversation",
      };
    }
  }
}
