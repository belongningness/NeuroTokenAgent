import { describe, expect, it } from "vitest";
import {
  INITIAL_NEUROTOKEN_SHELL_STATE,
  neuroTokenShellReducer,
  type NeuroTokenTask,
} from "./shellState";

const codingTask: NeuroTokenTask = {
  id: "coding-task",
  threadId: "coding-thread",
  title: "Author adapter",
  subtitle: "Coding",
  status: "coding",
  updatedAt: "2026-08-04T08:00:00Z",
};

const runningTask: NeuroTokenTask = {
  id: "running-task",
  threadId: "running-thread",
  title: "Run pipeline",
  subtitle: "Running",
  status: "running",
  updatedAt: "2026-08-04T07:00:00Z",
};

describe("neuroTokenShellReducer", () => {
  it("opens a selected coding task in its live coding workspace", () => {
    const state = neuroTokenShellReducer(INITIAL_NEUROTOKEN_SHELL_STATE, {
      type: "rehydrate",
      tasks: [runningTask, codingTask],
      selectedTaskId: runningTask.id,
    });
    const selected = neuroTokenShellReducer(
      neuroTokenShellReducer(state, { type: "open_workspace", workspace: "editor" }),
      { type: "select_task", taskId: codingTask.id },
    );

    expect(selected.selectedTaskId).toBe(codingTask.id);
    expect(selected.centerWorkspace).toBe("coding");
  });

  it("ignores unknown task selections", () => {
    const state = neuroTokenShellReducer(INITIAL_NEUROTOKEN_SHELL_STATE, {
      type: "select_task",
      taskId: "missing",
    });
    expect(state).toBe(INITIAL_NEUROTOKEN_SHELL_STATE);
  });

  it("rehydrates server tasks by recency into their default workspace", () => {
    const state = neuroTokenShellReducer(INITIAL_NEUROTOKEN_SHELL_STATE, {
      type: "rehydrate",
      tasks: [runningTask, codingTask],
      selectedTaskId: "stale-task",
    });

    expect(state.tasks.map((task) => task.id)).toEqual([codingTask.id, runningTask.id]);
    expect(state.selectedTaskId).toBe(codingTask.id);
    expect(state.centerWorkspace).toBe("coding");
  });

  it("supports explicit coding subviews without a context panel", () => {
    const withEditor = neuroTokenShellReducer(INITIAL_NEUROTOKEN_SHELL_STATE, {
      type: "open_workspace",
      workspace: "editor",
    });
    expect(withEditor.centerWorkspace).toBe("editor");
    expect("contextPanel" in withEditor).toBe(false);
  });

  it("upserts tasks without duplicating them", () => {
    const state = neuroTokenShellReducer(
      { ...INITIAL_NEUROTOKEN_SHELL_STATE, tasks: [runningTask], selectedTaskId: runningTask.id },
      { type: "upsert_task", task: { ...runningTask, status: "completed", subtitle: "Done" } },
    );
    expect(state.tasks).toHaveLength(1);
    expect(state.tasks[0]).toMatchObject({ status: "completed", subtitle: "Done" });
  });
});
