import { describe, expect, it } from "vitest";
import { INITIAL_HANDOFF_STATE, handoffReducer } from "./handoffState";

describe("handoffReducer", () => {
  it("moves an identity-only candidate through validation and approval", () => {
    const submitted = handoffReducer(INITIAL_HANDOFF_STATE, { type: "submit", taskId: "task-adapter-001", revision: 2 });
    const validated = handoffReducer(submitted, { type: "validated", taskId: "task-adapter-001", revision: 2 });
    expect(submitted.tasks["task-adapter-001"].status).toBe("validating");
    expect(validated.tasks["task-adapter-001"].status).toBe("awaiting_approval");
  });

  it("returns validation failures to the same task and increments on refine", () => {
    const failed = handoffReducer(INITIAL_HANDOFF_STATE, { type: "validated", taskId: "task-adapter-001", revision: 2, error: "Adapter import failed" });
    const refined = handoffReducer(failed, { type: "refine", taskId: "task-adapter-001", message: "Fix the import" });
    expect(failed.tasks["task-adapter-001"].status).toBe("needs_revision");
    expect(refined.tasks["task-adapter-001"]).toMatchObject({ revision: 3, status: "coding" });
  });

  it("ignores stale candidate revisions", () => {
    const state = handoffReducer(INITIAL_HANDOFF_STATE, { type: "submit", taskId: "task-adapter-001", revision: 1 });
    expect(state).toBe(INITIAL_HANDOFF_STATE);
  });

  it("does not reset a registered task after it starts validating", () => {
    const submitted = handoffReducer(INITIAL_HANDOFF_STATE, {
      type: "submit",
      taskId: "task-adapter-001",
      revision: 2,
    });
    const replayed = handoffReducer(submitted, {
      type: "register",
      taskId: "task-adapter-001",
      threadId: "thread-adapter-001",
      revision: 2,
    });

    expect(replayed).toBe(submitted);
    expect(replayed.tasks["task-adapter-001"].status).toBe("validating");
  });

  it("removes an invalidated obsolete coding task", () => {
    const next = handoffReducer(INITIAL_HANDOFF_STATE, { type: "invalidate", taskId: "task-adapter-001" });
    expect(next.tasks["task-adapter-001"]).toBeUndefined();
  });

  it("supports apply only and apply-and-commit with editable message", () => {
    const commit = handoffReducer(INITIAL_HANDOFF_STATE, { type: "set_apply", taskId: "task-adapter-001", mode: "apply_and_commit", commitMessage: "Implement EEG adapter" });
    expect(commit.tasks["task-adapter-001"]).toMatchObject({ applyMode: "apply_and_commit", commitMessage: "Implement EEG adapter" });
  });

  it("keeps concurrent tasks isolated", () => {
    const state = { tasks: { ...INITIAL_HANDOFF_STATE.tasks, other: { ...INITIAL_HANDOFF_STATE.tasks["task-adapter-001"], taskId: "other", threadId: "other-thread" } } };
    const next = handoffReducer(state, { type: "reject", taskId: "other" });
    expect(next.tasks.other.status).toBe("rejected");
    expect(next.tasks["task-adapter-001"].status).toBe("coding");
  });
});
