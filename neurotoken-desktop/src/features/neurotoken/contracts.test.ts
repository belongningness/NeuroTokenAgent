import { describe, expect, expectTypeOf, it } from "vitest";
import handoffFixture from "./fixtures/adapter-coding-handoff.json";
import workspaceFixture from "./fixtures/workspace-snapshot.json";
import type {
  AdapterCandidateSubmission,
  AdapterCodingHandoff,
  AdapterPatchDecision,
  NeuroTokenInterrupt,
  WorkspaceSnapshot,
} from "./contracts";
import type {
  CodingRuntime,
  LangGraphClient,
  SshManager,
} from "./runtime";

function asHandoff(value: unknown): AdapterCodingHandoff {
  const handoff = value as AdapterCodingHandoff;
  if (handoff.schema_version !== "1.0") {
    throw new Error(`Unsupported protocol version: ${String(handoff.schema_version)}`);
  }
  return handoff;
}

function asWorkspace(value: unknown): WorkspaceSnapshot {
  const workspace = value as WorkspaceSnapshot;
  if (workspace.schema_version !== "1.0") {
    throw new Error(`Unsupported protocol version: ${String(workspace.schema_version)}`);
  }
  return workspace;
}

describe("NeuroToken contracts", () => {
  it("loads the adapter handoff fixture without server paths", () => {
    const handoff = asHandoff(handoffFixture);
    const serialized = JSON.stringify(handoff);

    expect(handoff.task_id).toBe("adapter-task-001");
    expect(handoff.allowed_paths).toEqual(["agent_pipeline/adapters/"]);
    expect(serialized).not.toContain("worktree");
    expect(serialized).not.toContain("source_path");
  });

  it("loads the workspace snapshot fixture", () => {
    const workspace = asWorkspace(workspaceFixture);

    expect(workspace.mode).toBe("awaiting_plan_approval");
    expect(workspace.stages.map((stage) => stage.name)).toEqual([
      "preflight",
      "tokenize",
    ]);
  });

  it("rejects unsupported protocol versions at the boundary", () => {
    expect(() =>
      asHandoff({ ...handoffFixture, schema_version: "2.0" }),
    ).toThrow("Unsupported protocol version: 2.0");
  });

  it("keeps legacy snapshots valid while typing validation and full-run fields", () => {
    const legacy = asWorkspace(workspaceFixture);
    const phased: WorkspaceSnapshot = {
      ...legacy,
      active_phase: "validation",
      validation_target_h5: "/input/session-01.h5",
      full_input_root: "/input",
      linked_runs: [{ run_id: "validation-1", scope: "validation" }],
      validation_gate: {
        gate_id: "gate-1",
        status: "awaiting_decision",
        verdict: "pass",
        report: { metrics: { valid_chunks: 12 } },
      },
      plan: {
        stages: ["inspect", "tokenize"],
        active_phase: "validation",
        validation_target_h5: "/input/session-01.h5",
      },
      stages: [
        { name: "inspect", status: "completed", stage_id: "validation:inspect", scope: "validation", run_id: "validation-1" },
        { name: "inspect", status: "pending", stage_id: "full:inspect", scope: "full", run_id: "full-1" },
      ],
    };
    const interrupt: NeuroTokenInterrupt = {
      schema_version: "1.0",
      kind: "smoke_validation_review",
      owner: "local-user",
      thread_id: "thread-1",
      gate_id: "gate-1",
      validation_target_h5: phased.validation_target_h5,
      report: { metrics: { valid_chunks: 12 } },
      verdict: "pass",
      allowed_actions: ["start_full_run", "rerun_smoke", "pause", "cancel"],
    };

    expect(legacy.active_phase).toBeUndefined();
    expect(phased.stages.map((stage) => stage.scope)).toEqual(["validation", "full"]);
    expect(interrupt.gate_id).toBe("gate-1");
  });

  it("keeps candidate submissions identity-only", () => {
    const submission: AdapterCandidateSubmission = {
      schema_version: "1.0",
      action: "submit_candidate",
      task_id: "adapter-task-001",
      thread_id: "langgraph-thread-001",
      codex_thread_id: "codex-thread-001",
      revision: 2,
    };

    expect(Object.keys(submission).sort()).toEqual([
      "action",
      "codex_thread_id",
      "revision",
      "schema_version",
      "task_id",
      "thread_id",
    ]);
  });

  it("supports the apply-and-commit decision", () => {
    const decision: AdapterPatchDecision = {
      schema_version: "1.0",
      action: "apply_and_commit",
      task_id: "adapter-task-001",
      revision: 2,
      message: null,
      commit_message: "Add example EEG adapter",
    };

    expect(decision.action).toBe("apply_and_commit");
  });

  it("publishes replaceable runtime boundaries", () => {
    expectTypeOf<LangGraphClient>().toBeObject();
    expectTypeOf<CodingRuntime>().toBeObject();
    expectTypeOf<SshManager>().toBeObject();
  });
});
