import { describe, expect, it, vi } from "vitest";
import {
  MOCK_CODING_HANDOFF,
  MockCodingRuntime,
} from "./mockCodingRuntime";

describe("MockCodingRuntime", () => {
  it("lists and reads only allowed task files", async () => {
    const runtime = new MockCodingRuntime();
    const files = await runtime.listFiles(MOCK_CODING_HANDOFF.task_id);

    expect(files).toEqual(["agent_pipeline/adapters/example_eeg.py"]);
    await expect(
      runtime.readFile(MOCK_CODING_HANDOFF.task_id, "../secrets.env"),
    ).rejects.toThrow("Path escapes the allowed task scope.");
    await expect(
      runtime.readFile(MOCK_CODING_HANDOFF.task_id, "/etc/passwd"),
    ).rejects.toThrow("Path escapes the allowed task scope.");
  });

  it("saves with the expected hash and updates the diff", async () => {
    const runtime = new MockCodingRuntime();
    const file = await runtime.readFile(
      MOCK_CODING_HANDOFF.task_id,
      "agent_pipeline/adapters/example_eeg.py",
    );
    const result = await runtime.writeFile({
      taskId: MOCK_CODING_HANDOFF.task_id,
      path: file.path,
      content: `${file.content}\nADAPTER_NAME = "example_eeg"\n`,
      expectedContentHash: file.contentHash,
    });

    expect(result.status).toBe("saved");
    expect(await runtime.getDiff(MOCK_CODING_HANDOFF.task_id)).toContain(
      "ADAPTER_NAME",
    );
  });

  it("returns the server file when the expected hash is stale", async () => {
    const runtime = new MockCodingRuntime();
    const original = await runtime.readFile(
      MOCK_CODING_HANDOFF.task_id,
      "agent_pipeline/adapters/example_eeg.py",
    );
    await runtime.writeFile({
      taskId: MOCK_CODING_HANDOFF.task_id,
      path: original.path,
      content: `${original.content}\n# server change\n`,
      expectedContentHash: original.contentHash,
    });
    const conflict = await runtime.writeFile({
      taskId: MOCK_CODING_HANDOFF.task_id,
      path: original.path,
      content: `${original.content}\n# local change\n`,
      expectedContentHash: original.contentHash,
    });

    expect(conflict.status).toBe("conflict");
    if (conflict.status === "conflict") {
      expect(conflict.current.content).toContain("server change");
    }
  });

  it("accepts natural-language coding instructions through the shared contract", async () => {
    const runtime = new MockCodingRuntime();
    const listener = vi.fn();
    runtime.subscribe(MOCK_CODING_HANDOFF.task_id, listener);

    await runtime.sendInstruction(MOCK_CODING_HANDOFF.task_id, "Add strict validation");

    expect(listener).toHaveBeenLastCalledWith({
      type: "app_server",
      message: { kind: "assistant_message", text: "Working on: Add strict validation" },
    });
  });

  it("streams Codex events and records tool decisions", async () => {
    const runtime = new MockCodingRuntime();
    const listener = vi.fn();
    runtime.subscribe(MOCK_CODING_HANDOFF.task_id, listener);

    await runtime.approveTool(
      MOCK_CODING_HANDOFF.task_id,
      "tool-request-001",
      "deny",
    );

    expect(listener).toHaveBeenCalledTimes(5);
    expect(runtime.decisionFor("tool-request-001")).toBe("deny");
  });
});
