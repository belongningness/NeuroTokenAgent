/** @vitest-environment jsdom */
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CodingWorkspace } from "./CodingWorkspace";
import { MOCK_CODING_HANDOFF, MockCodingRuntime } from "./mockCodingRuntime";
import type { CodingEvent, Unsubscribe } from "./runtime";

vi.mock("@monaco-editor/react", () => ({
  default: ({ value, onChange }: { value: string; onChange: (value: string) => void }) => (
    <textarea aria-label="Adapter editor" value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}));

afterEach(cleanup);

describe("CodingWorkspace", () => {
  it("renders native Codex messages, command rows, and approval toasts", async () => {
    const runtime = new MockCodingRuntime();
    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);

    expect(await screen.findByText("I found the EEG adapter skeleton and inspected the allowed file.")).toBeTruthy();
    fireEvent.click(screen.getByText("Inspect HDF5 datasets"));
    expect(screen.getByText(/datasets: signal, channels, sampling_rate/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Approve/ }));
    await waitFor(() => expect(runtime.decisionFor("tool-request-001")).toBe("allow"));
  });

  it("keeps the native conversation at full workspace height instead of the legacy three-row layout", () => {
    const runtime = new MockCodingRuntime();
    const { container } = render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);
    const workspace = container.querySelector(".neurotoken-coding-native");
    expect(workspace?.classList.contains("neurotoken-coding")).toBe(true);
    expect(workspace?.querySelector(".neurotoken-conversation")).toBeTruthy();
  });

  it("keeps the diagnostics entry visible when the conversation has no output", async () => {
    const runtime = new MockCodingRuntime();
    vi.spyOn(runtime, "subscribe").mockImplementation(() => () => undefined);
    vi.spyOn(runtime, "openTask").mockImplementation(() => new Promise(() => undefined));

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);

    expect(screen.getByRole("button", { name: "Copy Codex diagnostics" })).toBeTruthy();
  });

  it("opens a selectable report when WebView clipboard access is denied", async () => {
    const runtime = Object.assign(new MockCodingRuntime(), {
      getDiagnostics: vi.fn().mockResolvedValue({
        generatedAt: "2026-08-10T04:00:00.000Z",
        taskId: MOCK_CODING_HANDOFF.task_id,
        revision: MOCK_CODING_HANDOFF.revision,
        session: null,
        normalizedHistoryCount: 1,
        rawHistoryCount: 1,
        rawEventMethods: ["item/agentMessage/delta"],
        timeline: [],
      }),
    });
    vi.stubGlobal("navigator", {
      ...navigator,
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) },
    });

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy Codex diagnostics" }));

    expect(await screen.findByRole("dialog", { name: "Codex diagnostics report" })).toBeTruthy();
    expect((screen.getByLabelText("Codex diagnostics text") as HTMLTextAreaElement).value).toContain("rawHistoryCount: 1");
    expect(screen.getByText(/Clipboard access is unavailable/)).toBeTruthy();
  });

  it("shows immediate startup feedback before the remote Codex workspace responds", async () => {
    const runtime = new MockCodingRuntime();
    vi.spyOn(runtime, "subscribe").mockImplementation(() => () => undefined);
    vi.spyOn(runtime, "openTask").mockImplementation(() => new Promise(() => undefined));

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);

    expect(screen.getByText("Connecting to the remote Codex workspace…")).toBeTruthy();
  });

  it("renders reasoning progress emitted while continuing a revision", async () => {
    const runtime = new MockCodingRuntime();
    const openTask = vi.spyOn(runtime, "openTask");
    let emitRaw!: (event: {
      workspace_id: string;
      message: Record<string, unknown>;
    }) => void;
    vi.spyOn(runtime, "subscribe").mockImplementation(() => () => undefined);
    Object.assign(runtime, {
      subscribeRaw: vi.fn((_taskId: string, onEvent: typeof emitRaw) => {
        emitRaw = onEvent;
        return () => undefined;
      }),
    });

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);
    await waitFor(() => expect(openTask).toHaveBeenCalled());
    await act(async () => emitRaw({
      workspace_id: `neurotoken:${MOCK_CODING_HANDOFF.task_id}`,
      message: {
        method: "item/reasoning/summaryTextDelta",
        params: {
          threadId: "codex-thread-001",
          itemId: "revision-reasoning",
          delta: "Reviewing the previous adapter changes",
        },
      },
    }));

    expect(await screen.findByText(/Reviewing the previous adapter changes/)).toBeTruthy();
  });

  it("starts the handoff instruction for a newly created Codex thread", async () => {
    const runtime = new MockCodingRuntime();
    vi.spyOn(runtime, "openTask").mockResolvedValue({
      taskId: MOCK_CODING_HANDOFF.task_id,
      codexThreadId: "codex-new",
      revision: MOCK_CODING_HANDOFF.revision,
      status: "idle",
      shouldStartInstruction: true,
    });
    const sendInstruction = vi.spyOn(runtime, "sendInstruction");
    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);
    await waitFor(() => expect(sendInstruction).toHaveBeenCalledWith(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.instruction));
  });

  it("does not report an old completed turn for a revision that never started", async () => {
    const runtime = new MockCodingRuntime();
    const onTurnCompleted = vi.fn();
    vi.spyOn(runtime, "openTask").mockResolvedValue({
      taskId: MOCK_CODING_HANDOFF.task_id,
      codexThreadId: "codex-existing",
      revision: MOCK_CODING_HANDOFF.revision,
      status: "idle",
      turnStarted: false,
      shouldStartInstruction: false,
    });
    vi.spyOn(runtime, "getHistory").mockReturnValue([{ type: "turn_completed" }]);

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} onTurnCompleted={onTurnCompleted} />);

    await waitFor(() => expect(runtime.openTask).toHaveBeenCalled());
    expect(onTurnCompleted).not.toHaveBeenCalled();
  });

  it("restores cached messages after switching back to the task", async () => {
    const runtime = new MockCodingRuntime();
    vi.spyOn(runtime, "getHistory").mockReturnValue([
      { type: "app_server", message: { kind: "assistant_message", text: "Restored response" } },
    ]);
    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);
    expect(await screen.findByText("Restored response")).toBeTruthy();
  });

  it("sends natural-language instructions through the native composer", async () => {
    const runtime = new MockCodingRuntime();
    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);
    const instruction = screen.getByPlaceholderText("Ask Codex to do something...");
    fireEvent.change(instruction, { target: { value: "Add channel validation" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Working on: Add channel validation")).toBeTruthy();
  });

  it.each([
    "Stale NeuroToken Bridge session revision: expected 10, received 9",
    "Failed to resolve task worktree: No such file or directory (os error 2)",
  ])("invalidates an obsolete coding session without offering retry: %s", async (message) => {
    const runtime = new MockCodingRuntime();
    const onSessionInvalidated = vi.fn();
    vi.spyOn(runtime, "openTask").mockRejectedValue(new Error(message));

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} onSessionInvalidated={onSessionInvalidated} />);

    await waitFor(() => expect(onSessionInvalidated).toHaveBeenCalledWith(
      MOCK_CODING_HANDOFF.task_id,
      MOCK_CODING_HANDOFF.revision,
      message,
    ));
    expect(screen.queryByText(/Codex turn failed|Failed to resolve task worktree/i)).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry Codex" })).toBeNull();
  });

  it("invalidates a session when cleanup races after the task was opened", async () => {
    const runtime = new MockCodingRuntime();
    const onSessionInvalidated = vi.fn();
    vi.spyOn(runtime, "listFiles").mockRejectedValue(new Error(
      "Failed to resolve task worktree: No such file or directory (os error 2)",
    ));

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} onSessionInvalidated={onSessionInvalidated} />);

    await waitFor(() => expect(onSessionInvalidated).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: "Retry Codex" })).toBeNull();
  });

  it("invalidates a session when a live runtime event reports a removed worktree", async () => {
    const runtime = new MockCodingRuntime();
    const onSessionInvalidated = vi.fn();
    const openTask = vi.spyOn(runtime, "openTask");
    const listFiles = vi.spyOn(runtime, "listFiles");
    vi.spyOn(runtime, "subscribe").mockImplementation((_taskId, onEvent) => {
      onEvent({
        type: "error",
        message: "Adapter coding task worktree is missing",
        retryable: false,
      });
      return () => undefined;
    });

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} onSessionInvalidated={onSessionInvalidated} />);

    await waitFor(() => expect(onSessionInvalidated).toHaveBeenCalled());
    expect(openTask).toHaveBeenCalledTimes(1);
    expect(listFiles).not.toHaveBeenCalled();
    expect(screen.queryByText(/Codex turn failed|Failed to resolve task worktree/i)).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry Codex" })).toBeNull();
  });

  it("shows a retry action and does not report completion after a Codex failure", async () => {
    const runtime = new MockCodingRuntime();
    const onTurnCompleted = vi.fn();
    vi.spyOn(runtime, "subscribe").mockImplementation((_taskId, onEvent) => {
      onEvent({ type: "turn_completed" });
      onEvent({ type: "error", message: "The quota has been exceeded.", retryable: true });
      return () => undefined;
    });
    const sendInstruction = vi.spyOn(runtime, "sendInstruction");

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} onTurnCompleted={onTurnCompleted} />);

    expect(await screen.findByText(/Codex Bridge account quota was rejected/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Retry Codex" }));
    await waitFor(() => expect(sendInstruction).toHaveBeenCalledWith(MOCK_CODING_HANDOFF.task_id, MOCK_CODING_HANDOFF.instruction));
    await new Promise((resolve) => window.setTimeout(resolve, 800));
    expect(onTurnCompleted).not.toHaveBeenCalled();
  });

  it("mounts Monaco only in the editor subview and saves allowed files", async () => {
    const runtime = new MockCodingRuntime();
    const { rerender } = render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} />);
    expect(screen.queryByLabelText("Adapter editor")).toBeNull();
    rerender(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} view="editor" />);
    const editor = await screen.findByLabelText("Adapter editor");
    fireEvent.change(editor, { target: { value: "ADAPTER_NAME = 'example_eeg'" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.getByText("saved")).toBeTruthy());
    expect(await runtime.getDiff("task-adapter-001")).toContain("ADAPTER_NAME");
  });

  it("renders remote patch through GitDiffViewer in the diff subview", async () => {
    const runtime = new MockCodingRuntime();
    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} view="diff" />);
    expect(await screen.findByLabelText("Candidate diff")).toBeTruthy();
    expect((await screen.findAllByText("example_eeg.py")).length).toBeGreaterThanOrEqual(1);
  });

  it("reloads Codex file changes and live diff after turn completion", async () => {
    const runtime = new MockCodingRuntime();
    let emit!: (event: CodingEvent) => void;
    vi.spyOn(runtime, "subscribe").mockImplementation((_taskId, onEvent): Unsubscribe => {
      emit = onEvent;
      return () => undefined;
    });
    const getDiff = vi.spyOn(runtime, "getDiff").mockResolvedValue(
      "diff --git a/agent_pipeline/adapters/example_eeg.py b/agent_pipeline/adapters/example_eeg.py\n" +
      "--- a/agent_pipeline/adapters/example_eeg.py\n" +
      "+++ b/agent_pipeline/adapters/example_eeg.py\n" +
      "@@ -1 +1 @@\n-before\n+after_codex",
    );
    const { rerender } = render(
      <CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} view="editor" />,
    );
    const editor = await screen.findByLabelText("Adapter editor") as HTMLTextAreaElement;
    expect(editor.value).toContain("read_session");

    const current = await runtime.readFile(MOCK_CODING_HANDOFF.task_id, "agent_pipeline/adapters/example_eeg.py");
    await runtime.writeFile({
      taskId: MOCK_CODING_HANDOFF.task_id,
      path: current.path,
      content: "ADAPTER_NAME = 'after_codex'",
      expectedContentHash: current.contentHash,
    });
    await act(async () => emit({ type: "turn_completed" }));

    await waitFor(() => expect((screen.getByLabelText("Adapter editor") as HTMLTextAreaElement).value).toContain("after_codex"));
    rerender(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} view="diff" />);
    await waitFor(() => expect(getDiff.mock.calls.length).toBeGreaterThanOrEqual(1));
    expect((await screen.findAllByText("example_eeg.py")).length).toBeGreaterThanOrEqual(1);
  });

  it("preserves a dirty editor draft when Codex changes the same file", async () => {
    const runtime = new MockCodingRuntime();
    let emit!: (event: CodingEvent) => void;
    vi.spyOn(runtime, "subscribe").mockImplementation((_taskId, onEvent): Unsubscribe => {
      emit = onEvent;
      return () => undefined;
    });
    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} view="editor" />);
    const editor = await screen.findByLabelText("Adapter editor") as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "LOCAL_DRAFT = True" } });

    const current = await runtime.readFile(MOCK_CODING_HANDOFF.task_id, "agent_pipeline/adapters/example_eeg.py");
    await runtime.writeFile({
      taskId: MOCK_CODING_HANDOFF.task_id,
      path: current.path,
      content: "REMOTE_CHANGE = True",
      expectedContentHash: current.contentHash,
    });
    await act(async () => emit({ type: "turn_completed" }));

    await waitFor(() => expect(screen.getByText(/server file changed/i)).toBeTruthy());
    expect((screen.getByLabelText("Adapter editor") as HTMLTextAreaElement).value).toBe("LOCAL_DRAFT = True");
  });

  it("loads approval files and frozen diff without reopening Codex", async () => {
    const runtime = new MockCodingRuntime();
    const openTask = vi.spyOn(runtime, "openTask");
    const frozenDiff = await runtime.getDiff(MOCK_CODING_HANDOFF.task_id);
    const { rerender } = render(
      <CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} view="editor" readOnly approvalDiff={frozenDiff} />,
    );

    expect(await screen.findByLabelText("Adapter editor")).toBeTruthy();
    expect(screen.getByText("read only")).toBeTruthy();
    expect(openTask).not.toHaveBeenCalled();

    rerender(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} view="diff" readOnly approvalDiff={frozenDiff} />);
    expect((await screen.findAllByText("example_eeg.py")).length).toBeGreaterThanOrEqual(1);
    expect(openTask).not.toHaveBeenCalled();
  });

  it("renders completed coding history as an archive without reopening a removed worktree", async () => {
    const runtime = new MockCodingRuntime();
    const openTask = vi.spyOn(runtime, "openTask");
    const listFiles = vi.spyOn(runtime, "listFiles");
    const readFile = vi.spyOn(runtime, "readFile");
    const getDiff = vi.spyOn(runtime, "getDiff");
    const sendInstruction = vi.spyOn(runtime, "sendInstruction");
    const subscribe = vi.spyOn(runtime, "subscribe");

    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} archived />);

    expect(await screen.findByText(/coding session is archived/i)).toBeTruthy();
    expect(openTask).not.toHaveBeenCalled();
    expect(listFiles).not.toHaveBeenCalled();
    expect(readFile).not.toHaveBeenCalled();
    expect(getDiff).not.toHaveBeenCalled();
    expect(sendInstruction).not.toHaveBeenCalled();
    expect(subscribe).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).toBeNull();
    expect((screen.getByRole("button", { name: "Send" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("preserves the frozen approval diff across equivalent handoff snapshots", async () => {
    const runtime = new MockCodingRuntime();
    const frozenDiff = await runtime.getDiff(MOCK_CODING_HANDOFF.task_id);
    const { rerender } = render(
      <CodingWorkspace
        runtime={runtime}
        handoff={MOCK_CODING_HANDOFF}
        view="diff"
        readOnly
        approvalDiff={frozenDiff}
      />,
    );

    expect((await screen.findAllByText("example_eeg.py")).length).toBeGreaterThanOrEqual(1);

    rerender(
      <CodingWorkspace
        runtime={runtime}
        handoff={{ ...MOCK_CODING_HANDOFF }}
        view="diff"
        readOnly
        approvalDiff={frozenDiff}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByText("The frozen candidate diff is unavailable")).toBeNull();
      expect(screen.getAllByText("example_eeg.py").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("does not replace an explicitly empty frozen approval diff with live worktree changes", async () => {
    const runtime = new MockCodingRuntime();
    const getDiff = vi.spyOn(runtime, "getDiff");

    render(
      <CodingWorkspace
        runtime={runtime}
        handoff={MOCK_CODING_HANDOFF}
        view="diff"
        readOnly
        approvalDiff=""
      />,
    );

    expect(await screen.findByText("The frozen candidate diff is unavailable")).toBeTruthy();
    expect(getDiff).not.toHaveBeenCalled();
  });

  it("disables the Codex composer while a candidate awaits approval", async () => {
    const runtime = new MockCodingRuntime();
    const sendInstruction = vi.spyOn(runtime, "sendInstruction");
    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} readOnly />);

    expect(await screen.findByText(/awaiting patch approval/i)).toBeTruthy();
    expect(screen.queryByPlaceholderText("Ask Codex to do something...")).toBeNull();
    expect(sendInstruction).not.toHaveBeenCalled();
  });

  it("collapses the allowed file panel in the editor subview", async () => {
    const runtime = new MockCodingRuntime();
    render(<CodingWorkspace runtime={runtime} handoff={MOCK_CODING_HANDOFF} view="editor" />);
    await screen.findByLabelText("Adapter editor");
    fireEvent.click(screen.getByRole("button", { name: "Collapse file panel" }));
    expect(screen.getByRole("button", { name: "Expand file panel" })).toBeTruthy();
  });
});
