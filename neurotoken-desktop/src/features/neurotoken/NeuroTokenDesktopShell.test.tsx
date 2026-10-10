/** @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { NeuroTokenDesktopShell } from "./NeuroTokenDesktopShell";
import { MockLangGraphClient } from "./mockLangGraph";
import { MockCodingRuntime } from "./mockCodingRuntime";
import type { LangGraphEvent, Unsubscribe } from "./runtime";

afterEach(cleanup);

function taskNavigation() {
  return screen.getByRole("navigation", { name: "NeuroToken tasks" });
}

describe("NeuroTokenDesktopShell unified UI", () => {
  it("renders a two-panel shell with native task rows and no context placeholder", () => {
    render(<NeuroTokenDesktopShell />);
    expect(screen.getByTestId("neurotoken-shell")).toBeTruthy();
    expect(within(taskNavigation()).getByText("Prepare EEG NeuroTokens")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Conversation" })).toBeTruthy();
    expect(screen.getByRole("complementary", { name: "Stage output" })).toBeTruthy();
    expect(screen.queryByLabelText("Task context")).toBeNull();
    expect(screen.queryByRole("separator", { name: "Resize context panel" })).toBeNull();
  });

  it("shows the selected conversation and waits for explicit deletion confirmation", async () => {
    const client = new MockLangGraphClient();
    const deleteThread = vi.spyOn(client, "deleteThread").mockResolvedValue(undefined);
    render(<NeuroTokenDesktopShell langGraphClient={client} />);
    expect(within(taskNavigation()).getByText("Prepare EEG NeuroTokens")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Delete selected conversation" }));

    const dialog = screen.getByRole("dialog", { name: "Confirm conversation deletion" });
    expect(within(dialog).getByText("Prepare EEG NeuroTokens")).toBeTruthy();
    expect(deleteThread).not.toHaveBeenCalled();
    fireEvent.click(within(dialog).getByRole("button", { name: "Keep conversation" }));
    expect(screen.queryByRole("dialog", { name: "Confirm conversation deletion" })).toBeNull();
    expect(within(taskNavigation()).getByText("Prepare EEG NeuroTokens")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Delete selected conversation" }));
    fireEvent.click(within(screen.getByRole("dialog", { name: "Confirm conversation deletion" })).getByRole("button", { name: "Delete conversation" }));
    await waitFor(() => expect(deleteThread).toHaveBeenCalledWith("thread-pipeline-001"));
    expect(within(taskNavigation()).queryByText("Prepare EEG NeuroTokens")).toBeNull();
  });

  it("creates tasks and keeps the native conversation available", async () => {
    render(<NeuroTokenDesktopShell />);
    fireEvent.click(screen.getByRole("button", { name: "New task" }));
    fireEvent.change(screen.getByLabelText("New task request"), {
      target: { value: "Analyze subject 0002 H5\nand generate EEG NeuroTokens" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create task" }));
    await waitFor(() => expect(within(taskNavigation()).getByText("Analyze subject 0002 H5")).toBeTruthy());
    expect(screen.getByLabelText("Task conversation")).toBeTruthy();
  });

  it("shrinks the Stage output grid column when the panel is collapsed", () => {
    const { container } = render(<NeuroTokenDesktopShell />);
    fireEvent.click(screen.getByRole("button", { name: "Collapse stage output" }));
    expect(container.querySelector(".neurotoken-conversation-layout")?.classList.contains("is-stage-output-collapsed")).toBe(true);
    expect(screen.getByRole("button", { name: "Expand stage output" })).toBeTruthy();
  });

  it("switches the shell and native CodexMonitor theme together", () => {
    render(<NeuroTokenDesktopShell />);
    const shell = screen.getByTestId("neurotoken-shell");
    expect(document.documentElement.dataset.theme).toBe("dark");
    fireEvent.click(screen.getByRole("button", { name: "Switch to light theme" }));
    expect(shell.getAttribute("data-theme")).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("edits and retries in the selected thread without creating a conversation", async () => {
    class FailedMockClient extends MockLangGraphClient {
      override subscribe(threadId: string, onEvent: (event: LangGraphEvent) => void): Unsubscribe {
        const unsubscribe = super.subscribe(threadId, onEvent);
        if (threadId === "thread-pipeline-001") {
          onEvent({ type: "error", message: "materialize failed", retryable: true });
        }
        return unsubscribe;
      }
    }
    const client = new FailedMockClient();
    const createThread = vi.spyOn(client, "createThread");
    const sendMessage = vi.spyOn(client, "sendMessage");
    render(<NeuroTokenDesktopShell langGraphClient={client} />);
    await screen.findByText("Run failed");
    fireEvent.click(screen.getByRole("button", { name: "Edit & retry" }));
    fireEvent.change(screen.getByLabelText("Edit request before retry"), {
      target: { value: "Write output under /home/luyilin/braintoken/agent/outputs/smoke" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run edited request" }));
    await waitFor(() => expect(sendMessage).toHaveBeenCalledWith(
      "thread-pipeline-001",
      "Write output under /home/luyilin/braintoken/agent/outputs/smoke",
    ));
    expect(createThread).not.toHaveBeenCalled();
    expect(within(taskNavigation()).getByText(/Write output under \/home\/luyilin/)).toBeTruthy();
  });

  it("renders plan and stage decisions as differentiated conversation cards", () => {
    render(<NeuroTokenDesktopShell />);
    expect(screen.getByTestId("workflow-decision-card").className).toContain("plan_approval");
    expect(screen.getByText("Approve workflow plan")).toBeTruthy();
    expect(screen.queryByText("Pipeline state")).toBeNull();

    fireEvent.click(within(taskNavigation()).getByText("Package subject 0001"));
    expect(screen.getByTestId("workflow-decision-card").className).toContain("stage_review");
    expect(screen.getByText("Review stage: preflight")).toBeTruthy();
  });

  it("supports natural-language plan refinement without retaining stale submission feedback", async () => {
    render(<NeuroTokenDesktopShell />);
    const refine = screen.getByRole("button", { name: "Request revision" });
    expect((refine as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByPlaceholderText("Describe how the workflow plan should change"), {
      target: { value: "只处理一个文件并启用严格验证" },
    });
    fireEvent.click(refine);
    await waitFor(() => expect(screen.getByText("Decision recorded: refine")).toBeTruthy());
    expect(screen.queryByText("Submitted: Request revision")).toBeNull();
    expect((screen.getByRole("button", { name: "Approve plan" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("keeps adapter approval beside the Codex reasoning and exposes the code editor", async () => {
    render(<NeuroTokenDesktopShell />);
    fireEvent.click(within(taskNavigation()).getByText("Author example EEG adapter"));
    expect(await screen.findByText("Review and apply adapter")).toBeTruthy();
    expect(screen.getByTestId("workflow-decision-card").className).toContain("adapter_patch_approval");
    expect(screen.queryByLabelText("Candidate diff")).toBeNull();
    expect(screen.queryByRole("button", { name: "Diff" })).toBeNull();
    expect(screen.getByLabelText("Patch review panel")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Collapse patch review" }));
    expect(screen.getByLabelText("Patch review panel").className).toContain("is-collapsed");
    fireEvent.click(screen.getByRole("button", { name: "Expand patch review" }));
    expect(screen.getByLabelText("Patch review panel").className).not.toContain("is-collapsed");

    fireEvent.click(screen.getByRole("button", { name: "Coding" }));
    expect(await screen.findByText("I found the EEG adapter skeleton and inspected the allowed file.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Editor" }));
    expect(await screen.findByLabelText("Allowed files")).toBeTruthy();
  });

  it("leaves patch approval immediately while the next revision is being prepared", async () => {
    let releaseRevision: (() => void) | undefined;
    class DelayedRevisionClient extends MockLangGraphClient {
      override async resume(threadId: string, decision: Parameters<MockLangGraphClient["resume"]>[1]): Promise<void> {
        if (decision.action !== "refine") return super.resume(threadId, decision);
        await new Promise<void>((resolve) => {
          releaseRevision = resolve;
        });
        return super.resume(threadId, decision);
      }
    }
    const client = new DelayedRevisionClient();
    render(<NeuroTokenDesktopShell langGraphClient={client} />);
    fireEvent.click(within(taskNavigation()).getByText("Author example EEG adapter"));
    await screen.findByText("Review and apply adapter");
    fireEvent.change(screen.getByPlaceholderText("Describe the revision you want"), {
      target: { value: "Keep the existing Codex context and revise validation" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Request revision" }));

    await waitFor(() => expect(screen.queryByLabelText("Patch review panel")).toBeNull());
    expect(screen.queryByText("Review and apply adapter")).toBeNull();
    expect(screen.getByText("authoring adapter")).toBeTruthy();

    releaseRevision?.();
    await waitFor(() => expect(screen.getByText("Candidate revision 3")).toBeTruthy());
    expect(screen.queryByLabelText("Patch review panel")).toBeNull();
  });

  it("keeps the diff panel removed when a new revision starts", async () => {
    window.localStorage.clear();
    const client = new MockLangGraphClient();
    const runtime = new MockCodingRuntime();
    render(<NeuroTokenDesktopShell langGraphClient={client} codingRuntime={runtime} />);
    fireEvent.click(within(taskNavigation()).getByText("Author example EEG adapter"));
    expect(await screen.findByText("Review and apply adapter")).toBeTruthy();

    fireEvent.change(screen.getByPlaceholderText("Describe the revision you want"), {
      target: { value: "Add stricter channel validation" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Request revision" }));

    await waitFor(() => expect(screen.getByText("Candidate revision 3")).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Diff" })).toBeNull();
    expect(screen.queryByLabelText("Candidate diff")).toBeNull();
  });

  it("archives the coding workspace after applying a patch without reopening the removed worktree", async () => {
    const runtime = new MockCodingRuntime();
    const openTask = vi.spyOn(runtime, "openTask");
    const listFiles = vi.spyOn(runtime, "listFiles");
    render(<NeuroTokenDesktopShell codingRuntime={runtime} />);
    fireEvent.click(within(taskNavigation()).getByText("Author example EEG adapter"));
    await screen.findByText("Review and apply adapter");
    fireEvent.click(screen.getByRole("button", { name: "Apply only" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Conversation" }).className).toContain("is-active"));
    const openCallsAfterApply = openTask.mock.calls.length;
    const listCallsAfterApply = listFiles.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Coding" }));

    expect(await screen.findByText(/coding session is archived/i)).toBeTruthy();
    expect(openTask).toHaveBeenCalledTimes(openCallsAfterApply);
    expect(listFiles).toHaveBeenCalledTimes(listCallsAfterApply);
    expect(screen.queryByText(/Failed to resolve task worktree/i)).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry Codex" })).toBeNull();
    expect((screen.getByRole("button", { name: "Codex archived" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole("button", { name: "Editor" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Diff" })).toBeNull();
  });

  it("supports patch ask, refine, and apply-and-commit controls", async () => {
    render(<NeuroTokenDesktopShell />);
    fireEvent.click(within(taskNavigation()).getByText("Author example EEG adapter"));
    await screen.findByText("Review and apply adapter");
    const ask = screen.getByRole("button", { name: "Ask" });
    fireEvent.change(screen.getByPlaceholderText("Ask why a patch change is needed"), { target: { value: "Why?" } });
    expect((ask as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(screen.getByLabelText("Apply and commit"));
    fireEvent.change(screen.getByText("Commit message").parentElement?.querySelector("input") as HTMLInputElement, {
      target: { value: "Implement EEG adapter" },
    });
    expect(screen.getByRole("button", { name: "Apply and commit" })).toBeTruthy();
  });

  it("resizes and collapses only the task panel", () => {
    render(<NeuroTokenDesktopShell />);
    const shell = screen.getByTestId("neurotoken-shell");
    const resizer = screen.getByRole("separator", { name: "Resize task panel" });
    fireEvent.keyDown(resizer, { key: "ArrowRight" });
    expect(shell.style.gridTemplateColumns).toContain("316px");
    fireEvent.click(screen.getByRole("button", { name: "Collapse task panel" }));
    expect(shell.style.gridTemplateColumns).toContain("44px 3px");
  });
});
