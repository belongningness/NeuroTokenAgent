/** @vitest-environment jsdom */
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { NeuroTokenConversation } from "./NeuroTokenConversation";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("NeuroTokenConversation", () => {
  it("advances the native working timer while processing", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-09T14:00:00Z"));
    render(
      <NeuroTokenConversation
        items={[]}
        threadId="thread-1"
        isLoading={false}
        isProcessing
        draft=""
        onDraftChange={() => undefined}
        onSend={() => undefined}
      />,
    );

    expect(screen.getByText("0:00")).toBeTruthy();
    act(() => {
      vi.advanceTimersByTime(3_000);
    });
    expect(screen.getByText("0:03")).toBeTruthy();
  });
});
