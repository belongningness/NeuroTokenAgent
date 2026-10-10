import { describe, expect, it, vi } from "vitest";
import type { SshConnectionState } from "./runtime";
import { SystemSshManager } from "./systemSshManager";

const SETTINGS = {
  host: "brain-server",
  langGraphLocalPort: 2024,
  langGraphRemotePort: 2024,
  codingLocalPort: 4732,
  codingRemotePort: 4732,
};

describe("SystemSshManager", () => {
  it("checks availability and invokes the two-port tunnel command", async () => {
    const invoke = vi.fn(async (command: string) => command === "neurotoken_ssh_available");
    const listen = vi.fn(async () => () => undefined);
    const manager = new SystemSshManager(
      invoke as <T>(command: string, args?: Record<string, unknown>) => Promise<T>,
      listen,
    );

    expect(await manager.checkAvailable()).toBe(true);
    await manager.connect(SETTINGS);
    expect(invoke).toHaveBeenCalledWith("neurotoken_ssh_connect", { settings: SETTINGS });
  });

  it("returns exportable native SSH lifecycle diagnostics", async () => {
    const diagnostics = {
      logPath: "/tmp/neurotoken-ssh.log",
      desktopPid: 10,
      watchdogPid: 11,
      generation: 2,
      state: { status: "connected" as const },
      settings: SETTINGS,
      recentLog: "parent pipe closed",
    };
    const invoke = vi.fn(async (command: string) => command === "neurotoken_ssh_diagnostics" ? diagnostics : undefined);
    const manager = new SystemSshManager(
      invoke as <T>(command: string, args?: Record<string, unknown>) => Promise<T>,
      vi.fn(async () => () => undefined),
    );

    await expect(manager.getDiagnostics()).resolves.toEqual(diagnostics);
    expect(invoke).toHaveBeenCalledWith("neurotoken_ssh_diagnostics");
  });

  it("forwards native reconnect states and disconnects cleanly", async () => {
    const eventListeners: Array<(event: { payload: SshConnectionState }) => void> = [];
    const invoke = vi.fn(async () => undefined);
    const listen = async (
      _event: string,
      listener: (event: { payload: SshConnectionState }) => void,
    ) => {
      eventListeners.push(listener);
      return () => undefined;
    };
    const manager = new SystemSshManager(
      invoke as <T>(command: string, args?: Record<string, unknown>) => Promise<T>,
      listen,
    );
    const states: SshConnectionState[] = [];
    manager.subscribe((state) => states.push(state));
    await Promise.resolve();

    eventListeners[0]?.({ payload: { status: "error", message: "connection lost" } });
    eventListeners[0]?.({ payload: { status: "connected" } });
    await manager.disconnect();

    expect(states).toContainEqual({ status: "error", message: "connection lost" });
    expect(states[states.length - 1]).toEqual({ status: "disconnected" });
    expect(invoke).toHaveBeenCalledWith("neurotoken_ssh_disconnect");
  });
});
