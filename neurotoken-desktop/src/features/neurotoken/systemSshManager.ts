import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  SshConnectionSettings,
  SshConnectionState,
  SshDiagnostics,
  SshManager,
  Unsubscribe,
} from "./runtime";

type Invoke = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;
type Listen = (
  event: string,
  listener: (event: { payload: SshConnectionState }) => void,
) => Promise<Unsubscribe>;

export class SystemSshManager implements SshManager {
  private state: SshConnectionState = { status: "disconnected" };
  private listeners = new Set<(state: SshConnectionState) => void>();
  private unlisten: Unsubscribe | null = null;

  constructor(
    private readonly invokeCommand: Invoke = invoke,
    private readonly listenEvent: Listen = listen,
  ) {}

  async checkAvailable(): Promise<boolean> {
    return this.invokeCommand<boolean>("neurotoken_ssh_available");
  }

  async connect(settings: SshConnectionSettings): Promise<void> {
    await this.ensureSubscribed();
    this.publish({ status: "connecting" });
    try {
      await this.invokeCommand("neurotoken_ssh_connect", { settings });
      this.publish({ status: "connected" });
    } catch (error) {
      const state = { status: "error", message: error instanceof Error ? error.message : String(error) } as const;
      this.publish(state);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    await this.invokeCommand("neurotoken_ssh_disconnect");
    this.publish({ status: "disconnected" });
  }

  async getDiagnostics(): Promise<SshDiagnostics> {
    return this.invokeCommand<SshDiagnostics>("neurotoken_ssh_diagnostics");
  }

  subscribe(onState: (state: SshConnectionState) => void): Unsubscribe {
    this.listeners.add(onState);
    onState(this.state);
    void this.ensureSubscribed();
    return () => {
      this.listeners.delete(onState);
      if (this.listeners.size === 0 && this.unlisten) {
        this.unlisten();
        this.unlisten = null;
      }
    };
  }

  private async ensureSubscribed() {
    if (this.unlisten) return;
    this.unlisten = await this.listenEvent(
      "neurotoken-ssh-state",
      (event) => this.publish(event.payload),
    );
  }

  private publish(state: SshConnectionState) {
    this.state = state;
    for (const listener of this.listeners) listener(state);
  }
}
