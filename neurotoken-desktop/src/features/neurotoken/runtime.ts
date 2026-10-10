import type {
  AdapterCandidateSubmission,
  AdapterCodingHandoff,
  AdapterPatchDecision,
  NeuroTokenInterrupt,
  WorkspaceSnapshot,
} from "./contracts";

export type Unsubscribe = () => void;

export type LangGraphThread = {
  threadId: string;
  title: string;
  updatedAt: string;
  workspaceMode?: string;
};

export type LangGraphSnapshot = {
  workspace: WorkspaceSnapshot | null;
  interrupt: NeuroTokenInterrupt | null;
  handoff: AdapterCodingHandoff | null;
  messages: Array<{ id?: string; role: "user" | "assistant"; content: string }>;
  error: string | null;
};

export type LangGraphEvent =
  | { type: "snapshot"; snapshot: LangGraphSnapshot }
  | { type: "workspace"; snapshot: WorkspaceSnapshot }
  | { type: "interrupt"; interrupt: NeuroTokenInterrupt }
  | { type: "handoff"; handoff: AdapterCodingHandoff }
  | { type: "message"; id?: string; role: "user" | "assistant"; content: string }
  | { type: "error"; message: string; retryable: boolean };

export type LangGraphDecision =
  | AdapterPatchDecision
  | {
      schema_version: "1.0";
      action: "approve" | "refine" | "reject" | "continue" | "pause" | "cancel";
      plan?: Record<string, unknown> | null;
      message?: string | null;
    }
  | {
      schema_version: "1.0";
      action: "start_full_run" | "rerun_smoke" | "pause" | "cancel";
      gate_id: string;
      plan?: null;
      message?: null;
    };

export type LangGraphArtifact = {
  artifactId: string;
  mediaType: string;
  fileName: string;
  content: string;
};

export interface LangGraphClient {
  listThreads(): Promise<LangGraphThread[]>;
  createThread(): Promise<LangGraphThread>;
  deleteThread(threadId: string): Promise<void>;
  sendMessage(threadId: string, content: string): Promise<void>;
  retry(threadId: string): Promise<void>;
  submitCandidate(submission: AdapterCandidateSubmission): Promise<void>;
  resume(threadId: string, decision: LangGraphDecision): Promise<void>;
  getArtifact(threadId: string, artifactId: string): Promise<LangGraphArtifact>;
  subscribe(
    threadId: string,
    onEvent: (event: LangGraphEvent) => void,
  ): Unsubscribe;
}

export type CodingSession = {
  taskId: string;
  codexThreadId: string;
  revision: number;
  status: "idle" | "running" | "awaiting_approval" | "failed";
  turnStarted?: boolean;
  shouldStartInstruction?: boolean;
};

export type CodingRawAppServerEvent = {
  workspace_id: string;
  message: Record<string, unknown>;
};

export type CodingDiagnosticEntry = {
  timestamp: string;
  taskId: string;
  revision: number | null;
  event: string;
  details: Record<string, unknown>;
};

export type CodingDiagnostics = {
  generatedAt: string;
  taskId: string;
  revision: number;
  session: CodingSession | null;
  normalizedHistoryCount: number;
  rawHistoryCount: number;
  rawEventMethods: string[];
  timeline: CodingDiagnosticEntry[];
};

export type CodingEvent =
  | { type: "session"; session: CodingSession }
  | {
      type: "app_server";
      message: Record<string, unknown>;
      requestId?: string | number;
      rawMessage?: Record<string, unknown>;
      workspaceId?: string;
    }
  | {
      type: "turn_completed";
      rawMessage?: Record<string, unknown>;
      workspaceId?: string;
    }
  | {
      type: "error";
      message: string;
      retryable: boolean;
      rawMessage?: Record<string, unknown>;
      workspaceId?: string;
    };

export type RemoteFile = {
  path: string;
  content: string;
  contentHash: string;
};

export type RemoteFileWrite = {
  taskId: string;
  path: string;
  content: string;
  expectedContentHash: string;
};

export type RemoteFileWriteResult =
  | { status: "saved"; file: RemoteFile }
  | { status: "conflict"; current: RemoteFile };

export interface CodingRuntime {
  openTask(handoff: AdapterCodingHandoff): Promise<CodingSession>;
  getHistory(taskId: string, revision: number): CodingEvent[];
  getRawHistory?(taskId: string, revision: number): CodingRawAppServerEvent[];
  refreshRawHistory?(taskId: string, revision: number): Promise<CodingRawAppServerEvent[]>;
  getDiagnostics?(taskId: string, revision: number): Promise<CodingDiagnostics>;
  subscribeRaw?(
    taskId: string,
    onEvent: (event: CodingRawAppServerEvent) => void,
  ): Unsubscribe;
  sendInstruction(taskId: string, text: string): Promise<void>;
  listFiles(taskId: string): Promise<string[]>;
  readFile(taskId: string, path: string): Promise<RemoteFile>;
  writeFile(input: RemoteFileWrite): Promise<RemoteFileWriteResult>;
  getDiff(taskId: string): Promise<string>;
  approveTool(
    taskId: string,
    requestId: string | number,
    decision: "allow" | "deny",
  ): Promise<void>;
  subscribe(
    taskId: string,
    onEvent: (event: CodingEvent) => void,
  ): Unsubscribe;
}

export type SshConnectionSettings = {
  host: string;
  langGraphLocalPort: number;
  langGraphRemotePort: number;
  codingLocalPort: number;
  codingRemotePort: number;
  remoteStartCommand?: string;
};

export type SshConnectionState =
  | { status: "disconnected" }
  | { status: "connecting" }
  | { status: "connected" }
  | { status: "error"; message: string };

export type SshDiagnostics = {
  logPath: string;
  desktopPid: number;
  watchdogPid: number | null;
  generation: number;
  state: SshConnectionState;
  settings: SshConnectionSettings | null;
  recentLog: string;
};

export interface SshManager {
  checkAvailable(): Promise<boolean>;
  connect(settings: SshConnectionSettings): Promise<void>;
  disconnect(): Promise<void>;
  getDiagnostics?(): Promise<SshDiagnostics>;
  subscribe(onState: (state: SshConnectionState) => void): Unsubscribe;
}
