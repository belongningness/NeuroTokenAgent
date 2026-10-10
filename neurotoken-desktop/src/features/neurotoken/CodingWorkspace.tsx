import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import ChevronLeft from "lucide-react/dist/esm/icons/chevron-left";
import ChevronRight from "lucide-react/dist/esm/icons/chevron-right";
import FileCode2 from "lucide-react/dist/esm/icons/file-code-2";
import ScrollText from "lucide-react/dist/esm/icons/scroll-text";
import { ApprovalToasts } from "../app/components/ApprovalToasts";
import { GitDiffViewer } from "../git/components/GitDiffViewer";
import { NeuroTokenConversation } from "./NeuroTokenConversation";
import { NativeCodingUiAdapter, type NativeCodingUiState } from "./nativeUiAdapter";
import { parseRemoteDiff } from "./remoteDiff";
import type { AdapterCodingHandoff } from "./contracts";
import type { CodingRuntime, CodingSession, RemoteFile } from "./runtime";
import { readableRuntimeError } from "./remoteCodingRuntime";

export type CodingWorkspaceView = "conversation" | "editor" | "diff";

type CodingWorkspaceProps = {
  runtime: CodingRuntime;
  handoff: AdapterCodingHandoff;
  onSessionOpened?: (session: CodingSession) => void;
  onTurnCompleted?: (session: CodingSession | null) => void;
  onSessionInvalidated?: (taskId: string, revision: number, message: string) => void;
  theme?: "dark" | "light";
  view?: CodingWorkspaceView;
  approvalDiff?: string | null;
  readOnly?: boolean;
  archived?: boolean;
};

const EMPTY_UI_STATE = (handoff: AdapterCodingHandoff): NativeCodingUiState => ({
  taskId: handoff.task_id,
  revision: handoff.revision,
  conversationItems: [],
  approvalRequests: [],
  isProcessing: false,
});

type StartupPhase = "opening" | "starting_instruction" | "waiting_for_codex" | "ready";

const STARTUP_LABELS: Record<Exclude<StartupPhase, "ready">, string> = {
  opening: "Connecting to the remote Codex workspace…",
  starting_instruction: "Codex workspace opened. Sending the adapter instruction…",
  waiting_for_codex: "Instruction accepted. Waiting for Codex's first response…",
};

function isInvalidCodingSession(message: string) {
  return /coding task is not open|task session is not open|task is not open for coding|stale neurotoken (?:bridge session|task) revision|failed to resolve task (?:worktree|baseline)|adapter coding task (?:worktree|baseline) is missing|task (?:worktree|baseline) is not a directory/i.test(message);
}

function formatCodexRuntimeError(error: unknown) {
  const message = readableRuntimeError(error);
  if (/quota|exceeded|rate.?limit/i.test(message)) {
    return `Codex Bridge account quota was rejected: ${message} This is separate from the NeuroToken pipeline API key. The candidate was not submitted; check the Codex account/model selected in Desktop, then retry.`;
  }
  if (/network|offline|connect|socket|fetch|timeout|timed out|econn|closed|name resolution|dns/i.test(message)) {
    return `Codex connection was lost: ${message} The current turn has stopped. Restore the connection, then retry.`;
  }
  return `Codex turn failed: ${message} The candidate was not submitted. You can retry this turn.`;
}

function rawTurnError(message: Record<string, unknown>) {
  const method = String(message.method ?? "");
  if (method !== "error" && method !== "turn/failed" && method !== "turn/cancelled") return null;
  const params = message.params && typeof message.params === "object" && !Array.isArray(message.params)
    ? message.params as Record<string, unknown>
    : {};
  return readableRuntimeError(params.message ?? params.error ?? message.error);
}

export function CodingWorkspace({
  runtime,
  handoff,
  onSessionOpened,
  onTurnCompleted,
  onSessionInvalidated,
  theme = "dark",
  view = "conversation",
  approvalDiff = null,
  readOnly = false,
  archived = false,
}: CodingWorkspaceProps) {
  const adapter = useMemo(
    () => new NativeCodingUiAdapter(handoff.task_id, handoff.revision),
    [handoff.revision, handoff.task_id],
  );
  const [uiState, setUiState] = useState<NativeCodingUiState>(() => EMPTY_UI_STATE(handoff));
  const [instruction, setInstruction] = useState("");
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [diagnosticsFeedback, setDiagnosticsFeedback] = useState<string | null>(null);
  const [diagnosticsReport, setDiagnosticsReport] = useState<string | null>(null);
  const [startupPhase, setStartupPhase] = useState<StartupPhase>("opening");
  const [startupStartedAt, setStartupStartedAt] = useState(() => Date.now());
  const [files, setFiles] = useState<string[]>([]);
  const [file, setFile] = useState<RemoteFile | null>(null);
  const [draft, setDraft] = useState("");
  const [saveState, setSaveState] = useState<"clean" | "dirty" | "saved" | "conflict">("clean");
  const [filePanelWidth, setFilePanelWidth] = useState(180);
  const [filePanelCollapsed, setFilePanelCollapsed] = useState(false);
  const [diff, setDiff] = useState("");
  const [diffLoading, setDiffLoading] = useState(false);
  const [workspaceRefreshVersion, setWorkspaceRefreshVersion] = useState(0);
  const openedSessionRef = useRef<CodingSession | null>(null);
  const fileRef = useRef<RemoteFile | null>(null);
  const saveStateRef = useRef(saveState);
  const syncedHistorySignatureRef = useRef("");
  const completionReportedRef = useRef(false);
  const completionTimerRef = useRef<number | null>(null);
  const turnFailedRef = useRef(false);
  const sessionInvalidatedRef = useRef(false);
  const hasMeaningfulEventRef = useRef(false);
  const callbacksRef = useRef({ onSessionOpened, onTurnCompleted, onSessionInvalidated });

  useEffect(() => {
    callbacksRef.current = { onSessionOpened, onTurnCompleted, onSessionInvalidated };
  }, [onSessionInvalidated, onSessionOpened, onTurnCompleted]);

  useEffect(() => {
    fileRef.current = file;
  }, [file]);

  useEffect(() => {
    saveStateRef.current = saveState;
  }, [saveState]);

  const handleRuntimeFailure = useCallback((error: unknown) => {
    const message = readableRuntimeError(error);
    if (isInvalidCodingSession(message)) {
      sessionInvalidatedRef.current = true;
      turnFailedRef.current = true;
      if (completionTimerRef.current !== null) {
        window.clearTimeout(completionTimerRef.current);
        completionTimerRef.current = null;
      }
      callbacksRef.current.onSessionInvalidated?.(handoff.task_id, handoff.revision, message);
      return true;
    }
    setRuntimeError(formatCodexRuntimeError(error));
    return false;
  }, [handoff.revision, handoff.task_id]);

  const reloadFiles = useCallback(async () => {
    if (archived || sessionInvalidatedRef.current) return;
    const paths = await runtime.listFiles(handoff.task_id);
    if (sessionInvalidatedRef.current) return;
    setFiles(paths);
    const selectedPath = fileRef.current?.path;
    const nextPath = selectedPath && paths.includes(selectedPath) ? selectedPath : paths[0];
    if (!nextPath) {
      if (saveStateRef.current !== "dirty") {
        setFile(null);
        setDraft("");
        setSaveState("clean");
      }
      return;
    }
    const loaded = await runtime.readFile(handoff.task_id, nextPath);
    if (sessionInvalidatedRef.current) return;
    if (saveStateRef.current === "dirty") {
      if (fileRef.current?.path === loaded.path && fileRef.current.contentHash !== loaded.contentHash) {
        saveStateRef.current = "conflict";
        setSaveState("conflict");
      }
      return;
    }
    fileRef.current = loaded;
    saveStateRef.current = "clean";
    setFile(loaded);
    setDraft(loaded.content);
    setSaveState("clean");
  }, [archived, handoff.task_id, runtime]);

  const refreshWorkspaceAfterTurn = useCallback(() => {
    setWorkspaceRefreshVersion((version) => version + 1);
    void reloadFiles().catch(handleRuntimeFailure);
  }, [handleRuntimeFailure, reloadFiles]);

  useEffect(() => {
    let active = true;
    const rawHistory = runtime.getRawHistory?.(handoff.task_id, handoff.revision) ?? [];
    const normalizedHistory = runtime.getHistory(handoff.task_id, handoff.revision);
    const visibleRawHistory = archived
      ? rawHistory.filter((event) => rawTurnError(event.message) === null)
      : rawHistory;
    const visibleHistory = archived
      ? normalizedHistory.filter((event) => event.type !== "error")
      : normalizedHistory;
    setUiState(visibleRawHistory.length
      ? adapter.replaceRawHistory(visibleRawHistory)
      : adapter.replaceHistory(visibleHistory));
    setRuntimeError(null);
    setDiagnosticsFeedback(null);
    setDiagnosticsReport(null);
    setStartupPhase("opening");
    setStartupStartedAt(Date.now());
    syncedHistorySignatureRef.current = "";
    completionReportedRef.current = false;
    turnFailedRef.current = false;
    sessionInvalidatedRef.current = false;
    if (completionTimerRef.current !== null) {
      window.clearTimeout(completionTimerRef.current);
      completionTimerRef.current = null;
    }
    hasMeaningfulEventRef.current = false;
    fileRef.current = null;
    saveStateRef.current = "clean";
    setFiles([]);
    setFile(null);
    setSaveState("clean");
    setDiff("");

    const usesRawEvents = Boolean(runtime.subscribeRaw);
    const unsubscribe = archived ? () => {} : runtime.subscribe(handoff.task_id, (event) => {
      if (!active) return;
      const meaningfulEvent = event.type === "turn_completed" || event.type === "error" || (
        event.type === "app_server" && ["assistant_message", "command_output", "tool_call"].includes(String(event.message.kind ?? ""))
      );
      if (meaningfulEvent) {
        hasMeaningfulEventRef.current = true;
        setStartupPhase("ready");
      }
      if (!usesRawEvents || event.type === "error") {
        setUiState(adapter.applyCodingEvent(event));
      }
      if (event.type === "error") {
        turnFailedRef.current = true;
        if (completionTimerRef.current !== null) {
          window.clearTimeout(completionTimerRef.current);
          completionTimerRef.current = null;
        }
        handleRuntimeFailure(event.message);
      }
      if (event.type === "turn_completed" && openedSessionRef.current?.turnStarted && !completionReportedRef.current && !turnFailedRef.current) {
        refreshWorkspaceAfterTurn();
        completionTimerRef.current = window.setTimeout(() => {
          completionTimerRef.current = null;
          if (!active || completionReportedRef.current || turnFailedRef.current) return;
          completionReportedRef.current = true;
          callbacksRef.current.onTurnCompleted?.(openedSessionRef.current);
        }, 750);
      }
    });
    const unsubscribeRaw = archived ? undefined : runtime.subscribeRaw?.(handoff.task_id, (event) => {
      if (active) {
        const method = String(event.message.method ?? "");
        if (
          method === "turn/started" ||
          method === "turn/completed" ||
          method === "turn/failed" ||
          method === "turn/cancelled" ||
          method === "error" ||
          method.startsWith("item/")
        ) {
          hasMeaningfulEventRef.current = true;
          setStartupPhase("ready");
        }
        if (method === "turn/completed" && openedSessionRef.current?.turnStarted && !completionReportedRef.current && !turnFailedRef.current) {
          refreshWorkspaceAfterTurn();
          completionTimerRef.current = window.setTimeout(() => {
            completionTimerRef.current = null;
            if (!active || completionReportedRef.current || turnFailedRef.current) return;
            completionReportedRef.current = true;
            callbacksRef.current.onTurnCompleted?.(openedSessionRef.current);
          }, 750);
        }
        const turnError = rawTurnError(event.message);
        if (turnError) {
          turnFailedRef.current = true;
          if (completionTimerRef.current !== null) {
            window.clearTimeout(completionTimerRef.current);
            completionTimerRef.current = null;
          }
          handleRuntimeFailure(turnError);
        }
        setUiState(adapter.applyRawEvent(event));
      }
    });

    const openWorkspace = readOnly || archived
      ? Promise.resolve(null)
      : runtime.openTask(handoff);
    void openWorkspace
      .then(async (session) => {
        if (!active || sessionInvalidatedRef.current) return [];
        if (session) {
          openedSessionRef.current = session;
          callbacksRef.current.onSessionOpened?.(session);
          if (session.shouldStartInstruction) {
            setStartupPhase("starting_instruction");
            await runtime.sendInstruction(handoff.task_id, handoff.instruction);
            if (active && !hasMeaningfulEventRef.current) setStartupPhase("waiting_for_codex");
          } else if (session.turnStarted && runtime.getHistory(handoff.task_id, handoff.revision).some((event) => event.type === "turn_completed")) {
            callbacksRef.current.onTurnCompleted?.(session);
          }
        }
        setStartupPhase("ready");
        if (active && !archived) await reloadFiles();
      })
      .catch((error) => {
        if (!active) return;
        handleRuntimeFailure(error);
      });

    return () => {
      active = false;
      if (completionTimerRef.current !== null) {
        window.clearTimeout(completionTimerRef.current);
        completionTimerRef.current = null;
      }
      openedSessionRef.current = null;
      unsubscribeRaw?.();
      unsubscribe();
    };
  }, [adapter, archived, handoff, handleRuntimeFailure, readOnly, refreshWorkspaceAfterTurn, reloadFiles, runtime]);

  useEffect(() => {
    if (archived || !runtime.refreshRawHistory || startupPhase === "opening") return;
    let active = true;
    let inFlight = false;
    const refresh = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const events = await runtime.refreshRawHistory?.(handoff.task_id, handoff.revision);
        if (!active || !events?.length) return;
        if (events.some((event) => {
          const method = String(event.message.method ?? "");
          return method.startsWith("item/") || method.startsWith("turn/") || method === "error";
        })) {
          hasMeaningfulEventRef.current = true;
          setStartupPhase("ready");
        }
        const signature = JSON.stringify(events.map((event) => event.message));
        if (signature !== syncedHistorySignatureRef.current) {
          syncedHistorySignatureRef.current = signature;
          setUiState(adapter.replaceRawHistory(events));
        }
        const latestTerminal = [...events].reverse().find((event) => {
          const method = String(event.message.method ?? "");
          return method === "turn/completed" || method === "turn/failed" || method === "turn/cancelled" || method === "error";
        });
        const historyError = latestTerminal ? rawTurnError(latestTerminal.message) : null;
        if (historyError) {
          turnFailedRef.current = true;
          handleRuntimeFailure(historyError);
        } else if (
          openedSessionRef.current?.turnStarted &&
          !completionReportedRef.current &&
          latestTerminal?.message.method === "turn/completed"
        ) {
          completionReportedRef.current = true;
          refreshWorkspaceAfterTurn();
          callbacksRef.current.onTurnCompleted?.(openedSessionRef.current);
        }
      } catch (error) {
        if (active) handleRuntimeFailure(error);
      } finally {
        inFlight = false;
      }
    };
    void refresh();
    const interval = window.setInterval(() => void refresh(), 1_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [adapter, archived, handoff.revision, handoff.task_id, handleRuntimeFailure, refreshWorkspaceAfterTurn, runtime, startupPhase]);

  useEffect(() => {
    if (archived || startupPhase === "ready" || runtimeError) return;
    const timeout = window.setTimeout(() => {
      turnFailedRef.current = true;
      setStartupPhase("ready");
      setRuntimeError(
        `Codex connection timed out after ${Math.round((Date.now() - startupStartedAt) / 1000)} seconds. ` +
        "No response was received, so this task will not be submitted automatically. Restore the connection, then retry.",
      );
    }, 45_000);
    return () => window.clearTimeout(timeout);
  }, [archived, runtimeError, startupPhase, startupStartedAt]);

  useEffect(() => {
    if (view !== "diff") return;
    let active = true;
    if (approvalDiff !== null || archived) {
      setDiffLoading(false);
      return () => { active = false; };
    }
    setDiffLoading(true);
    runtime.getDiff(handoff.task_id)
      .then((value) => { if (active) setDiff(value); })
      .catch((error) => { if (active) handleRuntimeFailure(error); })
      .finally(() => { if (active) setDiffLoading(false); });
    return () => { active = false; };
  }, [approvalDiff, archived, handoff.task_id, handleRuntimeFailure, runtime, view, workspaceRefreshVersion]);

  useEffect(() => {
    if (view !== "editor" || archived || saveStateRef.current === "dirty") return;
    void reloadFiles().catch(handleRuntimeFailure);
  }, [archived, handleRuntimeFailure, reloadFiles, view]);

  async function sendInstruction(text: string) {
    const message = text.trim();
    if (!message || readOnly || archived) return;
    setRuntimeError(null);
    turnFailedRef.current = false;
    try {
      await runtime.sendInstruction(handoff.task_id, message);
      setInstruction("");
    } catch (error) {
      turnFailedRef.current = true;
      handleRuntimeFailure(error);
    }
  }

  async function retryCodexTurn() {
    if (readOnly || archived) return;
    setRuntimeError(null);
    turnFailedRef.current = false;
    setStartupPhase("starting_instruction");
    try {
      await runtime.sendInstruction(handoff.task_id, handoff.instruction);
      if (!hasMeaningfulEventRef.current) setStartupPhase("waiting_for_codex");
    } catch (error) {
      turnFailedRef.current = true;
      setStartupPhase("ready");
      handleRuntimeFailure(error);
    }
  }

  async function copyCodexDiagnostics() {
    if (!runtime.getDiagnostics) return;
    setDiagnosticsFeedback("Collecting Codex diagnostics…");
    try {
      const diagnostics = await runtime.getDiagnostics(handoff.task_id, handoff.revision);
      const report = [
        "# NeuroToken Codex diagnostics",
        `generatedAt: ${diagnostics.generatedAt}`,
        `taskId: ${diagnostics.taskId}`,
        `revision: ${diagnostics.revision}`,
        `startupPhase: ${startupPhase}`,
        `startupElapsedSeconds: ${Math.round((Date.now() - startupStartedAt) / 1000)}`,
        `runtimeError: ${runtimeError ?? "none"}`,
        `uiConversationItems: ${uiState.conversationItems.length}`,
        `uiApprovalRequests: ${uiState.approvalRequests.length}`,
        `uiIsProcessing: ${uiState.isProcessing}`,
        `session: ${JSON.stringify(diagnostics.session)}`,
        `normalizedHistoryCount: ${diagnostics.normalizedHistoryCount}`,
        `rawHistoryCount: ${diagnostics.rawHistoryCount}`,
        `rawEventMethods: ${JSON.stringify(diagnostics.rawEventMethods)}`,
        "",
        "## Timeline",
        ...diagnostics.timeline.map((entry) =>
          `${entry.timestamp} event=${entry.event} revision=${entry.revision ?? "unknown"} details=${JSON.stringify(entry.details)}`
        ),
      ].join("\n");
      setDiagnosticsReport(report);
      try {
        await navigator.clipboard.writeText(report);
        setDiagnosticsFeedback("Codex diagnostics copied. Paste them into Cursor chat.");
      } catch {
        setDiagnosticsFeedback("Clipboard access is unavailable. The report is open below; use Command+A, Command+C.");
      }
    } catch (error) {
      setDiagnosticsFeedback(`Unable to collect Codex diagnostics: ${readableRuntimeError(error)}`);
    }
  }

  async function decideTool(requestId: string | number, decision: "accept" | "decline") {
    try {
      await runtime.approveTool(handoff.task_id, requestId, decision === "accept" ? "allow" : "deny");
      setUiState(adapter.resolveApproval(requestId));
    } catch (error) {
      handleRuntimeFailure(error);
    }
  }

  async function selectFile(path: string) {
    try {
      const loaded = await runtime.readFile(handoff.task_id, path);
      fileRef.current = loaded;
      saveStateRef.current = "clean";
      setFile(loaded);
      setDraft(loaded.content);
      setSaveState("clean");
    } catch (error) {
      handleRuntimeFailure(error);
    }
  }

  async function saveFile() {
    if (!file) return;
    const result = await runtime.writeFile({
      taskId: handoff.task_id,
      path: file.path,
      content: draft,
      expectedContentHash: file.contentHash,
    });
    if (result.status === "conflict") {
      saveStateRef.current = "conflict";
      setSaveState("conflict");
      return;
    }
    fileRef.current = result.file;
    saveStateRef.current = "saved";
    setFile(result.file);
    setSaveState("saved");
    setWorkspaceRefreshVersion((version) => version + 1);
  }

  if (view === "diff") {
    const displayedDiff = approvalDiff !== null ? approvalDiff : diff;
    const diffs = parseRemoteDiff(displayedDiff);
    return (
      <section className="neurotoken-native-diff" aria-label="Candidate diff">
        {!diffLoading && !runtimeError && diffs.length === 0 && (
          <div className="neurotoken-diff-empty">
            <strong>{readOnly || archived ? "The frozen candidate diff is unavailable" : "No candidate diff is available"}</strong>
            <p>{archived
              ? "The finished workflow did not preserve a reviewable candidate diff."
              : readOnly
                ? "This approval checkpoint does not contain a reviewable diff. Do not apply it; request a revision and submit the candidate again."
                : "Continue in Coding or save changes in Editor, then submit the candidate again."}</p>
          </div>
        )}
        <GitDiffViewer
          diffs={diffs}
          selectedPath={diffs[0]?.path ?? null}
          isLoading={diffLoading}
          error={runtimeError}
          diffStyle="unified"
          canRevert={false}
          collapsibleFiles
          virtualizeFiles={false}
        />
      </section>
    );
  }

  if (view === "editor") {
    return (
      <section className="neurotoken-editor neurotoken-editor-only" data-testid="coding-workspace">
        <header>
          <div><FileCode2 /><span>{file?.path ?? (archived ? "Archived files are unavailable" : "Loading allowed file…")}</span></div>
          <div><span className={`save-${saveState}`}>{readOnly || archived ? "read only" : saveState}</span><button type="button" disabled={readOnly || archived || !file || saveState === "clean"} onClick={saveFile}>Save</button></div>
        </header>
        <div className="neurotoken-editor-body" style={{ gridTemplateColumns: `${filePanelCollapsed ? 36 : filePanelWidth}px 3px minmax(0, 1fr)` }}>
          <aside className={filePanelCollapsed ? "is-collapsed" : undefined} aria-label="Allowed files">
            <button type="button" className="neurotoken-file-panel-toggle" aria-label={filePanelCollapsed ? "Expand file panel" : "Collapse file panel"} onClick={() => setFilePanelCollapsed((value) => !value)}>
              {filePanelCollapsed ? <ChevronRight /> : <ChevronLeft />}
            </button>
            {!filePanelCollapsed && files.map((path) => <button type="button" className={file?.path === path ? "is-active" : undefined} key={path} onClick={() => void selectFile(path)}>{path.split("/").pop()}</button>)}
          </aside>
          <div className={`neurotoken-file-resizer${filePanelCollapsed ? " is-disabled" : ""}`} role="separator" aria-label="Resize allowed files panel" aria-orientation="vertical" aria-valuemin={120} aria-valuemax={360} aria-valuenow={filePanelCollapsed ? 36 : filePanelWidth} tabIndex={filePanelCollapsed ? -1 : 0} onKeyDown={(event) => {
            if (event.key === "ArrowLeft") setFilePanelWidth((width) => Math.max(120, width - 20));
            if (event.key === "ArrowRight") setFilePanelWidth((width) => Math.min(360, width + 20));
          }} />
          <Editor height="100%" language="python" theme={theme === "dark" ? "vs-dark" : "light"} value={draft} onChange={(value) => { if (!readOnly && !archived) { setDraft(value ?? ""); saveStateRef.current = "dirty"; setSaveState("dirty"); } }} options={{ readOnly: readOnly || archived, minimap: { enabled: false }, fontSize: 13, automaticLayout: true }} />
        </div>
        {saveState === "conflict" && <div className="neurotoken-conflict">The server file changed. Reload it before saving your edits.</div>}
        {runtimeError && <div className="neurotoken-conflict">{runtimeError}</div>}
      </section>
    );
  }

  return (
    <div className="neurotoken-coding neurotoken-coding-native" data-testid="coding-workspace">
      <NeuroTokenConversation
        items={uiState.conversationItems}
        threadId={openedSessionRef.current?.codexThreadId ?? handoff.thread_id}
        isLoading={uiState.conversationItems.length === 0 && !runtimeError}
        isProcessing={uiState.isProcessing}
        draft={instruction}
        onDraftChange={setInstruction}
        onSend={(text) => void sendInstruction(text)}
        readOnly={readOnly || archived}
      >
        {(readOnly || archived) && (
          <div className="neurotoken-codex-readonly" role="status">
            {archived
              ? "This coding session is archived because the adapter workflow has finished. Its conversation and frozen diff remain available for reference."
              : "This candidate is awaiting patch approval. Review the frozen diff, or use Refine in the approval card to continue coding in a new revision."}
          </div>
        )}
        {!archived && startupPhase !== "ready" && !runtimeError && (
          <div className="neurotoken-codex-startup" role="status" aria-live="polite">
            <span className="neurotoken-codex-startup__spinner" aria-hidden="true" />
            <div>
              <strong>{STARTUP_LABELS[startupPhase]}</strong>
              <small>The first tool approval may take time, but progress will remain visible here.</small>
            </div>
          </div>
        )}
        {runtimeError && (
          <div className="neurotoken-codex-failure" role="alert">
            <span>{runtimeError}</span>
            {!readOnly && !archived && <button type="button" onClick={() => void retryCodexTurn()}>Retry Codex</button>}
          </div>
        )}
      </NeuroTokenConversation>
      <div className="neurotoken-codex-diagnostics">
        <button
          type="button"
          disabled={!runtime.getDiagnostics}
          title={runtime.getDiagnostics ? "Copy the current Codex task lifecycle and event history" : "Codex diagnostics require the remote runtime"}
          onClick={() => void copyCodexDiagnostics()}
        >
          <ScrollText />
          Copy Codex diagnostics
        </button>
        {diagnosticsFeedback && <span role="status">{diagnosticsFeedback}</span>}
      </div>
      {diagnosticsReport && (
        <div className="neurotoken-codex-diagnostics-report" role="dialog" aria-label="Codex diagnostics report">
          <header>
            <strong>Codex diagnostics</strong>
            <button type="button" onClick={() => setDiagnosticsReport(null)}>Close</button>
          </header>
          <textarea readOnly value={diagnosticsReport} aria-label="Codex diagnostics text" onFocus={(event) => event.currentTarget.select()} />
        </div>
      )}
      <ApprovalToasts
        approvals={uiState.approvalRequests}
        workspaces={[{
          id: `neurotoken:${handoff.task_id}`,
          name: handoff.adapter.name,
          path: handoff.allowed_paths[0] ?? "",
          connected: true,
          settings: { sidebarCollapsed: false },
        }]}
        onDecision={(request, decision) => void decideTool(request.request_id, decision)}
      />
    </div>
  );
}
