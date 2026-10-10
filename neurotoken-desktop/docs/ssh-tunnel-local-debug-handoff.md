# NeuroToken Desktop SSH tunnel local-debug handoff

## Goal

Find the root cause of the reported behavior on macOS:

- NeuroToken Desktop connects to the remote LangGraph and coding services through SSH.
- After the user believes the Desktop app has exited, one or more SSH tunnels still appear connected.
- Previous fixes added a Rust shutdown hook, but the observed behavior has not been resolved.

Debug this on the Mac where the packaged application actually runs. Do not assume that every `ssh` process belongs to NeuroToken Desktop.

## Repository and relevant files

Repository:

```text
neurotoken-desktop
```

Primary files:

```text
src/features/neurotoken/NeuroTokenDesktopShell.tsx
src/features/neurotoken/systemSshManager.ts
src/features/neurotoken/runtime.ts
src-tauri/src/neurotoken_ssh.rs
src-tauri/src/shared/neurotoken_ssh_core.rs
src-tauri/src/shared/process_core.rs
src-tauri/src/lib.rs
src-tauri/src/menu.rs
src-tauri/src/tray.rs
src-tauri/tauri.conf.json
```

## Important topology distinction

The confirmed Mac configuration uses one Desktop-managed `ssh` process with two local forwards:

```text
-L 127.0.0.1:12024:127.0.0.1:2024
-L 127.0.0.1:14732:127.0.0.1:4732
```

The local ports are configuration-driven. Always read `VITE_NEUROTOKEN_LANGGRAPH_LOCAL_PORT` and `VITE_NEUROTOKEN_CODING_LOCAL_PORT`; do not assume they equal the remote ports. On the investigated Mac, local port `2024` belongs to Cursor and is not part of this tunnel.

Its generated command also includes:

```text
-N -T
BatchMode=yes
ControlMaster=no
ControlPath=none
ExitOnForwardFailure=yes
ServerAliveInterval=15
ServerAliveCountMax=3
```

The manually created LLM proxy tunnel is different:

```text
-R 17890:127.0.0.1:7890
```

That reverse tunnel is not created or owned by NeuroToken Desktop. Quitting Desktop must not be expected to terminate a manually started `-R 17890` process.

Before changing code, identify the exact surviving command line.

## Current implementation

### Frontend lifecycle

`NeuroTokenDesktopShell.tsx` creates `SystemSshManager` only when:

```text
VITE_NEUROTOKEN_CODING_RUNTIME=remote
```

On component mount it:

1. subscribes to `neurotoken-ssh-state`;
2. checks whether system SSH is available;
3. calls `neurotoken_ssh_connect` with host and two forwarded ports.

The React effect cleanup only unsubscribes from state events. It does not call `sshManager.disconnect()`.

This is deliberate for ordinary React remounts, but it means WebView teardown itself is not a shutdown safety net. Tunnel cleanup currently depends entirely on the native application exit path.

### Rust tunnel ownership

`src-tauri/src/neurotoken_ssh.rs` stores the active process as:

```text
NeurotokenSshRuntime.child: Option<tokio::process::Child>
```

It also stores:

```text
generation
settings
state
```

`neurotoken_ssh_connect` starts the remote runtime first, spawns the watchdog and its SSH tunnel, waits for the configured local ports (`12024` and `14732` in the confirmed Mac setup) to become healthy, stores the watchdog child, and starts a reconnect monitor.

`disconnect_runtime`:

1. increments `generation` to cancel reconnect loops;
2. clears `settings`;
3. changes state to disconnected;
4. takes the stored child;
5. calls `kill_child_process_tree`;
6. clears the remote backend.

On macOS, `kill_child_process_tree` currently only calls:

```text
child.kill().await
```

There is no explicit process-group creation or process-group kill on Unix.

### Native application lifecycle

In `src-tauri/src/lib.rs`:

- macOS `WindowEvent::CloseRequested` is prevented and the window is hidden.
- Therefore clicking the red traffic-light button does not quit the application and should not close the tunnel.
- Command+Q, application-menu Quit, tray Quit, or another true app-exit request should produce `RunEvent::ExitRequested`.
- `ExitRequested` is prevented once, async cleanup runs, and then `app_handle.exit(0)` is called.
- `stop_managed_daemons_for_exit` always calls `neurotoken_ssh::disconnect_runtime`, independent of the “keep daemon running” setting.

The expected behavior is therefore:

```text
red close button -> app remains alive -> Desktop tunnel remains alive
Command+Q / Quit -> app exits -> Desktop tunnel is killed
```

Confirm that the user's “Desktop exit” means Command+Q/Quit and not merely closing the window.

## Confirmed root cause from the Mac investigation

The surviving Desktop tunnel was PID `69929`, with `PPID=1`, and listened on local ports `12024` and `14732`. Its command line contained both generated `-L` forwards. The original Tauri parent was terminated using `kill`/`pkill`, so it never received `RunEvent::ExitRequested`. The SSH process was reparented to PID 1 and continued running.

This evidence does not show that Command+Q or any normal Quit path is broken. It shows that the previous design had no parent-death protection when the Tauri host was killed, crashed, stopped by an IDE, or otherwise disappeared without asynchronous cleanup.

The implemented hardening launches a lightweight watchdog mode of the Desktop executable. The Desktop retains the watchdog's stdin pipe; the watchdog owns SSH. Kernel pipe closure notifies the watchdog when Desktop disappears, including after `SIGKILL`, and the watchdog terminates and waits for SSH before exiting. The normal `ExitRequested` cleanup remains in place.

## Other root causes to distinguish

Investigate in this order when a future residual process is reported.

### 1. The surviving process is the manual reverse proxy tunnel

If the process command contains `-R 17890`, this is expected and unrelated to the Desktop cleanup path.

### 2. The window was closed but the app was not quit

The macOS close handler explicitly hides the window. The tray/menu-bar process remains active. This is inherited CodexMonitor behavior.

If product requirements say the red close button must terminate NeuroToken's tunnel while keeping the app resident, implement tunnel disconnect on window close and reconnect on reopen. Do not describe this as app-exit cleanup.

### 3. The installed app is stale or is not the binary being tested

The source contains an `ExitRequested` cleanup hook, but an older installed `.app` may not. Confirm executable path, build timestamp, bundle version, and process path before debugging source.

### 4. `ExitRequested` cleanup is not reached for the actual quit path

There is currently no diagnostic logging around:

- received run event;
- exit phase transition;
- SSH generation;
- child PID;
- kill attempt;
- kill result;
- final child wait status.

Without these logs, source review cannot prove the callback executed in the packaged app.

### 5. Killing the direct child is insufficient on macOS

The tunnel is normally the direct `/usr/bin/ssh` child, so killing it should work. However, this is not guaranteed if command resolution or a wrapper changes. Unix process-group ownership would make cleanup stronger and easier to verify.

### 6. Force termination bypasses async cleanup

`kill -9`, crash, system shutdown, forced termination, or abrupt process death cannot rely on asynchronous `RunEvent::ExitRequested` cleanup. `kill_on_drop(true)` helps only if Tokio's child object is dropped through normal Rust destruction; it is not a substitute for OS-level orphan prevention in every crash case.

## Mandatory local reproduction procedure

Use a Development build first so native stderr is visible.

### 1. Confirm environment and built app

From the Mac repository:

```bash
pwd
git status --short
git rev-parse --short HEAD
node --version
npm --version
rustc --version
cargo --version
```

Record the exact app or executable path being launched. For an installed app:

```bash
ps -axo pid,ppid,lstart,command | rg -i 'Codex Monitor|NeuroToken|codex-monitor'
mdls -name kMDItemVersion -name kMDItemFSCreationDate -name kMDItemContentModificationDate "/Applications/Codex Monitor.app"
```

Adjust the application path if the bundle has been renamed.

### 2. Establish a process baseline

Before opening Desktop:

```bash
ps -axo pid,ppid,lstart,command | rg '(^|/)ssh( |$)|ssh -'
lsof -nP -iTCP:12024 -sTCP:LISTEN
lsof -nP -iTCP:14732 -sTCP:LISTEN
lsof -nP -iTCP:17890 -sTCP:LISTEN
```

Save the output. A process with `-R 17890` is the manual LLM proxy tunnel.

### 3. Launch the current source build

Use the project's normal Tauri development command. Ensure the environment includes the remote shell variables used by the app, especially:

```text
VITE_NEUROTOKEN_SHELL
VITE_NEUROTOKEN_CODING_RUNTIME=remote
VITE_NEUROTOKEN_SSH_HOST
VITE_NEUROTOKEN_LANGGRAPH_LOCAL_PORT
VITE_NEUROTOKEN_LANGGRAPH_REMOTE_PORT
VITE_NEUROTOKEN_CODING_LOCAL_PORT
VITE_NEUROTOKEN_CODING_REMOTE_PORT
VITE_NEUROTOKEN_REMOTE_START_COMMAND
```

After the UI reports connected, capture:

```bash
ps -axo pid,ppid,lstart,command | rg 'ssh .*127\.0\.0\.1:(12024|14732)'
lsof -nP -iTCP:12024 -sTCP:LISTEN
lsof -nP -iTCP:14732 -sTCP:LISTEN
```

Expected: exactly one Desktop-owned SSH process whose parent is the Tauri process and whose command contains both `-L` forwards.

### 4. Test red-window close separately

Click the red close button. Then inspect processes again.

Expected under current code: app and Desktop-owned tunnel remain alive because the window is only hidden.

Reopen from Dock/menu bar and confirm the same process is reused rather than a second SSH tunnel being created.

### 5. Test Command+Q separately

With the app visible, press Command+Q. Wait a few seconds, then run:

```bash
ps -axo pid,ppid,lstart,command | rg -i 'Codex Monitor|NeuroToken|codex-monitor|ssh .*127\.0\.0\.1:(12024|14732)'
lsof -nP -iTCP:12024 -sTCP:LISTEN
lsof -nP -iTCP:14732 -sTCP:LISTEN
```

Expected: both the Tauri app and its `-L` SSH child are gone. A separate `-R 17890` process may remain.

### 6. Test all real quit paths

Repeat independently for:

- Command+Q;
- application menu Quit;
- tray/menu-bar Quit;
- Dock “Quit”.

Do not combine this with the red-window close test.

## Required diagnostic instrumentation

Add temporary native logs before attempting another lifecycle fix. Prefix all lines with a stable marker such as `[neurotoken-ssh]`.

Log at least:

### In `spawn_tunnel`

```text
spawn requested: generation/host/ports
spawned pid
health check passed/failed
```

Do not log secrets or private-key contents.

### In `spawn_monitor`

```text
monitor started generation
child exited pid/status
reconnect attempt generation/index
reconnect child pid
monitor stopped because generation changed/settings cleared
```

### In `disconnect_runtime`

```text
disconnect entered generation/current pid
settings cleared/new generation
kill requested pid
kill returned
try_wait/wait result
remote backend cleared
```

### In the `app.run` callback

```text
RunEvent variant relevant to exit
ExitRequested received/current EXIT_PHASE
cleanup task spawned
cleanup complete
app_handle.exit called
second ExitRequested observed after cleanup
```

Run the development executable from Terminal and preserve the full output for one red-close test and one Command+Q test.

## Recommended code hardening after evidence collection

Do not implement all of these blindly. Select based on logs.

### A. Wait for confirmed child exit

After `child.kill().await`, explicitly call `child.wait().await` and log the status. A successful signal request is weaker evidence than observing process termination.

### B. Create and kill a Unix process group

On macOS/Linux, start SSH in its own process group and terminate the group during disconnect. This protects against wrappers or descendants. Keep the Windows `taskkill /T /F` path unchanged.

Be careful to use platform-gated code and avoid unsafe ad-hoc shell commands.

### C. Add native status diagnostics

Extend `neurotoken_ssh_status` during debugging to report non-sensitive fields such as:

```text
state
generation
child_pid
host alias
local ports
```

This allows the UI and local Cursor to compare Rust ownership with `ps` output.

### D. Define close semantics explicitly

Choose one product behavior:

1. Keep current CodexMonitor semantics: red close hides app and preserves tunnel; Quit closes tunnel.
2. Red close hides app but disconnects NeuroToken SSH; reopen reconnects.
3. Red close fully quits NeuroToken Desktop.

The current bug report may be a mismatch between option 1 and an expectation of option 2 or 3.

### E. Add tests at the correct level

Existing TypeScript tests only verify that `disconnect()` invokes the Tauri command. They do not test application exit.

Add Rust unit tests for state transitions where possible, but real process cleanup needs a macOS integration test with a harmless long-lived child process. Verify:

- disconnect cancels reconnect by generation;
- direct child exits;
- process group exits if group ownership is added;
- all Quit paths reach cleanup;
- red close behavior matches the selected product semantics.

## Current test limitation

`src/features/neurotoken/systemSshManager.test.ts` tests only:

- availability command invocation;
- connect command invocation;
- native state forwarding;
- explicit `disconnect()` command invocation.

It does not prove that:

- React cleanup calls disconnect (it currently does not);
- `RunEvent::ExitRequested` fires on macOS;
- the native child dies;
- a packaged `.app` contains the latest cleanup code;
- a surviving `ssh` process is Desktop-owned.

## Built-in SSH lifecycle diagnostics

The NeuroToken shell now has a **Copy SSH diagnostics** button next to the SSH status in the sidebar footer. It exports:

- the absolute persistent log path;
- Desktop and watchdog PIDs;
- runtime generation and connection state;
- non-secret host and port settings;
- the most recent 128 KiB of the shared Desktop/watchdog/SSH lifecycle log.

The log is named `neurotoken-ssh.log` under Tauri's application log directory. With the current macOS bundle identifier it is normally under `~/Library/Logs/com.dimillian.codexmonitor/`, but use the exported `log_path` as the authority.

After reproducing a Command+Q leak:

1. Reopen Desktop without manually killing the leaked tunnel.
2. Click **Copy SSH diagnostics** and paste the result into local Cursor or the main development session.
3. Also inspect the exported `log_path` directly if a larger timeline is needed.
4. Capture `lsof`/`ps` output for the leaked PID as described below.

The decisive lifecycle markers are:

- `RunEvent::ExitRequested received; cleanup scheduled`;
- `disconnect_runtime reason=application exit requested`;
- `stop requested ... closing parent pipe`;
- watchdog trigger: `parent pipe closed`, `Desktop parent process disappeared`, or `termination signal received`;
- `ssh stop completed ... wait_status=...`;
- `exit cleanup completed; calling app_handle.exit(0)`.

If an earlier marker exists but the next one does not, the exact failing boundary is identified. SSH stderr is appended to the same file. Secrets, API keys, private keys, and remote start command contents are not included in the exported structured settings.

## Evidence to return

Return all of the following to the main development session:

1. Exact app executable path and build timestamp.
2. Git commit checked out on the Mac.
3. Environment variable names and non-secret values for runtime mode, host alias, and ports.
4. `ps` output before launch, after connect, after red close, and after Command+Q.
5. `lsof` output for configured local ports 12024, 14732, and the manual proxy port 17890 at the same points.
6. Native diagnostic logs for connect and quit.
7. Identification of each surviving SSH PID as one of:
   - Desktop `-L` tunnel;
   - manual `-R 17890` tunnel;
   - unrelated SSH session;
   - unknown.
8. The exact quit path used.
9. Whether the Tauri parent process also remained alive.
10. A minimal root-cause statement tied to observed PID/PPID and logs.

## Acceptance criteria

The issue is resolved only when all of the following are demonstrated on the Mac:

- Launching Desktop creates exactly one SSH process with both local forwards.
- Reopening a hidden window does not create duplicate tunnel processes.
- Command+Q and every supported Quit command terminate the Desktop-owned SSH PID.
- The configured local ports (12024 and 14732 in the confirmed Mac setup) are no longer listening after true app exit or forced host termination.
- The manually managed `-R 17890` tunnel is correctly identified and is not confused with Desktop ownership.
- Behavior of the red close button matches the explicitly selected product requirement.
- Logs show that cleanup completed before final application exit.
