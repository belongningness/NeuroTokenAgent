use serde::Serialize;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::Child;
use tokio::time::{sleep, timeout};

use crate::shared::neurotoken_ssh_core::{
    availability_args, remote_start_args, tunnel_args, NeurotokenSshSettings,
};
use crate::shared::process_core::{kill_child_process_tree, tokio_command};
use crate::state::AppState;

const WATCHDOG_ARGUMENT: &str = "--neurotoken-ssh-watchdog";
const WATCHDOG_PARENT_PID_ARGUMENT: &str = "--neurotoken-parent-pid";
const WATCHDOG_LOG_PATH_ARGUMENT: &str = "--neurotoken-log-path";

const STATE_EVENT: &str = "neurotoken-ssh-state";
const START_TIMEOUT: Duration = Duration::from_secs(45);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);
const RECONNECT_DELAYS: [u64; 5] = [1, 2, 5, 10, 20];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub(crate) enum NeurotokenSshState {
    Disconnected,
    Connecting,
    Connected,
    Error { message: String },
}

pub(crate) struct NeurotokenSshRuntime {
    pub(crate) child: Option<Child>,
    pub(crate) generation: u64,
    pub(crate) settings: Option<NeurotokenSshSettings>,
    pub(crate) state: NeurotokenSshState,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NeurotokenSshDiagnosticSettings {
    host: String,
    lang_graph_local_port: u16,
    lang_graph_remote_port: u16,
    coding_local_port: u16,
    coding_remote_port: u16,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NeurotokenSshDiagnostics {
    log_path: String,
    desktop_pid: u32,
    watchdog_pid: Option<u32>,
    generation: u64,
    state: NeurotokenSshState,
    settings: Option<NeurotokenSshDiagnosticSettings>,
    recent_log: String,
}

impl Default for NeurotokenSshRuntime {
    fn default() -> Self {
        Self {
            child: None,
            generation: 0,
            settings: None,
            state: NeurotokenSshState::Disconnected,
        }
    }
}

fn emit_state(app: &AppHandle, state: &NeurotokenSshState) {
    let _ = app.emit(STATE_EVENT, state);
}

async fn tcp_healthy(port: u16) -> bool {
    timeout(
        Duration::from_secs(2),
        TcpStream::connect(("127.0.0.1", port)),
    )
    .await
    .is_ok_and(|result| result.is_ok())
}

async fn wait_for_health(settings: &NeurotokenSshSettings) -> Result<(), String> {
    let deadline = tokio::time::Instant::now() + HEALTH_TIMEOUT;
    loop {
        if tcp_healthy(settings.lang_graph_local_port).await
            && tcp_healthy(settings.coding_local_port).await
        {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(
                "SSH tunnel opened, but remote services did not become healthy in time."
                    .to_string(),
            );
        }
        sleep(Duration::from_millis(400)).await;
    }
}

async fn start_remote_runtime(settings: &NeurotokenSshSettings) -> Result<(), String> {
    let args = remote_start_args(settings)?;
    let output = timeout(START_TIMEOUT, {
        let mut command = tokio_command("ssh");
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        command.output()
    })
    .await
    .map_err(|_| "Timed out starting the remote NeuroToken runtime.".to_string())?
    .map_err(|err| format!("Failed to execute system ssh: {err}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(if stderr.is_empty() {
        "Remote NeuroToken runtime start command failed.".to_string()
    } else {
        stderr
    })
}

async fn ensure_local_ports_available(settings: &NeurotokenSshSettings) -> Result<(), String> {
    let ports = [settings.lang_graph_local_port, settings.coding_local_port];
    for port in ports {
        if tcp_healthy(port).await {
            return Err(format!(
                "Local port {port} is already in use. Stop the existing SSH tunnel or choose another local port."
            ));
        }
    }
    Ok(())
}

async fn spawn_tunnel(app: &AppHandle, settings: &NeurotokenSshSettings) -> Result<Child, String> {
    ensure_local_ports_available(settings).await?;
    let ssh_args = tunnel_args(settings)?;
    let log_path = crate::neurotoken_ssh_log::log_path(app)?;
    crate::neurotoken_ssh_log::append(
        &log_path,
        "desktop",
        format!(
            "spawning watchdog parent_pid={} host={} langgraph={}:{} coding={}:{}",
            std::process::id(),
            settings.host,
            settings.lang_graph_local_port,
            settings.lang_graph_remote_port,
            settings.coding_local_port,
            settings.coding_remote_port,
        ),
    );
    let executable = std::env::current_exe()
        .map_err(|err| format!("Failed to resolve Desktop executable for SSH watchdog: {err}"))?;
    let mut command = tokio_command(executable);
    command
        .arg(WATCHDOG_ARGUMENT)
        .arg(WATCHDOG_PARENT_PID_ARGUMENT)
        .arg(std::process::id().to_string())
        .arg(WATCHDOG_LOG_PATH_ARGUMENT)
        .arg(&log_path)
        .arg("--")
        .args(ssh_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    unsafe {
        command.as_std_mut().pre_exec(|| {
            if libc::setpgid(0, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|err| format!("Failed to start SSH watchdog: {err}"))?;
    crate::neurotoken_ssh_log::append(
        &log_path,
        "desktop",
        format!("watchdog spawned watchdog_pid={:?}", child.id()),
    );

    sleep(Duration::from_millis(300)).await;
    if let Some(status) = child
        .try_wait()
        .map_err(|err| format!("Failed to inspect SSH watchdog: {err}"))?
    {
        let mut stderr = String::new();
        if let Some(mut pipe) = child.stderr.take() {
            let _ = pipe.read_to_string(&mut stderr).await;
        }
        let detail = stderr.trim();
        return Err(if detail.is_empty() {
            format!("SSH watchdog exited immediately with {status}.")
        } else {
            detail.to_string()
        });
    }
    if let Some(mut stderr) = child.stderr.take() {
        tauri::async_runtime::spawn(async move {
            let mut output = String::new();
            if stderr.read_to_string(&mut output).await.is_ok() && !output.trim().is_empty() {
                eprintln!("[neurotoken-ssh-watchdog] {}", output.trim());
            }
        });
    }
    Ok(child)
}

async fn stop_tunnel(app: &AppHandle, child: &mut Child, reason: &str) {
    let pid = child.id();
    if let Ok(log_path) = crate::neurotoken_ssh_log::log_path(app) {
        crate::neurotoken_ssh_log::append(
            &log_path,
            "desktop",
            format!("stop requested reason={reason} watchdog_pid={pid:?}; closing parent pipe"),
        );
    }
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.shutdown().await;
    }
    match timeout(Duration::from_secs(5), child.wait()).await {
        Ok(Ok(status)) => {
            if let Ok(path) = crate::neurotoken_ssh_log::log_path(app) {
                crate::neurotoken_ssh_log::append(
                    &path,
                    "desktop",
                    format!("watchdog_pid={pid:?} exited status={status}"),
                );
            }
        }
        Ok(Err(error)) => {
            if let Ok(path) = crate::neurotoken_ssh_log::log_path(app) {
                crate::neurotoken_ssh_log::append(
                    &path,
                    "desktop",
                    format!("watchdog_pid={pid:?} wait failed error={error}"),
                );
            }
        }
        Err(_) => {
            if let Ok(path) = crate::neurotoken_ssh_log::log_path(app) {
                crate::neurotoken_ssh_log::append(&path, "desktop", format!("watchdog_pid={pid:?} did not exit in 5s; forcing process-group termination"));
            }
            #[cfg(unix)]
            if let Some(pid) = pid {
                unsafe {
                    libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
                }
            }
            kill_child_process_tree(child).await;
        }
    }
}

async fn connect_once(app: &AppHandle, settings: &NeurotokenSshSettings) -> Result<Child, String> {
    start_remote_runtime(settings).await?;
    let mut child = spawn_tunnel(app, settings).await?;
    if let Err(err) = wait_for_health(settings).await {
        stop_tunnel(app, &mut child, "health check failed").await;
        return Err(err);
    }
    if let Ok(path) = crate::neurotoken_ssh_log::log_path(app) {
        crate::neurotoken_ssh_log::append(&path, "desktop", "SSH tunnel health checks passed");
    }
    Ok(child)
}

fn spawn_monitor(app: AppHandle, generation: u64) {
    tauri::async_runtime::spawn(async move {
        loop {
            sleep(Duration::from_millis(500)).await;
            let settings = {
                let state = app.state::<AppState>();
                let mut runtime = state.neurotoken_ssh.lock().await;
                if runtime.generation != generation {
                    return;
                }
                let Some(child) = runtime.child.as_mut() else {
                    return;
                };
                match child.try_wait() {
                    Ok(Some(_)) | Err(_) => {
                        runtime.child = None;
                        runtime.settings.clone()
                    }
                    Ok(None) => continue,
                }
            };
            let Some(settings) = settings else {
                return;
            };

            let mut attempt = 0usize;
            loop {
                let delay = RECONNECT_DELAYS[attempt.min(RECONNECT_DELAYS.len() - 1)];
                sleep(Duration::from_secs(delay)).await;
                {
                    let state = app.state::<AppState>();
                    let runtime = state.neurotoken_ssh.lock().await;
                    if runtime.generation != generation || runtime.settings.is_none() {
                        return;
                    }
                }
                match connect_once(&app, &settings).await {
                    Ok(mut child) => {
                        let connected = NeurotokenSshState::Connected;
                        let state = app.state::<AppState>();
                        let mut runtime = state.neurotoken_ssh.lock().await;
                        if runtime.generation != generation || runtime.settings.is_none() {
                            drop(runtime);
                            stop_tunnel(&app, &mut child, "connection cancelled").await;
                            return;
                        }
                        runtime.child = Some(child);
                        runtime.state = connected.clone();
                        emit_state(&app, &connected);
                        break;
                    }
                    Err(message) => {
                        let failed = NeurotokenSshState::Error { message };
                        let state = app.state::<AppState>();
                        let mut runtime = state.neurotoken_ssh.lock().await;
                        if runtime.generation != generation {
                            return;
                        }
                        runtime.state = failed.clone();
                        emit_state(&app, &failed);
                        attempt += 1;
                    }
                }
            }
        }
    });
}

#[tauri::command]
pub(crate) async fn neurotoken_ssh_available() -> Result<bool, String> {
    match tokio_command("ssh")
        .args(availability_args())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
    {
        Ok(_) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(format!("Failed to check system ssh: {err}")),
    }
}

#[tauri::command]
pub(crate) async fn neurotoken_ssh_connect(
    settings: NeurotokenSshSettings,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<(), String> {
    settings.validate()?;
    let generation = {
        let mut runtime = state.neurotoken_ssh.lock().await;
        runtime.generation += 1;
        runtime.settings = Some(settings.clone());
        runtime.state = NeurotokenSshState::Connecting;
        if let Some(mut child) = runtime.child.take() {
            stop_tunnel(&app, &mut child, "connection cancelled").await;
        }
        emit_state(&app, &runtime.state);
        runtime.generation
    };

    match connect_once(&app, &settings).await {
        Ok(mut child) => {
            {
                let mut runtime = state.neurotoken_ssh.lock().await;
                if runtime.generation != generation || runtime.settings.is_none() {
                    drop(runtime);
                    stop_tunnel(&app, &mut child, "connection cancelled").await;
                    return Err("SSH connection was cancelled.".to_string());
                }
                runtime.child = Some(child);
                runtime.state = NeurotokenSshState::Connected;
                emit_state(&app, &runtime.state);
            }
            *state.remote_backend.lock().await = None;
            spawn_monitor(app, generation);
            Ok(())
        }
        Err(message) => {
            let failed = NeurotokenSshState::Error {
                message: message.clone(),
            };
            let mut runtime = state.neurotoken_ssh.lock().await;
            if runtime.generation == generation && runtime.settings.is_some() {
                runtime.state = failed.clone();
                emit_state(&app, &failed);
            }
            Err(message)
        }
    }
}

pub(crate) async fn disconnect_runtime(state: &AppState, app: &AppHandle, reason: &str) {
    let (child, generation) = {
        let mut runtime = state.neurotoken_ssh.lock().await;
        runtime.generation += 1;
        runtime.settings = None;
        runtime.state = NeurotokenSshState::Disconnected;
        (runtime.child.take(), runtime.generation)
    };
    if let Ok(path) = crate::neurotoken_ssh_log::log_path(app) {
        crate::neurotoken_ssh_log::append(
            &path,
            "desktop",
            format!(
                "disconnect_runtime reason={reason} generation={generation} watchdog_present={}",
                child.is_some()
            ),
        );
    }
    if let Some(mut child) = child {
        stop_tunnel(app, &mut child, reason).await;
    }
    *state.remote_backend.lock().await = None;
    emit_state(app, &NeurotokenSshState::Disconnected);
}

#[tauri::command]
pub(crate) async fn neurotoken_ssh_disconnect(
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<(), String> {
    disconnect_runtime(&state, &app, "user requested disconnect").await;
    Ok(())
}

#[tauri::command]
pub(crate) async fn neurotoken_ssh_status(
    state: State<'_, AppState>,
) -> Result<NeurotokenSshState, String> {
    Ok(state.neurotoken_ssh.lock().await.state.clone())
}

#[tauri::command]
pub(crate) async fn neurotoken_ssh_diagnostics(
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<NeurotokenSshDiagnostics, String> {
    let path = crate::neurotoken_ssh_log::log_path(&app)?;
    let runtime = state.neurotoken_ssh.lock().await;
    let recent_log = if path.exists() {
        crate::neurotoken_ssh_log::read_recent(&path)?
    } else {
        "No SSH lifecycle events have been recorded yet.\n".to_string()
    };
    Ok(NeurotokenSshDiagnostics {
        log_path: path.to_string_lossy().into_owned(),
        desktop_pid: std::process::id(),
        watchdog_pid: runtime.child.as_ref().and_then(Child::id),
        generation: runtime.generation,
        state: runtime.state.clone(),
        settings: runtime
            .settings
            .as_ref()
            .map(|settings| NeurotokenSshDiagnosticSettings {
                host: settings.host.clone(),
                lang_graph_local_port: settings.lang_graph_local_port,
                lang_graph_remote_port: settings.lang_graph_remote_port,
                coding_local_port: settings.coding_local_port,
                coding_remote_port: settings.coding_remote_port,
            }),
        recent_log,
    })
}
