use super::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TaskRequest {
    task_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenTaskRequest {
    task_id: String,
    revision: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HistoryRequest {
    task_id: String,
    revision: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileRequest {
    task_id: String,
    path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileWriteRequest {
    task_id: String,
    path: String,
    content: String,
    expected_content_hash: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexConnectRequest {
    task_id: String,
    revision: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexTurnRequest {
    task_id: String,
    revision: u32,
    text: String,
    model: Option<String>,
    effort: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexApprovalRequest {
    task_id: String,
    revision: u32,
    request_id: Value,
    result: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexInterruptRequest {
    task_id: String,
    revision: u32,
    turn_id: String,
}

fn parse<T: serde::de::DeserializeOwned>(params: &Value) -> Result<T, String> {
    serde_json::from_value(params.clone()).map_err(|err| err.to_string())
}

fn serialize<T: Serialize>(value: T) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|err| err.to_string())
}

async fn connect_codex(
    state: &DaemonState,
    request: CodexConnectRequest,
    client_version: &str,
) -> Result<Value, String> {
    let bridge_session = state
        .neurotoken_bridge
        .open_task(&request.task_id, request.revision)?;
    if let Some(live) = state.neurotoken_live_session(&request.task_id).await {
        if live.revision == request.revision {
            if let Some(thread_id) = bridge_session.codex_thread_id.clone() {
                return Ok(json!({
                    "session": bridge_session,
                    "codexThreadId": thread_id,
                    "recovered": true,
                    "reusedProcess": true
                }));
            }
        }
        state.neurotoken_remove_session(&request.task_id).await;
    }
    state.neurotoken_bridge.update_session(
        &request.task_id,
        request.revision,
        if bridge_session.codex_thread_id.is_some() {
            "recovering"
        } else {
            "starting"
        },
        None,
    )?;
    let session = match state
        .neurotoken_ensure_session(&request.task_id, client_version)
        .await
    {
        Ok(session) => session,
        Err(error) => {
            let _ = state.neurotoken_bridge.update_session(
                &request.task_id,
                request.revision,
                "error",
                None,
            );
            return Err(error);
        }
    };
    let task = state.neurotoken_bridge.task(&request.task_id)?;
    let workspace_id = neurotoken_codex_core::workspace_id(&request.task_id);
    let mut recovered = false;
    let thread_response = if let Some(thread_id) = bridge_session.codex_thread_id.as_deref() {
        let response = session
            .send_request_for_workspace(
                &workspace_id,
                "thread/resume",
                neurotoken_codex_core::thread_resume_params(thread_id),
            )
            .await?;
        if neurotoken_codex_core::response_error(&response).is_none() {
            recovered = true;
            response
        } else {
            session
                .send_request_for_workspace(
                    &workspace_id,
                    "thread/start",
                    neurotoken_codex_core::thread_start_params(&task.worktree_path),
                )
                .await?
        }
    } else {
        session
            .send_request_for_workspace(
                &workspace_id,
                "thread/start",
                neurotoken_codex_core::thread_start_params(&task.worktree_path),
            )
            .await?
    };
    if let Some(error) = neurotoken_codex_core::response_error(&thread_response) {
        let _ = state.neurotoken_bridge.update_session(
            &request.task_id,
            request.revision,
            "error",
            None,
        );
        return Err(error);
    }
    let thread_id = neurotoken_codex_core::extract_thread_id(&thread_response)
        .ok_or_else(|| "Codex thread response did not include a thread id".to_string())?;
    let bridge_session = state.neurotoken_bridge.update_session(
        &request.task_id,
        request.revision,
        "idle",
        Some(thread_id.clone()),
    )?;
    if bridge_session.turn_started {
        if let Some(live) = state.neurotoken_live_session(&request.task_id).await {
            if live.revision == request.revision {
                live.capture_events.store(true, Ordering::Release);
            }
        }
    }
    Ok(json!({
        "session": bridge_session,
        "codexThreadId": thread_id,
        "recovered": recovered,
        "thread": thread_response
    }))
}

async fn start_turn(
    state: &DaemonState,
    request: CodexTurnRequest,
    client_version: &str,
) -> Result<Value, String> {
    let has_current_session = state
        .neurotoken_live_session(&request.task_id)
        .await
        .is_some_and(|live| live.revision == request.revision);
    if !has_current_session {
        connect_codex(
            state,
            CodexConnectRequest {
                task_id: request.task_id.clone(),
                revision: request.revision,
            },
            client_version,
        )
        .await?;
    }
    let bridge_session = state
        .neurotoken_bridge
        .session(&request.task_id)?
        .ok_or_else(|| format!("NeuroToken task session is not open: {}", request.task_id))?;
    if bridge_session.revision != request.revision {
        return Err(format!(
            "Stale NeuroToken Bridge session revision: expected {}, received {}",
            bridge_session.revision, request.revision
        ));
    }
    let thread_id = bridge_session
        .codex_thread_id
        .ok_or_else(|| "NeuroToken task has no Codex thread".to_string())?;
    let task = state.neurotoken_bridge.task(&request.task_id)?;
    if !matches!(task.status.as_str(), "ready_for_coding" | "coding") {
        return Err(format!(
            "NeuroToken task is not accepting Codex turns: {}",
            task.status
        ));
    }
    let session = state
        .neurotoken_ensure_session(&request.task_id, client_version)
        .await?;
    let live = state
        .neurotoken_live_session(&request.task_id)
        .await
        .filter(|live| live.revision == request.revision)
        .ok_or_else(|| "NeuroToken Codex process is not running".to_string())?;
    state
        .neurotoken_bridge
        .begin_turn(&request.task_id, request.revision)?;
    live.capture_events.store(true, Ordering::Release);
    let turn = neurotoken_codex_core::turn_start_params(
        &thread_id,
        &request.text,
        &task.worktree_path,
        &task.allowed_paths,
        request.model.as_deref(),
        request.effort.as_deref(),
    );
    state
        .neurotoken_bridge
        .update_session(&request.task_id, request.revision, "running", None)?;
    let response = session
        .send_request_for_workspace(
            &neurotoken_codex_core::workspace_id(&request.task_id),
            "turn/start",
            turn,
        )
        .await;
    match response {
        Ok(value) => {
            if let Some(error) = neurotoken_codex_core::response_error(&value) {
                let _ = state.neurotoken_bridge.update_session(
                    &request.task_id,
                    request.revision,
                    "error",
                    None,
                );
                Err(error)
            } else {
                Ok(value)
            }
        }
        Err(error) => {
            let _ = state.neurotoken_bridge.update_session(
                &request.task_id,
                request.revision,
                "error",
                None,
            );
            Err(error)
        }
    }
}

pub(super) async fn try_handle(
    state: &DaemonState,
    method: &str,
    params: &Value,
    client_version: &str,
) -> Option<Result<Value, String>> {
    match method {
        "neurotoken_codex_connect" => {
            let request = match parse::<CodexConnectRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(connect_codex(state, request, client_version).await)
        }
        "neurotoken_codex_turn_start" => {
            let request = match parse::<CodexTurnRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(start_turn(state, request, client_version).await)
        }
        "neurotoken_codex_approval_respond" => {
            let request = match parse::<CodexApprovalRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let current = match state.neurotoken_bridge.session(&request.task_id) {
                Ok(Some(value)) => value,
                Ok(None) => {
                    return Some(Err(format!(
                        "NeuroToken task session is not open: {}",
                        request.task_id
                    )))
                }
                Err(err) => return Some(Err(err)),
            };
            if current.revision != request.revision {
                return Some(Err("Stale NeuroToken approval revision".to_string()));
            }
            let session = match state.neurotoken_live_session(&request.task_id).await {
                Some(value) if value.revision == request.revision => value.session,
                _ => return Some(Err("NeuroToken Codex process is not running".to_string())),
            };
            let result = session
                .send_response(request.request_id, request.result)
                .await
                .map(|_| json!({ "ok": true }));
            if result.is_ok() {
                let _ = state.neurotoken_bridge.update_session(
                    &request.task_id,
                    request.revision,
                    "running",
                    None,
                );
            }
            Some(result)
        }
        "neurotoken_codex_turn_interrupt" => {
            let request = match parse::<CodexInterruptRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            let current = match state.neurotoken_bridge.session(&request.task_id) {
                Ok(Some(value)) => value,
                Ok(None) => return Some(Err("NeuroToken task session is not open".to_string())),
                Err(err) => return Some(Err(err)),
            };
            if current.revision != request.revision {
                return Some(Err("Stale NeuroToken interrupt revision".to_string()));
            }
            let thread_id = match current.codex_thread_id {
                Some(value) => value,
                None => return Some(Err("NeuroToken task has no Codex thread".to_string())),
            };
            let session = match state.neurotoken_live_session(&request.task_id).await {
                Some(value) if value.revision == request.revision => value.session,
                _ => return Some(Err("NeuroToken Codex process is not running".to_string())),
            };
            Some(
                session
                    .send_request_for_workspace(
                        &neurotoken_codex_core::workspace_id(&request.task_id),
                        "turn/interrupt",
                        json!({ "threadId": thread_id, "turnId": request.turn_id }),
                    )
                    .await,
            )
        }
        "neurotoken_codex_history" => {
            let request = match parse::<HistoryRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(
                state
                    .neurotoken_bridge
                    .events(&request.task_id, request.revision)
                    .and_then(serialize),
            )
        }
        "neurotoken_codex_disconnect" => {
            let request = match parse::<TaskRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            state.neurotoken_remove_session(&request.task_id).await;
            Some(Ok(json!({ "ok": true })))
        }
        "neurotoken_task_open" => {
            let request = match parse::<OpenTaskRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(
                state
                    .neurotoken_bridge
                    .open_task(&request.task_id, request.revision)
                    .and_then(serialize),
            )
        }
        "neurotoken_session_read" => {
            let request = match parse::<TaskRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(
                state
                    .neurotoken_bridge
                    .session(&request.task_id)
                    .and_then(serialize),
            )
        }
        "neurotoken_file_list" => {
            let request = match parse::<TaskRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(
                state
                    .neurotoken_bridge
                    .list_files(&request.task_id)
                    .and_then(serialize),
            )
        }
        "neurotoken_file_read" => {
            let request = match parse::<FileRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(
                state
                    .neurotoken_bridge
                    .read_file(&request.task_id, &request.path)
                    .and_then(serialize),
            )
        }
        "neurotoken_file_write" => {
            let request = match parse::<FileWriteRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(
                state
                    .neurotoken_bridge
                    .write_file(
                        &request.task_id,
                        &request.path,
                        &request.content,
                        &request.expected_content_hash,
                    )
                    .and_then(serialize),
            )
        }
        "neurotoken_diff" => {
            let request = match parse::<TaskRequest>(params) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };
            Some(
                state
                    .neurotoken_bridge
                    .diff(&request.task_id)
                    .map(|diff| json!({ "diff": diff })),
            )
        }
        _ => None,
    }
}
