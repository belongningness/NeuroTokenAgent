use serde_json::{json, Value};
use tauri::{AppHandle, State};

use crate::remote_backend;
use crate::state::AppState;

async fn call(
    state: &AppState,
    app: AppHandle,
    method: &str,
    params: Value,
) -> Result<Value, String> {
    if !remote_backend::is_remote_mode(state).await {
        return Err("NeuroToken coding requires the remote backend".to_string());
    }
    remote_backend::call_remote(state, app, method, params).await
}

#[tauri::command]
pub(crate) async fn neurotoken_coding_open(
    task_id: String,
    revision: u32,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<Value, String> {
    call(
        &state,
        app,
        "neurotoken_codex_connect",
        json!({ "taskId": task_id, "revision": revision }),
    )
    .await
}

#[tauri::command]
pub(crate) async fn neurotoken_coding_history(
    task_id: String,
    revision: u32,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<Value, String> {
    call(
        &state,
        app,
        "neurotoken_codex_history",
        json!({ "taskId": task_id, "revision": revision }),
    )
    .await
}

#[tauri::command]
pub(crate) async fn neurotoken_coding_turn_start(
    task_id: String,
    revision: u32,
    text: String,
    model: Option<String>,
    effort: Option<String>,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<Value, String> {
    call(
        &state,
        app,
        "neurotoken_codex_turn_start",
        json!({
            "taskId": task_id,
            "revision": revision,
            "text": text,
            "model": model,
            "effort": effort,
        }),
    )
    .await
}

#[tauri::command]
pub(crate) async fn neurotoken_coding_approve(
    task_id: String,
    revision: u32,
    request_id: Value,
    decision: String,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<Value, String> {
    let result = match decision.as_str() {
        "allow" => json!({ "decision": "accept" }),
        "deny" => json!({ "decision": "decline" }),
        _ => return Err("Unsupported NeuroToken tool decision".to_string()),
    };
    call(
        &state,
        app,
        "neurotoken_codex_approval_respond",
        json!({
            "taskId": task_id,
            "revision": revision,
            "requestId": request_id,
            "result": result,
        }),
    )
    .await
}

#[tauri::command]
pub(crate) async fn neurotoken_coding_list_files(
    task_id: String,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<Value, String> {
    call(
        &state,
        app,
        "neurotoken_file_list",
        json!({ "taskId": task_id }),
    )
    .await
}

#[tauri::command]
pub(crate) async fn neurotoken_coding_read_file(
    task_id: String,
    path: String,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<Value, String> {
    call(
        &state,
        app,
        "neurotoken_file_read",
        json!({ "taskId": task_id, "path": path }),
    )
    .await
}

#[tauri::command]
pub(crate) async fn neurotoken_coding_write_file(
    task_id: String,
    path: String,
    content: String,
    expected_content_hash: String,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<Value, String> {
    call(
        &state,
        app,
        "neurotoken_file_write",
        json!({
            "taskId": task_id,
            "path": path,
            "content": content,
            "expectedContentHash": expected_content_hash,
        }),
    )
    .await
}

#[tauri::command]
pub(crate) async fn neurotoken_coding_diff(
    task_id: String,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<Value, String> {
    call(&state, app, "neurotoken_diff", json!({ "taskId": task_id })).await
}
