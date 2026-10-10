use serde_json::{json, Value};

pub(crate) fn workspace_id(task_id: &str) -> String {
    format!("neurotoken:{task_id}")
}

pub(crate) fn response_error(value: &Value) -> Option<String> {
    value
        .get("error")
        .and_then(|error| error.get("message").or(Some(error)))
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub(crate) fn extract_thread_id(value: &Value) -> Option<String> {
    value
        .get("result")
        .and_then(|result| result.get("threadId"))
        .or_else(|| {
            value
                .get("result")
                .and_then(|result| result.get("thread"))
                .and_then(|thread| thread.get("id"))
        })
        .or_else(|| value.get("threadId"))
        .or_else(|| value.get("thread").and_then(|thread| thread.get("id")))
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub(crate) fn thread_resume_params(thread_id: &str) -> Value {
    json!({ "threadId": thread_id })
}

pub(crate) fn thread_start_params(worktree_path: &str) -> Value {
    json!({
        "cwd": worktree_path,
        "approvalPolicy": "on-request"
    })
}

pub(crate) fn turn_start_params(
    thread_id: &str,
    text: &str,
    worktree_path: &str,
    allowed_paths: &[String],
    model: Option<&str>,
    effort: Option<&str>,
) -> Value {
    let _ = allowed_paths;
    let mut value = json!({
    "threadId": thread_id,
    "input": [{ "type": "text", "text": text }],
    "cwd": worktree_path,
    "approvalPolicy": "never",
        "sandboxPolicy": {
            "type": "dangerFullAccess"
        }
    });
    if let Some(model) = model {
        value["model"] = json!(model);
    }
    if let Some(effort) = effort {
        value["effort"] = json!(effort);
    }
    value
}

pub(crate) fn session_status_for_event(message: &Value) -> Option<&'static str> {
    let method = message.get("method").and_then(Value::as_str)?;
    if method.ends_with("requestApproval") || method == "item/tool/requestUserInput" {
        return Some("awaiting_approval");
    }
    match method {
        "turn/completed" | "turn/failed" | "turn/cancelled" => Some("idle"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_thread_ids_from_supported_response_shapes() {
        for response in [
            json!({"result": {"threadId": "thread-1"}}),
            json!({"result": {"thread": {"id": "thread-2"}}}),
            json!({"threadId": "thread-3"}),
        ] {
            assert!(extract_thread_id(&response).is_some());
        }
        assert!(extract_thread_id(&json!({"result": {}})).is_none());
    }

    #[test]
    fn detects_resume_failure_for_fallback() {
        let failure = json!({"error": {"message": "thread not found"}});
        assert_eq!(
            response_error(&failure).as_deref(),
            Some("thread not found")
        );
        assert!(response_error(&json!({"result": {"threadId": "ok"}})).is_none());
    }

    #[test]
    fn constrains_turn_to_allowed_writable_roots() {
        let params = turn_start_params(
            "thread-1",
            "Implement adapter",
            "/tmp/worktree",
            &["agent_pipeline/adapters".to_string()],
            Some("gpt-5"),
            Some("high"),
        );
        assert_eq!(params["approvalPolicy"], "never");
        assert_eq!(params["sandboxPolicy"]["type"], "dangerFullAccess");
        assert_eq!(params["model"], "gpt-5");
        assert_eq!(params["effort"], "high");
    }

    #[test]
    fn maps_approval_and_terminal_events_to_durable_status() {
        for method in [
            "item/commandExecution/requestApproval",
            "workspace/requestApproval",
            "item/tool/requestUserInput",
        ] {
            assert_eq!(
                session_status_for_event(&json!({"method": method})),
                Some("awaiting_approval")
            );
        }
        assert_eq!(
            session_status_for_event(&json!({"method": "turn/completed"})),
            Some("idle")
        );
        assert_eq!(
            session_status_for_event(&json!({"method": "item/agentMessage/delta"})),
            None
        );
    }

    #[test]
    fn builds_stable_task_workspace_and_thread_requests() {
        assert_eq!(workspace_id("task-1"), "neurotoken:task-1");
        assert_eq!(
            thread_resume_params("thread-1"),
            json!({"threadId": "thread-1"})
        );
        assert_eq!(
            thread_start_params("/tmp/worktree"),
            json!({"cwd": "/tmp/worktree", "approvalPolicy": "on-request"})
        );
    }
}
