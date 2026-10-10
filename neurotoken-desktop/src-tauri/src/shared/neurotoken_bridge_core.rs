use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(1);

const REGISTRY_VERSION: &str = "1.0";
const SESSION_VERSION: &str = "1.0";

#[derive(Debug, Clone, Deserialize)]
struct TaskRegistry {
    schema_version: String,
    tasks: HashMap<String, TaskRecord>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TaskRecord {
    pub(crate) task_id: String,
    pub(crate) thread_id: String,
    pub(crate) revision: u32,
    pub(crate) status: String,
    pub(crate) allowed_paths: Vec<String>,
    pub(crate) instruction: String,
    pub(crate) worktree_path: String,
    pub(crate) baseline_path: String,
    #[serde(default)]
    pub(crate) codex_thread_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BridgeSession {
    pub(crate) task_id: String,
    pub(crate) thread_id: String,
    pub(crate) revision: u32,
    pub(crate) status: String,
    pub(crate) codex_thread_id: Option<String>,
    #[serde(default)]
    pub(crate) turn_started: bool,
    pub(crate) updated_at_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionRegistry {
    schema_version: String,
    sessions: HashMap<String, BridgeSession>,
    #[serde(default)]
    events: HashMap<String, Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BridgeFile {
    pub(crate) path: String,
    pub(crate) content: String,
    pub(crate) content_hash: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum WriteResult {
    Saved { file: BridgeFile },
    Conflict { current: BridgeFile },
}

pub(crate) struct NeurotokenBridge {
    registry_path: Option<PathBuf>,
    sessions_path: PathBuf,
    state_lock: Mutex<()>,
}

impl NeurotokenBridge {
    pub(crate) fn new(registry_path: Option<PathBuf>, data_dir: &Path) -> Self {
        Self {
            registry_path,
            sessions_path: data_dir.join("neurotoken-sessions.json"),
            state_lock: Mutex::new(()),
        }
    }

    pub(crate) fn task(&self, task_id: &str) -> Result<TaskRecord, String> {
        validate_task_id(task_id)?;
        let registry_path = self
            .registry_path
            .as_ref()
            .ok_or_else(|| "NeuroToken task registry is not configured".to_string())?;
        let payload = fs::read_to_string(registry_path)
            .map_err(|err| format!("Failed to read NeuroToken task registry: {err}"))?;
        let registry: TaskRegistry = serde_json::from_str(&payload)
            .map_err(|err| format!("Invalid NeuroToken task registry: {err}"))?;
        if registry.schema_version != REGISTRY_VERSION {
            return Err(format!(
                "Unsupported NeuroToken task registry version: {}",
                registry.schema_version
            ));
        }
        let task = registry
            .tasks
            .get(task_id)
            .cloned()
            .ok_or_else(|| format!("Unknown NeuroToken task: {task_id}"))?;
        if task.task_id != task_id {
            return Err("NeuroToken registry task key does not match task_id".to_string());
        }
        validate_task(&task)?;
        Ok(task)
    }

    pub(crate) fn open_task(&self, task_id: &str, revision: u32) -> Result<BridgeSession, String> {
        let task = self.task(task_id)?;
        if task.revision != revision {
            return Err(format!(
                "Stale NeuroToken task revision: expected {}, received {revision}",
                task.revision
            ));
        }
        if !matches!(
            task.status.as_str(),
            "ready_for_coding" | "coding" | "validating"
        ) {
            return Err(format!(
                "NeuroToken task is not open for coding: {}",
                task.status
            ));
        }
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "Bridge state lock poisoned")?;
        let mut registry = self.read_sessions()?;
        let existing = registry.sessions.get(task_id);
        let existing_revision = existing.filter(|session| session.revision == revision);
        let session = BridgeSession {
            task_id: task.task_id,
            thread_id: task.thread_id,
            revision,
            status: existing_revision
                .map(|session| session.status.clone())
                .unwrap_or_else(|| "idle".to_string()),
            codex_thread_id: task
                .codex_thread_id
                .or_else(|| existing.and_then(|session| session.codex_thread_id.clone())),
            turn_started: existing_revision.is_some_and(|session| session.turn_started),
            updated_at_ms: now_ms(),
        };
        registry
            .sessions
            .insert(task_id.to_string(), session.clone());
        self.write_sessions(&registry)?;
        Ok(session)
    }

    pub(crate) fn session(&self, task_id: &str) -> Result<Option<BridgeSession>, String> {
        validate_task_id(task_id)?;
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "Bridge state lock poisoned")?;
        Ok(self.read_sessions()?.sessions.get(task_id).cloned())
    }

    pub(crate) fn record_event(
        &self,
        task_id: &str,
        revision: u32,
        message: &serde_json::Value,
    ) -> Result<(), String> {
        validate_task_id(task_id)?;
        let task = self.task(task_id)?;
        if task.revision != revision {
            return Err(format!(
                "Stale NeuroToken task revision: expected {}, received {revision}",
                task.revision
            ));
        }
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "Bridge state lock poisoned")?;
        let mut registry = self.read_sessions()?;
        let events = registry
            .events
            .entry(revision_event_key(task_id, revision))
            .or_default();
        events.push(message.clone());
        if events.len() > 500 {
            events.drain(..events.len() - 500);
        }
        self.write_sessions(&registry)
    }

    pub(crate) fn begin_turn(&self, task_id: &str, revision: u32) -> Result<BridgeSession, String> {
        validate_task_id(task_id)?;
        let task = self.task(task_id)?;
        if task.revision != revision {
            return Err(format!(
                "Stale NeuroToken task revision: expected {}, received {revision}",
                task.revision
            ));
        }
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "Bridge state lock poisoned")?;
        let mut registry = self.read_sessions()?;
        let current = registry
            .sessions
            .get(task_id)
            .ok_or_else(|| format!("NeuroToken task session is not open: {task_id}"))?;
        if current.revision != revision {
            return Err(format!(
                "Stale NeuroToken Bridge session revision: expected {}, received {revision}",
                current.revision
            ));
        }
        let session = BridgeSession {
            task_id: current.task_id.clone(),
            thread_id: current.thread_id.clone(),
            revision,
            status: "running".to_string(),
            codex_thread_id: current.codex_thread_id.clone(),
            turn_started: true,
            updated_at_ms: now_ms(),
        };
        registry
            .events
            .remove(&revision_event_key(task_id, revision));
        registry
            .sessions
            .insert(task_id.to_string(), session.clone());
        self.write_sessions(&registry)?;
        Ok(session)
    }

    pub(crate) fn events(
        &self,
        task_id: &str,
        revision: u32,
    ) -> Result<Vec<serde_json::Value>, String> {
        validate_task_id(task_id)?;
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "Bridge state lock poisoned")?;
        let registry = self.read_sessions()?;
        let session = registry
            .sessions
            .get(task_id)
            .ok_or_else(|| format!("NeuroToken task session is not open: {task_id}"))?;
        if session.revision != revision {
            return Err(format!(
                "Stale NeuroToken Bridge session revision: expected {}, received {revision}",
                session.revision
            ));
        }
        let revision_events = registry.events.get(&revision_event_key(task_id, revision));
        let legacy_events = (revision == 1)
            .then(|| registry.events.get(task_id))
            .flatten();
        Ok(revision_events
            .or(legacy_events)
            .cloned()
            .unwrap_or_default())
    }

    pub(crate) fn update_session(
        &self,
        task_id: &str,
        revision: u32,
        status: &str,
        codex_thread_id: Option<String>,
    ) -> Result<BridgeSession, String> {
        validate_task_id(task_id)?;
        validate_session_status(status)?;
        let task = self.task(task_id)?;
        if task.revision != revision {
            return Err(format!(
                "Stale NeuroToken task revision: expected {}, received {revision}",
                task.revision
            ));
        }
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "Bridge state lock poisoned")?;
        let mut registry = self.read_sessions()?;
        let current = registry
            .sessions
            .get(task_id)
            .ok_or_else(|| format!("NeuroToken task session is not open: {task_id}"))?;
        if current.revision != revision {
            return Err(format!(
                "Stale NeuroToken Bridge session revision: expected {}, received {revision}",
                current.revision
            ));
        }
        let session = BridgeSession {
            task_id: current.task_id.clone(),
            thread_id: current.thread_id.clone(),
            revision,
            status: status.to_string(),
            codex_thread_id: codex_thread_id.or_else(|| current.codex_thread_id.clone()),
            turn_started: current.turn_started,
            updated_at_ms: now_ms(),
        };
        registry
            .sessions
            .insert(task_id.to_string(), session.clone());
        self.write_sessions(&registry)?;
        Ok(session)
    }

    pub(crate) fn list_files(&self, task_id: &str) -> Result<Vec<String>, String> {
        let task = self.task(task_id)?;
        let root = canonical_directory(Path::new(&task.worktree_path), "task worktree")?;
        let mut files = Vec::new();
        for allowed in &task.allowed_paths {
            let candidate = root.join(allowed);
            if !candidate.exists() {
                continue;
            }
            collect_files(&root, &candidate, &mut files)?;
        }
        files.sort();
        files.dedup();
        Ok(files)
    }

    pub(crate) fn read_file(
        &self,
        task_id: &str,
        relative_path: &str,
    ) -> Result<BridgeFile, String> {
        let task = self.task(task_id)?;
        let path = resolve_existing_allowed_file(&task, relative_path)?;
        let content =
            fs::read_to_string(&path).map_err(|err| format!("Failed to read task file: {err}"))?;
        Ok(BridgeFile {
            path: normalize_relative_path(relative_path)?,
            content_hash: content_hash(&content),
            content,
        })
    }

    pub(crate) fn write_file(
        &self,
        task_id: &str,
        relative_path: &str,
        content: &str,
        expected_content_hash: &str,
    ) -> Result<WriteResult, String> {
        let task = self.task(task_id)?;
        if !matches!(task.status.as_str(), "ready_for_coding" | "coding") {
            return Err(format!(
                "NeuroToken task is not accepting file writes: {}",
                task.status
            ));
        }
        let relative = normalize_relative_path(relative_path)?;
        let path = resolve_allowed_write_path(&task, &relative)?;
        let current_content = if path.exists() {
            fs::read_to_string(&path).map_err(|err| format!("Failed to read task file: {err}"))?
        } else {
            String::new()
        };
        let current = BridgeFile {
            path: relative,
            content_hash: content_hash(&current_content),
            content: current_content,
        };
        if current.content_hash != expected_content_hash {
            return Ok(WriteResult::Conflict { current });
        }
        atomic_write(&path, content.as_bytes())?;
        Ok(WriteResult::Saved {
            file: BridgeFile {
                path: current.path,
                content: content.to_string(),
                content_hash: content_hash(content),
            },
        })
    }

    pub(crate) fn diff(&self, task_id: &str) -> Result<String, String> {
        let task = self.task(task_id)?;
        let worktree = canonical_directory(Path::new(&task.worktree_path), "task worktree")?;
        let baseline = canonical_directory(Path::new(&task.baseline_path), "task baseline")?;
        let worktree_parent = worktree
            .parent()
            .ok_or_else(|| "Task worktree has no parent directory".to_string())?;
        if baseline.parent() != Some(worktree_parent) {
            return Err("Task worktree and baseline must share a parent directory".to_string());
        }
        let worktree_name = worktree
            .file_name()
            .ok_or_else(|| "Task worktree has no directory name".to_string())?;
        let baseline_name = baseline
            .file_name()
            .ok_or_else(|| "Task baseline has no directory name".to_string())?;
        let mut result = String::new();
        for allowed in &task.allowed_paths {
            let baseline_path = baseline.join(allowed);
            let worktree_path = worktree.join(allowed);
            if !baseline_path.exists() && !worktree_path.exists() {
                continue;
            }
            let before = baseline_path
                .exists()
                .then(|| Path::new(baseline_name).join(allowed));
            let after = worktree_path
                .exists()
                .then(|| Path::new(worktree_name).join(allowed));
            let output = Command::new("git")
                .args(["diff", "--no-index", "--binary", "--"])
                .arg(before.as_deref().unwrap_or_else(|| Path::new("/dev/null")))
                .arg(after.as_deref().unwrap_or_else(|| Path::new("/dev/null")))
                .current_dir(worktree_parent)
                .output()
                .map_err(|err| format!("Failed to compute task diff: {err}"))?;
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if !matches!(output.status.code(), Some(0 | 1))
                || (output.status.code() == Some(1)
                    && output.stdout.is_empty()
                    && !stderr.is_empty())
            {
                return Err(stderr);
            }
            let mut diff = String::from_utf8(output.stdout)
                .map_err(|_| "Task diff is not valid UTF-8".to_string())?;
            for prefix in ["a", "b"] {
                for root_name in [baseline_name, worktree_name] {
                    diff = diff.replace(
                        &format!("{prefix}/{}/", root_name.to_string_lossy()),
                        &format!("{prefix}/"),
                    );
                }
            }
            result.push_str(&diff);
        }
        Ok(result)
    }

    fn read_sessions(&self) -> Result<SessionRegistry, String> {
        if !self.sessions_path.exists() {
            return Ok(SessionRegistry {
                schema_version: SESSION_VERSION.to_string(),
                sessions: HashMap::new(),
                events: HashMap::new(),
            });
        }
        let payload = fs::read_to_string(&self.sessions_path)
            .map_err(|err| format!("Failed to read NeuroToken Bridge state: {err}"))?;
        let registry: SessionRegistry = serde_json::from_str(&payload)
            .map_err(|err| format!("Invalid NeuroToken Bridge state: {err}"))?;
        if registry.schema_version != SESSION_VERSION {
            return Err(format!(
                "Unsupported NeuroToken Bridge state version: {}",
                registry.schema_version
            ));
        }
        Ok(registry)
    }

    fn write_sessions(&self, registry: &SessionRegistry) -> Result<(), String> {
        let payload = serde_json::to_vec_pretty(registry)
            .map_err(|err| format!("Failed to serialize NeuroToken Bridge state: {err}"))?;
        atomic_write(&self.sessions_path, &payload)
    }
}

fn validate_session_status(status: &str) -> Result<(), String> {
    if matches!(
        status,
        "idle" | "starting" | "running" | "awaiting_approval" | "recovering" | "error"
    ) {
        Ok(())
    } else {
        Err(format!(
            "Invalid NeuroToken Bridge session status: {status}"
        ))
    }
}

fn validate_task_id(task_id: &str) -> Result<(), String> {
    if task_id.is_empty()
        || task_id.len() > 96
        || !task_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(format!("Invalid NeuroToken task id: {task_id:?}"));
    }
    Ok(())
}

fn validate_task(task: &TaskRecord) -> Result<(), String> {
    if task.allowed_paths.is_empty() {
        return Err("NeuroToken task has no allowed paths".to_string());
    }
    canonical_directory(Path::new(&task.worktree_path), "task worktree")?;
    canonical_directory(Path::new(&task.baseline_path), "task baseline")?;
    for allowed in &task.allowed_paths {
        normalize_relative_path(allowed)?;
    }
    Ok(())
}

fn normalize_relative_path(path: &str) -> Result<String, String> {
    if path.is_empty() || path.contains('\0') || path.contains('\\') {
        return Err("Invalid task-relative path".to_string());
    }
    let parsed = Path::new(path);
    if parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("Invalid task-relative path".to_string());
    }
    Ok(parsed.to_string_lossy().replace('\\', "/"))
}

fn canonical_directory(path: &Path, context: &str) -> Result<PathBuf, String> {
    let canonical = path
        .canonicalize()
        .map_err(|err| format!("Failed to resolve {context}: {err}"))?;
    if !canonical.is_dir() {
        return Err(format!("{context} is not a directory"));
    }
    Ok(canonical)
}

fn path_is_allowed(path: &str, allowed_paths: &[String]) -> bool {
    let candidate = Path::new(path);
    allowed_paths.iter().any(|allowed| {
        let root = Path::new(allowed);
        candidate == root || candidate.starts_with(root)
    })
}

fn resolve_existing_allowed_file(
    task: &TaskRecord,
    relative_path: &str,
) -> Result<PathBuf, String> {
    let relative = normalize_relative_path(relative_path)?;
    if !path_is_allowed(&relative, &task.allowed_paths) {
        return Err("Path escapes the allowed task scope".to_string());
    }
    let root = canonical_directory(Path::new(&task.worktree_path), "task worktree")?;
    let candidate = root.join(&relative);
    let canonical = candidate
        .canonicalize()
        .map_err(|err| format!("Failed to resolve task file: {err}"))?;
    if !canonical.starts_with(&root) || !canonical.is_file() {
        return Err("Invalid task file path".to_string());
    }
    Ok(canonical)
}

fn resolve_allowed_write_path(task: &TaskRecord, relative_path: &str) -> Result<PathBuf, String> {
    if !path_is_allowed(relative_path, &task.allowed_paths) {
        return Err("Path escapes the allowed task scope".to_string());
    }
    let root = canonical_directory(Path::new(&task.worktree_path), "task worktree")?;
    let candidate = root.join(relative_path);
    if candidate.exists() {
        let canonical = candidate
            .canonicalize()
            .map_err(|err| format!("Failed to resolve task file: {err}"))?;
        if !canonical.starts_with(&root) || !canonical.is_file() {
            return Err("Invalid task file path".to_string());
        }
        return Ok(canonical);
    }
    let parent = candidate
        .parent()
        .ok_or_else(|| "Task file has no parent directory".to_string())?
        .canonicalize()
        .map_err(|err| format!("Failed to resolve task file parent: {err}"))?;
    if !parent.starts_with(&root) || !parent.is_dir() {
        return Err("Invalid task file parent path".to_string());
    }
    Ok(parent.join(
        candidate
            .file_name()
            .ok_or_else(|| "Task file has no filename".to_string())?,
    ))
}

fn collect_files(root: &Path, candidate: &Path, files: &mut Vec<String>) -> Result<(), String> {
    let metadata = fs::symlink_metadata(candidate)
        .map_err(|err| format!("Failed to inspect allowed path: {err}"))?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_file() {
        let canonical = candidate
            .canonicalize()
            .map_err(|err| format!("Failed to resolve allowed file: {err}"))?;
        if !canonical.starts_with(root) {
            return Err("Allowed file escapes task worktree".to_string());
        }
        let relative = canonical
            .strip_prefix(root)
            .map_err(|_| "Allowed file escapes task worktree".to_string())?;
        files.push(relative.to_string_lossy().replace('\\', "/"));
        return Ok(());
    }
    if metadata.is_dir() {
        let canonical = candidate
            .canonicalize()
            .map_err(|err| format!("Failed to resolve allowed directory: {err}"))?;
        if !canonical.starts_with(root) {
            return Err("Allowed directory escapes task worktree".to_string());
        }
        for entry in
            fs::read_dir(&canonical).map_err(|err| format!("Failed to list allowed path: {err}"))?
        {
            collect_files(root, &entry.map_err(|err| err.to_string())?.path(), files)?;
        }
    }
    Ok(())
}

fn revision_event_key(task_id: &str, revision: u32) -> String {
    format!("{task_id}:{revision}")
}

fn content_hash(content: &str) -> String {
    let mut hash: u32 = 2_166_136_261;
    for byte in content.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    format!("fnv1a-{hash:08x}")
}

fn atomic_write(path: &Path, content: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "Atomic write target has no parent".to_string())?;
    fs::create_dir_all(parent).map_err(|err| format!("Failed to create state directory: {err}"))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        unique_suffix()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|err| format!("Failed to create temporary file: {err}"))?;
    file.write_all(content)
        .and_then(|_| file.sync_all())
        .map_err(|err| format!("Failed to persist temporary file: {err}"))?;
    fs::rename(&temporary, path).map_err(|err| format!("Failed to install atomic file: {err}"))?;
    FileSync::sync_directory(parent)
}

struct FileSync;

impl FileSync {
    #[cfg(unix)]
    fn sync_directory(path: &Path) -> Result<(), String> {
        fs::File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(|err| format!("Failed to persist atomic directory update: {err}"))
    }

    #[cfg(not(unix))]
    fn sync_directory(_path: &Path) -> Result<(), String> {
        Ok(())
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn unique_suffix() -> String {
    format!(
        "{}-{}-{}",
        std::process::id(),
        now_ms(),
        UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::sync::Arc;
    use std::thread;

    fn temp_dir(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("neurotoken-bridge-{name}-{}", unique_suffix()));
        fs::create_dir_all(&path).expect("create temp directory");
        path
    }

    fn fixture() -> (PathBuf, PathBuf, PathBuf, NeurotokenBridge) {
        let root = temp_dir("core");
        let worktree = root.join("worktree");
        let baseline = root.join("baseline");
        let adapter_dir = Path::new("agent_pipeline/adapters");
        fs::create_dir_all(worktree.join(adapter_dir)).expect("create worktree adapter dir");
        fs::create_dir_all(baseline.join(adapter_dir)).expect("create baseline adapter dir");
        fs::write(worktree.join(adapter_dir).join("example.py"), "initial\n")
            .expect("seed worktree");
        fs::write(baseline.join(adapter_dir).join("example.py"), "initial\n")
            .expect("seed baseline");
        let registry_path = root.join("tasks.json");
        let registry = json!({
            "schema_version": "1.0",
            "tasks": {
                "task-1": {
                    "schema_version": "1.0",
                    "task_id": "task-1",
                    "owner": "alice",
                    "thread_id": "thread-1",
                    "revision": 1,
                    "status": "coding",
                    "adapter": {"name": "example", "modality": "eeg", "notes": ""},
                    "allowed_paths": ["agent_pipeline/adapters", "ADAPTER_DESIGN.md"],
                    "instruction": "Implement adapter",
                    "repo_root": root,
                    "worktree_path": worktree,
                    "baseline_path": baseline,
                    "codex_thread_id": null,
                    "created_at": "2026-08-05T00:00:00Z",
                    "updated_at": "2026-08-05T00:00:00Z",
                    "error": null
                }
            }
        });
        fs::write(
            &registry_path,
            serde_json::to_vec_pretty(&registry).expect("serialize registry"),
        )
        .expect("write registry");
        let bridge = NeurotokenBridge::new(Some(registry_path), &root);
        (root, worktree, baseline, bridge)
    }

    #[test]
    fn task_lookup_and_session_survive_bridge_restart() {
        let (root, _worktree, _baseline, bridge) = fixture();
        let session = bridge.open_task("task-1", 1).expect("open task");
        assert_eq!(session.thread_id, "thread-1");
        assert!(!root.join("neurotoken-sessions.json.tmp").exists());

        let restarted = NeurotokenBridge::new(Some(root.join("tasks.json")), &root);
        assert_eq!(
            restarted.session("task-1").expect("read session"),
            Some(session)
        );
        let running = restarted
            .update_session("task-1", 1, "running", Some("codex-1".to_string()))
            .expect("persist running session");
        assert_eq!(running.status, "running");
        assert_eq!(running.codex_thread_id.as_deref(), Some("codex-1"));
        assert!(restarted
            .update_session("task-1", 1, "invalid", None)
            .is_err());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn persisted_events_are_isolated_by_task_revision() {
        let (root, _worktree, _baseline, bridge) = fixture();
        bridge.open_task("task-1", 1).expect("open first revision");
        bridge
            .record_event(
                "task-1",
                1,
                &json!({"method": "turn/completed", "revision": 1}),
            )
            .expect("record first revision event");

        let registry_path = root.join("tasks.json");
        let payload = fs::read_to_string(&registry_path).expect("read registry");
        let mut registry: Value = serde_json::from_str(&payload).expect("parse registry");
        registry["tasks"]["task-1"]["revision"] = json!(2);
        registry["tasks"]["task-1"]["codex_thread_id"] = json!("codex-1");
        fs::write(
            &registry_path,
            serde_json::to_vec_pretty(&registry).expect("serialize registry"),
        )
        .expect("write next revision");

        let second_session = bridge.open_task("task-1", 2).expect("open second revision");
        assert_eq!(second_session.codex_thread_id.as_deref(), Some("codex-1"));
        assert!(!second_session.turn_started);
        assert!(bridge
            .events("task-1", 2)
            .expect("read empty history")
            .is_empty());
        bridge
            .record_event(
                "task-1",
                2,
                &json!({"method": "turn/completed", "revision": 1, "replayed": true}),
            )
            .expect("record resumed thread replay");
        let running = bridge.begin_turn("task-1", 2).expect("begin revision turn");
        assert!(running.turn_started);
        assert!(bridge
            .events("task-1", 2)
            .expect("replay cleared at turn boundary")
            .is_empty());
        bridge
            .record_event(
                "task-1",
                2,
                &json!({"method": "turn/completed", "revision": 2}),
            )
            .expect("record second revision event");
        assert_eq!(
            bridge
                .events("task-1", 2)
                .expect("read second revision history"),
            vec![json!({"method": "turn/completed", "revision": 2})]
        );

        let sessions: Value = serde_json::from_str(
            &fs::read_to_string(root.join("neurotoken-sessions.json")).expect("read sessions"),
        )
        .expect("parse sessions");
        assert!(sessions["events"]["task-1:1"].is_array());
        assert!(sessions["events"]["task-1:2"].is_array());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn stale_revision_and_terminal_task_are_rejected() {
        let (root, _worktree, _baseline, bridge) = fixture();
        let stale = bridge
            .open_task("task-1", 2)
            .expect_err("reject stale revision");
        assert!(stale.contains("Stale"));

        let registry_path = root.join("tasks.json");
        let payload = fs::read_to_string(&registry_path).expect("read registry");
        fs::write(&registry_path, payload.replace("\"coding\"", "\"applied\""))
            .expect("write terminal registry");
        let terminal = bridge
            .open_task("task-1", 1)
            .expect_err("reject terminal task");
        assert!(terminal.contains("not open for coding"));
        let write = bridge
            .write_file(
                "task-1",
                "agent_pipeline/adapters/example.py",
                "late change\n",
                &content_hash("initial\n"),
            )
            .expect_err("reject terminal write");
        assert!(write.contains("not accepting file writes"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn allowed_files_hash_conflict_and_diff_round_trip() {
        let (root, _worktree, _baseline, bridge) = fixture();
        let files = bridge.list_files("task-1").expect("list files");
        assert_eq!(files, vec!["agent_pipeline/adapters/example.py"]);
        let file = bridge
            .read_file("task-1", "agent_pipeline/adapters/example.py")
            .expect("read file");
        assert_eq!(file.content_hash, "fnv1a-d46e15cb");

        let conflict = bridge
            .write_file(
                "task-1",
                "agent_pipeline/adapters/example.py",
                "changed\n",
                "fnv1a-stale",
            )
            .expect("conflict response");
        assert!(matches!(conflict, WriteResult::Conflict { .. }));
        let saved = bridge
            .write_file(
                "task-1",
                "agent_pipeline/adapters/example.py",
                "changed\n",
                &file.content_hash,
            )
            .expect("save file");
        assert!(matches!(saved, WriteResult::Saved { .. }));
        let diff = bridge.diff("task-1").expect("compute diff");
        assert!(diff.contains("example.py"));
        assert!(diff.contains("+changed"));
        assert!(!diff.contains(&root.to_string_lossy().to_string()));

        let design = bridge
            .write_file(
                "task-1",
                "ADAPTER_DESIGN.md",
                "# Adapter Design\n",
                &content_hash(""),
            )
            .expect("create design document");
        assert!(matches!(design, WriteResult::Saved { .. }));
        let diff = bridge.diff("task-1").expect("diff design document");
        assert!(diff.contains("diff --git a/ADAPTER_DESIGN.md b/ADAPTER_DESIGN.md"));
        assert!(diff.contains("+# Adapter Design"));

        let empty_hash = content_hash("");
        let created = bridge
            .write_file(
                "task-1",
                "agent_pipeline/adapters/new_adapter.py",
                "new adapter\n",
                &empty_hash,
            )
            .expect("create allowed file");
        assert!(matches!(created, WriteResult::Saved { .. }));
        assert_eq!(
            bridge
                .read_file("task-1", "agent_pipeline/adapters/new_adapter.py")
                .expect("read created file")
                .content,
            "new adapter\n"
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn traversal_and_component_prefix_escape_are_rejected() {
        let (root, worktree, _baseline, bridge) = fixture();
        fs::create_dir_all(worktree.join("agent_pipeline/adapters_evil")).expect("create evil dir");
        fs::write(
            worktree.join("agent_pipeline/adapters_evil/example.py"),
            "evil\n",
        )
        .expect("seed evil file");
        for path in [
            "../outside.py",
            "/tmp/outside.py",
            "agent_pipeline/adapters_evil/example.py",
        ] {
            assert!(bridge.read_file("task-1", path).is_err(), "accepted {path}");
        }
        fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_hidden_and_rejected() {
        use std::os::unix::fs::symlink;

        let (root, worktree, _baseline, bridge) = fixture();
        let outside = root.join("outside.py");
        fs::write(&outside, "outside\n").expect("seed outside");
        let link = worktree.join("agent_pipeline/adapters/link.py");
        symlink(&outside, &link).expect("create symlink");
        assert!(!bridge
            .list_files("task-1")
            .expect("list files")
            .contains(&"agent_pipeline/adapters/link.py".to_string()));
        assert!(bridge
            .read_file("task-1", "agent_pipeline/adapters/link.py")
            .is_err());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn concurrent_tasks_preserve_independent_session_records() {
        let (root, worktree, baseline, bridge) = fixture();
        let registry_path = root.join("tasks.json");
        let payload = fs::read_to_string(&registry_path).expect("read registry");
        let mut registry: Value = serde_json::from_str(&payload).expect("parse registry");
        let mut second = registry["tasks"]["task-1"].clone();
        second["task_id"] = Value::String("task-2".to_string());
        second["thread_id"] = Value::String("thread-2".to_string());
        second["worktree_path"] = Value::String(worktree.to_string_lossy().to_string());
        second["baseline_path"] = Value::String(baseline.to_string_lossy().to_string());
        registry["tasks"]["task-2"] = second;
        fs::write(
            &registry_path,
            serde_json::to_vec_pretty(&registry).expect("serialize registry"),
        )
        .expect("write registry");

        let shared = Arc::new(bridge);
        let handles = ["task-1", "task-2"].map(|task_id| {
            let bridge = Arc::clone(&shared);
            thread::spawn(move || bridge.open_task(task_id, 1).expect("open concurrent task"))
        });
        for handle in handles {
            handle.join().expect("join task");
        }
        let restarted = NeurotokenBridge::new(Some(registry_path), &root);
        assert!(restarted.session("task-1").expect("session one").is_some());
        assert!(restarted.session("task-2").expect("session two").is_some());
        fs::remove_dir_all(root).ok();
    }
}
