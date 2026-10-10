use chrono::Utc;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

const LOG_FILE_NAME: &str = "neurotoken-ssh.log";
const MAX_EXPORT_BYTES: u64 = 128 * 1024;

pub(crate) fn log_path(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_log_dir()
        .map(|directory| directory.join(LOG_FILE_NAME))
        .map_err(|error| format!("Failed to resolve SSH diagnostic log directory: {error}"))
}

pub(crate) fn append(path: &Path, component: &str, message: impl AsRef<str>) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let sanitized = message.as_ref().replace(['\n', '\r'], " ");
    let _ = writeln!(
        file,
        "{} component={} pid={} {}",
        Utc::now().to_rfc3339(),
        component,
        std::process::id(),
        sanitized
    );
    let _ = file.flush();
}

pub(crate) fn read_recent(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("Failed to open SSH diagnostic log: {error}"))?;
    let length = file
        .metadata()
        .map_err(|error| format!("Failed to inspect SSH diagnostic log: {error}"))?
        .len();
    let start = length.saturating_sub(MAX_EXPORT_BYTES);
    file.seek(SeekFrom::Start(start))
        .map_err(|error| format!("Failed to seek SSH diagnostic log: {error}"))?;
    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|error| format!("Failed to read SSH diagnostic log: {error}"))?;
    if start > 0 {
        if let Some(first_newline) = content.find('\n') {
            content.drain(..=first_newline);
        }
        content.insert_str(0, "[earlier SSH diagnostics omitted]\n");
    }
    Ok(content)
}
