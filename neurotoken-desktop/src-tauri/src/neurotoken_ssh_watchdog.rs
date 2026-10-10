use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

pub const WATCHDOG_ARGUMENT: &str = "--neurotoken-ssh-watchdog";
const PARENT_PID_ARGUMENT: &str = "--neurotoken-parent-pid";
const LOG_PATH_ARGUMENT: &str = "--neurotoken-log-path";

struct WatchdogArguments {
    parent_pid: u32,
    log_path: PathBuf,
    ssh_args: Vec<OsString>,
}

fn parse_watchdog_arguments(
    mut args: impl Iterator<Item = OsString>,
) -> Result<WatchdogArguments, String> {
    if args.next().as_deref() != Some(std::ffi::OsStr::new(PARENT_PID_ARGUMENT)) {
        return Err(format!("expected {PARENT_PID_ARGUMENT}"));
    }
    let parent_pid = args
        .next()
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|pid| *pid > 0)
        .ok_or_else(|| "invalid Desktop parent pid".to_string())?;
    if args.next().as_deref() != Some(std::ffi::OsStr::new(LOG_PATH_ARGUMENT)) {
        return Err(format!("expected {LOG_PATH_ARGUMENT}"));
    }
    let log_path = args
        .next()
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| "invalid SSH diagnostic log path".to_string())?;
    if args.next().as_deref() != Some(std::ffi::OsStr::new("--")) {
        return Err("expected -- before SSH arguments".to_string());
    }
    let ssh_args: Vec<_> = args.collect();
    if ssh_args.is_empty() {
        return Err("no SSH arguments were provided".to_string());
    }
    Ok(WatchdogArguments {
        parent_pid,
        log_path,
        ssh_args,
    })
}

fn append_log(path: &Path, message: impl AsRef<str>) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let message = message.as_ref().replace(['\n', '\r'], " ");
    let _ = writeln!(
        file,
        "{} component=watchdog pid={} {}",
        chrono::Utc::now().to_rfc3339(),
        std::process::id(),
        message,
    );
    let _ = file.flush();
}

#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn process_exists(_pid: u32) -> bool {
    true
}

#[cfg(unix)]
fn install_termination_handlers(termination_requested: &Arc<AtomicBool>) {
    static mut TERMINATION_FLAG: *const AtomicBool = std::ptr::null();
    extern "C" fn handle_signal(_signal: libc::c_int) {
        unsafe {
            if !TERMINATION_FLAG.is_null() {
                (*TERMINATION_FLAG).store(true, Ordering::SeqCst);
            }
        }
    }
    unsafe {
        TERMINATION_FLAG = Arc::as_ptr(termination_requested);
        for signal in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
            libc::signal(signal, handle_signal as libc::sighandler_t);
        }
    }
}

#[cfg(not(unix))]
fn install_termination_handlers(_termination_requested: &Arc<AtomicBool>) {}

fn stop_ssh(log_path: &Path, ssh: &mut Child, reason: &str) -> i32 {
    append_log(
        log_path,
        format!("trigger={reason}; stopping ssh_pid={}", ssh.id()),
    );
    let kill_result = ssh.kill();
    let status = ssh.wait();
    append_log(
        log_path,
        format!("ssh stop completed kill_result={kill_result:?} wait_status={status:?}"),
    );
    0
}

pub fn run_from_process_args() -> Option<i32> {
    let mut args = std::env::args_os();
    let _executable = args.next();
    if args.next().as_deref() != Some(std::ffi::OsStr::new(WATCHDOG_ARGUMENT)) {
        return None;
    }
    let arguments = match parse_watchdog_arguments(args) {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("[neurotoken-ssh-watchdog] {error}");
            return Some(2);
        }
    };

    append_log(
        &arguments.log_path,
        format!("watchdog started parent_pid={}", arguments.parent_pid),
    );
    let ssh_stderr = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&arguments.log_path)
        .ok()
        .map(Stdio::from)
        .unwrap_or_else(Stdio::null);
    let mut ssh = match Command::new("ssh")
        .args(&arguments.ssh_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(ssh_stderr)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            append_log(
                &arguments.log_path,
                format!("failed to start SSH error={error}"),
            );
            return Some(2);
        }
    };

    append_log(
        &arguments.log_path,
        format!(
            "SSH started parent_pid={} ssh_pid={}",
            arguments.parent_pid,
            ssh.id()
        ),
    );

    let termination_requested = Arc::new(AtomicBool::new(false));
    install_termination_handlers(&termination_requested);
    let (parent_gone_tx, parent_gone_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut stdin = std::io::stdin();
        let mut buffer = [0_u8; 1];
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) | Err(_) => {
                    let _ = parent_gone_tx.send(());
                    return;
                }
                Ok(_) => {}
            }
        }
    });

    loop {
        match ssh.try_wait() {
            Ok(Some(status)) => {
                append_log(
                    &arguments.log_path,
                    format!("SSH exited independently status={status}"),
                );
                return Some(status.code().unwrap_or(1));
            }
            Ok(None) => {}
            Err(error) => {
                append_log(
                    &arguments.log_path,
                    format!("failed to inspect SSH error={error}"),
                );
                return Some(stop_ssh(
                    &arguments.log_path,
                    &mut ssh,
                    "SSH inspection failed",
                ));
            }
        }

        if termination_requested.load(Ordering::SeqCst) {
            return Some(stop_ssh(
                &arguments.log_path,
                &mut ssh,
                "termination signal received",
            ));
        }
        if parent_gone_rx.try_recv().is_ok() {
            return Some(stop_ssh(
                &arguments.log_path,
                &mut ssh,
                "parent pipe closed",
            ));
        }
        if !process_exists(arguments.parent_pid) {
            return Some(stop_ssh(
                &arguments.log_path,
                &mut ssh,
                "Desktop parent process disappeared",
            ));
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_argument_is_stable() {
        assert_eq!(WATCHDOG_ARGUMENT, "--neurotoken-ssh-watchdog");
    }

    #[test]
    fn parses_parent_pid_and_ssh_arguments() {
        let parsed = parse_watchdog_arguments(
            [
                PARENT_PID_ARGUMENT,
                "42",
                LOG_PATH_ARGUMENT,
                "/tmp/ssh.log",
                "--",
                "-N",
                "example-host",
            ]
            .into_iter()
            .map(OsString::from),
        )
        .expect("parse watchdog arguments");
        assert_eq!(parsed.parent_pid, 42);
        assert_eq!(
            parsed.ssh_args,
            [OsString::from("-N"), OsString::from("example-host")]
        );
    }

    #[test]
    fn rejects_missing_parent_pid() {
        assert!(parse_watchdog_arguments(
            ["--", "-N", "example-host"].into_iter().map(OsString::from)
        )
        .is_err());
    }
}
