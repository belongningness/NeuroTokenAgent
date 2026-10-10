// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod neurotoken_ssh_watchdog;

fn main() {
    if let Some(code) = neurotoken_ssh_watchdog::run_from_process_args() {
        std::process::exit(code);
    }
    if let Err(err) = fix_path_env::fix() {
        eprintln!("Failed to sync PATH from shell: {err}");
    }
    codex_monitor_lib::run()
}
