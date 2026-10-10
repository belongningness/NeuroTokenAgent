#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${NEUROTOKEN_RUNTIME_DIR:-$HOME/.local/state/neurotoken-runtime}"
DESKTOP_DIR="${NEUROTOKEN_DESKTOP_DIR:-$ROOT/../neurotoken-desktop}"
DEFAULT_DAEMON_BIN="$DESKTOP_DIR/src-tauri/target/release/codex_monitor_daemon"
if [[ ! -x "$DEFAULT_DAEMON_BIN" && -x "$HOME/.local/bin/codex_monitor_daemon" ]]; then
  DEFAULT_DAEMON_BIN="$HOME/.local/bin/codex_monitor_daemon"
fi
DAEMON_BIN="${NEUROTOKEN_DAEMON_BIN:-$DEFAULT_DAEMON_BIN}"
LANGGRAPH_PORT="${NEUROTOKEN_API_PORT:-2024}"
CODING_PORT="${NEUROTOKEN_CODING_PORT:-4732}"
DATA_DIR="${NEUROTOKEN_DAEMON_DATA_DIR:-$RUNTIME_DIR/daemon-data}"
TOKEN="${CODEX_MONITOR_DAEMON_TOKEN:-}"
REGISTRY="${NEUROTOKEN_ADAPTER_TASK_REGISTRY:-$(git -C "$ROOT" rev-parse --git-path neurotoken-sdk-authoring/tasks.json)}"
if [[ "$REGISTRY" != /* ]]; then
  REGISTRY="$ROOT/$REGISTRY"
fi
export NEUROTOKEN_ADAPTER_TASK_REGISTRY="$REGISTRY"

ENV_FILE="${NEUROTOKEN_ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

NODE_BIN_DIR="${NEUROTOKEN_NODE_BIN_DIR:-}"
if [[ -n "$NODE_BIN_DIR" ]]; then
  [[ -x "$NODE_BIN_DIR/node" ]] || {
    echo "Node.js is not executable: $NODE_BIN_DIR/node" >&2
    exit 1
  }
  export PATH="$NODE_BIN_DIR:$PATH"
fi
command -v tmux >/dev/null 2>&1 || {
  echo "tmux is required for durable package, tokenize, and reconstruct stages" >&2
  exit 1
}
export NEUROTOKEN_TMUX_SOCKET="${NEUROTOKEN_TMUX_SOCKET:-$RUNTIME_DIR/tmux/neurotoken.sock}"

mkdir -p "$RUNTIME_DIR" "$DATA_DIR" "$(dirname "$REGISTRY")" "$(dirname "$NEUROTOKEN_TMUX_SOCKET")"
exec 9>"$RUNTIME_DIR/start.lock"
flock -w 20 9 || { echo "Timed out waiting for NeuroToken runtime lock" >&2; exit 1; }

pid_running() {
  local pid_file="$1" marker="$2"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  local cmdline
  cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  [[ "$cmdline" == *"$marker"* ]]
}

repair_pid_file() {
  local pid_file="$1" marker="$2"
  if pid_running "$pid_file" "$marker"; then
    return 0
  fi
  local proc pid cmdline
  for proc in /proc/[0-9]*; do
    [[ -r "$proc/cmdline" ]] || continue
    pid="${proc##*/}"
    cmdline="$(tr '\0' ' ' <"$proc/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"$marker"* ]]; then
      printf '%s\n' "$pid" >"$pid_file"
      return 0
    fi
  done
  rm -f "$pid_file"
  return 1
}

port_ready() {
  local port="$1"
  python3 - "$port" <<'PY' 2>/dev/null
import socket
import sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
}

langgraph_ready() {
  local port="$1"
  python3 - "$port" <<'PY' 2>/dev/null
import http.client
import sys
connection = http.client.HTTPConnection("127.0.0.1", int(sys.argv[1]), timeout=2)
connection.request("GET", "/ok")
response = connection.getresponse()
response.read()
if response.status != 200:
    raise SystemExit(1)
PY
}

service_ready() {
  local name="$1" port="$2"
  if [[ "$name" == "LangGraph" ]]; then
    langgraph_ready "$port"
  else
    port_ready "$port"
  fi
}

wait_ready() {
  local name="$1" port="$2" pid_file="$3" marker="$4" log_file="$5"
  for _ in $(seq 1 60); do
    if service_ready "$name" "$port"; then
      if pid_running "$pid_file" "$marker"; then
        return 0
      fi
      echo "Port $port is occupied by a process not owned by $name" >&2
      return 1
    fi
    if ! pid_running "$pid_file" "$marker"; then
      echo "$name exited before port $port became ready; inspect $log_file" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "$name did not become ready on port $port; inspect $log_file" >&2
  return 1
}

start_langgraph() {
  local pid_file="$RUNTIME_DIR/langgraph.pid" log_file="$RUNTIME_DIR/langgraph.log"
  if port_ready "$LANGGRAPH_PORT"; then
    if repair_pid_file "$pid_file" "langgraph dev"; then
      if langgraph_ready "$LANGGRAPH_PORT"; then
        return 0
      fi
      echo "LangGraph owns port $LANGGRAPH_PORT but its health endpoint is unresponsive" >&2
      return 1
    fi
    echo "Port $LANGGRAPH_PORT is already occupied by another process" >&2
    return 1
  fi
  rm -f "$pid_file"
  (
    cd "$ROOT"
    nohup "$ROOT/run_agent_server.sh" >>"$log_file" 2>&1 </dev/null 9>&- &
    echo $! >"$pid_file"
  )
  wait_ready "LangGraph" "$LANGGRAPH_PORT" "$pid_file" "langgraph dev" "$log_file"
}

start_coding_daemon() {
  local pid_file="$RUNTIME_DIR/coding-daemon.pid" log_file="$RUNTIME_DIR/coding-daemon.log"
  if port_ready "$CODING_PORT"; then
    if repair_pid_file "$pid_file" "$DAEMON_BIN"; then
      return 0
    fi
    echo "Port $CODING_PORT is already occupied by another process" >&2
    return 1
  fi
  [[ -x "$DAEMON_BIN" ]] || {
    echo "Codex daemon binary is not executable: $DAEMON_BIN" >&2
    exit 1
  }
  rm -f "$pid_file"
  local args=(
    --listen "127.0.0.1:$CODING_PORT"
    --data-dir "$DATA_DIR"
    --neurotoken-registry "$REGISTRY"
  )
  if [[ -n "$TOKEN" ]]; then
    args+=(--token "$TOKEN")
  else
    args+=(--insecure-no-auth)
  fi
  nohup "$DAEMON_BIN" "${args[@]}" >>"$log_file" 2>&1 </dev/null 9>&- &
  echo $! >"$pid_file"
  wait_ready "Codex daemon" "$CODING_PORT" "$pid_file" "$DAEMON_BIN" "$log_file"
}

start_langgraph
start_coding_daemon
printf '{"status":"running","langGraphPort":%s,"codingPort":%s}\n' "$LANGGRAPH_PORT" "$CODING_PORT"
