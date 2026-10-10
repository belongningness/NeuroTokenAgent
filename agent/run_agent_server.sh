#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ENV="${NEUROTOKEN_SERVER_ENV:-}"
if [[ -n "$SERVER_ENV" ]]; then
  export PATH="$SERVER_ENV/bin:$PATH"
fi
export NEUROTOKEN_PIPELINE_PYTHON="${NEUROTOKEN_PIPELINE_PYTHON:-python}"
export NEUROTOKEN_CURSOR_PYTHON="${NEUROTOKEN_CURSOR_PYTHON:-python}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

cd "$ROOT"
exec langgraph dev --host "${NEUROTOKEN_API_HOST:-127.0.0.1}" --port "${NEUROTOKEN_API_PORT:-2024}" --no-browser --no-reload
