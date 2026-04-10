#!/usr/bin/env bash
# Load repo-root .env then start TeamMemory MCP (stdio). Keeps secrets out of mcp.json.
# Usage: from repo root, or set MCP cwd to repo root and command to this script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

ENV_FILE="$REPO_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "run_mcp_with_dotenv: missing $ENV_FILE" >&2
  echo "  Copy example/env.team-memory.example to .env and set at least TEAM_MEMORY_API_KEY and TEAM_MEMORY_PROJECT." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -f "$PYTHON" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
    echo "run_mcp_with_dotenv: warning: using PATH python3 (not .venv): $PYTHON" >&2
  else
    echo "run_mcp_with_dotenv: no interpreter at $REPO_ROOT/.venv/bin/python and no python3 on PATH" >&2
    exit 1
  fi
fi

exec "$PYTHON" -m team_memory.server
