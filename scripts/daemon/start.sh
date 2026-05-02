#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Python interpreter resolution (order of precedence):
#   1. .env PYTHON variable
#   2. .venv/bin/python
#   3. conda run -n team_memory
#   4. system python
PYTHON=""
if [ -f .env ]; then
    PYTHON=$(grep -E '^PYTHON=' .env 2>/dev/null | sed 's/^PYTHON=//' | tr -d '"' | tr -d "'")
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    if [ -x .venv/bin/python ]; then
        PYTHON=.venv/bin/python
    elif command -v conda >/dev/null 2>&1 && conda run -n team_memory python -c "pass" 2>/dev/null; then
        PYTHON="conda run -n team_memory python"
    else
        PYTHON=python
    fi
fi

export PYTHONPATH=src:scripts
exec $PYTHON -m daemon
