#!/usr/bin/env bash
# Pre-Bash guard hook: blocks dangerous and interactive commands
# Reads tool_input JSON from stdin, checks command field
# Exit 0 = allow, Exit 2 = block (with reason on stderr)

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

if [ -z "$COMMAND" ]; then
  exit 0
fi

# === Dangerous commands (High risk) ===

# Block: rm -rf with root or wildcard paths
if echo "$COMMAND" | grep -qE 'rm\s+-rf\s+/($|\s|[^.])'; then
  echo "BLOCKED: 'rm -rf /' is not allowed" >&2
  exit 2
fi

# Block: sudo
if echo "$COMMAND" | grep -qE '^\s*sudo\s'; then
  echo "BLOCKED: sudo commands are not allowed" >&2
  exit 2
fi

# Block: pipe to shell (curl/wget | sh/bash)
if echo "$COMMAND" | grep -qE '(curl|wget)\s.*\|\s*(sh|bash)'; then
  echo "BLOCKED: piping downloads to shell is not allowed" >&2
  exit 2
fi

# Block: DROP DATABASE
if echo "$COMMAND" | grep -qiE 'DROP\s+DATABASE'; then
  echo "BLOCKED: DROP DATABASE is not allowed" >&2
  exit 2
fi

# Block: git push --force to main/master
if echo "$COMMAND" | grep -qE 'git\s+push\s+.*--force.*\s+(main|master)'; then
  echo "BLOCKED: force push to main/master is not allowed" >&2
  exit 2
fi

# === Interactive commands (would hang the agent) ===

# Block: interactive editors
if echo "$COMMAND" | grep -qE '^\s*(vim|vi|nano|emacs)\b'; then
  echo "BLOCKED: Interactive editor — use Edit/Write tool instead" >&2
  exit 2
fi

# Block: interactive REPLs
if echo "$COMMAND" | grep -qE '^\s*(python3?|node)\s+-i\b'; then
  echo "BLOCKED: Interactive REPL — use Bash with a script file instead" >&2
  exit 2
fi

# Block: bare database CLIs (allow with -c/-e flag for non-interactive use)
if echo "$COMMAND" | grep -qE '^\s*(psql|mysql|sqlite3)\s*$'; then
  echo "BLOCKED: Interactive DB CLI — use 'psql -c \"SQL\"' or 'mysql -e \"SQL\"' instead" >&2
  exit 2
fi

# Block: pagers and monitors
if echo "$COMMAND" | grep -qE '^\s*(less|more|top|htop)\b'; then
  echo "BLOCKED: Interactive pager/monitor — use cat/head/tail instead" >&2
  exit 2
fi

exit 0
