#!/usr/bin/env bash
# Post-Edit/Write format hook: auto-formats edited files
# Reads tool_input JSON from stdin, applies formatter by file extension
# Exit 0 always (formatting failure should not block edits)

set -uo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

EXT="${FILE_PATH##*.}"

case "$EXT" in
  py)
    # Auto-format Python with ruff
    ruff format --quiet "$FILE_PATH" 2>/dev/null || true
    ;;
  # js|ts|jsx|tsx)
  #   # Uncomment if you want JS auto-formatting
  #   npx prettier --write "$FILE_PATH" 2>/dev/null || true
  #   ;;
esac

exit 0
