#!/usr/bin/env bash
# Verify harness doc references after exec-plans migration.
# Validates: docs/exec-plans/**/*.md
# Exit 1 if any check fails.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
FAIL=0

# docs/exec-plans/**/*.md
if [ -d "$ROOT/docs/exec-plans" ] && find "$ROOT/docs/exec-plans" -name "*.md" -type f 2>/dev/null | grep -q .; then
  echo "OK: docs/exec-plans/**/*.md"
else
  echo "FAIL: docs/exec-plans/**/*.md - path missing or no .md files"
  FAIL=1
fi

if [ "$FAIL" -eq 1 ]; then
  echo "=== harness_ref_verify FAILED ==="
  exit 1
fi
echo "=== harness_ref_verify PASSED ==="
exit 0
