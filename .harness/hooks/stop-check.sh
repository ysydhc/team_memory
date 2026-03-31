#!/usr/bin/env bash
# Stop hook: checks completion criteria before session ends
# Reports warnings on stderr (does NOT block — exit 0 always)

set -uo pipefail

WARNINGS=""

# Check for uncommitted changes in src/
if git diff --name-only -- src/ tests/ 2>/dev/null | grep -q .; then
  WARNINGS="${WARNINGS}\n⚠️  有未提交的 src/ 或 tests/ 改动"
fi

# Check for ruff lint errors
if command -v ruff &>/dev/null; then
  LINT_ERRORS=$(ruff check src/ --quiet 2>/dev/null | head -5)
  if [ -n "$LINT_ERRORS" ]; then
    WARNINGS="${WARNINGS}\n⚠️  存在 ruff lint 错误:\n${LINT_ERRORS}"
  fi
fi

if [ -n "$WARNINGS" ]; then
  echo -e "\n📋 完成标准检查:${WARNINGS}" >&2
fi

exit 0
