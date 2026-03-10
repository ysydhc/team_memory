#!/usr/bin/env bash
# Verify harness doc references after exec-plans migration.
# Validates: docs/exec-plans/code-arch-viz-gitnexus/, docs/exec-plans/**/*.md,
#            schemas_architecture.py and base.py docstrings -> docs/exec-plans/
# Exit 1 if any check fails.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
FAIL=0

# 1. docs/exec-plans/completed/code-arch-viz-gitnexus/*.md (provider-interface 等)
if [ -d "$ROOT/docs/exec-plans/completed/code-arch-viz-gitnexus" ] && [ -f "$ROOT/docs/exec-plans/completed/code-arch-viz-gitnexus/code-arch-viz-provider-interface.md" ]; then
  echo "OK: docs/exec-plans/completed/code-arch-viz-gitnexus/*.md"
else
  echo "FAIL: docs/exec-plans/completed/code-arch-viz-gitnexus/ - path missing or provider-interface not found"
  FAIL=1
fi

# 2. docs/exec-plans/**/*.md
if [ -d "$ROOT/docs/exec-plans" ] && find "$ROOT/docs/exec-plans" -name "*.md" -type f 2>/dev/null | grep -q .; then
  echo "OK: docs/exec-plans/**/*.md"
else
  echo "FAIL: docs/exec-plans/**/*.md - path missing or no .md files"
  FAIL=1
fi

# 3. schemas_architecture.py docstring -> docs/exec-plans/ (provider-interface 契约)
if grep -q "docs/exec-plans/.*code-arch-viz-gitnexus" "$ROOT/src/team_memory/schemas_architecture.py" 2>/dev/null; then
  echo "OK: schemas_architecture.py docstring -> docs/exec-plans/"
else
  echo "FAIL: schemas_architecture.py docstring does not point to provider-interface"
  FAIL=1
fi

# 4. base.py docstring -> docs/exec-plans/ (provider-interface 契约)
if grep -q "docs/exec-plans/.*code-arch-viz-gitnexus" "$ROOT/src/team_memory/architecture/base.py" 2>/dev/null; then
  echo "OK: base.py docstring -> docs/exec-plans/"
else
  echo "FAIL: base.py docstring does not point to provider-interface"
  FAIL=1
fi

if [ "$FAIL" -eq 1 ]; then
  echo "=== harness_ref_verify FAILED ==="
  exit 1
fi
echo "=== harness_ref_verify PASSED ==="
exit 0
