#!/usr/bin/env bash
# Verify harness doc references after exec-plans migration.
# Validates: docs/design-docs/, docs/exec-plans/, .cursor/plans/workflows/,
#            architecture_models.py and base.py docstrings -> docs/design-docs/
# Exit 1 if any check fails.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
FAIL=0

# 1. docs/design-docs/code-arch-viz/*.md
if [ -d "$ROOT/docs/design-docs/code-arch-viz" ] && [ -n "$(find "$ROOT/docs/design-docs/code-arch-viz" -maxdepth 1 -name '*.md' -type f 2>/dev/null)" ]; then
  echo "OK: docs/design-docs/code-arch-viz/*.md"
else
  echo "FAIL: docs/design-docs/code-arch-viz/*.md - path missing or empty"
  FAIL=1
fi

# 2. docs/exec-plans/**/*.md
if [ -d "$ROOT/docs/exec-plans" ] && find "$ROOT/docs/exec-plans" -name "*.md" -type f 2>/dev/null | grep -q .; then
  echo "OK: docs/exec-plans/**/*.md"
else
  echo "FAIL: docs/exec-plans/**/*.md - path missing or no .md files"
  FAIL=1
fi

# 3. .cursor/plans/workflows/*.yaml
if [ -d "$ROOT/.cursor/plans/workflows" ] && [ -n "$(find "$ROOT/.cursor/plans/workflows" -maxdepth 1 -name '*.yaml' -type f 2>/dev/null)" ]; then
  echo "OK: .cursor/plans/workflows/*.yaml"
else
  echo "FAIL: .cursor/plans/workflows/*.yaml - path missing or no .yaml files"
  FAIL=1
fi

# 4. architecture_models.py docstring -> docs/design-docs/
if grep -q "docs/design-docs/" "$ROOT/src/team_memory/web/architecture_models.py" 2>/dev/null; then
  echo "OK: architecture_models.py docstring -> docs/design-docs/"
else
  echo "FAIL: architecture_models.py docstring does not point to docs/design-docs/"
  FAIL=1
fi

# 5. base.py docstring -> docs/design-docs/
if grep -q "docs/design-docs/" "$ROOT/src/team_memory/architecture/base.py" 2>/dev/null; then
  echo "OK: base.py docstring -> docs/design-docs/"
else
  echo "FAIL: base.py docstring does not point to docs/design-docs/"
  FAIL=1
fi

if [ "$FAIL" -eq 1 ]; then
  echo "=== harness_ref_verify FAILED ==="
  exit 1
fi
echo "=== harness_ref_verify PASSED ==="
exit 0
