#!/usr/bin/env bash
# Scan all references to docs/, .debug/, .cursor/plans in code and docs
# Requires: ripgrep (rg). Tested with rg >= 13.0.
set -e
echo "=== docs/ references ==="
rg -l "docs/" --type-add 'code:*.{py,md,yaml,yml,mdc,json}' -t code 2>/dev/null || true
echo "=== .debug/ references ==="
rg -l "\.debug/" --type-add 'code:*.{py,md,yaml,yml,mdc,json}' -t code 2>/dev/null || true
echo "=== .cursor/plans references ==="
rg -l "\.cursor/plans|plans/" --type-add 'code:*.{py,md,yaml,yml,mdc,json}' -t code 2>/dev/null || true
