#!/usr/bin/env python3
"""Check web/static/js/*.js for duplicate function/async function declarations.

Catches issues like 'async function collectFilesFromEntry' declared twice,
which causes 'Identifier has already been declared' at runtime.
Only checks function/async function (not const/let/var, which may be valid in different scopes).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JS_DIR = ROOT / "src" / "team_memory" / "web" / "static" / "js"

# Only function and async function - these cause "already declared" when duplicated at module level
FUNC_PATTERN = re.compile(r"^\s*(async\s+)?function\s+(\w+)\s*\(", re.MULTILINE)


def check_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, msg) for duplicate function declarations."""
    content = path.read_text(encoding="utf-8")
    seen: dict[str, list[int]] = {}
    for m in FUNC_PATTERN.finditer(content):
        name = m.group(2)
        line_no = content[: m.start()].count("\n") + 1
        if name not in seen:
            seen[name] = []
        seen[name].append(line_no)
    issues = []
    for name, lines in seen.items():
        if len(lines) > 1:
            issues.append((lines[0], f"Duplicate function: '{name}' at lines {lines}"))
    return issues


def main() -> int:
    if not JS_DIR.exists():
        print(f"SKIP: {JS_DIR} not found")
        return 0
    failed = False
    for js in sorted(JS_DIR.glob("*.js")):
        issues = check_file(js)
        for line_no, msg in issues:
            print(f"{js}:{line_no}: {msg}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
