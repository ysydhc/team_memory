#!/usr/bin/env python3
"""Import direction checker for team_memory architecture layers.

Validates that imports respect the layer rules defined in
docs/design-docs/harness/project-extension.md.

Usage:
  python scripts/harness_import_check.py [--root PATH]
  Exit 0 = pass, non-zero = violations found.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Layer mapping (from docs/design-docs/harness/project-extension.md)
# ---------------------------------------------------------------------------

# (path_pattern, layer) - order matters for prefix matching
LAYER_MAP: list[tuple[str, str]] = [
    ("src/team_memory/schemas.py", "L0"),
    ("src/team_memory/schemas_architecture.py", "L0"),
    ("src/team_memory/config.py", "L0"),
    ("src/team_memory/storage/", "L1"),
    ("src/team_memory/services/", "L2"),
    ("src/team_memory/auth/", "L2"),
    ("src/team_memory/embedding/", "L2"),
    ("src/team_memory/reranker/", "L2"),
    ("src/team_memory/architecture/", "L2"),
    ("src/team_memory/web/", "L3"),
    ("src/team_memory/server.py", "L3"),
    ("src/team_memory/bootstrap.py", "L3"),
    ("src/team_memory/workflow_oracle.py", "L3"),
]

# L3 modules that L0-L2 must not import (L2 cannot import any L3)
FORBIDDEN_L3_MODULES = {
    "team_memory.web",
    "team_memory.server",
    "team_memory.bootstrap",
    "team_memory.workflow_oracle",
}

# Rule IDs for output
RULE_L2_IMPORTS_L3 = "L2_IMPORTS_L3"
RULE_L0_L2_IMPORTS_BOOTSTRAP_SERVER = "L0_L2_IMPORTS_BOOTSTRAP_SERVER"

# Exclude patterns (relative to root)
EXCLUDE_DIRS = {"tests"}
EXCLUDE_PREFIXES = ("migrate_", "migration_")


def get_layer_for_path(file_path: Path, root: Path) -> str | None:
    """Map file path to layer (L0/L1/L2/L3) or None if not in layer table."""
    try:
        rel = file_path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if "src" not in parts or "team_memory" not in parts:
        return None
    # Build path like src/team_memory/xxx
    idx = parts.index("src")
    rel_str = str(Path(*parts[idx:])).replace("\\", "/")
    if rel_str.endswith("/"):
        rel_str = rel_str.rstrip("/")
    for pattern, layer in LAYER_MAP:
        if pattern.endswith("/"):
            if rel_str.startswith(pattern) or rel_str == pattern.rstrip("/"):
                return layer
        elif rel_str == pattern:
            return layer
    return None


def is_excluded_path(file_path: Path, root: Path) -> bool:
    """Check if path should be excluded from checking."""
    try:
        rel = file_path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    parts = rel.parts
    if "tests" in parts:
        return True
    name = file_path.name
    if name.startswith(EXCLUDE_PREFIXES):
        return True
    for d in EXCLUDE_DIRS:
        if d in parts:
            return True
    return False


def get_imported_module(node: ast.ImportFrom) -> str | None:
    """Extract team_memory.* module from ImportFrom node."""
    if node.module is None:
        return None
    if not node.module.startswith("team_memory."):
        return None
    # Normalize: team_memory.web.architecture_models -> team_memory.web
    # team_memory.server -> team_memory.server
    # team_memory.bootstrap -> team_memory.bootstrap
    mod = node.module
    if mod.startswith("team_memory.web"):
        return "team_memory.web"
    if mod.startswith("team_memory.server"):
        return "team_memory.server"
    if mod.startswith("team_memory.bootstrap"):
        return "team_memory.bootstrap"
    if mod.startswith("team_memory.workflow_oracle"):
        return "team_memory.workflow_oracle"
    return None


def is_forbidden_import(imported_mod: str) -> bool:
    """Check if imported module is forbidden for L0-L2."""
    return imported_mod in FORBIDDEN_L3_MODULES


def has_noqa_layer_check(line: str) -> bool:
    """Check if line has # noqa: layer-check exemption."""
    return "noqa: layer-check" in line or "noqa:layer-check" in line


def collect_imports_with_context(
    tree: ast.AST, source_lines: list[str]
) -> list[tuple[int, ast.ImportFrom, bool, bool]]:
    """Collect ImportFrom nodes with (line, node, in_type_checking, has_noqa)."""
    results: list[tuple[int, ast.ImportFrom, bool, bool]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._type_checking_stack: list[bool] = []

        def visit_If(self, node: ast.If) -> None:
            in_type_checking = False
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                in_type_checking = True
            self._type_checking_stack.append(in_type_checking)
            self.generic_visit(node)
            self._type_checking_stack.pop()

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            in_tc = any(self._type_checking_stack)
            line = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
            has_noqa = has_noqa_layer_check(line)
            results.append((node.lineno, node, in_tc, has_noqa))
            self.generic_visit(node)

    v = Visitor()
    v.visit(tree)
    return results


def check_file(file_path: Path, root: Path) -> list[tuple[int, str, str]]:
    """Check a single file. Returns list of (line, rule_id, message)."""
    violations: list[tuple[int, str, str]] = []
    layer = get_layer_for_path(file_path, root)
    if layer is None:
        return violations
    if layer == "L3":
        return violations  # L3 internal not validated

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"parse failed: {file_path}: {e}", file=sys.stderr)
        return violations

    source_lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"syntax error: {file_path}:{e.lineno}", file=sys.stderr)
        return violations

    imports = collect_imports_with_context(tree, source_lines)
    for lineno, node, in_type_checking, has_noqa in imports:
        if in_type_checking or has_noqa:
            continue
        imported = get_imported_module(node)
        if imported is None:
            continue
        if not is_forbidden_import(imported):
            continue

        if layer in ("L0", "L1", "L2") and imported in ("team_memory.server", "team_memory.bootstrap"):
            violations.append((lineno, RULE_L0_L2_IMPORTS_BOOTSTRAP_SERVER, f"L0-L2 must not import {imported}"))
        elif layer == "L2" and imported == "team_memory.web":
            violations.append((lineno, RULE_L2_IMPORTS_L3, "L2 must not import L3 (web)"))
        elif layer in ("L0", "L1") and imported == "team_memory.web":
            violations.append((lineno, RULE_L2_IMPORTS_L3, f"{layer} must not import L3 (web)"))

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Check import direction for team_memory layers")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: cwd)")
    args = parser.parse_args()
    root = args.root.resolve()

    src = root / "src" / "team_memory"
    if not src.exists():
        print(f"Error: {src} not found", file=sys.stderr)
        return 2

    all_violations: list[tuple[Path, int, str, str]] = []
    for py_path in sorted(src.rglob("*.py")):
        if is_excluded_path(py_path, root):
            continue
        for line, rule_id, msg in check_file(py_path, root):
            all_violations.append((py_path, line, rule_id, msg))

    for path, line, rule_id, msg in all_violations:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        print(f"{rel}:{line}: {rule_id}: {msg}")

    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
