#!/usr/bin/env python3
"""Plan structure check: scan exec-plans for DOC_PLAN_* rule violations.

Scans docs/exec-plans/wait, executing, completed for:
- DOC_PLAN_EXECUTING_MISSING_EXECUTE: executing/{主题} lacks execute file
- DOC_PLAN_COMPLETED_MISSING_EXECUTE: completed/{主题} lacks execute file
- DOC_PLAN_RESEARCH_MISSING_BRIEF: 1-research/ exists but brief.md missing
- DOC_PLAN_PLAN_MISSING_EXECUTE: 2-plan/ exists but execute file missing
- DOC_PLAN_LEGACY_STRUCTURE: topic not using 1-research/2-plan/3-retro or plan+execute structure
- DOC_PLAN_EXCESS_FILES: topic has > 25 .md files

Output: path: rule_id: message
Exit 0 = no issues, non-zero = issues found.

Usage:
  python scripts/harness_plan_structure_check.py [--root PATH] [--whitelist PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Rule IDs (from docs/design-docs/harness/plan-document-structure.md)
RULE_EXECUTING_MISSING_EXECUTE = "DOC_PLAN_EXECUTING_MISSING_EXECUTE"
RULE_COMPLETED_MISSING_EXECUTE = "DOC_PLAN_COMPLETED_MISSING_EXECUTE"
RULE_RESEARCH_MISSING_BRIEF = "DOC_PLAN_RESEARCH_MISSING_BRIEF"
RULE_PLAN_MISSING_EXECUTE = "DOC_PLAN_PLAN_MISSING_EXECUTE"
RULE_LEGACY_STRUCTURE = "DOC_PLAN_LEGACY_STRUCTURE"
RULE_EXCESS_FILES = "DOC_PLAN_EXCESS_FILES"

EXCESS_FILES_THRESHOLD = 25
LIGHTWEIGHT_FILE_THRESHOLD = 5  # 轻量版：文件数 ≤ 5，超过则不算轻量
EXCLUDED_TOPICS = ("archive",)
DEFAULT_WHITELIST_NAME = "plan-structure-whitelist.txt"


def _load_whitelist(whitelist_path: Path | None) -> set[str]:
    """Load whitelist entries. Returns set of 'path' or 'path:rule_id'."""
    if whitelist_path is None or not whitelist_path.exists():
        return set()
    entries: set[str] = set()
    for line in whitelist_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _is_whitelisted(rel_path: str, rule_id: str, whitelist: set[str]) -> bool:
    """Check if this violation is whitelisted."""
    if rel_path in whitelist:
        return True
    if f"{rel_path}:{rule_id}" in whitelist:
        return True
    return False


def _has_execute_file(dir_path: Path) -> bool:
    """True if any .md file under dir_path has 'execute' in its name (case-insensitive)."""
    for p in dir_path.rglob("*.md"):
        if "execute" in p.name.lower():
            return True
    return False


def _has_plan_file(dir_path: Path) -> bool:
    """True if any .md file under dir_path has 'plan' in its name (case-insensitive)."""
    for p in dir_path.rglob("*.md"):
        if "plan" in p.name.lower():
            return True
    return False


def _uses_standard_structure(topic_path: Path) -> bool:
    """True if topic uses 1-research/ or 2-plan/ (full structure)."""
    return (topic_path / "1-research").exists() or (topic_path / "2-plan").exists()


def _uses_lightweight_structure(topic_path: Path) -> bool:
    """True if topic follows lightweight structure: plan.md + execute at root, file count ≤ 5."""
    plan_md = topic_path / "plan.md"
    if not plan_md.exists() or not _has_execute_file(topic_path):
        return False
    return _count_md_files(topic_path) <= LIGHTWEIGHT_FILE_THRESHOLD


def _count_md_files(dir_path: Path) -> int:
    """Count .md files under dir_path (recursive)."""
    return sum(1 for _ in dir_path.rglob("*.md"))


def check_topic(
    topic_path: Path,
    root: Path,
    phase: str,
    whitelist: set[str],
) -> list[tuple[str, str, str]]:
    """Check a single topic dir. Returns list of (rel_path, rule_id, message)."""
    violations: list[tuple[str, str, str]] = []
    try:
        rel = topic_path.resolve().relative_to(root.resolve())
        rel_str = str(rel).replace("\\", "/")
    except ValueError:
        rel_str = str(topic_path)

    # DOC_PLAN_EXECUTING_MISSING_EXECUTE
    if phase == "executing" and topic_path.is_dir():
        if not _has_execute_file(topic_path):
            rule_id = RULE_EXECUTING_MISSING_EXECUTE
            if not _is_whitelisted(rel_str, rule_id, whitelist):
                violations.append(
                    (rel_str, rule_id, "No execute file found")
                )

    # DOC_PLAN_COMPLETED_MISSING_EXECUTE
    if phase == "completed" and topic_path.is_dir():
        if not _has_execute_file(topic_path):
            rule_id = RULE_COMPLETED_MISSING_EXECUTE
            if not _is_whitelisted(rel_str, rule_id, whitelist):
                violations.append(
                    (rel_str, rule_id, "No execute file found")
                )

    # DOC_PLAN_RESEARCH_MISSING_BRIEF
    research_dir = topic_path / "1-research"
    if research_dir.exists() and research_dir.is_dir():
        brief = research_dir / "brief.md"
        if not brief.exists():
            rule_id = RULE_RESEARCH_MISSING_BRIEF
            path_for_rule = f"{rel_str}/1-research"
            if not _is_whitelisted(path_for_rule, rule_id, whitelist):
                violations.append(
                    (path_for_rule, rule_id, "1-research/ exists but brief.md missing")
                )

    # DOC_PLAN_PLAN_MISSING_EXECUTE
    plan_dir = topic_path / "2-plan"
    if plan_dir.exists() and plan_dir.is_dir():
        if not _has_execute_file(plan_dir):
            rule_id = RULE_PLAN_MISSING_EXECUTE
            path_for_rule = f"{rel_str}/2-plan"
            if not _is_whitelisted(path_for_rule, rule_id, whitelist):
                violations.append(
                    (
                        path_for_rule,
                        rule_id,
                        "2-plan/ exists but no execute file found",
                    )
                )

    # DOC_PLAN_LEGACY_STRUCTURE
    if topic_path.is_dir() and not _uses_standard_structure(topic_path) and not _uses_lightweight_structure(topic_path):
        rule_id = RULE_LEGACY_STRUCTURE
        if not _is_whitelisted(rel_str, rule_id, whitelist):
            violations.append(
                (
                    rel_str,
                    rule_id,
                    "Not using 1-research/2-plan/3-retro or plan+execute structure",
                )
            )

    # DOC_PLAN_EXCESS_FILES
    md_count = _count_md_files(topic_path)
    if md_count > EXCESS_FILES_THRESHOLD:
        rule_id = RULE_EXCESS_FILES
        if not _is_whitelisted(rel_str, rule_id, whitelist):
            violations.append(
                (
                    rel_str,
                    rule_id,
                    f"{md_count} .md files (threshold {EXCESS_FILES_THRESHOLD})",
                )
            )

    return violations


def collect_topic_dirs(root: Path) -> list[tuple[Path, str]]:
    """Collect (topic_path, phase) for wait, executing, completed. Excludes archive."""
    results: list[tuple[Path, str]] = []
    base = root / "docs" / "exec-plans"
    if not base.exists():
        return results

    for phase in ("wait", "executing", "completed"):
        phase_dir = base / phase
        if not phase_dir.exists() or not phase_dir.is_dir():
            continue
        for item in phase_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name in EXCLUDED_TOPICS:
                continue
            results.append((item, phase))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan structure check: scan exec-plans for DOC_PLAN_* violations"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: cwd)",
    )
    parser.add_argument(
        "--whitelist",
        type=Path,
        default=None,
        help="Whitelist file (default: scripts/plan-structure-whitelist.txt)",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    whitelist_path = args.whitelist
    if whitelist_path is None:
        whitelist_path = root / "scripts" / DEFAULT_WHITELIST_NAME
    else:
        whitelist_path = whitelist_path.resolve()
    whitelist = _load_whitelist(whitelist_path)

    topic_dirs = collect_topic_dirs(root)
    all_violations: list[tuple[str, str, str]] = []
    for topic_path, phase in topic_dirs:
        for rel_path, rule_id, msg in check_topic(topic_path, root, phase, whitelist):
            all_violations.append((rel_path, rule_id, msg))

    for rel_path, rule_id, msg in all_violations:
        print(f"{rel_path}: {rule_id}: {msg}")

    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
