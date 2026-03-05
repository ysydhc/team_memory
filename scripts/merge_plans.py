#!/usr/bin/env python3
"""
Merge .debug/plan/*.plan.md by theme: keep latest per theme (by mtime).
- Confirmed TM -> .debug/plan/{theme}.plan.md
- Non-TM or uncertain -> ~/.cursor/plans/{theme}.plan.md
"""
from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path

PLAN_DIR = Path(__file__).resolve().parents[1] / ".debug" / "plan"
CURSOR_PLANS = Path.home() / ".cursor" / "plans"
HASH_PATTERN = re.compile(r"^(.+)_([0-9a-f]{8})\.plan\.md$", re.I)

# Confirmed TM themes (prefix or contains match)
TM_PREFIXES = (
    "team_doc_",
    "experience_",
    "database_doc_",
    "mcp_server_",
    "retrieval_",
    "pageindex-lite",
    "installable",
    "父子经验",
    "检索质量",
    "痛点修复",
    "ruff_fixes",
    "multi-perspective",
    "priority_split",
    "p1-p4_",
    "通用化平台",
    "验证管道",
    "ollama_embedding",
    "tm经验",
    "teammemory",
    "workflow_",
    "phase_6",
    "task_6",
    "skills_rules",
    "experience_",
    "multi-user",
    "git_history",
    "readme_",
    "settings_ui",
    "web_ui",
    "ui_fix",
    "data_fix",
    "api_key_",
    "mcp_",
    "三层架构",
    "个人记忆",
    "去重检测",
    "任务组模式",
    "阶段",
    "经验组_",
    "使用统计",
    "项目管理",
    "架构重构",
    "拆分_",
    "截图元素",
    "错误自愈",
    "driver层",
)

# Explicit non-TM themes (prefix or exact)
NON_TM_PREFIXES = (
    "hmdriver2",
    "harmony-android",
    "gravity_ad",
    "phone-pilot",
    "代码整洁与文档同步",
)


def extract_theme(filename: str) -> str | None:
    """Extract theme from filename like 'theme_hash.plan.md'. Returns None for non-plan files."""
    if not filename.endswith(".plan.md"):
        return None
    m = HASH_PATTERN.match(filename)
    if m:
        return m.group(1)
    # Single-version: whole name minus .plan.md
    return filename[:-9] if filename.endswith(".plan.md") else None


def is_tm(theme: str) -> bool:
    """True if theme is confirmed TM."""
    for p in TM_PREFIXES:
        if theme.startswith(p) or p in theme:
            return True
    return False


def is_non_tm(theme: str) -> bool:
    """True if theme is explicitly non-TM."""
    for p in NON_TM_PREFIXES:
        if theme.startswith(p) or p in theme or theme == p:
            return True
    return False


def main() -> None:
    CURSOR_PLANS.mkdir(parents=True, exist_ok=True)

    # Collect plan files (exclude D-architecture-detail.md)
    by_theme: dict[str, list[Path]] = defaultdict(list)
    for f in PLAN_DIR.glob("*.plan.md"):
        if f.name == "D-architecture-detail.md":
            continue
        theme = extract_theme(f.name)
        if theme:
            by_theme[theme].append(f)

    kept = 0
    deleted = 0
    to_tm = []
    to_cursor = []

    for theme, files in by_theme.items():
        # Sort by mtime desc, keep newest
        files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        keep_path = files_sorted[0]
        to_delete = files_sorted[1:]

        dest_name = f"{theme}.plan.md"
        if is_tm(theme):
            dest = PLAN_DIR / dest_name
            to_tm.append((keep_path, dest, theme))
        else:
            dest = CURSOR_PLANS / dest_name
            to_cursor.append((keep_path, dest, theme))

        kept += 1
        deleted += len(to_delete)

    # Execute: copy kept file to dest, then delete originals (never delete dest)
    for src, dest, theme in to_tm:
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        for old in by_theme[theme]:
            if old.resolve() != dest.resolve():
                old.unlink()

    for src, dest, theme in to_cursor:
        shutil.copy2(src, dest)
        for old in by_theme[theme]:
            old.unlink()  # all in plan dir, dest is in cursor

    print(f"TM -> .debug/plan: {len(to_tm)}")
    print(f"Non-TM/uncertain -> ~/.cursor/plans: {len(to_cursor)}")
    print(f"Kept: {kept}, Deleted: {deleted}")


if __name__ == "__main__":
    main()
