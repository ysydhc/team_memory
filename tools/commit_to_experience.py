#!/usr/bin/env python3
"""从 Git 提交生成带 file_locations 的经验并可选写入 tm。

用法:
  python tools/commit_to_experience.py --range 397735e^..72c28f3 [--dry-run]
  python tools/commit_to_experience.py --range 397735e^..72c28f3 --save

约定:
- 只记录「核心目录」改动：src/team_memory/、migrations/
- 排除：仅 docs/、仅 tests/、仅注释/格式的提交（本脚本按「是否含核心路径」过滤）
- 每条经验对应一笔提交，file_locations 为该提交在核心路径上的 diff 行范围（新文件侧）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# 核心目录：这些路径的修改必须记录
CORE_PREFIXES = ("src/team_memory/", "migrations/")

# 排除的路径：仅改这些不生成经验（与核心路径取差后若为空则整笔提交可跳过）
SKIP_ONLY_PREFIXES = ("docs/", "tests/", "scripts/")

# 文件位置绑定需求：用于生成「为什么、注意点」的上下文
FEATURE_CONTEXT = """
本批提交实现「经验文件位置绑定」：经验与文件路径+行范围绑定，内容指纹稳定匹配；
TTL 默认 30 天、访问时刷新；检索用 location_weight 融合位置得分；
管道禁止 N×M 调用，需 list_bindings_by_paths 批量查后在内存算 location_score；
过期绑定由定时任务清理。常量与指纹仅由 utils/location_fingerprint 单点实现。
"""

# 按 subject 子串匹配补充「做了什么、注意点」（来自计划与执行过程）；先匹配的优先
COMMIT_NOTES: list[tuple[str, tuple[str, str]]] = [
    ("feat(repository): 文件位置 TTL", (
        "list_bindings_by_paths 支持 refresh_on_access 并批量更新 expires_at；过期不返回；测试过期不返回与 refresh 延后。",
        "ttl_days、refresh_on_access 由调用方传入。",
    )),
    ("feat(repository)", (
        "实现 replace/get/list_bindings_by_paths/find_experience_ids_by_location/delete_expired；得分用 LOCATION_SCORE_EXACT/SAME_FILE。",
        "content_fingerprint 必须由调用方传入；禁止仓储内算指纹；批量查询返回 path->list[binding]。",
    )),
    ("feat(config)", (
        "在 SearchConfig 增加 location_weight、新增 FileLocationBindingConfig 与 Settings.file_location_binding。",
        "location_weight 推荐 0.1～0.25，默认 0.15；cleanup_interval_hours 控制清理周期。",
    )),
    ("feat(storage)", (
        "新增 ExperienceFileLocation 模型与迁移；Experience 上增加 file_locations relationship。",
        "回滚会删除 experience_file_locations 全表数据；索引 (experience_id,path)、(path,content_fingerprint)。",
    )),
    ("feat(utils)", (
        "location_fingerprint.py：归一化、content_fingerprint、find_fingerprint_in_lines(window_size=20)、lines_overlap、overlap_score；导出常量。",
        "全计划唯一指纹与 window 单点；仓储/管道只引用不重算。",
    )),
    ("feat(experience)", (
        "save()/update() 接受 file_locations；用 utils 算 content_fingerprint；调用 replace_file_location_bindings，expires_at=now+ttl_days。",
        "bootstrap 传入 file_location_config；无 snippet 则不填 content_fingerprint。",
    )),
    ("feat(mcp)", (
        "tm_save/tm_save_typed 增加 file_locations 参数并透传；tm_search/tm_solve 增加 current_file_locations 并传入 SearchRequest。",
        "工具描述中需说明 path、start_line、end_line 及可选 snippet/file_mtime/file_content_hash。",
    )),
    ("feat(web)", (
        "API schema 与路由支持 file_locations 创建/更新/详情；设置项 location_weight 与 file_location TTL/refresh/cleanup；录入 path 或 path:start-end。",
        "首版仅手动输入；详情展示「关联位置」只读列表。",
    )),
    ("feat(search)", (
        "SearchRequest.current_file_locations；管道 _apply_location_boost：一次 list_bindings_by_paths，内存算 location_score，final_score=rrf+location_weight*location_score；refresh_file_location_bindings。",
        "禁止 N×M 调用；可观测性日志记录 current_file_locations 数、绑定数、加分候选数、耗时。",
    )),
    ("feat(ops)", (
        "bootstrap 中 _file_location_cleanup_loop，按 interval_hours 周期调用 delete_expired_file_location_bindings(batch_size=500)。",
        "清理轮次打日志：deleted、duration；cleanup_enabled=False 时不启动任务。",
    )),
    ("docs:", (
        "设计文档实现小节、GETTING-STARTED、mcp-patterns 文件位置绑定参数与清理/可观测性说明。",
        "MCP 工具列表需与 server 参数一致。",
    )),
    ("chore:", (
        "E2E test_file_locations_e2e_location_score_and_final_score；重叠 1.0、同文件不重叠 0.7、final_score 抬升断言。",
        "有 PostgreSQL 时跑完整 E2E 验证。",
    )),
]


def run_git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout or ""


def get_commits_in_range(repo_root: Path, rev_range: str) -> list[tuple[str, str, str]]:
    """返回 (hash, subject, body) 列表，旧到新。"""
    out = run_git(
        ["log", "--reverse", "--format=%H%n%s%n%b<<<END>>>", rev_range],
        repo_root,
    )
    commits = []
    for block in out.split("<<<END>>>"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        h, subj = lines[0], lines[1]
        body = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
        commits.append((h, subj, body))
    return commits


def parse_diff_for_core_hunks(repo_root: Path, commit_hash: str) -> list[dict]:
    """解析该提交的 diff，只保留核心路径的 (path, start_line, end_line)。"""
    out = run_git(["diff", f"{commit_hash}^", commit_hash], repo_root)
    locations = []
    current_path = None
    for line in out.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+?)(?:\s|$)", line)
            if m:
                current_path = m.group(2)
            continue
        if line.startswith("@@ "):
            if not current_path:
                continue
            if not any(current_path.startswith(p) for p in CORE_PREFIXES):
                continue
            # @@ -old_s,old_c +new_s,new_c @@
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if not m:
                continue
            new_start = int(m.group(1))
            new_count = int(m.group(2) or 1)
            if new_count <= 0:
                continue
            end_line = new_start + new_count - 1
            locations.append({
                "path": current_path,
                "start_line": new_start,
                "end_line": end_line,
            })
    return locations


def has_core_changes(repo_root: Path, commit_hash: str) -> bool:
    """该提交是否涉及核心路径。"""
    out = run_git(["show", commit_hash, "--name-only", "--format="], repo_root)
    for name in out.splitlines():
        name = name.strip()
        if any(name.startswith(p) for p in CORE_PREFIXES):
            return True
    return False


def _commit_notes(subject: str) -> tuple[str, str] | None:
    for key, (what, note) in COMMIT_NOTES:
        if key in subject:
            return (what, note)
    return None


def build_experience(
    commit_hash: str,
    subject: str,
    body: str,
    file_locations: list[dict],
) -> dict:
    """生成单条经验：做了什么、为什么、注意点。"""
    title = subject
    problem = (
        f"实现/落地本提交：{subject}\n\n"
        f"提交：{commit_hash}\n"
        f"涉及核心路径 {len(file_locations)} 处改动，需在后续编辑或检索时能关联到该经验。"
    )
    parts = []
    if body.strip():
        parts.append(body.strip())
    custom = _commit_notes(subject)
    if custom:
        what, note = custom
        parts.append(f"做了什么：{what}\n注意点：{note}")
    solution = "\n\n".join(parts) if parts else "见提交信息与 diff。"
    if FEATURE_CONTEXT.strip():
        solution += "\n\n【功能上下文】" + FEATURE_CONTEXT.strip()
    solution += (
        "\n\n通用注意点：核心目录（src/team_memory/、migrations/）的修改已绑定行号；"
        "检索时传入 current_file_locations 可提升相关经验排序；"
        "管道层禁止按「每候选×每位置」逐次查库，需批量 list_bindings_by_paths。"
    )
    return {
        "title": title,
        "problem": problem,
        "solution": solution,
        "file_locations": file_locations,
        "tags": ["file_location_binding", "git_commit"],
        "experience_type": "tech_design",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从 Git 提交生成带 file_locations 的经验")
    parser.add_argument(
        "--range",
        default="397735e^..72c28f3",
        help="git rev-range，默认为位置绑定需求全部提交",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="仓库根目录",
    )
    parser.add_argument("--dry-run", action="store_true", help="只输出 JSON，不写入 tm")
    parser.add_argument("--save", action="store_true", help="调用 tm 写入经验")
    args = parser.parse_args()
    repo_root = args.repo

    commits = get_commits_in_range(repo_root, args.range)
    experiences = []
    for h, subj, body in commits:
        if not has_core_changes(repo_root, h):
            continue
        locations = parse_diff_for_core_hunks(repo_root, h)
        if not locations:
            continue
        exp = build_experience(h, subj, body, locations)
        exp["_commit"] = h
        experiences.append(exp)

    if args.dry_run or not args.save:
        print(json.dumps(experiences, ensure_ascii=False, indent=2))
        return 0

    if not args.save:
        return 0

    # --save: 在仓库根目录 bootstrap 后调用 service.save
    os.chdir(repo_root)
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    from team_memory.bootstrap import bootstrap, get_context

    bootstrap(enable_background=False)
    ctx = get_context()
    service = ctx.service

    import asyncio

    # 与 MCP 一致：从环境变量读取（项目级 mcp.json 的 env 或脚本前导 export）
    default_user = os.environ.get("TEAM_MEMORY_USER", "system")
    default_project = os.environ.get("TEAM_MEMORY_PROJECT", "") or get_context().settings.default_project or "default"

    async def save_all() -> None:
        for exp in experiences:
            out = await service.save(
                title=exp["title"],
                problem=exp["problem"],
                solution=exp["solution"],
                file_locations=exp["file_locations"],
                tags=exp.get("tags", []),
                experience_type=exp.get("experience_type", "tech_design"),
                publish_status="published",
                created_by=default_user,
                project=default_project,
            )
            print("Saved:", out.get("id"), exp["title"][:50], file=sys.stderr)

    asyncio.run(save_all())
    return 0


if __name__ == "__main__":
    sys.exit(main())
