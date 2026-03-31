#!/usr/bin/env python3
"""One-shot smoke: save a session-level archive and verify search can find it.

Run from repo root:
  PYTHONPATH=src python scripts/smoke_archive_session.py

Requires: DB per config.development.yaml / TEAM_MEMORY_DB_URL, embedding provider (e.g. Ollama).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Synthetic transcript for this chat thread (schema cleanup + archive experiment)
SESSION_DOC = """
## 会话主题：Team Memory MVP 表结构梳理、废弃字段与档案馆实验

### 1. 表结构 / 废弃列
- 当前 MVP 以 models.py 与 001_initial_mvp 为准：experiences、experience_feedbacks、archives、
  archive_experience_links、archive_attachments、document_tree_nodes、personal_memories、api_keys。
- 002_mvp_cleanup 已从旧库删除多表（tool_usage_logs、experience_links 等）及 experiences 上多列
  （avg_rating、source_context、embedding_status 等）。
- 代码曾写 avg_rating、source_context：已修复为不写废列；tm_claim 改为 tags 前缀 agent_claim。

### 2. 后续清理项（已完成部分）
- ruff I001；hooks UsageTrackingHandler no-op；遗留脚本头注释；RetrievalConfig 补 min_avg_rating 等；
  002 中 group_key 幂等添加；文档与 Web hint 对齐。

### 3. 可再讨论的库对象
- document_tree_nodes：表存在但业务未接线 PageIndex。
- experience_feedbacks.fitness_score：API 有，检索排序未用。

### 4. 档案馆实验计划（本脚本即执行项）
- tm_archive_save：title + solution_doc + 可选 overview/conversation_summary；需 embedding 才能被
  include_archives 向量检索命中。
- 校验：tm_get_archive(id)；tm_search(..., include_archives=true)，同 project、created_by 可见 draft。
""".strip()

SUMMARY = (
    "梳理 MVP 表与 002 已删列；修复 feedback/tm_claim；hooks 与脚本标注；"
    "档案馆藏对话与 tm_search include_archives 烟雾测试。"
)


async def main() -> int:
    from team_memory.bootstrap import bootstrap, reset_context

    reset_context()
    ctx = bootstrap(enable_background=False)
    db_url = ctx.db_url
    project = ctx.settings.default_project or "default"
    user = "archive_smoke_user"

    arch = ctx.archive_service
    svc = ctx.service

    title = "MVP 表结构 / 档案馆烟雾测试 — 2026-03-P30"
    scope_ref = "chat-archive-smoke-001"

    print("1) archive_save ...", flush=True)
    aid = await arch.archive_save(
        title=title,
        solution_doc=SESSION_DOC,
        created_by=user,
        project=project,
        scope="session",
        scope_ref=scope_ref,
        overview="DB 废弃列、代码修复、档案馆实验与校验步骤。",
        conversation_summary=SUMMARY,
        linked_experience_ids=None,
        attachments=None,
    )
    print(f"   archive_id={aid}", flush=True)

    print("2) get_archive ...", flush=True)
    got = await arch.get_archive(aid, viewer=user, project=project)
    if not got or "solution_doc" not in got:
        print("   FAIL: get_archive empty", flush=True)
        return 1
    print(f"   title={got.get('title')!r} project={got.get('project')}", flush=True)

    queries = [
        "document_tree_nodes 未接线 PageIndex",
        "tm_archive_save include_archives 向量",
        "agent_claim 认领标签",
    ]
    print("3) search include_archives ...", flush=True)
    for q in queries:
        results = await svc.search(
            query=q,
            tags=None,
            max_results=8,
            min_similarity=0.25,
            user_name=user,
            source="smoke_script",
            grouped=False,
            top_k_children=0,
            project=project,
            include_archives=True,
        )
        archives = [r for r in results if r.get("type") == "archive"]
        hit = any(str(r.get("id")) == str(aid) for r in archives)
        print(
            f"   q={q[:40]!r}... total={len(results)} archives={len(archives)} hit_self={hit}",
            flush=True,
        )

    print("OK", flush=True)
    print(json.dumps({"archive_id": str(aid), "scope_ref": scope_ref}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
