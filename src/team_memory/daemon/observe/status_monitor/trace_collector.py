"""Recall pipeline trace reconstruction from search_logs, PG knowledge base, and local Wiki."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from team_memory.daemon.observe.status_monitor.collectors import (
    _clean_query_display,
    _lookup_local_wiki_entries,
)
from team_memory.daemon.observe.status_monitor.trace_helpers import (
    _analyze_trace_query_and_graph,
    _resolve_trace_target,
)
from team_memory.daemon.observe.tm_stats import _resolve_dsn


async def inspect_recall_trace(
    selector: str | int,
    recent_recalls: list[dict[str, Any]] | None = None,
    timeout: float = 4.0,
    limit: int = 20,
) -> dict[str, Any]:
    """Inspect a selected recall log and reconstruct its complete 6-node pipeline trace."""
    dsn = _resolve_dsn()
    import asyncpg

    conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout)
    try:
        target = await _resolve_trace_target(conn, selector, recent_recalls, limit)
        if target is None:
            return {"error": f"未找到对应的召回记录: {selector}"}

        raw_log_query = str(target.get("query") or "")
        query_raw = _clean_query_display(raw_log_query, 240).removeprefix("[Ctx] ").strip()
        project = str(target.get("project") or "default")
        result_ids = list(target.get("result_ids") or [])

        # Node ②: Query Expansion, Entity Extraction, Graph Paths, FTS estimation
        (
            expanded_query,
            added_synonyms,
            candidate_entities,
            matched_entities,
            graph_paths,
            graph_exp_ids,
            fts_exp_ids,
            fts_total_estimate,
        ) = await _analyze_trace_query_and_graph(conn, query_raw)

        # Node ③ & ④: Stage counts and metrics (with intelligent reconstruction if historical None)
        raw_counts = dict(target.get("stage_counts") or {})
        raw_metrics = dict(target.get("stage_metrics") or {})
        counts_reconstructed = False

        if not raw_counts:
            counts_reconstructed = True
            hit_n = len(result_ids)
            vec_est = max(hit_n * 2, 10) if hit_n > 0 else 0
            fts_est = fts_total_estimate
            graph_est = len(graph_exp_ids)
            graph_uniq = len(graph_exp_ids - fts_exp_ids)
            fused_est = max(hit_n, len(set(result_ids) | fts_exp_ids | graph_exp_ids))
            raw_counts = {
                "vec_count": vec_est,
                "fts_count": fts_est,
                "topic_count": 0,
                "graph_count": graph_est,
                "graph_unique_added": graph_uniq,
                "fused_count": fused_est,
                "final_count": hit_n,
            }

        # Node ⑤: Detailed experience cards for result_ids
        recalled_experiences: list[dict[str, Any]] = []
        superseded_hits = 0
        if result_ids:
            exp_rows = await conn.fetch(
                """
                SELECT id::text AS id,
                       title,
                       COALESCE(project, 'default') AS project,
                       COALESCE(exp_status, 'published') AS exp_status,
                       COALESCE(experience_type, 'general') AS experience_type,
                       tags,
                       description,
                       solution,
                       valid_from,
                       valid_until,
                       superseded_by_id::text AS superseded_by_id,
                       COALESCE(used_count, 0) AS use_count,
                       COALESCE(recall_count, 0) AS recall_count,
                       (last_verified_at IS NOT NULL) AS verified
                FROM experiences
                WHERE id::text = ANY($1::text[])
                """,
                result_ids,
            )
            exp_map = {str(er["id"]): er for er in exp_rows}

            missing_arch_ids = [eid for eid in result_ids if eid not in exp_map]
            arch_map: dict[str, Any] = {}
            pm_map: dict[str, Any] = {}
            wiki_map: dict[str, dict[str, Any]] = {}
            if missing_arch_ids:
                try:
                    arch_rows = await conn.fetch(
                        """
                        SELECT id::text AS id,
                               title,
                               COALESCE(project, 'default') AS project,
                               COALESCE(status, 'published') AS exp_status,
                               COALESCE(content_type, 'archive') AS experience_type,
                               tags,
                               COALESCE(overview, value_summary, '') AS description,
                               COALESCE(solution_doc, '') AS solution
                        FROM archives
                        WHERE id::text = ANY($1::text[])
                        """,
                        missing_arch_ids,
                    )
                    arch_map = {str(ar["id"]): ar for ar in arch_rows}
                except Exception:
                    pass

                still_missing = [eid for eid in missing_arch_ids if eid not in arch_map]
                if still_missing:
                    try:
                        pm_rows = await conn.fetch(
                            """
                            SELECT id::text AS id,
                                   user_id,
                                   content,
                                   COALESCE(scope, 'generic') AS scope,
                                   COALESCE(persona, 'general') AS persona,
                                   COALESCE(context_hint, '') AS context_hint
                            FROM personal_memories
                            WHERE id::text = ANY($1::text[])
                            """,
                            still_missing,
                        )
                        pm_map = {str(pr["id"]): pr for pr in pm_rows}
                    except Exception:
                        pass
                    wiki_missing = [eid for eid in still_missing if eid not in pm_map]
                    if wiki_missing:
                        wiki_map = _lookup_local_wiki_entries(wiki_missing)

            # Fetch linked entities per experience
            ee_rows = await conn.fetch(
                """
                SELECT ee.experience_id::text AS exp_id,
                       e.id::text AS entity_id,
                       e.name,
                       e.entity_type,
                       COALESCE(e.source_count, 1) AS source_count,
                       ee.role
                FROM experience_entities ee
                JOIN entities e ON e.id = ee.entity_id
                WHERE ee.experience_id::text = ANY($1::text[])
                ORDER BY e.source_count DESC
                """,
                result_ids,
            )
            exp_entities_map: dict[str, list[str]] = {}
            for eer in ee_rows:
                eid_str = str(eer["exp_id"])
                exp_entities_map.setdefault(eid_str, [])
                ename = str(eer["name"])
                if ename not in exp_entities_map[eid_str] and len(exp_entities_map[eid_str]) < 6:
                    exp_entities_map[eid_str].append(ename)

            # Populate active graph paths from the top entities bridged by these results if needed
            if not matched_entities and ee_rows and int(raw_counts.get("graph_count", 0)) > 0:
                seen_eids: set[str] = set()
                for eer in ee_rows:
                    en_id = str(eer["entity_id"])
                    if en_id not in seen_eids and len(matched_entities) < 4:
                        seen_eids.add(en_id)
                        matched_entities.append(
                            {
                                "id": en_id,
                                "name": str(eer["name"]),
                                "entity_type": str(eer["entity_type"] or "concept"),
                                "project": project,
                                "source_count": int(eer["source_count"] or 1),
                            }
                        )
                if seen_eids and not graph_paths:
                    rel_rows2 = await conn.fetch(
                        """
                        SELECT e1.name AS src_name, r.relation_type, e2.name AS tgt_name,
                               COALESCE(r.weight, 1.0) AS weight
                        FROM relationships r
                        JOIN entities e1 ON e1.id = r.source_entity_id
                        JOIN entities e2 ON e2.id = r.target_entity_id
                        WHERE r.valid_until IS NULL
                          AND (
                              r.source_entity_id::text = ANY($1::text[])
                              OR r.target_entity_id::text = ANY($1::text[])
                          )
                        ORDER BY r.weight DESC
                        LIMIT 6
                        """,
                        list(seen_eids),
                    )
                    for rr in rel_rows2:
                        graph_paths.append(
                            {
                                "src": str(rr["src_name"]),
                                "rel": str(rr["relation_type"]),
                                "tgt": str(rr["tgt_name"]),
                                "weight": round(float(rr["weight"] or 1.0), 2),
                            }
                        )

            for rank, exp_id in enumerate(result_ids, start=1):
                er = exp_map.get(exp_id)
                if not er and exp_id in arch_map:
                    ar = arch_map[exp_id]
                    prob_a = " ".join(str(ar["description"] or "").split())
                    sol_a = " ".join(str(ar["solution"] or "").split())
                    recalled_experiences.append(
                        {
                            "rank": rank,
                            "id": exp_id,
                            "short_id": exp_id[:8],
                            "title": f"[📦L3归档] {ar['title'] or ''}",
                            "project": str(ar["project"] or project),
                            "exp_status": str(ar["exp_status"] or "published"),
                            "experience_type": str(ar["experience_type"] or "archive"),
                            "channels": ["🔵Vector(Archive)"],
                            "is_superseded": False,
                            "verified": False,
                            "tags": [str(t) for t in (ar["tags"] or [])][:6],
                            "entities": [],
                            "problem_snippet": (
                                (prob_a[:120] + "…") if len(prob_a) > 120 else prob_a
                            ),
                            "solution_snippet": (
                                (sol_a[:140] + "…") if len(sol_a) > 140 else sol_a
                            ),
                            "use_count": 0,
                            "recall_count": 1,
                        }
                    )
                    continue
                if not er and exp_id in pm_map:
                    pr = pm_map[exp_id]
                    pm_txt = " ".join(str(pr["content"] or "").split())
                    recalled_experiences.append(
                        {
                            "rank": rank,
                            "id": exp_id,
                            "short_id": exp_id[:8],
                            "title": f"[👤L1个人记忆] {pm_txt[:56]}",
                            "project": f"user:{pr['user_id']}",
                            "exp_status": "active",
                            "experience_type": str(pr["persona"] or "personal"),
                            "channels": ["🟣L1Profile"],
                            "is_superseded": False,
                            "verified": True,
                            "tags": [str(pr["scope"]), str(pr["persona"])],
                            "entities": [],
                            "problem_snippet": str(pr["context_hint"] or "用户个人偏好/全局记忆注入"),
                            "solution_snippet": (
                                (pm_txt[:140] + "…") if len(pm_txt) > 140 else pm_txt
                            ),
                            "use_count": 1,
                            "recall_count": 1,
                        }
                    )
                    continue
                if not er and exp_id in wiki_map:
                    wr = wiki_map[exp_id]
                    w_txt = " ".join(str(wr.get("content") or "").split())
                    recalled_experiences.append(
                        {
                            "rank": rank,
                            "id": exp_id,
                            "short_id": exp_id[:8],
                            "title": f"[📘Wiki卡片] {wr.get('title') or ''}",
                            "project": str(wr.get("project") or project),
                            "exp_status": "published",
                            "experience_type": "wiki_card",
                            "channels": ["📘Wiki(FTS5)"],
                            "is_superseded": False,
                            "verified": True,
                            "tags": list(wr.get("tags") or []),
                            "entities": [],
                            "problem_snippet": str(wr.get("source_url") or "本地 SQLite LLMWiki 知识卡片检索"),
                            "solution_snippet": (
                                (w_txt[:140] + "…") if len(w_txt) > 140 else w_txt
                            ),
                            "use_count": 0,
                            "recall_count": 1,
                        }
                    )
                    continue
                if not er:
                    recalled_experiences.append(
                        {
                            "rank": rank,
                            "id": exp_id,
                            "short_id": exp_id[:8],
                            "title": "(已删除或未找到的经验条目)",
                            "project": project,
                            "channels": ["🔵Vector"],
                            "is_superseded": False,
                            "verified": False,
                            "tags": [],
                            "entities": [],
                            "problem_snippet": "",
                            "solution_snippet": "",
                            "use_count": 0,
                            "recall_count": 0,
                        }
                    )
                    continue

                is_sup = er["valid_until"] is not None or er["superseded_by_id"] is not None
                if is_sup:
                    superseded_hits += 1

                channels = ["🔵Vector"]
                if exp_id in fts_exp_ids:
                    channels.append("🟣FTS")
                if exp_id in graph_exp_ids:
                    channels.append("🟢Graph(2-Hop)")

                tags_val = er["tags"]
                if isinstance(tags_val, str):
                    try:
                        tags_val = json.loads(tags_val)
                    except Exception:
                        tags_val = [tags_val]
                tags_list = [str(t) for t in (tags_val or [])][:6]

                prob_clean = " ".join(str(er["description"] or "").split())
                sol_clean = " ".join(str(er["solution"] or "").split())

                recalled_experiences.append(
                    {
                        "rank": rank,
                        "id": exp_id,
                        "short_id": exp_id[:8],
                        "title": str(er["title"] or ""),
                        "project": str(er["project"] or "default"),
                        "exp_status": str(er["exp_status"] or "published"),
                        "experience_type": str(er["experience_type"] or "general"),
                        "channels": channels,
                        "is_superseded": is_sup,
                        "superseded_by": str(er["superseded_by_id"] or "")[:8],
                        "verified": bool(er["verified"]),
                        "tags": tags_list,
                        "entities": exp_entities_map.get(exp_id, []),
                        "problem_snippet": (
                            (prob_clean[:120] + "…") if len(prob_clean) > 120 else prob_clean
                        ),
                        "solution_snippet": (
                            (sol_clean[:140] + "…") if len(sol_clean) > 140 else sol_clean
                        ),
                        "use_count": int(er["use_count"] or 0),
                        "recall_count": int(er["recall_count"] or 0),
                    }
                )

        # Estimate token count if historical row had returned_tokens=None
        ret_tokens = target.get("returned_tokens")
        tokens_estimated = False
        if ret_tokens is None:
            tokens_estimated = True
            total_chars = sum(
                len(x.get("title", ""))
                + len(x.get("problem_snippet", ""))
                + len(x.get("solution_snippet", ""))
                for x in recalled_experiences
            )
            ret_tokens = (
                max(60 * len(recalled_experiences), int(total_chars * 0.6))
                if recalled_experiences
                else 0
            )

        return {
            "log": target,
            "query_analysis": {
                "raw_query": query_raw,
                "expanded_query": expanded_query,
                "added_synonyms": added_synonyms,
                "candidate_entities": candidate_entities,
                "matched_entities": matched_entities,
                "graph_paths": graph_paths,
            },
            "stage_counts": raw_counts,
            "counts_reconstructed": counts_reconstructed,
            "stage_metrics": raw_metrics,
            "superseded_hits": superseded_hits,
            "returned_tokens": ret_tokens,
            "tokens_estimated": tokens_estimated,
            "recalled_experiences": recalled_experiences,
        }
    finally:
        await conn.close()
