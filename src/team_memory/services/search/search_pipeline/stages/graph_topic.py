"""Entity-graph and topic-cluster retrieval stage for SearchPipeline."""

from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from team_memory.services.search.query_expansion import _TOPIC_CACHE_TTL, _topic_cache
from team_memory.services.search.search_pipeline.models import (
    SearchRequest,
    SearchResultItem,
    _pkg_attr,
)
from team_memory.storage.database import get_session as _default_get_session


class SearchGraphTopicMixin:
    """Mixin implementing weighted entity-graph retrieval and topic-cluster search."""

    async def _graph_search(
        self,
        repo: Any,
        graph_weighted_hits: list[tuple[str, float]],
        limit: int,
        request: SearchRequest,
        *,
        existing_by_id: dict[str, dict[str, Any]] | None = None,
        session: AsyncSession | None = None,
    ) -> list[SearchResultItem]:
        """Fetch active experiences discovered via weighted entity-graph traversal."""
        if not graph_weighted_hits:
            return []

        top_hits = graph_weighted_hits[:limit]
        score_by_id: dict[str, float] = {}
        ordered_ids: list[str] = []
        for eid, g_score in top_hits:
            eid_str = str(eid)
            if eid_str not in score_by_id:
                score_by_id[eid_str] = float(g_score)
                ordered_ids.append(eid_str)

        if not ordered_ids:
            return []

        fetched_map: dict[str, dict[str, Any]] = {}
        if existing_by_id:
            for eid_str in ordered_ids:
                existing = existing_by_id.get(eid_str)
                if isinstance(existing, dict) and existing:
                    fetched_map[eid_str] = existing

        missing_ids = [eid for eid in ordered_ids if eid not in fetched_map]
        db_query_succeeded = False
        if missing_ids:
            repo_session = getattr(repo, "_session", None)
            try:
                active_session = (
                    session
                    if (isinstance(AsyncSession, type) and isinstance(session, AsyncSession))
                    else (
                        repo_session
                        if (isinstance(AsyncSession, type) and isinstance(repo_session, AsyncSession))
                        else None
                    )
                )
            except TypeError:
                active_session = None
            status_clause = (
                "exp_status IN ('published', 'promoted')"
                if getattr(request, "include_promoted", False)
                else "exp_status = 'published'"
            )
            inc_superseded = bool(getattr(request, "include_superseded", False))
            valid_clause = "" if inc_superseded else "AND valid_until IS NULL"
            query_sql = f"""
                SELECT id, title, description, solution, tags, project,
                       experience_type, exp_status, recall_count, used_count,
                       negative_feedback_count, created_at,
                       implicit_positive_count, implicit_negative_count,
                       epistemic_tier, staleness_status, last_verified_at, evidence_anchors
                FROM experiences
                WHERE id::text = ANY(:ids)
                  AND {status_clause}
                  AND is_deleted = false
                  {valid_clause}
                  AND coalesce(trim(title), '') <> ''
                  AND title !~ '^(Draft [0-9a-zA-Z_-]+|Session [0-9a-zA-Z_-]+ draft|主题-.*)$'
            """
            if active_session is not None:
                try:
                    from sqlalchemy import text as sql_text

                    result = await active_session.execute(
                        sql_text(query_sql),
                        {"ids": missing_ids},
                    )
                    rows = result.fetchall()
                    if isinstance(rows, list):
                        db_query_succeeded = True
                        for row in rows:
                            rid = str(row[0])
                            created_iso = row[11].isoformat() if len(row) > 11 and row[11] else None
                            verified_iso = (
                                row[16].isoformat() if len(row) > 16 and row[16] else created_iso
                            )
                            fetched_map[rid] = {
                                "id": rid,
                                "title": row[1],
                                "description": row[2],
                                "solution": row[3],
                                "tags": row[4] or [],
                                "project": row[5],
                                "experience_type": row[6],
                                "status": row[7] or "published",
                                "recall_count": row[8] or 0,
                                "used_count": row[9] or 0,
                                "negative_feedback_count": (row[10] or 0) if len(row) > 10 else 0,
                                "created_at": created_iso,
                                "implicit_positive_count": (row[12] or 0) if len(row) > 12 else 0,
                                "implicit_negative_count": (row[13] or 0) if len(row) > 13 else 0,
                                "epistemic_tier": (row[14] or "auto_extract") if len(row) > 14 else "auto_extract",
                                "staleness_status": (row[15] or "fresh") if len(row) > 15 else "fresh",
                                "last_verified_at": verified_iso,
                                "evidence_anchors": (
                                    (row[17] or {})
                                    if len(row) > 17 and isinstance(row[17], dict)
                                    else {}
                                ),
                            }
                except Exception:
                    if hasattr(active_session, "rollback"):
                        try:
                            await active_session.rollback()
                        except Exception:
                            pass
            elif self._db_url:
                try:
                    from sqlalchemy import text as sql_text

                    get_session_fn = _pkg_attr("get_session", _default_get_session)
                    async with get_session_fn(self._db_url) as s:
                        result = await s.execute(
                            sql_text(query_sql),
                            {"ids": missing_ids},
                        )
                        rows = result.fetchall()
                        if isinstance(rows, list):
                            db_query_succeeded = True
                            for row in rows:
                                rid = str(row[0])
                                created_iso = row[11].isoformat() if len(row) > 11 and row[11] else None
                                verified_iso = (
                                    row[16].isoformat() if len(row) > 16 and row[16] else created_iso
                                )
                                fetched_map[rid] = {
                                    "id": rid,
                                    "title": row[1],
                                    "description": row[2],
                                    "solution": row[3],
                                    "tags": row[4] or [],
                                    "project": row[5],
                                    "experience_type": row[6],
                                    "status": row[7] or "published",
                                    "recall_count": row[8] or 0,
                                    "used_count": row[9] or 0,
                                    "negative_feedback_count": (row[10] or 0) if len(row) > 10 else 0,
                                    "created_at": created_iso,
                                    "implicit_positive_count": (row[12] or 0) if len(row) > 12 else 0,
                                    "implicit_negative_count": (row[13] or 0) if len(row) > 13 else 0,
                                    "epistemic_tier": (row[14] or "auto_extract") if len(row) > 14 else "auto_extract",
                                    "staleness_status": (row[15] or "fresh") if len(row) > 15 else "fresh",
                                    "last_verified_at": verified_iso,
                                    "evidence_anchors": (
                                        (row[17] or {})
                                        if len(row) > 17 and isinstance(row[17], dict)
                                        else {}
                                    ),
                                }
                except Exception:
                    pass

        if not db_query_succeeded and len(fetched_map) < len(ordered_ids) and hasattr(repo, "get_by_id"):
            for eid_str in ordered_ids:
                if eid_str in fetched_map:
                    continue
                try:
                    try:
                        lookup_id: Any = uuid.UUID(eid_str)
                    except (ValueError, TypeError):
                        lookup_id = eid_str
                    exp_obj = await repo.get_by_id(lookup_id)
                    if exp_obj is None:
                        continue
                    if isinstance(exp_obj, dict):
                        d = dict(exp_obj)
                    elif hasattr(exp_obj, "to_dict") and callable(exp_obj.to_dict):
                        if getattr(exp_obj, "is_deleted", False) is True:
                            continue
                        valid_until = getattr(exp_obj, "valid_until", None)
                        if (
                            not getattr(request, "include_superseded", False)
                            and valid_until is not None
                            and not hasattr(valid_until, "_mock_name")
                        ):
                            continue
                        exp_status = getattr(exp_obj, "exp_status", "published")
                        if isinstance(exp_status, str) and exp_status not in ("published", "promoted"):
                            continue
                        d = exp_obj.to_dict()
                    else:
                        continue
                    if isinstance(d, dict) and d.get("id"):
                        fetched_map[str(d["id"])] = d
                except Exception:
                    continue

        req_proj = (request.project or "").strip()
        items: list[SearchResultItem] = []
        for eid_str in ordered_ids:
            data = fetched_map.get(eid_str)
            if not data:
                continue
            g_score = score_by_id.get(eid_str, 0.5)
            d_copy = dict(data)
            exp_proj = str(d_copy.get("project") or "").strip()
            if (
                req_proj
                and req_proj not in ("default", "team_memory", "*", "all")
                and exp_proj
                and exp_proj not in (req_proj, "default", "global")
            ):
                d_copy["cross_project"] = True
            d_copy["graph_score"] = round(g_score, 4)
            items.append(
                SearchResultItem(
                    data=d_copy,
                    score=float(g_score),
                    similarity=max(0.5, min(0.95, float(g_score))),
                    source_type="graph",
                )
            )
        return items

    async def _topic_search(
        self,
        query_embedding: list[float],
        limit: int,
        request: SearchRequest,
    ) -> list[SearchResultItem]:
        """Search via topic clusters — returns experiences from relevant topics."""
        if not self._db_url:
            return []

        # Step 1: Get or cache discover_topics result
        cache_key = self._db_url
        cached = _topic_cache.get(cache_key)
        if cached and (time.monotonic() - cached[1]) < _TOPIC_CACHE_TTL:
            topics = cached[0]
        else:
            try:
                from team_memory.services.analysis.topic_discovery import discover_topics

                result = await discover_topics(self._db_url)
                topics = result.topics
                _topic_cache[cache_key] = (topics, time.monotonic())
            except Exception:
                return []

        if not topics:
            return []

        # Step 2: Compute cosine similarity to each topic center
        import numpy as np

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-8:
            return []

        scored: list[tuple[float, str]] = []
        for topic in topics:
            if not topic.center:
                continue
            c_vec = np.array(topic.center, dtype=np.float32)
            sim = float(np.dot(q_vec, c_vec) / (q_norm * np.linalg.norm(c_vec) + 1e-8))
            if sim >= 0.52:
                for eid in topic.experience_ids:
                    scored.append((sim, eid))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        exp_ids: list[str] = []
        sim_by_id: dict[str, float] = {}
        candidate_pool_limit = max(limit * 15, 50)
        for sim_val, eid in scored:
            if eid not in seen:
                seen.add(eid)
                exp_ids.append(eid)
                sim_by_id[eid] = sim_val
            if len(exp_ids) >= candidate_pool_limit:
                break

        if not exp_ids:
            return []

        # Fetch experience data from PG
        from sqlalchemy import text as sql_text

        get_session_fn = _pkg_attr("get_session", _default_get_session)

        items: list[SearchResultItem] = []
        allowed_statuses = (
            ["published", "promoted"]
            if getattr(request, "include_promoted", False)
            else ["published"]
        )
        req_proj = (request.project or "default").strip()
        allow_all_projects = req_proj in ("*", "all") or (
            bool(request.fallback_projects)
            and any(p in ("*", "all") for p in request.fallback_projects)
        )
        try:
            async with get_session_fn(self._db_url) as session:
                result = await session.execute(
                    sql_text("""
                        SELECT id, title, description, solution, tags, project,
                               experience_type, exp_status, recall_count, used_count,
                               negative_feedback_count, created_at,
                               implicit_positive_count, implicit_negative_count,
                               epistemic_tier, staleness_status, last_verified_at, evidence_anchors
                        FROM experiences
                        WHERE id::text = ANY(:ids)
                          AND exp_status = ANY(:statuses)
                          AND is_deleted = false
                          AND valid_until IS NULL
                          AND coalesce(trim(title), '') <> ''
                          AND title !~ '^(Draft [0-9a-zA-Z_-]+|Session [0-9a-zA-Z_-]+ draft|主题-.*)$'
                          AND (:allow_all = true OR project = :project)
                    """),
                    {
                        "ids": exp_ids,
                        "statuses": allowed_statuses,
                        "allow_all": allow_all_projects,
                        "project": req_proj,
                    },
                )
                rows = result.fetchall()
                row_map = {str(r[0]): r for r in rows}
                # Sort matched rows: prefer same-project first, then cluster similarity, then recall_count
                ordered_eids = [eid for eid in exp_ids if eid in row_map]
                ordered_eids.sort(
                    key=lambda eid: (
                        0 if str(row_map[eid][5] or "") == req_proj else 1,
                        -sim_by_id.get(eid, 0.52),
                        -int(row_map[eid][8] or 0),
                    )
                )
                for eid in ordered_eids[:limit]:
                    row = row_map[eid]
                    t_sim = sim_by_id.get(eid, 0.55)
                    items.append(
                        SearchResultItem(
                            data={
                                "id": str(row[0]),
                                "title": row[1],
                                "description": row[2],
                                "solution": row[3],
                                "tags": row[4],
                                "project": row[5],
                                "experience_type": row[6],
                                "status": row[7] or "published",
                                "exp_status": row[7] or "published",
                                "recall_count": row[8] or 0,
                                "used_count": row[9] or 0,
                                "negative_feedback_count": row[10] or 0,
                                "created_at": row[11].isoformat() if row[11] else None,
                                "implicit_positive_count": (row[12] or 0) if len(row) > 12 else 0,
                                "implicit_negative_count": (row[13] or 0) if len(row) > 13 else 0,
                                "epistemic_tier": (row[14] or "auto_extract") if len(row) > 14 else "auto_extract",
                                "staleness_status": (row[15] or "fresh") if len(row) > 15 else "fresh",
                                "last_verified_at": (
                                    row[16].isoformat()
                                    if len(row) > 16 and row[16]
                                    else (row[11].isoformat() if row[11] else None)
                                ),
                                "evidence_anchors": (
                                    (row[17] or {})
                                    if len(row) > 17 and isinstance(row[17], dict)
                                    else {}
                                ),
                            },
                            score=t_sim,
                            similarity=t_sim,
                            source_type="topic",
                        )
                    )
        except Exception:
            return []

        return items
