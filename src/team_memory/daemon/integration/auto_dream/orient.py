"""Phase 1 (Orient) and Phase 2 (Gather) mixin for AutoDreamWorker."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from team_memory.daemon.integration.auto_dream.state import _get_daemon_context

logger = logging.getLogger("daemon.auto_dream")


class AutoDreamOrientMixin:
    """Mixin implementing Orient (candidate sampling) and Gather (pgvector clustering)."""

    async def _phase_orient(
        self,
        last_dream: datetime | None,
        *,
        project: str | None = None,
    ) -> tuple[list[dict], int]:
        """Scan recent active experiences (all tiers) + high-recall/zero-adoption experiences.

        Raises on fatal DB/service failure so watermark is not silently advanced.
        """
        recent = await self._get_dream_candidates(last_dream, project=project)

        low_adoption: list[dict] = []
        try:
            low_adoption = await self._get_low_adoption_candidates(project=project)
        except Exception:
            logger.debug("Low-adoption candidate sampling unavailable, continuing with recent", exc_info=True)

        stale_items: list[dict] = []
        try:
            stale_items = await self._get_stale_candidates(project=project)
        except Exception:
            logger.debug("Stale candidate sampling unavailable, continuing with recent", exc_info=True)

        merged: list[dict] = []
        seen_ids: set[str] = set()
        low_count = 0

        for item in recent:
            cid = str(item.get("id") or "")
            if cid and cid not in seen_ids and item.get("valid_until") is None:
                seen_ids.add(cid)
                merged.append(item)

        for item in stale_items:
            cid = str(item.get("id") or "")
            if not cid or item.get("valid_until") is not None:
                continue
            item_copy = dict(item)
            item_copy["_signal"] = "possibly_stale"
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(item_copy)
            else:
                for existing in merged:
                    if str(existing.get("id") or "") == cid:
                        existing["_signal"] = "possibly_stale"
                        break

        for item in low_adoption:
            cid = str(item.get("id") or "")
            if not cid or item.get("valid_until") is not None:
                continue
            item_copy = dict(item)
            item_copy["_signal"] = "low_adoption"
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(item_copy)
                low_count += 1
            else:
                # Tag existing entry as also having low-adoption signal (preserve possibly_stale if already tagged)
                for existing in merged:
                    if str(existing.get("id") or "") == cid:
                        if existing.get("_signal") != "possibly_stale":
                            existing["_signal"] = "low_adoption"
                        low_count += 1
                        break

        return merged, low_count

    async def _get_dream_candidates(
        self,
        last_dream: datetime | None,
        *,
        project: str | None = None,
    ) -> list[dict]:
        """Fetch active published/promoted experiences across ALL quality tiers since last_dream."""
        ctx = _get_daemon_context()
        try:
            candidates = await ctx.service.list_published(
                project=project,
                since=last_dream,
                limit=200,
                include_embedding=True,
                active_only=True,
            )
        except TypeError:
            try:
                candidates = await ctx.service.list_published(
                    project=project,
                    since=last_dream,
                    limit=200,
                    include_embedding=True,
                )
            except TypeError:
                candidates = await ctx.service.list_published(
                    since=last_dream,
                    limit=200,
                )
        return [c for c in candidates if c.get("valid_until") is None]

    async def _get_low_adoption_candidates(self, *, project: str | None = None) -> list[dict]:
        """Sample active experiences with high recall_count but zero used_count."""
        ctx = _get_daemon_context()
        if not hasattr(ctx.service, "list_low_adoption"):
            return []
        result = await ctx.service.list_low_adoption(
            min_recall_count=self._cfg.low_adoption_min_recall,
            max_used_count=0,
            project=project,
            limit=self._cfg.low_adoption_sample_limit,
            include_embedding=True,
        )
        return result if isinstance(result, list) else []

    async def _get_stale_candidates(self, *, project: str | None = None) -> list[dict]:
        """Sample active experiences flagged as possibly_stale for Defense-3 AutoDream verification."""
        ctx = _get_daemon_context()
        if not hasattr(ctx.service, "list_stale_candidates"):
            return []
        result = await ctx.service.list_stale_candidates(
            project=project,
            limit=self._cfg.low_adoption_sample_limit,
            include_embedding=True,
        )
        return result if isinstance(result, list) else []

    async def _find_neighbors(
        self,
        candidate: dict[str, Any],
        *,
        threshold: float,
        exclude_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Find similar active experiences using pgvector directly (zero search_logs pollution).

        Falls back to sink.recall() if embedding is absent or service method is unavailable.
        """
        cand_id = str(candidate.get("id") or "")
        project = candidate.get("project")
        embedding = candidate.get("embedding")

        if embedding:
            try:
                ctx = _get_daemon_context()
                if hasattr(ctx.service, "find_similar_active"):
                    hits = await ctx.service.find_similar_active(
                        embedding,
                        threshold=threshold,
                        limit=10,
                        project=project,
                        include_cross_project=False,
                        exclude_ids=list(exclude_ids | {cand_id}),
                    )
                    if isinstance(hits, list):
                        if self._cfg.enable_cross_project:
                            cross_thr = min(0.92, threshold + 0.08)
                            seen = {str(h.get("id") or "") for h in hits}
                            cross_hits = await ctx.service.find_similar_active(
                                embedding,
                                threshold=cross_thr,
                                limit=5,
                                project=None,
                                include_cross_project=True,
                                exclude_ids=list(exclude_ids | {cand_id} | seen),
                            )
                            if isinstance(cross_hits, list):
                                hits = hits + [h for h in cross_hits if str(h.get("id") or "") not in seen]
                        return hits
            except Exception:
                logger.debug(
                    "Gather: pgvector find_similar_active failed for %s, falling back to recall()",
                    cand_id[:8],
                    exc_info=True,
                )

        query_text = candidate.get("title", "") or candidate.get("description", "")
        if not query_text.strip():
            return []
        try:
            return await self._sink.recall(
                query=query_text[:500],
                project=project,
                max_results=10,
            )
        except Exception:
            logger.warning("Gather: recall fallback failed for %s", cand_id[:8], exc_info=True)
            return []

    async def _phase_gather(
        self,
        candidates: list[dict],
        *,
        max_clusters: int | None = None,
    ) -> list[list[dict]]:
        """Build semantic clusters of related active experiences, capped at max_clusters_per_run."""
        threshold = self._cfg.merge_similarity_threshold
        cap = max_clusters if max_clusters is not None else self._cfg.max_clusters_per_run
        scored_clusters: list[tuple[tuple[int, int, float, int], list[dict]]] = []
        processed_ids: set[str] = set()

        for candidate in candidates:
            cand_id = str(candidate.get("id") or "")
            if not cand_id or cand_id in processed_ids:
                continue

            similar = await self._find_neighbors(
                candidate,
                threshold=threshold,
                exclude_ids=processed_ids,
            )

            cluster_members: list[dict] = [candidate]
            member_ids: set[str] = {cand_id}
            max_sim = 0.0

            for hit in similar:
                hit_id = str(hit.get("id") or "")
                if not hit_id or hit_id == cand_id or hit_id in processed_ids or hit_id in member_ids:
                    continue
                if hit.get("valid_until") is not None:
                    continue
                sim = float(hit.get("similarity", hit.get("score", 0.0)) or 0.0)
                if sim >= threshold:
                    cluster_members.append(hit)
                    member_ids.add(hit_id)
                    if sim > max_sim:
                        max_sim = sim

            if len(cluster_members) >= 2:
                has_stale = int(
                    any(
                        m.get("_signal") == "possibly_stale"
                        or str(m.get("staleness_status") or "fresh") in ("possibly_stale", "stale")
                        for m in cluster_members
                    )
                )
                has_low_adopt = int(any(m.get("_signal") == "low_adoption" for m in cluster_members))
                priority = (has_stale, has_low_adopt, max_sim, len(cluster_members))
                scored_clusters.append((priority, cluster_members))
                processed_ids.update(member_ids)
                logger.debug(
                    "Gather: cluster formed (%d members, max_sim=%.3f, seed=%s)",
                    len(cluster_members),
                    max_sim,
                    cand_id[:8],
                )
            else:
                processed_ids.add(cand_id)

        scored_clusters.sort(key=lambda pair: pair[0], reverse=True)
        if cap > 0 and len(scored_clusters) > cap:
            logger.info(
                "Gather: capping clusters from %d to %d (max_clusters_per_run)",
                len(scored_clusters),
                cap,
            )
            scored_clusters = scored_clusters[:cap]

        return [cluster for _, cluster in scored_clusters]
