"""Auto Dream worker: periodic background consolidation of near-duplicate experiences.

Inspired by Claude Code's 4-phase Auto Dream process, adapted for our PG-based system.

Phases:
  1. Orient  — scan recent changes, identify candidates
  2. Gather  — find related memories via vector search
  3. Consolidate — merge near-duplicates
  4. Prune   — soft-delete stale/unused experiences

Trigger conditions (both must be met):
  - At least min_hours_since_last hours since last dream run
  - At least min_new_experiences created since last dream run
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from team_memory.daemon.integration.tm_sink import TMSink, _daemon_context
from team_memory.daemon.runtime.config import AutoDreamSettings, DaemonConfig

logger = logging.getLogger("daemon.auto_dream")

# Default cache directory for dream state
_CACHE_DIR = Path(os.path.expanduser("~/.cache/tm-pipeline"))
_LAST_DREAM_FILE = _CACHE_DIR / "last_dream.txt"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _read_last_dream() -> datetime | None:
    """Read last dream timestamp from file. Returns None if never run."""
    try:
        if _LAST_DREAM_FILE.exists():
            text = _LAST_DREAM_FILE.read_text().strip()
            if text:
                return datetime.fromisoformat(text)
    except Exception:
        logger.debug("Failed to read last_dream file", exc_info=True)
    return None


def _write_last_dream(ts: datetime) -> None:
    """Persist last dream timestamp."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_DREAM_FILE.write_text(ts.isoformat())
    except Exception:
        logger.warning("Failed to write last_dream file", exc_info=True)


# ---------------------------------------------------------------------------
# AutoDreamWorker
# ---------------------------------------------------------------------------


class AutoDreamWorker:
    """Periodic background worker that consolidates and prunes experiences.

    Usage::

        worker = AutoDreamWorker(config, sink)
        task = asyncio.create_task(worker.run())
        ...
        worker.stop()
        await task
    """

    def __init__(
        self,
        config: DaemonConfig,
        sink: TMSink,
    ) -> None:
        self._cfg: AutoDreamSettings = config.auto_dream
        self._sink = sink
        self._running = False
        self._paused = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle

    def start(self) -> asyncio.Task:
        """Start the background worker."""
        if not self._cfg.enabled:
            logger.info("AutoDreamWorker is disabled in config")
            self._task = asyncio.create_task(self._noop())
            return self._task

        self._running = True
        self._paused = False
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "AutoDreamWorker started "
            "(scan_interval=%dm, min_hours=%d, min_new=%d, merge_threshold=%.2f, prune_age=%dd)",
            self._cfg.scan_interval_minutes,
            self._cfg.min_hours_since_last,
            self._cfg.min_new_experiences,
            self._cfg.merge_similarity_threshold,
            self._cfg.prune_max_age_days,
        )
        return self._task

    async def _noop(self) -> None:
        """No-op task when worker is disabled."""
        pass

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        if self._task is not None:
            self._task.cancel()

    async def _loop(self) -> None:
        """Main loop: sleep, check triggers, dream, repeat."""
        while self._running:
            if self._paused:
                await asyncio.sleep(60)
                continue
            try:
                await self._tick()
            except asyncio.CancelledError:
                logger.debug("AutoDreamWorker loop cancelled")
                raise
            except Exception:
                logger.exception("AutoDreamWorker tick failed")

            try:
                await asyncio.sleep(self._cfg.scan_interval_minutes * 60)
            except asyncio.CancelledError:
                logger.debug("AutoDreamWorker sleep cancelled")
                raise

    # ------------------------------------------------------------------
    # Per-tick logic
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """Check trigger conditions and run dream if met."""
        last_dream = _read_last_dream()
        now = _utcnow()

        # Check trigger condition 1: time since last dream
        if last_dream is not None:
            hours_since = (now - last_dream).total_seconds() / 3600
            if hours_since < self._cfg.min_hours_since_last:
                logger.debug(
                    "AutoDream: only %.1fh since last dream (need %dh), skipping",
                    hours_since,
                    self._cfg.min_hours_since_last,
                )
                return
        else:
            # Never run before — check if there are enough experiences
            logger.info("AutoDream: no previous dream run found")

        # Check trigger condition 2: new experiences since last dream
        new_count = await self._count_new_experiences(last_dream)
        if new_count < self._cfg.min_new_experiences:
            logger.debug(
                "AutoDream: only %d new experiences (need %d), skipping",
                new_count,
                self._cfg.min_new_experiences,
            )
            return

        logger.info(
            "AutoDream: trigger conditions met (new_count=%d, hours_since=%.1f), starting dream cycle",
            new_count,
            (now - last_dream).total_seconds() / 3600 if last_dream else 0,
        )

        # Run the 4-phase dream process
        stats = await self._dream(last_dream)

        # Update last dream timestamp
        _write_last_dream(now)

        logger.info(
            "AutoDream: cycle completed — candidates=%d, clusters=%d, merged=%d, pruned=%d",
            stats.get("candidates", 0),
            stats.get("clusters", 0),
            stats.get("merged", 0),
            stats.get("pruned", 0),
        )

    # ------------------------------------------------------------------
    # 4-Phase Dream Process
    # ------------------------------------------------------------------

    async def _dream(self, last_dream: datetime | None) -> dict[str, int]:
        """Execute the full 4-phase dream cycle.

        Returns stats dict with keys: candidates, clusters, merged, pruned.
        """
        stats = {"candidates": 0, "clusters": 0, "merged": 0, "pruned": 0}

        # Phase 1: Orient — scan recent changes
        candidates = await self._phase_orient(last_dream)
        stats["candidates"] = len(candidates)
        if not candidates:
            logger.info("AutoDream Phase 1 (Orient): no candidates found")
            # Still try prune even if no merge candidates
            pruned = await self._phase_prune()
            stats["pruned"] = pruned
            return stats

        logger.info("AutoDream Phase 1 (Orient): %d candidates identified", len(candidates))

        # Phase 2: Gather — find related memories via vector search
        clusters = await self._phase_gather(candidates)
        stats["clusters"] = len(clusters)
        logger.info(
            "AutoDream Phase 2 (Gather): %d clusters found from %d candidates",
            len(clusters),
            len(candidates),
        )

        # Phase 3: Consolidate — merge near-duplicates
        merged = await self._phase_consolidate(clusters)
        stats["merged"] = merged
        logger.info("AutoDream Phase 3 (Consolidate): merged %d near-duplicates", merged)

        # Phase 4: Prune — remove stale/unused experiences
        pruned = await self._phase_prune()
        stats["pruned"] = pruned
        logger.info("AutoDream Phase 4 (Prune): pruned %d stale experiences", pruned)

        return stats

    # ------------------------------------------------------------------
    # Phase 1: Orient
    # ------------------------------------------------------------------

    async def _phase_orient(self, last_dream: datetime | None) -> list[dict]:
        """Scan recent experiences and identify consolidation candidates.

        Candidates are experiences that are:
        - Created since last dream run (or all if never run)
        - AND have low quality_tier (Bronze) or low recall_count

        Returns list of candidate dicts with keys: id, title, description, project, etc.
        """
        try:
            candidates = await self._query_candidates(last_dream)
        except Exception:
            logger.exception("AutoDream Phase 1 (Orient): failed to query candidates")
            return []

        return candidates

    async def _query_candidates(self, last_dream: datetime | None) -> list[dict]:
        """Query candidate experiences via ExperienceService."""
        try:
            return await self._get_dream_candidates(last_dream)
        except Exception:
            logger.warning(
                "Service query for candidates failed, falling back to recall-based scan",
                exc_info=True,
            )
            return await self._query_candidates_recall(last_dream)

    async def _get_dream_candidates(self, last_dream: datetime | None) -> list[dict]:
        """Fetch candidate experiences via ExperienceService."""
        ctx = _daemon_context()
        try:
            candidates = await ctx.service.list_published(
                quality_tier="Bronze",
                since=last_dream,
                limit=200,
                include_embedding=True,
            )
        except TypeError:
            candidates = await ctx.service.list_published(
                quality_tier="Bronze",
                since=last_dream,
                limit=200,
            )
        return [c for c in candidates if c.get("valid_until") is None]

    async def _query_candidates_recall(self, last_dream: datetime | None) -> list[dict]:
        """Fallback: use recall() to discover candidate experiences.

        This is less efficient but works for remote mode.
        """
        candidates: list[dict] = []
        try:
            # Use broad queries to discover recent experiences
            for query_text in ["problem solution", "debug fix", "configuration setup", "error troubleshooting"]:
                results = await self._sink.recall(
                    query=query_text,
                    max_results=10,
                )
                for r in results:
                    r.get("id", "")
                    quality_tier = r.get("quality_tier", "Silver")
                    recall_count = r.get("recall_count", 0)
                    # Filter for Bronze tier or low recall
                    if quality_tier == "Bronze" or recall_count == 0:
                        candidates.append(r)
        except Exception:
            logger.warning("Recall-based candidate scan failed", exc_info=True)

        # Deduplicate by id
        seen: set[str] = set()
        unique: list[dict] = []
        for c in candidates:
            cid = c.get("id", "")
            if cid and cid not in seen:
                seen.add(cid)
                unique.append(c)

        return unique

    # ------------------------------------------------------------------
    # Phase 2: Gather
    # ------------------------------------------------------------------

    async def _phase_gather(self, candidates: list[dict]) -> list[list[dict]]:
        """Build clusters of related experiences using vector similarity.

        For each candidate, use recall() to find similar experiences.
        Group them into clusters where similarity > merge_similarity_threshold.

        Returns list of clusters, each cluster is a list of experience dicts.
        Only returns clusters with 2+ experiences.
        """
        threshold = self._cfg.merge_similarity_threshold
        # Track which experiences belong to which cluster (by id)
        id_to_cluster: dict[str, int] = {}
        clusters: list[list[dict]] = []
        processed_ids: set[str] = set()

        for candidate in candidates:
            cand_id = candidate.get("id", "")
            if not cand_id or cand_id in processed_ids:
                continue

            # Use candidate's content as search query
            query_text = candidate.get("title", "") or candidate.get("description", "")
            if not query_text.strip():
                continue

            project = candidate.get("project")
            try:
                similar = await self._sink.recall(
                    query=query_text[:500],
                    project=project,
                    max_results=10,
                )
            except Exception:
                logger.warning("Gather: recall failed for candidate %s", cand_id[:8], exc_info=True)
                continue

            # Find experiences above similarity threshold
            cluster_members: list[dict] = [candidate]
            member_ids: set[str] = {cand_id}

            for hit in similar:
                hit_id = hit.get("id", "")
                if not hit_id or hit_id == cand_id:
                    continue
                # Skip already-processed or deleted
                if hit_id in processed_ids:
                    continue

                similarity = hit.get("similarity", 0.0)
                if similarity > threshold:
                    cluster_members.append(hit)
                    member_ids.add(hit_id)

            # Only create cluster if we found similar experiences
            if len(cluster_members) >= 2:
                cluster_idx = len(clusters)
                clusters.append(cluster_members)
                for mid in member_ids:
                    id_to_cluster[mid] = cluster_idx
                processed_ids.update(member_ids)
                logger.debug(
                    "Gather: cluster %d has %d members (seed=%s, sim_threshold=%.2f)",
                    cluster_idx,
                    len(cluster_members),
                    cand_id[:8],
                    threshold,
                )
            else:
                # Mark as processed even if no cluster formed
                processed_ids.add(cand_id)

        return clusters

    # ------------------------------------------------------------------
    # Phase 3: Consolidate
    # ------------------------------------------------------------------

    async def _phase_consolidate(self, clusters: list[list[dict]]) -> int:
        """Merge near-duplicate clusters into primary experiences.

        For each cluster:
        - Pick the experience with highest quality_score as the "primary"
        - Merge information from others into the primary via update_experience
        - Soft-delete the others (is_deleted=true)

        Returns total number of experiences merged (soft-deleted).
        """
        total_merged = 0

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            try:
                merged = await self._merge_cluster(cluster)
                total_merged += merged
            except Exception:
                logger.exception(
                    "Consolidate: failed to merge cluster (seed=%s)",
                    cluster[0].get("id", "?")[:8],
                )

        return total_merged

    async def _merge_cluster(self, cluster: list[dict]) -> int:
        """Merge a single cluster of near-duplicate experiences.

        Returns the number of secondary experiences soft-deleted.
        """
        # Pick primary: highest quality_score, then highest used_count, then recall_count as tiebreaker
        primary = max(
            cluster,
            key=lambda x: (x.get("quality_score", 0), x.get("used_count", 0), x.get("recall_count", 0)),
        )
        primary_id = primary.get("id", "")
        primary_title = primary.get("title", "Untitled")

        # Collect secondary experiences
        secondaries = [e for e in cluster if e.get("id") != primary_id]
        if not secondaries:
            return 0

        # Merge: enrich primary with information from secondaries
        merged_solution_parts = []
        merged_tags = set(primary.get("tags", []) or [])

        primary_solution = primary.get("solution") or ""
        if primary_solution:
            merged_solution_parts.append(primary_solution)

        for sec in secondaries:
            # Collect unique solution text
            sec_solution = sec.get("solution") or ""
            if sec_solution and sec_solution not in primary_solution:
                # Only add if substantively different (not a substring)
                if sec_solution not in primary_solution and primary_solution not in sec_solution:
                    merged_solution_parts.append(sec_solution)

            # Merge tags
            sec_tags = sec.get("tags") or []
            merged_tags.update(t for t in sec_tags if t)

        # Update primary if we have new information
        update_kwargs: dict[str, Any] = {}
        if len(merged_solution_parts) > 1:
            update_kwargs["solution"] = "\n\n---\n\n".join(merged_solution_parts)
        elif len(merged_solution_parts) == 1 and merged_solution_parts[0] != primary_solution:
            update_kwargs["solution"] = merged_solution_parts[0]

        if merged_tags and merged_tags != set(primary.get("tags", []) or []):
            update_kwargs["tags"] = sorted(merged_tags)

        if update_kwargs:
            try:
                await self._sink.update_experience(
                    experience_id=primary_id,
                    **update_kwargs,
                )
                logger.debug(
                    "Consolidate: updated primary %s with merged content",
                    primary_id[:8],
                )
            except Exception:
                logger.warning(
                    "Consolidate: failed to update primary %s, skipping merge",
                    primary_id[:8],
                    exc_info=True,
                )
                # Don't soft-delete secondaries if we couldn't update primary
                return 0

        # Soft-delete secondaries
        deleted_count = 0
        for sec in secondaries:
            sec_id = sec.get("id", "")
            if not sec_id:
                continue
            try:
                await self._soft_delete(sec_id)
                deleted_count += 1
                logger.debug(
                    "Consolidate: soft-deleted secondary %s (merged into %s)",
                    sec_id[:8],
                    primary_id[:8],
                )
            except Exception:
                logger.warning(
                    "Consolidate: failed to soft-delete secondary %s",
                    sec_id[:8],
                    exc_info=True,
                )

        if deleted_count > 0:
            logger.info(
                "merged %d near-duplicates into '%s'",
                deleted_count,
                primary_title[:60],
            )

        return deleted_count

    # ------------------------------------------------------------------
    # Phase 4: Prune
    # ------------------------------------------------------------------

    async def _phase_prune(self) -> int:
        """Soft-delete stale experiences.

        Criteria: Bronze quality_tier + recall_count=0 + older than prune_max_age_days.

        Returns the number of experiences pruned.
        """
        try:
            return await self._prune_stale()
        except Exception:
            logger.warning("Service prune failed, skipping prune phase", exc_info=True)
            return 0

    async def _prune_stale(self) -> int:
        """Find and soft-delete stale experiences via ExperienceService."""
        cutoff = _utcnow() - timedelta(days=self._cfg.prune_max_age_days)
        ctx = _daemon_context()
        seen_ids: set[str] = set()
        candidates: list[dict] = []
        for tier in ("Bronze", "Outdated"):
            items = await ctx.service.list_published(quality_tier=tier, limit=500)
            for item in items:
                exp_id = item.get("id")
                if exp_id and exp_id not in seen_ids:
                    seen_ids.add(exp_id)
                    candidates.append(item)
        pruned = 0
        for exp in candidates:
            if (
                exp.get("recall_count", 1) == 0
                and exp.get("used_count", 0) == 0
                and not exp.get("is_pinned")
                and exp.get("created_at") is not None
                and exp["created_at"] < cutoff
            ):
                await ctx.service.soft_delete_by_id(exp["id"])
                pruned += 1
        if pruned > 0:
            logger.info("pruned %d stale experiences", pruned)
        return pruned

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _soft_delete(self, experience_id: str) -> None:
        """Soft-delete an experience via ExperienceService."""
        await self._soft_delete_experience(experience_id)

    async def _soft_delete_experience(self, experience_id: str) -> None:
        """Soft-delete an experience by ID via ExperienceService."""
        ctx = _daemon_context()
        await ctx.service.soft_delete_by_id(experience_id)

    async def _count_new_experiences(self, last_dream: datetime | None) -> int:
        """Count experiences created since last dream run."""
        try:
            ctx = _daemon_context()
            return await ctx.service.count_published(since=last_dream)
        except Exception:
            logger.debug("Service count failed, using estimate", exc_info=True)
            # Rough estimate: if we can't count, assume enough
            return self._cfg.min_new_experiences
