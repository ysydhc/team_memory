"""AutoDreamWorker lifecycle, LLM orchestration, and 4-phase dream runner."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from team_memory.daemon.integration.auto_dream.consolidate import AutoDreamConsolidateMixin
from team_memory.daemon.integration.auto_dream.distill import AutoDreamDistillMixin
from team_memory.daemon.integration.auto_dream.llm import AutoDreamLLMMixin
from team_memory.daemon.integration.auto_dream.orient import AutoDreamOrientMixin
from team_memory.daemon.integration.auto_dream.state import (
    _get_lock_file,
    _NonBlockingFileLock,
    _read_last_dream,
    _utcnow,
    _write_last_dream,
    _write_last_dream_stats,
)
from team_memory.daemon.integration.tm_sink import TMSink
from team_memory.daemon.runtime.config import AutoDreamSettings, DaemonConfig, RefinementSettings

logger = logging.getLogger("daemon.auto_dream")


class AutoDreamWorker(
    AutoDreamLLMMixin,
    AutoDreamOrientMixin,
    AutoDreamConsolidateMixin,
    AutoDreamDistillMixin,
):
    """Periodic & on-demand background worker that consolidates, evolves, and distills experiences."""

    def __init__(
        self,
        config: DaemonConfig,
        sink: TMSink,
        *,
        llm_client: Any = None,
    ) -> None:
        self._daemon_config: DaemonConfig = config
        self._cfg: AutoDreamSettings = config.auto_dream
        self._refinement_cfg: RefinementSettings = getattr(config, "refinement", RefinementSettings())
        self._sink = sink
        self._llm_client = llm_client
        self._running = False
        self._paused = False
        self._task: asyncio.Task | None = None
        self._run_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> asyncio.Task:
        """Start the background worker."""
        if not self._cfg.enabled:
            logger.info("AutoDreamWorker is disabled in config")
            self._task = asyncio.create_task(self._noop())
            return self._task

        self._running = True
        self._paused = False
        self._task = asyncio.create_task(self._loop())
        model, fallback, _, _, _, _ = self._resolve_llm_settings()
        logger.info(
            "AutoDreamWorker started "
            "(scan_interval=%dm, min_hours=%d, min_new=%d, threshold=%.2f, "
            "max_clusters=%d, model=%s, fallback=%s)",
            self._cfg.scan_interval_minutes,
            self._cfg.min_hours_since_last,
            self._cfg.min_new_experiences,
            self._cfg.merge_similarity_threshold,
            self._cfg.max_clusters_per_run,
            model,
            fallback or "none",
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
    # Per-tick & Public run_once entry point
    # ------------------------------------------------------------------

    async def _tick(self) -> dict[str, Any]:
        """Scheduled tick: check trigger conditions and run dream if met."""
        return await self.run_once(dry_run=False, force=False, trigger="scheduled")

    async def run_once(
        self,
        *,
        dry_run: bool = False,
        force: bool = False,
        project: str | None = None,
        max_clusters: int | None = None,
        model: str | None = None,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        """Execute a single AutoDream cycle with concurrency lock, watermark safety, and audit stats."""
        now = _utcnow()
        started = time.monotonic()

        # In-process non-blocking guard
        if self._run_lock.locked():
            logger.info("AutoDream: already running in-process, skipping")
            return {
                "ran_at": now.isoformat(),
                "trigger": trigger,
                "dry_run": dry_run,
                "success": True,
                "skipped_run": True,
                "skipped_reason": "locked",
                "watermark_updated": False,
            }

        lock_file = _get_lock_file()
        async with self._run_lock:
            with _NonBlockingFileLock(lock_file) as acquired:
                if not acquired:
                    logger.info("AutoDream: lock held by another process (%s), skipping", lock_file)
                    return {
                        "ran_at": now.isoformat(),
                        "trigger": trigger,
                        "dry_run": dry_run,
                        "success": True,
                        "skipped_run": True,
                        "skipped_reason": "locked",
                        "watermark_updated": False,
                    }

                last_dream = _read_last_dream()

                # Scheduled trigger gates (bypassed when force=True or dry_run=True)
                if not force and not dry_run:
                    if last_dream is not None:
                        hours_since = (now - last_dream).total_seconds() / 3600
                        if hours_since < self._cfg.min_hours_since_last:
                            logger.debug(
                                "AutoDream: only %.1fh since last dream (need %dh), skipping",
                                hours_since,
                                self._cfg.min_hours_since_last,
                            )
                            return {
                                "ran_at": now.isoformat(),
                                "trigger": trigger,
                                "dry_run": False,
                                "success": True,
                                "skipped_run": True,
                                "skipped_reason": "min_hours_not_met",
                                "watermark_updated": False,
                            }
                    else:
                        logger.info("AutoDream: no previous dream run found")

                    try:
                        new_count = await self._count_new_experiences(last_dream, project=project)
                    except Exception as exc:
                        logger.warning("AutoDream: count_new_experiences failed: %s", exc)
                        return {
                            "ran_at": now.isoformat(),
                            "trigger": trigger,
                            "dry_run": False,
                            "success": False,
                            "skipped_run": True,
                            "skipped_reason": "count_failed",
                            "error": str(exc),
                            "watermark_updated": False,
                        }

                    if new_count < self._cfg.min_new_experiences:
                        logger.debug(
                            "AutoDream: only %d new experiences (need %d), skipping",
                            new_count,
                            self._cfg.min_new_experiences,
                        )
                        return {
                            "ran_at": now.isoformat(),
                            "trigger": trigger,
                            "dry_run": False,
                            "success": True,
                            "skipped_run": True,
                            "skipped_reason": "min_new_not_met",
                            "new_count": new_count,
                            "watermark_updated": False,
                        }

                    logger.info(
                        "AutoDream: trigger conditions met (new_count=%d, hours_since=%.1f), starting cycle",
                        new_count,
                        (now - last_dream).total_seconds() / 3600 if last_dream else 0,
                    )

                resolved_model, resolved_fallback, _, _, _, _ = self._resolve_llm_settings(model)
                scan_since = None if (force and project is not None) else last_dream
                # When manual --force without prior watermark or explicit full scan, scan all if last_dream is None
                if force and last_dream is not None:
                    # Force still respects last_dream unless caller wants full scan; if last_dream is recent,
                    # fall back to scanning all active experiences so --force always inspects meaningful data
                    scan_since = None

                fatal_error: str | None = None
                try:
                    stats = await self._dream(
                        scan_since,
                        dry_run=dry_run,
                        project=project,
                        max_clusters=max_clusters,
                        model=model,
                    )
                except Exception as exc:
                    fatal_error = str(exc)
                    logger.exception("AutoDream cycle aborted due to fatal error")
                    stats = {
                        "candidates": 0,
                        "low_adoption_candidates": 0,
                        "clusters": 0,
                        "merged": 0,
                        "superseded": 0,
                        "kept": 0,
                        "distilled": 0,
                        "pruned": 0,
                        "skipped": 0,
                        "errors": 1,
                        "estimated_tokens": {"input": 0, "output": 0},
                        "plan": [],
                        "distill_plan": [],
                    }

                duration_ms = int((time.monotonic() - started) * 1000)
                clusters_n = stats.get("clusters", 0)
                all_clusters_failed = (
                    clusters_n > 0
                    and stats.get("skipped", 0) >= clusters_n
                    and stats.get("errors", 0) >= clusters_n
                    and stats.get("merged", 0) == 0
                    and stats.get("superseded", 0) == 0
                    and stats.get("kept", 0) == 0
                )
                if all_clusters_failed and fatal_error is None:
                    fatal_error = "All clusters failed LLM consolidation"

                success = fatal_error is None
                watermark_updated = False

                # Only advance global watermark on non-dry-run, non-fatal, global (non-project-filtered) runs
                if not dry_run and success and project is None:
                    _write_last_dream(now)
                    watermark_updated = True

                summary: dict[str, Any] = {
                    "ran_at": now.isoformat(),
                    "duration_ms": duration_ms,
                    "trigger": trigger,
                    "dry_run": dry_run,
                    "project_filter": project,
                    "model": resolved_model,
                    "fallback_model": resolved_fallback,
                    "success": success,
                    "skipped_run": False,
                    "watermark_updated": watermark_updated,
                    "candidates": stats.get("candidates", 0),
                    "low_adoption_candidates": stats.get("low_adoption_candidates", 0),
                    "clusters": stats.get("clusters", 0),
                    "merged": stats.get("merged", 0),
                    "superseded": stats.get("superseded", 0),
                    "kept": stats.get("kept", 0),
                    "distilled": stats.get("distilled", 0),
                    "pruned": stats.get("pruned", 0),
                    "skipped": stats.get("skipped", 0),
                    "errors": stats.get("errors", 0),
                    "estimated_tokens": stats.get("estimated_tokens", {"input": 0, "output": 0}),
                }
                if fatal_error:
                    summary["error"] = fatal_error

                # Persist audit summary (without verbose plan arrays) on real runs
                if not dry_run:
                    _write_last_dream_stats(summary)

                summary["plan"] = stats.get("plan", [])
                summary["distill_plan"] = stats.get("distill_plan", [])

                logger.info(
                    "AutoDream (%s%s): done in %dms — candidates=%d (low_adopt=%d), "
                    "clusters=%d, merged=%d, superseded=%d, kept=%d, distilled=%d, pruned=%d, errors=%d",
                    trigger,
                    " DRY-RUN" if dry_run else "",
                    duration_ms,
                    summary["candidates"],
                    summary["low_adoption_candidates"],
                    summary["clusters"],
                    summary["merged"],
                    summary["superseded"],
                    summary["kept"],
                    summary["distilled"],
                    summary["pruned"],
                    summary["errors"],
                )
                return summary

    # ------------------------------------------------------------------
    # 4-Phase Dream Process
    # ------------------------------------------------------------------

    async def _dream(
        self,
        last_dream: datetime | None,
        *,
        dry_run: bool = False,
        project: str | None = None,
        max_clusters: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Execute the full 4-phase dream cycle."""
        stats: dict[str, Any] = {
            "candidates": 0,
            "low_adoption_candidates": 0,
            "clusters": 0,
            "merged": 0,
            "superseded": 0,
            "kept": 0,
            "distilled": 0,
            "pruned": 0,
            "skipped": 0,
            "errors": 0,
            "estimated_tokens": {"input": 0, "output": 0},
            "plan": [],
            "distill_plan": [],
        }

        # Phase 1: Orient — scan recent active experiences + low-adoption signals
        candidates, low_adopt_count = await self._phase_orient(last_dream, project=project)
        stats["candidates"] = len(candidates)
        stats["low_adoption_candidates"] = low_adopt_count

        if not candidates:
            logger.info("AutoDream Phase 1 (Orient): no candidates found")
            stats["pruned"] = await self._phase_prune(dry_run=dry_run, project=project)
            return stats

        logger.info(
            "AutoDream Phase 1 (Orient): %d candidates identified (%d low-adoption)",
            len(candidates),
            low_adopt_count,
        )

        # Phase 2: Gather — pgvector semantic clustering
        clusters = await self._phase_gather(candidates, max_clusters=max_clusters)
        stats["clusters"] = len(clusters)
        logger.info(
            "AutoDream Phase 2 (Gather): %d clusters found from %d candidates",
            len(clusters),
            len(candidates),
        )

        # Phase 3: Consolidate — LLM semantic rewrite / bi-temporal supersede / keep_both
        cons = await self._phase_consolidate(clusters, dry_run=dry_run, model=model)
        stats["merged"] = cons["merged"]
        stats["superseded"] = cons["superseded"]
        stats["kept"] = cons["kept"]
        stats["skipped"] = cons["skipped"]
        stats["errors"] = cons["errors"]
        stats["plan"] = cons["plan"]
        stats["estimated_tokens"]["input"] += cons["input_tokens"]
        stats["estimated_tokens"]["output"] += cons["output_tokens"]

        # Phase 4a: Distill — synthesize >=3-item project clusters into compact Playbooks
        # Exclude experiences retired (merged/superseded) in Phase 3 so obsolete advice is never distilled
        if self._cfg.distill_enabled and clusters:
            dist = await self._phase_distill(
                clusters,
                dry_run=dry_run,
                model=model,
                retired_ids=cons.get("retired_ids", set()),
            )
            stats["distilled"] = dist["distilled"]
            stats["errors"] += dist["errors"]
            stats["distill_plan"] = dist["plan"]
            stats["estimated_tokens"]["input"] += dist["input_tokens"]
            stats["estimated_tokens"]["output"] += dist["output_tokens"]

        # Phase 4b: Prune — lightweight Outdated orphan cleanup (heavy GC stays in Janitor)
        stats["pruned"] = await self._phase_prune(dry_run=dry_run, project=project)

        return stats
