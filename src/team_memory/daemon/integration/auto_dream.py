"""Auto Dream worker: offline semantic memory consolidation, bi-temporal evolution, and distillation.

Inspired by Claude Code's Auto Dream / Sleep-Time Compute, adapted for PostgreSQL + pgvector.

Division of labor:
  - MemoryJanitor: physical GC (score decay, outdated sweep, soft-delete purge, log dedup).
  - AutoDreamWorker: offline semantic thinking:
      Phase 1 (Orient)      — scan recent active experiences + high-recall/zero-adoption signals
      Phase 2 (Gather)      — pgvector cosine clustering (zero search_logs pollution)
      Phase 3 (Consolidate) — LLM structured decision: MERGE (semantic rewrite + absolute dates)
                              / SUPERSEDE (bi-temporal valid_until + superseded_by_id) / KEEP_BOTH
      Phase 4 (Distill & Prune) — distill >=3-item project clusters into compact Playbooks
                              (<=1500 tokens) + lightweight Outdated orphan cleanup
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from team_memory.daemon.integration.tm_sink import TMSink, _daemon_context
from team_memory.daemon.runtime.config import AutoDreamSettings, DaemonConfig, RefinementSettings
from team_memory.services.llm.llm_client import LLMClient, LLMError

logger = logging.getLogger("daemon.auto_dream")

# Default cache directory and state files
_CACHE_DIR = Path(os.path.expanduser("~/.cache/tm-pipeline"))
_LAST_DREAM_FILE = _CACHE_DIR / "last_dream.txt"
_LAST_DREAM_STATS_FILE = _CACHE_DIR / "last_dream_stats.json"
_LOCK_FILE = _CACHE_DIR / "auto_dream.lock"

_CONSOLIDATE_SYSTEM_PROMPT = """\
You are AutoDream, an offline semantic memory consolidation engine for a software engineering knowledge base.
Given a cluster of related technical experiences (each with id, title, problem, solution, tags, created_at,
quality_tier, is_pinned, epistemic_tier, staleness_status, evidence_anchors, recall_count, used_count),
choose ONE action:

1. "MERGE": Experiences are near-duplicates or complementary facets of the SAME stable solution/fact.
   - Rewrite into ONE compact, high-signal experience (DO NOT concatenate with "---" or repeat boilerplate).
   - Temporal normalization: convert relative time phrases ("yesterday", "last week", "recently", "today",
     "昨天", "上周", "最近", "今天") into absolute dates (YYYY-MM-DD) anchored to each item's created_at.
   - Choose `primary_id` (MUST prefer is_pinned=true, quality_tier="Gold", or fresh epistemic_tier="verified";
     otherwise highest quality/adoption). Put remaining IDs in `secondary_ids`.
2. "SUPERSEDE": Experiences conflict or a newer/fresh experience replaces an older or `possibly_stale` approach
   (e.g., library migration, changed config, fixed misconception, updated API, deleted/refactored anchored files).
   - Pay special attention to items marked `staleness_status="possibly_stale"`: prioritize verifying whether a newer
     `fresh` experience in the cluster supersedes them.
   - Choose `winner_id` (the accurate/current experience) and `superseded_ids` (obsolete/contradicted ones;
     NEVER supersede an item with is_pinned=true, quality_tier="Gold", or fresh epistemic_tier="verified").
3. "KEEP_BOTH": Experiences apply to different scopes, constraints, or distinct problems — keep untouched.

Respond with ONLY valid JSON matching this schema:
{
  "action": "MERGE" | "SUPERSEDE" | "KEEP_BOTH",
  "reason": "concise explanation",
  "primary_id": "<id, required for MERGE>",
  "secondary_ids": ["<id>", ...],
  "merged": {
    "title": "<concise title>",
    "problem": "<distilled problem>",
    "solution": "<distilled, non-redundant solution with absolute dates>",
    "tags": ["tag1", "tag2"]
  },
  "winner_id": "<id, required for SUPERSEDE>",
  "superseded_ids": ["<id>", ...]
}"""

_DISTILL_SYSTEM_PROMPT = """\
You are AutoDream's knowledge distiller. Synthesize the given cluster of related experiences
from the same project into a concise, actionable Project Playbook.
Rules:
- Only distill if the cluster forms a coherent reusable operational/architectural theme;
  otherwise set should_distill=false.
- Focus on invariants, recommended workflow, pitfalls to avoid, and verified commands/configs.
- Convert relative dates to YYYY-MM-DD based on created_at.
- Keep the solution strictly concise (under 1200 tokens, bullet points, zero fluff).
- Respond with ONLY valid JSON:
{
  "should_distill": true | false,
  "reason": "brief explanation",
  "title": "[Playbook] <project>: <topic>",
  "problem": "<when to consult this playbook>",
  "solution": "<compact structured markdown playbook>",
  "tags": ["playbook", "auto-dream", ...]
}"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(val: Any) -> datetime | None:
    """Safely parse datetime or ISO string into a timezone-aware UTC datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _estimate_tokens(text: str) -> int:
    """Rough cross-lingual token estimator (CJK ~1.2 tok/char, ASCII ~0.28 tok/char)."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk * 1.2 + other * 0.28))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text so its estimated token count does not exceed max_tokens."""
    if not text or max_tokens <= 0 or _estimate_tokens(text) <= max_tokens:
        return text
    # Binary-shrink by line or char boundary
    ratio = max_tokens / max(1, _estimate_tokens(text))
    target_len = max(100, int(len(text) * ratio * 0.92))
    truncated = text[:target_len].rstrip()
    if "\n" in truncated:
        truncated = truncated.rsplit("\n", 1)[0].rstrip()
    return f"{truncated}\n\n_(truncated to <= {max_tokens} tokens)_"


def _is_protected(exp: dict[str, Any]) -> bool:
    """Return True if an experience must not be auto-overwritten or superseded."""
    if exp.get("is_pinned"):
        return True
    tier = str(exp.get("quality_tier") or "")
    if tier == "Gold":
        return True
    epistemic = str(exp.get("epistemic_tier") or "auto_extract").lower()
    staleness = str(exp.get("staleness_status") or "fresh").lower()
    return epistemic == "verified" and staleness == "fresh"


# ---------------------------------------------------------------------------
# Persistence & Lock helpers
# ---------------------------------------------------------------------------


def _read_last_dream() -> datetime | None:
    """Read last dream timestamp from file. Returns None if never run."""
    try:
        if _LAST_DREAM_FILE.exists():
            text = _LAST_DREAM_FILE.read_text().strip()
            if text:
                return _parse_dt(text)
    except Exception:
        logger.debug("Failed to read last_dream file", exc_info=True)
    return None


def _write_last_dream(ts: datetime) -> None:
    """Persist last dream timestamp atomically."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _LAST_DREAM_FILE.with_suffix(".tmp")
        tmp.write_text(ts.isoformat())
        tmp.replace(_LAST_DREAM_FILE)
    except Exception:
        logger.warning("Failed to write last_dream file", exc_info=True)


def read_last_dream_stats() -> dict[str, Any] | None:
    """Read last persisted AutoDream audit summary from JSON file."""
    try:
        if _LAST_DREAM_STATS_FILE.exists():
            raw = _LAST_DREAM_STATS_FILE.read_text(encoding="utf-8").strip()
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
    except Exception:
        logger.debug("Failed to read last_dream_stats file", exc_info=True)
    return None


def _write_last_dream_stats(stats: dict[str, Any]) -> None:
    """Persist AutoDream audit summary atomically."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _LAST_DREAM_STATS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_LAST_DREAM_STATS_FILE)
    except Exception:
        logger.warning("Failed to write last_dream_stats file", exc_info=True)


class _NonBlockingFileLock:
    """Cross-process non-blocking file lock using fcntl.flock."""

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._fd: int | None = None
        self.acquired: bool = False

    def __enter__(self) -> bool:
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.acquired = True
            return True
        except BlockingIOError:
            self.acquired = False
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False
        except OSError:
            logger.debug("File lock unavailable for %s, proceeding with in-process lock", self._lock_path)
            self.acquired = True
            return True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


# ---------------------------------------------------------------------------
# AutoDreamWorker
# ---------------------------------------------------------------------------


class AutoDreamWorker:
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
    # LLM configuration & calling
    # ------------------------------------------------------------------

    def _resolve_llm_settings(
        self,
        model_override: str | None = None,
    ) -> tuple[str, str, str, str, str, float]:
        """Resolve (model, fallback_model, base_url, api_key, provider, timeout).

        Priority for model:
          1. CLI/call model_override
          2. TM_AUTO_DREAM_MODEL env var
          3. auto_dream.llm_model in config.yaml
          4. refinement.model in config.yaml
          5. ClientConfig.llm.model
          6. default ("gbu-gemini-3.8-flash")
        """
        client_llm = None
        try:
            from team_memory.client_config import ClientConfig

            client_llm = ClientConfig.load().llm
        except Exception:
            client_llm = None

        ref = self._refinement_cfg
        model = (
            (model_override or "").strip()
            or os.environ.get("TM_AUTO_DREAM_MODEL", "").strip()
            or (self._cfg.llm_model or "").strip()
            or (ref.model or "").strip()
            or (getattr(client_llm, "model", "") if client_llm else "").strip()
            or "gbu-gemini-3.8-flash"
        )
        fallback = (
            os.environ.get("TM_AUTO_DREAM_FALLBACK_MODEL", "").strip()
            or (self._cfg.llm_fallback_model or "").strip()
            or (ref.fallback_model or "").strip()
        )
        if fallback == model:
            fallback = ""

        base_url = (
            os.environ.get("TM_LLM_BASE_URL", "").strip()
            or (ref.base_url or "").strip()
            or (getattr(client_llm, "base_url", "") if client_llm else "").strip()
            or "https://gw.gburouter.com/v1"
        )
        api_key = ""
        if ref.api_key_env:
            api_key = os.environ.get(ref.api_key_env, "").strip()
        if not api_key and client_llm is not None:
            api_key = (client_llm.resolved_api_key or "").strip()

        provider = (ref.provider or "litellm").strip()
        timeout = float(self._cfg.llm_timeout or ref.timeout or 60)
        return model, fallback, base_url, api_key, provider, timeout

    async def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model_override: str | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Call LLM expecting JSON output, with fallback_model retry.

        Returns (parsed_dict, estimated_input_tokens, estimated_output_tokens).
        """
        in_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)

        if self._llm_client is not None:
            result = await self._llm_client.chat_json(
                system_prompt,
                user_prompt,
                temperature=0.2,
                timeout=float(self._cfg.llm_timeout or 60),
            )
            if not isinstance(result, dict):
                raise LLMError("LLM returned non-object JSON")
            out_tokens = _estimate_tokens(json.dumps(result, ensure_ascii=False))
            return result, in_tokens, out_tokens

        model, fallback, base_url, api_key, provider, timeout = self._resolve_llm_settings(model_override)
        models_to_try = [model] + ([fallback] if fallback else [])
        last_exc: Exception | None = None

        for idx, cur_model in enumerate(models_to_try):
            client = LLMClient(
                base_url=base_url,
                model=cur_model,
                api_key=api_key,
                provider=provider,
            )
            try:
                result = await client.chat_json(
                    system_prompt,
                    user_prompt,
                    temperature=0.2,
                    timeout=timeout,
                )
                if not isinstance(result, dict):
                    raise LLMError("LLM returned non-object JSON")
                out_tokens = _estimate_tokens(json.dumps(result, ensure_ascii=False))
                return result, in_tokens, out_tokens
            except Exception as exc:
                last_exc = exc
                if idx < len(models_to_try) - 1:
                    logger.warning(
                        "AutoDream LLM model %s failed (%s), falling back to %s",
                        cur_model,
                        exc,
                        models_to_try[idx + 1],
                    )

        raise last_exc or LLMError("AutoDream LLM call failed")

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

        async with self._run_lock:
            with _NonBlockingFileLock(_LOCK_FILE) as acquired:
                if not acquired:
                    logger.info("AutoDream: lock held by another process (%s), skipping", _LOCK_FILE)
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

    # ------------------------------------------------------------------
    # Phase 1: Orient
    # ------------------------------------------------------------------

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
        ctx = _daemon_context()
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
        ctx = _daemon_context()
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
        ctx = _daemon_context()
        if not hasattr(ctx.service, "list_stale_candidates"):
            return []
        result = await ctx.service.list_stale_candidates(
            project=project,
            limit=self._cfg.low_adoption_sample_limit,
            include_embedding=True,
        )
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Phase 2: Gather
    # ------------------------------------------------------------------

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
                ctx = _daemon_context()
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
        scored_clusters: list[tuple[tuple[int, float, int], list[dict]]] = []
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

    # ------------------------------------------------------------------
    # Phase 3: Consolidate (LLM Semantic Rewrite & Bi-Temporal Supersede)
    # ------------------------------------------------------------------

    def _build_consolidate_prompt(self, cluster: list[dict]) -> str:
        """Serialize a cluster into a compact prompt payload for LLM consolidation."""
        items: list[dict[str, Any]] = []
        for exp in cluster:
            items.append(
                {
                    "id": str(exp.get("id") or ""),
                    "project": exp.get("project", "default"),
                    "title": (exp.get("title") or "")[:300],
                    "problem": (exp.get("description") or exp.get("problem") or "")[:1200],
                    "solution": (exp.get("solution") or "")[:2000],
                    "tags": exp.get("tags") or [],
                    "created_at": str(exp.get("created_at") or ""),
                    "quality_tier": exp.get("quality_tier", "Silver"),
                    "quality_score": exp.get("quality_score", 0),
                    "is_pinned": bool(exp.get("is_pinned", False)),
                    "epistemic_tier": exp.get("epistemic_tier", "auto_extract"),
                    "staleness_status": exp.get("staleness_status", "fresh"),
                    "evidence_anchors": exp.get("evidence_anchors") or {},
                    "recall_count": int(exp.get("recall_count", 0) or 0),
                    "used_count": int(exp.get("used_count", 0) or 0),
                    "experience_type": exp.get("experience_type", "general"),
                }
            )
        return (
            f"Analyze this cluster of {len(items)} related experiences and return your JSON decision:\n\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}"
        )

    def _enforce_protection_guardrails(
        self,
        decision: dict[str, Any],
        cluster: list[dict],
    ) -> dict[str, Any]:
        """Enforce hard code-level guardrails for is_pinned / Gold experiences and valid IDs."""
        by_id = {str(e.get("id") or ""): e for e in cluster if e.get("id")}
        valid_ids = set(by_id.keys())
        protected_ids = {eid for eid, e in by_id.items() if _is_protected(e)}

        action = str(decision.get("action") or "").upper().strip()
        reason = str(decision.get("reason") or "").strip()

        if action == "MERGE":
            primary_id = str(decision.get("primary_id") or "").strip()
            raw_secs = decision.get("secondary_ids") or []
            secondary_ids = [str(sid).strip() for sid in raw_secs if str(sid).strip() in valid_ids]

            if primary_id not in valid_ids:
                # Pick highest-priority member as fallback primary
                primary = max(
                    cluster,
                    key=lambda x: (
                        int(_is_protected(x)),
                        float(x.get("quality_score", 0) or 0),
                        int(x.get("used_count", 0) or 0),
                        int(x.get("recall_count", 0) or 0),
                    ),
                )
                primary_id = str(primary.get("id") or "")

            secondary_ids = [sid for sid in dict.fromkeys(secondary_ids) if sid != primary_id]

            # Guardrail: >=2 protected items in cluster cannot be auto-merged into one
            protected_in_merge = [eid for eid in [primary_id] + secondary_ids if eid in protected_ids]
            if len(protected_in_merge) >= 2:
                return {
                    "action": "KEEP_BOTH",
                    "reason": (
                        f"Guardrail: multiple protected (pinned/Gold) items "
                        f"({len(protected_in_merge)}); {reason}"
                    ),
                    "guardrail_applied": True,
                }

            # Guardrail: single protected item MUST be primary and never in secondary_ids
            if len(protected_in_merge) == 1:
                prot_id = protected_in_merge[0]
                if primary_id != prot_id:
                    old_primary = primary_id
                    primary_id = prot_id
                    secondary_ids = [sid for sid in secondary_ids if sid != prot_id]
                    if old_primary in valid_ids and old_primary != prot_id:
                        secondary_ids.append(old_primary)

            secondary_ids = [sid for sid in secondary_ids if sid not in protected_ids]
            if not secondary_ids:
                return {
                    "action": "KEEP_BOTH",
                    "reason": f"Guardrail: no unprotected secondary items left to merge; {reason}",
                    "guardrail_applied": True,
                }

            merged = decision.get("merged")
            if not isinstance(merged, dict):
                raise LLMError("MERGE decision missing 'merged' object")
            m_title = str(merged.get("title") or "").strip()
            m_problem = str(merged.get("problem") or "").strip()
            m_solution = str(merged.get("solution") or "").strip()
            if not m_title or (not m_solution and not m_problem):
                raise LLMError("MERGE decision missing required merged title/content")

            raw_tags = merged.get("tags")
            if isinstance(raw_tags, list):
                m_tags = sorted({str(t).strip() for t in raw_tags if str(t).strip()})
            else:
                tag_set: set[str] = set()
                for eid in [primary_id] + secondary_ids:
                    tag_set.update(t for t in (by_id[eid].get("tags") or []) if t)
                m_tags = sorted(tag_set)

            return {
                "action": "MERGE",
                "reason": reason,
                "primary_id": primary_id,
                "secondary_ids": secondary_ids,
                "merged": {
                    "title": m_title,
                    "problem": m_problem,
                    "solution": m_solution,
                    "tags": m_tags,
                },
            }

        if action == "SUPERSEDE":
            winner_id = str(decision.get("winner_id") or "").strip()
            raw_sup = decision.get("superseded_ids") or []
            superseded_ids = [
                str(sid).strip()
                for sid in dict.fromkeys(raw_sup)
                if str(sid).strip() in valid_ids and str(sid).strip() != winner_id
            ]
            if winner_id not in valid_ids:
                raise LLMError("SUPERSEDE decision missing valid winner_id")

            # Guardrail: never supersede pinned or Gold experiences
            superseded_ids = [sid for sid in superseded_ids if sid not in protected_ids]
            if not superseded_ids:
                return {
                    "action": "KEEP_BOTH",
                    "reason": f"Guardrail: target experience is protected (pinned/Gold); {reason}",
                    "guardrail_applied": True,
                }

            return {
                "action": "SUPERSEDE",
                "reason": reason or "Superseded by newer/more accurate experience via AutoDream",
                "winner_id": winner_id,
                "superseded_ids": superseded_ids,
            }

        if action == "KEEP_BOTH":
            return {
                "action": "KEEP_BOTH",
                "reason": reason or "Distinct scopes or constraints",
            }

        raise LLMError(f"Unsupported AutoDream action: {action!r}")

    async def _phase_consolidate(
        self,
        clusters: list[list[dict]],
        *,
        dry_run: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run LLM consolidation on each cluster with strict safety guardrails."""
        res: dict[str, Any] = {
            "merged": 0,
            "superseded": 0,
            "kept": 0,
            "skipped": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "retired_ids": set(),
            "plan": [],
        }

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            by_id = {str(e.get("id") or ""): e for e in cluster if e.get("id")}
            cluster_ids = list(by_id.keys())
            cluster_titles = {eid: (e.get("title") or "")[:80] for eid, e in by_id.items()}

            try:
                user_prompt = self._build_consolidate_prompt(cluster)
                raw_decision, in_tok, out_tok = await self._call_llm_json(
                    _CONSOLIDATE_SYSTEM_PROMPT,
                    user_prompt,
                    model_override=model,
                )
                res["input_tokens"] += in_tok
                res["output_tokens"] += out_tok
                decision = self._enforce_protection_guardrails(raw_decision, cluster)
            except Exception as exc:
                logger.warning(
                    "Consolidate: LLM/decision error for cluster seed=%s (%s) — skipping safely",
                    cluster_ids[0][:8] if cluster_ids else "?",
                    exc,
                )
                res["errors"] += 1
                res["skipped"] += 1
                res["plan"].append(
                    {
                        "action": "SKIPPED_ERROR",
                        "cluster_ids": cluster_ids,
                        "titles": cluster_titles,
                        "reason": str(exc)[:200],
                    }
                )
                continue

            action = decision["action"]
            plan_entry: dict[str, Any] = {
                "action": action,
                "cluster_ids": cluster_ids,
                "titles": cluster_titles,
                "reason": decision.get("reason", ""),
            }

            if action == "KEEP_BOTH":
                res["kept"] += 1
                res["plan"].append(plan_entry)
                continue

            if action == "MERGE":
                primary_id = decision["primary_id"]
                secondary_ids = decision["secondary_ids"]
                merged_payload = decision["merged"]
                plan_entry["primary_id"] = primary_id
                plan_entry["secondary_ids"] = secondary_ids
                plan_entry["merged_preview"] = {
                    "title": merged_payload["title"],
                    "tags": merged_payload["tags"],
                    "solution_preview": merged_payload["solution"][:240],
                }
                res["plan"].append(plan_entry)

                if dry_run:
                    res["merged"] += len(secondary_ids)
                    res["retired_ids"].update(secondary_ids)
                    continue

                update_kwargs: dict[str, Any] = {
                    "title": merged_payload["title"],
                    "tags": merged_payload["tags"],
                }
                if merged_payload["problem"]:
                    update_kwargs["problem"] = merged_payload["problem"]
                if merged_payload["solution"]:
                    update_kwargs["solution"] = merged_payload["solution"]

                try:
                    await self._sink.update_experience(
                        experience_id=primary_id,
                        **update_kwargs,
                    )
                except Exception:
                    logger.warning(
                        "Consolidate: failed to update primary %s, aborting merge for cluster",
                        primary_id[:8],
                        exc_info=True,
                    )
                    res["errors"] += 1
                    continue

                for sec_id in secondary_ids:
                    try:
                        await self._soft_delete(sec_id)
                        res["merged"] += 1
                        res["retired_ids"].add(sec_id)
                    except Exception:
                        logger.warning(
                            "Consolidate: failed to soft-delete secondary %s",
                            sec_id[:8],
                            exc_info=True,
                        )
                        res["errors"] += 1
                continue

            if action == "SUPERSEDE":
                winner_id = decision["winner_id"]
                superseded_ids = decision["superseded_ids"]
                reason = decision["reason"]
                plan_entry["winner_id"] = winner_id
                plan_entry["superseded_ids"] = superseded_ids
                res["plan"].append(plan_entry)

                if dry_run:
                    res["superseded"] += len(superseded_ids)
                    res["retired_ids"].update(superseded_ids)
                    continue

                for old_id in superseded_ids:
                    try:
                        sup_res = await self._sink.supersede_experience(
                            old_experience_id=old_id,
                            new_experience_id=winner_id,
                            reason=reason,
                        )
                        if isinstance(sup_res, dict) and sup_res.get("resolved") is False:
                            logger.warning(
                                "Consolidate: supersede_experience returned resolved=False for %s -> %s",
                                old_id[:8],
                                winner_id[:8],
                            )
                            res["errors"] += 1
                        else:
                            res["superseded"] += 1
                            res["retired_ids"].add(old_id)
                    except Exception:
                        logger.warning(
                            "Consolidate: failed to supersede %s -> %s",
                            old_id[:8],
                            winner_id[:8],
                            exc_info=True,
                        )
                        res["errors"] += 1

        return res

    # ------------------------------------------------------------------
    # Phase 4a: Distill (Project Playbooks)
    # ------------------------------------------------------------------

    async def _find_existing_playbook(self, project: str, title: str) -> str | None:
        """Find an existing active playbook in the same project with matching topic (zero search_logs)."""
        norm_target = " ".join(title.lower().split())
        target_tokens = {w for w in norm_target.replace(":", " ").replace("[playbook]", "").split() if len(w) > 1}
        try:
            ctx = _daemon_context()
            items = await ctx.service.list_published(project=project, limit=100, active_only=True)
            for item in items:
                if str(item.get("experience_type") or "") != "playbook":
                    continue
                cand_title = " ".join(str(item.get("title") or "").lower().split())
                if cand_title == norm_target:
                    return str(item.get("id") or "")
                cand_tokens = {w for w in cand_title.replace(":", " ").replace("[playbook]", "").split() if len(w) > 1}
                if target_tokens and cand_tokens:
                    overlap = len(target_tokens & cand_tokens) / len(target_tokens | cand_tokens)
                    if overlap >= 0.7:
                        return str(item.get("id") or "")
        except Exception:
            logger.debug("Existing playbook lookup failed for project=%s", project, exc_info=True)
        return None

    async def _phase_distill(
        self,
        clusters: list[list[dict]],
        *,
        dry_run: bool = False,
        model: str | None = None,
        retired_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Distill >=distill_min_cluster_size same-project clusters into compact Project Playbooks."""
        res: dict[str, Any] = {
            "distilled": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "plan": [],
        }

        min_size = max(2, int(self._cfg.distill_min_cluster_size))
        max_distill = max(0, int(self._cfg.max_distill_per_run))
        max_tokens = max(200, int(self._cfg.distill_max_tokens))
        retired = retired_ids or set()

        if max_distill == 0:
            return res

        for cluster in clusters:
            if res["distilled"] >= max_distill:
                break

            # Group active non-playbook experiences by project
            by_project: dict[str, list[dict]] = {}
            for exp in cluster:
                eid = str(exp.get("id") or "")
                if eid in retired or str(exp.get("experience_type") or "") == "playbook":
                    continue
                proj = str(exp.get("project") or "default")
                by_project.setdefault(proj, []).append(exp)

            for proj, items in by_project.items():
                if res["distilled"] >= max_distill:
                    break
                if len(items) < min_size:
                    continue

                payload = [
                    {
                        "id": str(e.get("id") or ""),
                        "title": (e.get("title") or "")[:200],
                        "problem": (e.get("description") or e.get("problem") or "")[:600],
                        "solution": (e.get("solution") or "")[:1000],
                        "tags": e.get("tags") or [],
                        "created_at": str(e.get("created_at") or ""),
                    }
                    for e in items[:8]
                ]
                user_prompt = (
                    f"Project: {proj}\n"
                    f"Experiences ({len(payload)}):\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                )

                try:
                    decision, in_tok, out_tok = await self._call_llm_json(
                        _DISTILL_SYSTEM_PROMPT,
                        user_prompt,
                        model_override=model,
                    )
                    res["input_tokens"] += in_tok
                    res["output_tokens"] += out_tok
                except Exception as exc:
                    logger.warning("Distill: LLM call failed for project=%s (%s)", proj, exc)
                    res["errors"] += 1
                    continue

                if not decision.get("should_distill"):
                    continue

                title = str(decision.get("title") or "").strip()
                problem = str(decision.get("problem") or "").strip()
                solution = _truncate_to_tokens(str(decision.get("solution") or "").strip(), max_tokens)
                if not title or not solution:
                    continue

                raw_tags = decision.get("tags") or []
                tags = sorted({str(t).strip() for t in raw_tags if str(t).strip()} | {"playbook", "auto-dream"})

                plan_item = {
                    "project": proj,
                    "source_ids": [str(e.get("id") or "") for e in items],
                    "title": title,
                    "tags": tags,
                    "estimated_solution_tokens": _estimate_tokens(solution),
                    "reason": str(decision.get("reason") or ""),
                }
                res["plan"].append(plan_item)

                if dry_run:
                    res["distilled"] += 1
                    continue

                try:
                    existing_id = await self._find_existing_playbook(proj, title)

                    if existing_id:
                        await self._sink.update_experience(
                            experience_id=existing_id,
                            title=title,
                            problem=problem or f"Project playbook for {proj}",
                            solution=solution,
                            tags=tags,
                            experience_type="playbook",
                        )
                    else:
                        await self._sink.save(
                            title=title,
                            problem=problem or f"Project playbook for {proj}",
                            solution=solution,
                            tags=tags,
                            project=proj,
                            experience_type="playbook",
                        )
                    res["distilled"] += 1
                except Exception:
                    logger.warning("Distill: failed to persist playbook '%s'", title[:60], exc_info=True)
                    res["errors"] += 1

        return res

    # ------------------------------------------------------------------
    # Phase 4b: Prune (Lightweight Outdated Orphan Cleanup)
    # ------------------------------------------------------------------

    async def _phase_prune(self, *, dry_run: bool = False, project: str | None = None) -> int:
        """Lightweight cleanup of Outdated, never-used, unpinned experiences older than prune_max_age_days.

        Heavy GC (score decay, Bronze sweep, soft-delete purge) is owned by MemoryJanitor.
        """
        try:
            return await self._prune_stale(dry_run=dry_run, project=project)
        except Exception:
            logger.warning("Service prune failed, skipping prune phase", exc_info=True)
            return 0

    async def _prune_stale(self, *, dry_run: bool = False, project: str | None = None) -> int:
        """Find and soft-delete Outdated unused orphan experiences via ExperienceService."""
        cutoff = _utcnow() - timedelta(days=self._cfg.prune_max_age_days)
        ctx = _daemon_context()
        try:
            candidates = await ctx.service.list_published(
                project=project,
                quality_tier="Outdated",
                limit=500,
                active_only=True,
            )
        except TypeError:
            candidates = await ctx.service.list_published(
                quality_tier="Outdated",
                limit=500,
            )

        pruned = 0
        for exp in candidates:
            created_dt = _parse_dt(exp.get("created_at"))
            if (
                exp.get("valid_until") is None
                and int(exp.get("recall_count", 1) or 0) == 0
                and int(exp.get("used_count", 0) or 0) == 0
                and not exp.get("is_pinned")
                and created_dt is not None
                and created_dt < cutoff
            ):
                if not dry_run:
                    await ctx.service.soft_delete_by_id(exp["id"])
                pruned += 1
        if pruned > 0:
            logger.info("%s %d outdated orphan experiences", "would prune" if dry_run else "pruned", pruned)

        # Defense 3: Escalate orphaned possibly_stale experiences (>14d, 0 used, unpinned) to stale
        if hasattr(ctx.service, "list_stale_candidates"):
            try:
                stale_cutoff = _utcnow() - timedelta(days=14)
                stale_cands = await ctx.service.list_stale_candidates(
                    project=project,
                    limit=200,
                    include_embedding=False,
                )
                escalated = 0
                for exp in stale_cands or []:
                    if str(exp.get("staleness_status") or "") != "possibly_stale":
                        continue
                    if exp.get("is_pinned") or int(exp.get("used_count", 0) or 0) > 0:
                        continue
                    upd_dt = _parse_dt(exp.get("updated_at") or exp.get("created_at"))
                    if upd_dt is not None and upd_dt < stale_cutoff:
                        if not dry_run and hasattr(ctx.service, "update"):
                            await ctx.service.update(exp["id"], user="auto_dream", staleness_status="stale")
                        escalated += 1
                if escalated > 0:
                    logger.info(
                        "%s %d orphaned possibly_stale experiences to stale",
                        "would escalate" if dry_run else "escalated",
                        escalated,
                    )
            except Exception:
                logger.debug("Orphan possibly_stale escalation skipped", exc_info=True)

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

    async def _count_new_experiences(
        self,
        last_dream: datetime | None,
        *,
        project: str | None = None,
    ) -> int:
        """Count published/promoted experiences created since last dream run."""
        ctx = _daemon_context()
        try:
            return await ctx.service.count_published(project=project, since=last_dream)
        except TypeError:
            return await ctx.service.count_published(since=last_dream)
