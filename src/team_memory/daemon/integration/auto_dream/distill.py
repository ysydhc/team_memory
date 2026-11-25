"""Phase 4 (Distill Playbooks & Prune) and count/delete helpers for AutoDreamWorker."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from team_memory.daemon.integration.auto_dream.state import (
    _DISTILL_SYSTEM_PROMPT,
    _estimate_tokens,
    _get_daemon_context,
    _parse_dt,
    _truncate_to_tokens,
    _utcnow,
)

logger = logging.getLogger("daemon.auto_dream")


class AutoDreamDistillMixin:
    """Mixin implementing Phase 4a (Distill), Phase 4b (Prune), and service helpers."""

    async def _find_existing_playbook(self, project: str, title: str) -> str | None:
        """Find an existing active playbook in the same project with matching topic (zero search_logs)."""
        norm_target = " ".join(title.lower().split())
        target_tokens = {w for w in norm_target.replace(":", " ").replace("[playbook]", "").split() if len(w) > 1}
        try:
            ctx = _get_daemon_context()
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
        ctx = _get_daemon_context()
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

    async def _soft_delete(self, experience_id: str) -> None:
        """Soft-delete an experience via ExperienceService."""
        await self._soft_delete_experience(experience_id)

    async def _soft_delete_experience(self, experience_id: str) -> None:
        """Soft-delete an experience by ID via ExperienceService."""
        ctx = _get_daemon_context()
        await ctx.service.soft_delete_by_id(experience_id)

    async def _count_new_experiences(
        self,
        last_dream: datetime | None,
        *,
        project: str | None = None,
    ) -> int:
        """Count published/promoted experiences created since last dream run."""
        ctx = _get_daemon_context()
        try:
            return await ctx.service.count_published(project=project, since=last_dream)
        except TypeError:
            return await ctx.service.count_published(since=last_dream)
