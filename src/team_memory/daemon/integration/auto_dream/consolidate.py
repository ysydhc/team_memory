"""Phase 3 (Consolidate) mixin for AutoDreamWorker."""

from __future__ import annotations

import json
import logging
from typing import Any

from team_memory.daemon.integration.auto_dream.state import (
    _CONSOLIDATE_SYSTEM_PROMPT,
    _is_protected,
)
from team_memory.services.llm.llm_client import LLMError

logger = logging.getLogger("daemon.auto_dream")


class AutoDreamConsolidateMixin:
    """Mixin implementing LLM semantic consolidation and protection guardrails."""

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
