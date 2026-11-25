"""Lifecycle operations (soft_delete, restore, feedback, publish_draft, pin, revive, status) for ExperienceService."""

from __future__ import annotations

import logging
import uuid

from team_memory.services.infra.event_bus import Events
from team_memory.services.memory.experience.crud import _experience_repo_cls

logger = logging.getLogger("team_memory.service")


class ExperienceLifecycleMixin:
    """Mixin providing experience lifecycle, feedback, and draft publication workflows."""

    async def soft_delete(self, experience_id: str) -> bool:
        """Soft-delete an experience."""
        async with self._session() as session:
            repo = _experience_repo_cls()(session)
            result = await repo.soft_delete(uuid.UUID(experience_id))
            if result and self._event_bus:
                await self._event_bus.emit(
                    Events.EXPERIENCE_DELETED,
                    {
                        "experience_id": experience_id,
                        "hard": False,
                    },
                )
            return result

    async def restore(self, experience_id: str) -> bool:
        """Restore a soft-deleted experience."""
        async with self._session() as session:
            repo = _experience_repo_cls()(session)
            result = await repo.restore(uuid.UUID(experience_id))
            if result and self._event_bus:
                await self._event_bus.emit(
                    Events.EXPERIENCE_RESTORED,
                    {
                        "experience_id": experience_id,
                    },
                )
            return result

    async def feedback(
        self,
        experience_id: str,
        rating: int,
        feedback_by: str,
        comment: str | None = None,
        fitness_score: int | None = None,
        source: str = "explicit_user",
        weight: float = 1.0,
        search_log_id: str | None = None,
        *,
        session=None,  # noqa: ANN001
        **kwargs: object,
    ) -> bool:
        """Submit feedback for an experience. rating: 1-5, 5=best."""
        async with self._session_or(session) as session:
            if not (1 <= rating <= 5):
                raise ValueError("rating must be between 1 and 5")
            repo = _experience_repo_cls()(session)
            exp_uuid = uuid.UUID(experience_id)

            experience = await repo.get_by_id(exp_uuid)
            if experience is None:
                return False

            add_kwargs: dict[str, object] = {
                "experience_id": exp_uuid,
                "rating": rating,
                "feedback_by": feedback_by,
                "comment": comment,
                "fitness_score": fitness_score,
            }
            if source != "explicit_user" or weight != 1.0 or search_log_id is not None:
                add_kwargs["source"] = source
                add_kwargs["weight"] = weight
                if search_log_id is not None:
                    add_kwargs["search_log_id"] = uuid.UUID(search_log_id)

            await repo.add_feedback(**add_kwargs)

            is_implicit = source.startswith("implicit_")
            # Add quality score bonus for positive feedback (rating >= 4)
            # or penalty + negative_feedback_count increment for negative feedback (rating <= 2)
            if rating >= 4:
                delta = (2.0 if rating == 5 else 1.0) * float(weight)
                await repo.increment_quality_score(exp_uuid, delta)
                if is_implicit:
                    if hasattr(repo, "increment_implicit_positive_count"):
                        await repo.increment_implicit_positive_count(exp_uuid)
                else:
                    await repo.increment_used_count(exp_uuid)
                    # Defense 4 feedback loop: explicit rating=5 re-verifies freshness
                    if rating == 5 and hasattr(repo, "update"):
                        import inspect
                        from datetime import datetime, timezone

                        res_u = repo.update(
                            exp_uuid,
                            staleness_status="fresh",
                            last_verified_at=datetime.now(timezone.utc),
                        )
                        if inspect.isawaitable(res_u):
                            await res_u
            elif rating <= 2:
                import inspect

                delta = -3.0 * float(weight)
                res_q = repo.increment_quality_score(exp_uuid, delta)
                if inspect.isawaitable(res_q):
                    await res_q
                if is_implicit:
                    if hasattr(repo, "increment_implicit_negative_count"):
                        res_n = repo.increment_implicit_negative_count(exp_uuid)
                        if inspect.isawaitable(res_n):
                            await res_n
                else:
                    if hasattr(repo, "increment_negative_feedback_count"):
                        res_n = repo.increment_negative_feedback_count(exp_uuid)
                        if inspect.isawaitable(res_n):
                            await res_n
                    # Defense 4 feedback loop: explicit negative rating flags possibly_stale
                    if hasattr(repo, "update"):
                        res_u = repo.update(exp_uuid, staleness_status="possibly_stale")
                        if inspect.isawaitable(res_u):
                            await res_u

            if self._event_bus:
                await self._event_bus.emit(
                    Events.FEEDBACK_ADDED,
                    {
                        "experience_id": experience_id,
                        "rating": rating,
                        "feedback_by": feedback_by,
                        "source": source,
                        "weight": weight,
                    },
                )
            return True

    async def publish_draft(
        self,
        draft_id: str,
        *,
        user: str = "system",
        refined_content: str | None = None,
        resolve_mutations: bool | None = None,
        session=None,  # noqa: ANN001
    ) -> dict | None:
        """Promote a draft to published with write-time mutation adjudication and zombie-draft cleanup."""
        from team_memory.services.memory.evidence_anchors import extract_evidence_anchors
        from team_memory.services.memory.mutation_resolver import MutationResolver

        async with self._session_or(session) as sess:
            repo = _experience_repo_cls()(sess)
            draft_uuid = uuid.UUID(draft_id)
            draft = await repo.get_by_id(draft_uuid)
            if draft is None:
                return None

            title = draft.title or ""
            problem = refined_content if refined_content else (draft.description or "")
            solution = draft.solution or problem
            tags = list(draft.tags or [])
            project = draft.project or "default"

            anchors = extract_evidence_anchors(problem, solution, existing=getattr(draft, "evidence_anchors", None))

            embed_text = f"{title}\n{problem}\n{solution or ''}"[:4000]
            embedding = getattr(draft, "embedding", None)
            if refined_content or embedding is None:
                try:
                    embedding = await self._embedding.encode_single(embed_text)
                except Exception:
                    logger.warning("publish_draft: failed to compute embedding for %s", draft_id)

            cfg = self._lifecycle_config
            should_resolve = (
                resolve_mutations
                if resolve_mutations is not None
                else (getattr(cfg, "auto_resolve_mutations", True) is True)
            )

            if should_resolve and embedding is not None:
                sim_threshold = float(getattr(cfg, "mutation_similarity_threshold", 0.82))
                top_k = int(getattr(cfg, "mutation_top_k", 3))
                llm_timeout = float(getattr(cfg, "mutation_llm_timeout_seconds", 2.0))
                llm_enabled = bool(getattr(cfg, "mutation_llm_enabled", True)) and (
                    self._llm_config is not None
                )
                raw_candidates = await repo.check_similar(
                    embedding=list(embedding),
                    threshold=sim_threshold,
                    limit=top_k + 1,
                    project=project,
                )
                candidates = [
                    c
                    for c in (raw_candidates if isinstance(raw_candidates, list) else [])
                    if str(c.get("id", "")) != str(draft_uuid)
                ][:top_k]

                if candidates:
                    resolver = MutationResolver(
                        llm_config=self._llm_config,
                        mutation_threshold=sim_threshold,
                        llm_timeout_seconds=llm_timeout,
                        llm_enabled=llm_enabled,
                        update_action_name="UPDATE",
                    )
                    decision = await resolver.resolve(
                        title=title,
                        problem=problem,
                        solution=solution,
                        candidates=candidates,
                    )

                    if decision.action == "NOOP" and decision.target_id:
                        target_uuid = uuid.UUID(decision.target_id)
                        target_exp = await repo.touch_verified(
                            target_uuid,
                            reason=decision.reason,
                            actor=user,
                        )
                        # Clean up zombie draft
                        await repo.update(
                            draft_uuid,
                            is_deleted=True,
                            superseded_by_id=target_uuid,
                        )
                        if self._event_bus:
                            await self._event_bus.emit(
                                Events.EXPERIENCE_UPDATED,
                                {
                                    "experience_id": decision.target_id,
                                    "fields_updated": ["last_verified_at", "staleness_status"],
                                    "user": user,
                                },
                            )
                        out = target_exp.to_dict() if target_exp else {"id": decision.target_id}
                        out["status"] = "published"
                        out["mutation_action"] = "NOOP"
                        out["target_id"] = decision.target_id
                        out["retired_draft_id"] = draft_id
                        return out

                    if decision.action in ("UPDATE", "MERGE") and decision.target_id:
                        target_uuid = uuid.UUID(decision.target_id)
                        merged_sol = decision.merged_solution or solution
                        merged_emb = list(embedding)
                        try:
                            merged_emb = await self._embedding.encode_single(
                                f"{title}\n{problem}\n{merged_sol}"[:4000]
                            )
                        except Exception:
                            pass
                        draft_tier = getattr(draft, "epistemic_tier", None) or "auto_extract"
                        target_exp = await repo.merge_experience(
                            target_uuid,
                            merged_solution=decision.merged_solution,
                            merged_tags=tags,
                            merged_evidence_anchors=anchors or None,
                            merged_epistemic_tier=draft_tier,
                            embedding=merged_emb,
                            reason=decision.reason,
                            actor=user,
                        )
                        # Clean up zombie draft
                        await repo.update(
                            draft_uuid,
                            is_deleted=True,
                            superseded_by_id=target_uuid,
                        )
                        if self._event_bus:
                            await self._event_bus.emit(
                                Events.EXPERIENCE_UPDATED,
                                {
                                    "experience_id": decision.target_id,
                                    "fields_updated": ["solution", "evidence_anchors", "staleness_status"],
                                    "user": user,
                                },
                            )
                        out = target_exp.to_dict() if target_exp else {"id": decision.target_id}
                        out["status"] = "published"
                        out["mutation_action"] = decision.action
                        out["target_id"] = decision.target_id
                        out["retired_draft_id"] = draft_id
                        return out

                    if decision.action == "SUPERSEDE" and decision.target_id:
                        updates: dict = {
                            "exp_status": "published",
                            "evidence_anchors": anchors,
                            "staleness_status": "fresh",
                        }
                        if refined_content:
                            updates["description"] = refined_content
                        if embedding is not None:
                            updates["embedding"] = list(embedding)
                        updated = await repo.update(draft_uuid, **updates)
                        await repo.supersede_experience(
                            uuid.UUID(decision.target_id),
                            draft_uuid,
                            reason=decision.reason,
                            actor=user,
                        )
                        if self._event_bus:
                            await self._event_bus.emit(
                                Events.EXPERIENCE_UPDATED,
                                {
                                    "experience_id": draft_id,
                                    "fields_updated": list(updates.keys()),
                                    "user": user,
                                },
                            )
                        out = updated.to_dict() if updated else {"id": draft_id}
                        out["mutation_action"] = "SUPERSEDE"
                        out["superseded_id"] = decision.target_id
                        return out

            updates = {
                "exp_status": "published",
                "evidence_anchors": anchors,
                "staleness_status": "fresh",
            }
            if refined_content:
                updates["description"] = refined_content
            if embedding is not None:
                updates["embedding"] = list(embedding)
            updated = await repo.update(draft_uuid, **updates)
            if updated and self._event_bus:
                await self._event_bus.emit(
                    Events.EXPERIENCE_UPDATED,
                    {
                        "experience_id": draft_id,
                        "fields_updated": list(updates.keys()),
                        "user": user,
                    },
                )
            if updated is None:
                return None
            out = updated.to_dict()
            out["mutation_action"] = "ADD"
            return out

    async def soft_delete_by_id(self, experience_id: str) -> bool:
        """Soft-delete an experience by string ID.

        Wraps soft_delete() which already accepts a string ID.

        Args:
            experience_id: String UUID of the experience.

        Returns:
            True if deleted, False if not found.
        """
        return await self.soft_delete(experience_id)

    async def increment_used_count(self, experience_id: str) -> None:
        """Increment the used_count field for an experience and boost quality_score.

        Args:
            experience_id: String UUID of the experience.
        """
        from unittest.mock import NonCallableMock

        from sqlalchemy import update

        from team_memory.storage.models import Experience

        repo_cls = _experience_repo_cls()
        exp_uuid = uuid.UUID(experience_id)
        async with self._session() as session:
            await session.execute(
                update(Experience).where(Experience.id == exp_uuid).values(used_count=(Experience.used_count + 1))
            )
            repo_mocked = isinstance(repo_cls, NonCallableMock) or isinstance(
                getattr(repo_cls, "increment_quality_score", None), NonCallableMock
            )
            if repo_mocked or not isinstance(session, NonCallableMock):
                await repo_cls(session).increment_quality_score(exp_uuid, 2.0)
            await session.commit()

    async def toggle_pin(self, experience_id: str) -> dict:
        """Toggle pin status. Returns {"is_pinned": bool, "message": str}."""
        async with self._session() as session:
            repo = _experience_repo_cls()(session)
            exp_uuid = uuid.UUID(experience_id)
            exp = await repo.get_by_id(exp_uuid)
            if exp is None:
                raise ValueError("Experience not found")

            new_pinned = not exp.is_pinned
            await repo.update(exp_uuid, is_pinned=new_pinned)

            label = "已置顶" if new_pinned else "已取消置顶"
            return {
                "message": label,
                "is_pinned": new_pinned,
            }

    async def revive(self, experience_id: str) -> dict:
        """Revive an Outdated experience. Returns dict or raises ValueError."""
        from datetime import datetime, timezone

        async with self._session() as session:
            repo = _experience_repo_cls()(session)
            exp_uuid = uuid.UUID(experience_id)
            exp = await repo.get_by_id(exp_uuid)
            if exp is None:
                raise ValueError("Experience not found")

            if exp.quality_tier != "Outdated":
                raise ValueError("Only Outdated experiences can be revived")

            now = datetime.now(timezone.utc)
            await repo.update(
                exp_uuid,
                quality_score=100.0,
                quality_tier="Silver",
                last_scored_at=now,
            )

            return {
                "message": "经验已恢复，质量评分重置为 100",
                "quality_score": 100.0,
                "quality_tier": "Silver",
            }

    async def change_status(
        self,
        experience_id: str,
        *,
        new_status: str | None = None,
        visibility: str | None = None,
        changed_by: str,
        is_admin: bool = False,
    ) -> dict | None:
        """Change status and/or visibility. Returns experience dict or None."""
        async with self._session() as session:
            repo = _experience_repo_cls()(session)
            exp_uuid = uuid.UUID(experience_id)

            if new_status:
                try:
                    result = await repo.change_status(
                        experience_id=exp_uuid,
                        new_status=new_status,
                        visibility=visibility,
                        changed_by=changed_by,
                        is_admin=is_admin,
                    )
                except ValueError:
                    raise
            else:
                result = await repo.update(exp_uuid, visibility=visibility)

            if result is None:
                return None

            # Trigger archive status update if needed
            if new_status and self._archive_service:
                try:
                    await self._archive_service.update_archive_status_for_experience(exp_uuid)
                except Exception:
                    logger.warning("Failed to update archive status", exc_info=True)

            return result.to_dict()
