"""CRUD write operations (save, update, embedding text builder) for ExperienceService."""

from __future__ import annotations

import logging
import sys
import time
import uuid

from team_memory import io_logger
from team_memory.services.infra.event_bus import Events
from team_memory.storage.repository import ExperienceRepository as _DefaultExperienceRepository

logger = logging.getLogger("team_memory.service")


def _validate_source_status(source: str, exp_status: str) -> None:
    """Validate that draft status is only allowed for source='pipeline' or 'obsidian'."""
    if exp_status == "draft" and source not in ("pipeline", "obsidian"):
        raise ValueError(f"exp_status='draft' requires source='pipeline' or 'obsidian', got source='{source}'")


def _experience_repo_cls():
    pkg = sys.modules.get("team_memory.services.memory.experience")
    return getattr(pkg, "ExperienceRepository", _DefaultExperienceRepository)


class ExperienceCrudMixin:
    """Mixin providing save and update write operations."""

    async def save(
        self,
        title: str,
        problem: str,
        solution: str | None = None,
        created_by: str = "system",
        tags: list[str] | None = None,
        source: str = "auto_extract",
        exp_status: str = "published",
        visibility: str = "project",
        skip_dedup: bool = False,
        experience_type: str = "general",
        project: str | None = None,
        group_key: str | None = None,
        *,
        resolve_mutations: bool = False,
        epistemic_tier: str | None = None,
        evidence: str | None = None,
        evidence_anchors: dict | None = None,
        session=None,  # noqa: ANN001
    ) -> dict:
        """Save a new experience to the database.

        If group_key is provided, auto-groups under a shared parent.
        If write-time mutation resolution is enabled (and exp_status != 'draft'),
        adjudicates ADD / UPDATE / SUPERSEDE / NOOP against top-3 active neighbors.

        Returns:
            Experience dict. If dedup is triggered, returns a dict with
            "status": "duplicate_detected" and "candidates" list.
        """
        from team_memory.services.memory.evidence_anchors import (
            extract_evidence_anchors,
            infer_epistemic_tier,
        )
        from team_memory.utils.secret_redactor import redact_fields

        redacted_fields, _ = redact_fields(
            {
                "title": title,
                "problem": problem,
                "solution": solution,
                "evidence": evidence,
            },
            context="experience.save",
            actor=created_by,
        )
        title = redacted_fields["title"] or ""
        problem = redacted_fields["problem"] or ""
        solution = redacted_fields["solution"]
        evidence = redacted_fields["evidence"]

        async with self._session_or(session) as session:
            # Validate source/status constraint
            _validate_source_status(source, exp_status)

            prepare_start = time.monotonic()
            repo = _experience_repo_cls()(session)

            # Defense 2: 0ms local evidence anchor extraction & Defense 1: epistemic tiering
            computed_anchors = extract_evidence_anchors(
                problem,
                solution,
                evidence,
                existing=evidence_anchors,
            )
            resolved_epistemic_tier = infer_epistemic_tier(
                explicit_tier=epistemic_tier,
                source=source,
                evidence=evidence,
                text=f"{title}\n{problem}\n{solution or ''}",
            )

            # Build text for embedding
            # Truncate to 4000 chars to stay within embedding model's
            # 8192-token context limit (Chinese text ~1 char/token).
            embed_text = f"{title}\n{problem}\n{solution or ''}"
            if tags:
                embed_text += f"\n{' '.join(tags)}"
            embed_text = embed_text[:4000]

            # Generate embedding with one retry on failure.
            # If embedding still fails after retry, save with embedding=None
            # so the experience is not silently lost.
            import asyncio as _asyncio

            embedding = None
            for _attempt in range(2):
                try:
                    embedding = await self._embedding.encode_single(embed_text)
                    break
                except Exception as e:
                    logger.warning("Failed to generate embedding (attempt %d): %s", _attempt + 1, e)
                    if _attempt == 0:
                        await _asyncio.sleep(1.0)

            if embedding is None:
                logger.warning(
                    "Embedding failed after retry — saving experience without embedding. "
                    "The record will lack vector search until backfill runs."
                )

            mutation_decision = None
            cfg = self._lifecycle_config
            auto_resolve = getattr(cfg, "auto_resolve_mutations", False) is True
            use_mutation_resolver = (resolve_mutations or auto_resolve) and exp_status != "draft"
            if not skip_dedup and embedding is not None and use_mutation_resolver:
                from team_memory.services.memory.mutation_resolver import MutationResolver

                sim_threshold = float(getattr(cfg, "mutation_similarity_threshold", 0.82))
                top_k = int(getattr(cfg, "mutation_top_k", 3))
                llm_timeout = float(getattr(cfg, "mutation_llm_timeout_seconds", 2.0))
                llm_enabled = bool(getattr(cfg, "mutation_llm_enabled", True)) and (
                    self._llm_config is not None
                )

                raw_candidates = await repo.check_similar(
                    embedding=embedding,
                    threshold=sim_threshold,
                    limit=top_k,
                    project=project,
                )
                candidates = raw_candidates if isinstance(raw_candidates, list) else []
                resolver = MutationResolver(
                    llm_config=self._llm_config,
                    mutation_threshold=sim_threshold,
                    llm_timeout_seconds=llm_timeout,
                    llm_enabled=llm_enabled,
                    update_action_name="UPDATE",
                )
                mutation_decision = await resolver.resolve(
                    title=title,
                    problem=problem,
                    solution=solution,
                    candidates=candidates,
                )

                if mutation_decision.action == "NOOP" and mutation_decision.target_id:
                    target_uuid = uuid.UUID(mutation_decision.target_id)
                    exp = await repo.touch_verified(
                        target_uuid,
                        reason=mutation_decision.reason,
                        actor=created_by,
                    )
                    if self._event_bus:
                        await self._event_bus.emit(
                            Events.EXPERIENCE_UPDATED,
                            {
                                "experience_id": mutation_decision.target_id,
                                "fields_updated": ["last_verified_at", "staleness_status"],
                                "user": created_by,
                            },
                        )
                    base_dict = (
                        exp.to_dict()
                        if exp is not None
                        else {"id": mutation_decision.target_id, "title": title}
                    )
                    base_dict["mutation_action"] = "NOOP"
                    base_dict["target_id"] = mutation_decision.target_id
                    base_dict["mutation_reason"] = mutation_decision.reason
                    return base_dict

                if mutation_decision.action in ("MERGE", "UPDATE") and mutation_decision.target_id:
                    target_uuid = uuid.UUID(mutation_decision.target_id)
                    merged_sol = mutation_decision.merged_solution or solution or ""
                    merged_embed_text = f"{title}\n{problem}\n{merged_sol}"[:4000]
                    merged_embedding = embedding
                    try:
                        merged_embedding = await self._embedding.encode_single(merged_embed_text)
                    except Exception:
                        logger.warning("Failed to re-encode merged experience; reusing incoming embedding")
                    merge_kwargs: dict = {
                        "merged_solution": mutation_decision.merged_solution,
                        "merged_tags": tags,
                        "merged_epistemic_tier": resolved_epistemic_tier,
                        "embedding": merged_embedding,
                        "reason": mutation_decision.reason,
                        "actor": created_by,
                    }
                    if computed_anchors:
                        merge_kwargs["merged_evidence_anchors"] = computed_anchors
                    try:
                        exp = await repo.merge_experience(target_uuid, **merge_kwargs)
                    except TypeError:
                        merge_kwargs.pop("merged_evidence_anchors", None)
                        merge_kwargs.pop("merged_epistemic_tier", None)
                        exp = await repo.merge_experience(target_uuid, **merge_kwargs)
                    if self._event_bus:
                        await self._event_bus.emit(
                            Events.EXPERIENCE_UPDATED,
                            {
                                "experience_id": mutation_decision.target_id,
                                "fields_updated": ["solution", "evidence_anchors", "staleness_status"],
                                "user": created_by,
                            },
                        )
                    base_dict = (
                        exp.to_dict()
                        if exp is not None
                        else {"id": mutation_decision.target_id, "title": title}
                    )
                    base_dict["mutation_action"] = mutation_decision.action
                    base_dict["target_id"] = mutation_decision.target_id
                    base_dict["mutation_reason"] = mutation_decision.reason
                    return base_dict

            # Dedup-on-save check (fallback when mutation resolver is disabled or saving draft)
            elif (
                not skip_dedup
                and embedding is not None
                and self._lifecycle_config is not None
                and self._lifecycle_config.dedup_on_save
            ):
                try:
                    candidates = await repo.check_similar(
                        embedding=embedding,
                        threshold=self._lifecycle_config.dedup_on_save_threshold,
                        limit=5,
                        project=project,
                    )
                    if isinstance(candidates, list) and candidates:
                        return {
                            "status": "duplicate_detected",
                            "candidates": candidates,
                            "message": f"发现 {len(candidates)} 条相似经验，请确认是否仍要保存",
                            "experience_id": None,
                        }
                except Exception:
                    logger.warning("Dedup-on-save check failed, proceeding with save")

            prepare_duration_ms = int((time.monotonic() - prepare_start) * 1000)
            io_logger.log_internal(
                "save_prepare",
                {"title": (title or "")[:50]},
                duration_ms=prepare_duration_ms,
            )

            # Auto-group via group_key
            parent_id = None
            if group_key:
                parent = await repo.find_or_create_group_parent(
                    project=project or "default",
                    group_key=group_key,
                    created_by=created_by,
                )
                parent_id = parent.id

            persist_start = time.monotonic()
            create_kwargs: dict = {
                "title": title,
                "description": problem,
                "solution": solution,
                "created_by": created_by,
                "tags": tags,
                "embedding": embedding,
                "source": source,
                "exp_status": exp_status,
                "visibility": visibility,
                "experience_type": experience_type,
                "project": project or "default",
                "group_key": group_key,
                "parent_id": parent_id,
                "epistemic_tier": resolved_epistemic_tier,
                "evidence_anchors": computed_anchors,
                "staleness_status": "fresh",
            }
            try:
                experience = await repo.create(**create_kwargs)
            except TypeError:
                for extra_k in ("epistemic_tier", "evidence_anchors", "staleness_status"):
                    create_kwargs.pop(extra_k, None)
                experience = await repo.create(**create_kwargs)

            if (
                mutation_decision is not None
                and mutation_decision.action == "SUPERSEDE"
                and mutation_decision.target_id
            ):
                await repo.supersede_experience(
                    uuid.UUID(mutation_decision.target_id),
                    experience.id,
                    reason=mutation_decision.reason,
                    actor=created_by,
                )

            if self._event_bus:
                await self._event_bus.emit(
                    Events.EXPERIENCE_CREATED,
                    {
                        "experience_id": str(experience.id),
                        "title": title,
                        "description": problem,
                        "solution": solution or "",
                        "tags": tags or [],
                        "project": project or "default",
                        "created_by": created_by,
                        "status": exp_status,
                        "visibility": visibility,
                    },
                )
            persist_duration_ms = int((time.monotonic() - persist_start) * 1000)
            io_logger.log_internal(
                "save_persist",
                {
                    "experience_id": str(experience.id),
                    "title": (title or "")[:50],
                },
                duration_ms=persist_duration_ms,
            )

            # Re-embed parent if this child was added to a group
            if parent_id:
                try:
                    parent_exp = await repo.get_with_children(parent_id)
                    if parent_exp:
                        parent_embed_text = self._build_parent_embed_text(
                            parent_exp.title,
                            parent_exp.description,
                            parent_exp.solution or "",
                            tags=parent_exp.tags,
                            children=parent_exp.children,
                        )
                        parent_embedding = await self._embedding.encode_single(parent_embed_text)
                        await repo.update(parent_id, embedding=parent_embedding)
                except Exception:
                    logger.warning("Failed to re-embed group parent %s", parent_id)

            out = experience.to_dict()
            if mutation_decision is not None and mutation_decision.action == "SUPERSEDE":
                out["mutation_action"] = "SUPERSEDE"
                out["superseded_id"] = mutation_decision.target_id
                out["mutation_reason"] = mutation_decision.reason
            else:
                out["mutation_action"] = "ADD"
            return out

    async def update(
        self,
        experience_id: str,
        user: str = "system",
        *,
        session=None,  # noqa: ANN001
        **kwargs,
    ) -> dict | None:
        """Update an existing experience (in-place).

        Only kwargs that are explicitly provided will be applied.
        Supported kwargs:
            title, description, solution, tags, exp_status, visibility,
            experience_type, solution_addendum (legacy append mode).
        """
        async with self._session_or(session) as session:
            repo = _experience_repo_cls()(session)
            exp_uuid = uuid.UUID(experience_id)

            experience = await repo.get_by_id(exp_uuid)
            if experience is None:
                return None

            from team_memory.utils.secret_redactor import redact_fields

            redact_candidates = {
                k: kwargs[k]
                for k in ("title", "description", "solution", "solution_addendum")
                if k in kwargs and isinstance(kwargs[k], str)
            }
            if redact_candidates:
                cleaned_updates, _ = redact_fields(
                    redact_candidates,
                    context="experience.update",
                    actor=user,
                )
                kwargs = {**kwargs, **cleaned_updates}

            updates: dict = {}
            need_reembed = False

            if "title" in kwargs:
                updates["title"] = kwargs["title"]
                need_reembed = True

            if "description" in kwargs:
                updates["description"] = kwargs["description"]
                need_reembed = True

            solution_addendum = kwargs.get("solution_addendum")
            if "solution" in kwargs:
                updates["solution"] = kwargs["solution"]
                need_reembed = True
            elif solution_addendum:
                updates["solution"] = f"{experience.solution or ''}\n\n---\n\n{solution_addendum}"
                need_reembed = True

            if "tags" in kwargs:
                updates["tags"] = kwargs["tags"]

            if "exp_status" in kwargs:
                updates["exp_status"] = kwargs["exp_status"]
            if "visibility" in kwargs:
                updates["visibility"] = kwargs["visibility"]
            if "experience_type" in kwargs:
                updates["experience_type"] = kwargs["experience_type"]
            if "epistemic_tier" in kwargs:
                updates["epistemic_tier"] = kwargs["epistemic_tier"]
            if "staleness_status" in kwargs:
                updates["staleness_status"] = kwargs["staleness_status"]
            if "evidence_anchors" in kwargs:
                updates["evidence_anchors"] = kwargs["evidence_anchors"]
            if "last_verified_at" in kwargs:
                updates["last_verified_at"] = kwargs["last_verified_at"]

            if not updates:
                return experience.to_dict()

            # Re-generate embedding & refresh evidence anchors if content fields changed
            if need_reembed:
                from team_memory.services.memory.evidence_anchors import extract_evidence_anchors

                final_title = updates.get("title", experience.title)
                final_desc = updates.get("description", experience.description)
                final_sol = updates.get("solution", experience.solution) or ""
                final_tags = updates.get("tags", experience.tags)
                if "evidence_anchors" not in updates:
                    re_anchors = extract_evidence_anchors(final_desc, final_sol)
                    if re_anchors:
                        updates["evidence_anchors"] = re_anchors
                if "staleness_status" not in updates:
                    updates["staleness_status"] = "fresh"
                children = await repo.get_children(exp_uuid)
                embed_text = self._build_parent_embed_text(
                    final_title,
                    final_desc,
                    final_sol,
                    tags=final_tags,
                    children=children if children else None,
                )
                try:
                    embedding = await self._embedding.encode_single(embed_text)
                    updates["embedding"] = embedding
                except Exception:
                    logger.warning("Failed to regenerate embedding during update")

            updated = await repo.update(exp_uuid, **updates)

            if updated and updates:
                fields_updated = list(updates.keys())
                if self._event_bus:
                    await self._event_bus.emit(
                        Events.EXPERIENCE_UPDATED,
                        {
                            "experience_id": experience_id,
                            "fields_updated": fields_updated,
                            "user": user,
                            # Include content fields for entity extraction
                            "title": updated.title if updated else None,
                            "description": updated.description if updated else None,
                            "solution": updated.solution if updated else None,
                            "tags": updated.tags if updated else None,
                            "project": updated.project if updated else None,
                            "status": updated.exp_status if updated else None,
                            "visibility": updated.visibility if updated else None,
                        },
                    )

                # Update archive status if exp_status or visibility changed
                if self._archive_service and ("exp_status" in updates or "visibility" in updates):
                    try:
                        await self._archive_service.update_archive_status_for_experience(exp_uuid)
                    except Exception:
                        logger.warning("Failed to update archive status", exc_info=True)

            return updated.to_dict() if updated else None
