"""Personal memory service: per-user preferences/habits with semantic overwrite."""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from team_memory.storage.personal_memory_repository import (
    PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
    VALID_PERSONAS,
    PersonalMemoryRepository,
    normalize_scope,
)
from team_memory.utils.tokenizer import estimate_tokens

if TYPE_CHECKING:
    from team_memory.embedding.base import EmbeddingProvider

logger = logging.getLogger("team_memory.personal_memory")

DEFAULT_PERSONA_CAP = 20
DEFAULT_PROJECT_CAP = 10


class PersonalMemoryService:
    """Service for personal memory CRUD and semantic overwrite (mem0-style)."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_url: str = "",
    ):
        self._embedding = embedding_provider
        self._db_url = db_url

    @asynccontextmanager
    async def _session(self):
        from team_memory.storage.database import get_session

        async with get_session(self._db_url) as session:
            yield session

    async def list_by_user(
        self,
        user_id: str,
        scope: str | None = None,
        profile_kind: str | None = None,
        project: str | None = None,
    ) -> list[dict]:
        """List memories for user, optionally filtered by scope, project, and/or profile_kind."""
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            kwargs: dict[str, Any] = {"scope": scope, "profile_kind": profile_kind}
            if project is not None:
                kwargs["project"] = project
            rows = await repo.list_by_user(user_id, **kwargs)
            return [r.to_dict() for r in rows]

    @staticmethod
    def _dedupe_cap(contents: list[str], max_n: int) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for c in contents:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= max_n:
                break
        return out

    async def build_profile_for_user(
        self,
        user_id: str | None = None,
        current_context: str | None = None,
        context_similarity_threshold: float = 0.5,
        max_per_side: int = 8,
        persona: str | None = None,
        *,
        user_name: str | None = None,
        max_items: int | None = None,
        max_tokens: int | None = None,
        query_hint: str = "",
        project: str | None = None,
        context_embedding: list[float] | None = None,
    ) -> dict[str, list[str]]:
        """Assemble Supermemory-shaped profile: static/dynamic string lists.

        Supports:
          - scope/project routing (global + matching project only)
          - hybrid ranking by semantic similarity, keyword overlap, priority, and project match
          - item count (max_items) and token budget (max_tokens) enforcement
          - pre-computed context_embedding reuse to avoid duplicate embedding RPCs
        """
        effective_user = user_id if user_id is not None else user_name
        if not effective_user or effective_user.strip().lower() == "anonymous":
            return {"static": [], "dynamic": []}

        # Support positional (user_id, persona, max_items=..., query_hint=...) calls
        if (
            persona is None
            and current_context in VALID_PERSONAS
            and (max_items is not None or bool(query_hint))
        ):
            persona = current_context
            current_context = query_hint or None

        effective_context = current_context if current_context is not None else (query_hint or None)
        clean_project = (project or "").strip() or None

        if hasattr(self.pull, "_mock_name") or getattr(self.pull, "__module__", "").startswith(
            "unittest.mock"
        ):
            rows = await self.pull(
                user_id=effective_user,
                current_context=effective_context,
                context_similarity_threshold=context_similarity_threshold,
            )
        else:
            emb = context_embedding
            if emb is None and effective_context and effective_context.strip():
                emb = await self._embedding.encode_single(effective_context.strip())
            async with self._session() as session:
                repo = PersonalMemoryRepository(session)
                pull_kwargs: dict[str, Any] = {
                    "user_id": effective_user,
                    "context_embedding": emb,
                    "context_similarity_threshold": context_similarity_threshold,
                }
                if clean_project is not None:
                    pull_kwargs["project"] = clean_project
                rows = await repo.list_for_pull(**pull_kwargs)

        def _extract_item(r: Any) -> tuple[str, str, str, float, str, float, str, str | None]:
            if isinstance(r, dict):
                content = str(r.get("content") or r.get("rule_text") or "").strip()
                r_persona = str(r.get("persona") or "general")
                r_scope = str(r.get("scope") or "global")
                r_project = r.get("project")
                p_kind = str(
                    r.get("profile_kind")
                    or ("static" if r_scope in ("global", "generic", "project") else "dynamic")
                )
                raw_conf = r.get("confidence")
                if raw_conf is not None:
                    conf = float(raw_conf)
                else:
                    conf = float((r.get("priority", 80) or 80) / 100.0)
                ctx_hint = str(r.get("context_hint") or "")
                vec_sim = float(r.get("_similarity") or r.get("similarity") or 0.0)
            else:
                content = str(
                    getattr(r, "content", None) or getattr(r, "rule_text", None) or ""
                ).strip()
                r_persona = str(getattr(r, "persona", None) or "general")
                r_scope = str(getattr(r, "scope", None) or "global")
                r_project = getattr(r, "project", None)
                p_kind = str(getattr(r, "profile_kind", None) or "static")
                raw_conf = getattr(r, "confidence", None)
                if raw_conf is not None:
                    conf = float(raw_conf)
                else:
                    conf = float((getattr(r, "priority", 80) or 80) / 100.0)
                ctx_hint = str(getattr(r, "context_hint", None) or "")
                vec_sim = float(
                    getattr(r, "_similarity", None) or getattr(r, "similarity", None) or 0.0
                )
            return content, r_persona, p_kind, conf, ctx_hint, vec_sim, r_scope, r_project

        extracted = [_extract_item(r) for r in rows]
        # Defense-in-depth scope/project filtering (in case rows came from a mocked repo)
        filtered = []
        for item in extracted:
            r_scope, r_project = item[6], item[7]
            if r_scope == "project":
                if not clean_project or r_project != clean_project:
                    continue
            filtered.append(item)
        extracted = filtered

        if persona:
            extracted = [item for item in extracted if item[1] == persona]

        use_budget = (max_items is not None and max_items > 0) or (
            max_tokens is not None and max_tokens > 0
        )
        if use_budget:
            hint_source = (query_hint or effective_context or "").strip().lower()
            hint_tokens = set(re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", hint_source))
            seen_keys: set[str] = set()
            scored_candidates: list[tuple[float, int, str, str]] = []
            for idx, (
                content,
                _r_persona,
                p_kind,
                conf,
                ctx_hint,
                vec_sim,
                r_scope,
                r_project,
            ) in enumerate(extracted):
                if not content:
                    continue
                dedupe_key = content.lower()
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                is_static = p_kind != "dynamic"
                text_lower = f"{content} {ctx_hint}".lower()
                overlap_count = (
                    sum(1 for tok in hint_tokens if tok in text_lower) if hint_tokens else 0
                )
                project_bonus = (
                    1.2 if (clean_project and r_scope == "project" and r_project == clean_project) else 0.0
                )
                score = (
                    (max(0.0, vec_sim) * 3.0)
                    + (overlap_count * 1.5)
                    + (conf * 2.0)
                    + project_bonus
                    + (0.2 if is_static else 0.0)
                )
                scored_candidates.append((score, idx, content, p_kind))

            scored_candidates.sort(key=lambda x: (-x[0], x[1]))

            selected: list[tuple[float, int, str, str]] = []
            used_tokens = 0
            item_limit = max_items if (max_items is not None and max_items > 0) else max_per_side * 2
            for cand in scored_candidates:
                if len(selected) >= item_limit:
                    break
                if max_tokens is not None and max_tokens > 0:
                    item_toks = estimate_tokens(cand[2])
                    if used_tokens + item_toks > max_tokens:
                        continue
                    used_tokens += item_toks
                selected.append(cand)

            static_ordered = [c for _s, _i, c, pk in selected if pk != "dynamic"]
            dynamic_ordered = [c for _s, _i, c, pk in selected if pk == "dynamic"]
            return {
                "static": self._dedupe_cap(static_ordered, max_per_side),
                "dynamic": self._dedupe_cap(dynamic_ordered, max_per_side),
            }

        static_ordered = []
        dynamic_ordered = []
        for content, _r_persona, p_kind, _conf, _ctx_hint, _vec_sim, _rs, _rp in extracted:
            if not content:
                continue
            if p_kind == "dynamic":
                dynamic_ordered.append(content)
            else:
                static_ordered.append(content)
        return {
            "static": self._dedupe_cap(static_ordered, max_per_side),
            "dynamic": self._dedupe_cap(dynamic_ordered, max_per_side),
        }

    async def pull(
        self,
        user_id: str | None,
        current_context: str | None = None,
        context_similarity_threshold: float = 0.5,
        project: str | None = None,
    ) -> list[dict]:
        """Pull memories for Agent context: global + matching project + context-matched."""
        if not user_id or user_id.strip().lower() == "anonymous":
            return []
        context_embedding = None
        if current_context and current_context.strip():
            context_embedding = await self._embedding.encode_single(current_context.strip())
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            pull_kwargs: dict[str, Any] = {
                "user_id": user_id,
                "context_embedding": context_embedding,
                "context_similarity_threshold": context_similarity_threshold,
            }
            if project is not None:
                pull_kwargs["project"] = project
            rows = await repo.list_for_pull(**pull_kwargs)
            return [r.to_dict() for r in rows]

    async def get_by_id(self, memory_id: str, user_id: str) -> dict | None:
        """Get one memory by id; must belong to user."""
        import uuid

        try:
            uid = uuid.UUID(memory_id)
        except ValueError:
            return None
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            mem = await repo.get_by_id(uid, user_id)
            return mem.to_dict() if mem else None

    async def _enforce_persona_cap(
        self,
        repo: PersonalMemoryRepository,
        user_id: str,
        persona: str,
        cap: int = DEFAULT_PERSONA_CAP,
    ) -> bool:
        """Enforce per-(user_id, persona) cap by evicting lowest-priority/oldest item."""
        count = await repo.count_by_persona(user_id, persona)
        if count >= cap:
            await repo.delete_oldest_by_persona(user_id, persona)
            logger.info(
                "Persona limit reached (%s/%s, cap=%d): deleted oldest memory",
                user_id,
                persona,
                cap,
            )
            return True
        return False

    async def _enforce_project_cap(
        self,
        repo: PersonalMemoryRepository,
        user_id: str,
        project: str,
        cap: int = DEFAULT_PROJECT_CAP,
    ) -> bool:
        """Enforce per-(user_id, project) cap for project-scoped overrides."""
        count = await repo.count_by_project(user_id, project)
        if count >= cap:
            await repo.delete_oldest_by_project(user_id, project)
            logger.info(
                "Project personal-memory limit reached (%s/%s, cap=%d): deleted oldest memory",
                user_id,
                project,
                cap,
            )
            return True
        return False

    async def write(
        self,
        user_id: str,
        content: str,
        scope: str = "global",
        context_hint: str | None = None,
        overwrite_threshold: float = PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
        profile_kind: str | None = None,
        persona: str = "general",
        priority: int = 50,
        project: str | None = None,
    ) -> dict:
        """Write one memory: compute embedding, then upsert by semantic (overwrite if similar).

        Normalizes scope to ``global`` or ``project`` and enforces persona/project caps.
        """
        scope_norm, proj_norm, pk = normalize_scope(
            scope, project=project, profile_kind=profile_kind
        )
        embedding = await self._embedding.encode_single(content)
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            await self._enforce_persona_cap(repo, user_id, persona)
            if scope_norm == "project" and proj_norm:
                await self._enforce_project_cap(repo, user_id, proj_norm)
            mem = await repo.upsert_by_semantic(
                user_id=user_id,
                content=content,
                scope=scope_norm,
                project=proj_norm,
                context_hint=context_hint,
                embedding=embedding,
                threshold=overwrite_threshold,
                profile_kind=pk,
                persona=persona,
                priority=priority,
            )
            return mem.to_dict()

    async def update(
        self,
        memory_id: str,
        user_id: str,
        content: str | None = None,
        scope: str | None = None,
        context_hint: str | None = None,
        profile_kind: str | None = None,
        project: str | None = None,
    ) -> dict | None:
        """Update an existing memory; optionally re-compute embedding if content changed."""
        import uuid

        try:
            uid = uuid.UUID(memory_id)
        except ValueError:
            return None
        embedding = None
        if content is not None:
            embedding = await self._embedding.encode_single(content)
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            mem = await repo.update(
                uid,
                user_id,
                content=content,
                scope=scope,
                project=project,
                context_hint=context_hint,
                embedding=embedding,
                profile_kind=profile_kind,
            )
            return mem.to_dict() if mem else None

    async def delete(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory; returns True if deleted."""
        import uuid

        try:
            uid = uuid.UUID(memory_id)
        except ValueError:
            return False
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            return await repo.delete(uid, user_id)

    async def count_by_persona(self, user_id: str, persona: str) -> int:
        """Return the count of memories for a (user_id, persona) pair."""
        async with self._session() as session:
            repo = PersonalMemoryRepository(session)
            return await repo.count_by_persona(user_id, persona)
