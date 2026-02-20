"""AI Orchestration layer — unified entry point for all AI-powered operations.

Composes LLMClient, PromptLoader, and ExperienceService to provide
high-level AI capabilities: parsing, enrichment, review, and suggestions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from team_memory.services.llm_client import LLMClient, LLMError

if TYPE_CHECKING:
    from team_memory.services.experience import ExperienceService
    from team_memory.services.prompt_loader import PromptLoader

logger = logging.getLogger("team_memory.ai_orchestrator")


@dataclass
class TeamContext:
    """Team context injected into AI prompts for personalization."""

    preset: str = "software-dev"
    recent_topics: list[str] = field(default_factory=list)
    user_role: str = "member"
    project: str | None = None
    custom_instructions: str = ""


class AIOrchestrator:
    """Orchestrates all AI interactions for TeamMemory."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        service: ExperienceService | None = None,
    ):
        self._llm = llm_client
        self._prompts = prompt_loader
        self._service = service

    async def parse_document(
        self,
        content: str,
        *,
        as_group: bool = False,
        max_input_chars: int = 8000,
        context: TeamContext | None = None,
    ) -> dict:
        """Parse free-form text into structured experience(s)."""
        prompt_name = "parse_group" if as_group else "parse_single"
        system = self._prompts.load(prompt_name)
        reply = await self._llm.chat_json(
            system, content[:max_input_chars], temperature=0.3
        )
        from team_memory.services.llm_parser import (
            _normalize_group,
            _normalize_single,
        )

        return _normalize_group(reply) if as_group else _normalize_single(reply)

    async def suggest_type(
        self, title: str, problem: str = ""
    ) -> dict:
        """Suggest experience type for given title/problem."""
        system = self._prompts.load("suggest_type")
        user_content = f"标题：{title}"
        if problem:
            user_content += f"\n问题描述：{problem}"

        try:
            parsed = await self._llm.chat_json(
                system, user_content[:2000], temperature=0.2, timeout=30.0
            )
        except LLMError as e:
            logger.warning("Type suggestion failed: %s", e)
            return {
                "suggested_type": "general",
                "confidence": 0.0,
                "reason": "LLM 不可用",
                "fallback_types": [],
            }

        from team_memory.schemas import get_schema_registry

        registry = get_schema_registry()
        suggested = str(parsed.get("type", "general")).strip()
        if not registry.is_valid_type(suggested):
            suggested = "general"

        return {
            "suggested_type": suggested,
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
            "reason": str(parsed.get("reason", "")).strip(),
            "fallback_types": [
                str(f).strip()
                for f in (parsed.get("fallback_types") or [])
                if registry.is_valid_type(str(f).strip())
            ][:2],
        }

    async def generate_summary(
        self, content: str, *, max_input_chars: int = 4000
    ) -> str:
        """Generate a concise summary for experience content."""
        system = self._prompts.load("summary")
        return await self._llm.chat(
            system, content[:max_input_chars], temperature=0.2, timeout=60.0
        )

    async def enrich_experience(self, experience: dict) -> dict:
        """AI-assisted field completion for missing fields."""
        system = self._prompts.load("enrich")
        import json

        user_content = json.dumps(experience, ensure_ascii=False, indent=2)
        try:
            result = await self._llm.chat_json(
                system, user_content, temperature=0.3, timeout=60.0
            )
            return {**experience, **{k: v for k, v in result.items() if v is not None}}
        except LLMError as e:
            logger.warning("Enrich failed: %s", e)
            return experience

    async def review_experience(self, experience: dict) -> dict:
        """AI-assisted quality review with completeness check."""
        system = self._prompts.load("review")
        import json

        user_content = json.dumps(experience, ensure_ascii=False, indent=2)
        try:
            return await self._llm.chat_json(
                system, user_content, temperature=0.2, timeout=60.0
            )
        except LLMError as e:
            logger.warning("Review failed: %s", e)
            return {
                "score": 0,
                "suggestions": [],
                "error": str(e),
            }

    async def suggest_related(
        self, experience_id: str, *, max_results: int = 5
    ) -> list[dict]:
        """Suggest related experiences based on semantic similarity."""
        if not self._service:
            return []

        from team_memory.storage.repository import ExperienceRepository

        async with self._service._session() as session:
            repo = ExperienceRepository(session)
            exp = await repo.get_by_id(experience_id)
            if not exp:
                return []

            query = " ".join(
                filter(None, [exp.title, exp.description, exp.solution])
            )[:500]

        results = await self._service.search(
            query=query, max_results=max_results + 1
        )
        return [r for r in results if r.get("id") != experience_id][:max_results]

    async def build_team_context(
        self, user_role: str = "member", project: str | None = None
    ) -> TeamContext:
        """Build TeamContext from current state."""
        recent_topics: list[str] = []
        if self._service:
            try:
                stats = await self._service.get_query_stats()
                recent_topics = list(
                    (stats.get("search_type_distribution") or {}).keys()
                )[:5]
            except Exception:
                pass

        return TeamContext(
            recent_topics=recent_topics,
            user_role=user_role,
            project=project,
        )
