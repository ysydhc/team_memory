from __future__ import annotations

import asyncio
import logging
from pathlib import PurePosixPath

from team_memory.services.memory.ops.recall import _infer_recall_project
from team_memory.utils.project import resolve_project as _resolve_project

logger = logging.getLogger("team_memory")


async def op_context(
    user: str,
    *,
    search_orchestrator: object,
    settings: object,
    embedding_provider: object,
    db_url: str,
    file_paths: list[str] | None = None,
    task_description: str | None = None,
    project: str | None = None,
) -> dict:
    """Return user profile + relevant experiences for current context."""
    resolved_project = _infer_recall_project(
        project,
        _resolve_project(project),
        file_path=file_paths[0] if file_paths else None,
        query_text=task_description,
    )
    max_side = getattr(settings.mcp, "profile_max_strings_per_side", None) or 20
    result: dict = {
        "user": user,
        "project": resolved_project,
        "profile": {"static": [], "dynamic": []},
        "relevant_experiences": [],
    }

    context_text = task_description
    if not context_text and file_paths:
        context_text = " ".join(PurePosixPath(fp).name for fp in file_paths[:5])

    query_parts: list[str] = []
    if task_description:
        query_parts.append(task_description)
    if file_paths:
        for fp in file_paths[:3]:
            p = PurePosixPath(fp)
            query_parts.append(p.name)

    async def _fetch_profile() -> dict:
        try:
            from team_memory.services.memory.personal_memory import PersonalMemoryService

            pm_svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url=db_url)
            return await pm_svc.build_profile_for_user(
                user_id=user,
                current_context=context_text,
                max_per_side=max_side,
            )
        except Exception:
            logger.debug("Failed to build user profile", exc_info=True)
            return {"static": [], "dynamic": []}

    async def _fetch_experiences() -> tuple[bool | None, list[dict]]:
        if not query_parts:
            return None, []
        try:
            query = " ".join(query_parts)
            osearch = await search_orchestrator.search(
                query=query,
                max_results=3,
                min_similarity=0.4,
                user_name=user,
                source="mcp",
                grouped=True,
                top_k_children=1,
                project=resolved_project,
                include_promoted=True,
            )
            experiences: list[dict] = []
            for r in osearch.results:
                parent = r.get("parent", r)
                raw_summary = (
                    parent.get("solution")
                    or parent.get("description")
                    or parent.get("problem")
                    or r.get("solution")
                    or r.get("description")
                    or ""
                )
                cleaned_summary = " ".join(str(raw_summary).split())
                summary = (
                    cleaned_summary[:160] + "..."
                    if len(cleaned_summary) > 160
                    else cleaned_summary
                )
                experiences.append(
                    {
                        "id": r.get("group_id") or parent.get("id", ""),
                        "title": parent.get("title", "Untitled"),
                        "tags": parent.get("tags", []),
                        "confidence": r.get("confidence", "medium"),
                        "summary": summary,
                    }
                )
            return osearch.reranked, experiences
        except Exception:
            logger.debug("Failed to search relevant experiences", exc_info=True)
            return None, []

    profile_data, (reranked, rel_exps) = await asyncio.gather(
        _fetch_profile(),
        _fetch_experiences(),
    )
    result["profile"] = profile_data
    result["relevant_experiences"] = rel_exps
    if reranked is not None:
        result["search_reranked"] = reranked

    return result
