from __future__ import annotations

import logging
import os

from team_memory.storage.database import get_session
from team_memory.utils.project import resolve_project as _resolve_project

logger = logging.getLogger("team_memory")


async def _try_extract_and_save_personal_memory(
    conversation: str,
    user: str | None,
    settings: object,
    *,
    embedding_provider: object,
    db_url: str,
) -> None:
    """Extract personal preferences from conversation and write to personal memory.

    Only runs for logged-in non-anonymous user. Failure is logged and
    does not block the caller.
    """
    if not user or str(user).strip().lower() == "anonymous":
        logger.info("personal_memory: skipped — user is anonymous or empty")
        return
    try:
        from team_memory.services.llm.llm_parser import parse_personal_memory
        from team_memory.services.memory.personal_memory import PersonalMemoryService

        logger.debug(
            "personal_memory: parsing preferences (conversation_chars=%s user=%s)",
            len(conversation or ""),
            user,
        )
        items = await parse_personal_memory(conversation, llm_config=settings.llm, timeout=25.0)
        if not items:
            logger.info("personal_memory: no rows to save — LLM returned no preference items")
            return
        pm_svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url=db_url)
        n_static = 0
        n_dynamic = 0
        for item in items:
            pk = item.get("profile_kind")
            if pk not in ("static", "dynamic"):
                pk = "dynamic" if item.get("scope") == "context" else "static"
            if pk == "dynamic":
                n_dynamic += 1
            else:
                n_static += 1
            await pm_svc.write(
                user_id=user,
                content=item["content"],
                scope=item.get("scope") or "generic",
                context_hint=item.get("context_hint"),
                profile_kind=pk,
            )
        logger.info(
            "personal_memory: saved %d item(s) for user=%s (static=%d dynamic=%d)",
            len(items),
            user,
            n_static,
            n_dynamic,
        )
    except Exception as e:
        logger.warning(
            "personal_memory: extract/save failed: %s",
            e,
            exc_info=os.environ.get("TEAM_MEMORY_DEBUG", "0") == "1",
        )


async def _save_from_content(
    *,
    content: str,
    tags: list[str] | None,
    user: str,
    project: str,
    settings: object,
    service: object,
    db_url: str,
    embedding_provider: object,
) -> dict:
    """LLM parse + save helper (tm_learn logic)."""
    from team_memory.services.llm.llm_parser import (
        LLMParseError,
        parse_content,
    )

    ext = getattr(settings, "extraction", None)
    quality_min = ext.quality_gate if ext is not None else 2
    retry_once = (ext.max_retries > 0) if ext is not None else True

    try:
        parsed = await parse_content(
            content=content,
            llm_config=settings.llm,
            as_group=False,
            quality_min_score=quality_min,
            quality_retry_once=retry_once,
        )
    except LLMParseError as e:
        return {
            "error": True,
            "message": f"Failed to parse content: {e}",
            "code": "embedding_failed",
        }

    extracted_tags = parsed.get("tags", [])
    if tags:
        extracted_tags = list(set(extracted_tags + [t.lower() for t in tags]))

    async with get_session(db_url) as session:
        result = await service.save(
            session=session,
            title=parsed.get("title", "Untitled"),
            problem=parsed.get("problem", ""),
            solution=parsed.get("solution"),
            created_by=user,
            tags=extracted_tags,
            source="pipeline",
            exp_status="draft",
            visibility="project",
            experience_type=parsed.get("experience_type") or "general",
            project=project,
        )

    if result.get("error"):
        return {
            "error": True,
            "message": result.get("message", "Save failed."),
            "code": "internal_error",
        }

    if result.get("status") == "duplicate_detected":
        return {
            "message": (f"Found {len(result['candidates'])} similar experiences. The knowledge may already exist."),
            "duplicate_detected": True,
            "data": {"candidates": result["candidates"]},
        }

    await _try_extract_and_save_personal_memory(
        content,
        user,
        settings,
        embedding_provider=embedding_provider,
        db_url=db_url,
    )
    return {
        "message": f"Knowledge extracted and saved: {parsed.get('title', 'Untitled')}",
        "data": {
            "id": result.get("id"),
            "title": result.get("title"),
            "tags": result.get("tags", []),
            "status": result.get("exp_status"),
        },
    }


async def op_save(
    user: str,
    *,
    service: object,
    settings: object,
    db_url: str,
    embedding_provider: object,
    title: str | None = None,
    problem: str | None = None,
    solution: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    scope: str = "project",
    experience_type: str | None = None,
    project: str | None = None,
    group_key: str | None = None,
) -> dict:
    """Unified write: route to direct save or LLM parse based on input."""
    resolved_project = _resolve_project(project)

    if content and len(content) > settings.mcp.max_content_chars:
        return {
            "error": True,
            "message": f"Content too long. Max {settings.mcp.max_content_chars} characters.",
            "code": "content_too_long",
        }

    if tags:
        if len(tags) > settings.mcp.max_tags:
            return {
                "error": True,
                "message": f"Too many tags. Maximum {settings.mcp.max_tags} allowed.",
                "code": "validation_error",
            }
        for tag in tags:
            if len(tag) > settings.mcp.max_tag_length:
                return {
                    "error": True,
                    "message": (f"Tag too long (max {settings.mcp.max_tag_length} chars): {tag[:20]}..."),
                    "code": "validation_error",
                }

    if scope == "archive":
        return {
            "error": True,
            "message": (
                "scope='archive' is no longer supported. Use the /archive skill or POST /api/v1/archives instead."
            ),
            "code": "scope_removed",
        }

    if content:
        return await _save_from_content(
            content=content,
            tags=tags,
            user=user,
            project=resolved_project,
            settings=settings,
            service=service,
            db_url=db_url,
            embedding_provider=embedding_provider,
        )

    if not title or not problem:
        return {
            "error": True,
            "message": ("Provide either: (1) title + problem for direct save, or (2) content for LLM extraction."),
            "code": "validation_error",
        }

    async with get_session(db_url) as session:
        result = await service.save(
            session=session,
            title=title,
            problem=problem,
            solution=solution,
            created_by=user,
            tags=tags,
            source="auto_extract",
            exp_status="published",
            visibility="project" if scope == "project" else "private",
            experience_type=experience_type or "general",
            project=resolved_project,
            group_key=group_key,
        )

    if result.get("error"):
        return {
            "error": True,
            "message": result.get("message", "Save failed."),
            "code": "internal_error",
        }

    if result.get("status") == "duplicate_detected":
        return {
            "message": (f"Found {len(result['candidates'])} similar experiences. The knowledge may already exist."),
            "duplicate_detected": True,
            "data": {"candidates": result["candidates"]},
        }

    return {
        "message": "Knowledge saved successfully.",
        "data": {
            "id": result.get("id"),
            "title": result.get("title"),
            "status": result.get("exp_status"),
        },
    }


async def op_draft_save(
    user: str,
    *,
    service: object,
    db_url: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    project: str | None = None,
    group_key: str | None = None,
    conversation_id: str | None = None,
    skip_dedup: bool = False,
) -> dict:
    """Pipeline write: save draft. Forces source='pipeline', exp_status='draft'."""
    resolved_project = _resolve_project(project)

    async with get_session(db_url) as session:
        result = await service.save(
            session=session,
            title=title,
            problem=content,
            tags=tags,
            project=resolved_project,
            source="pipeline",
            exp_status="draft",
            visibility="project",
            experience_type="general",
            group_key=group_key,
            created_by=user,
            skip_dedup=skip_dedup,
        )

    if result.get("error"):
        return {
            "error": True,
            "message": result.get("message", "Save failed."),
            "code": "internal_error",
            "id": result.get("id"),
        }

    if result.get("status") == "duplicate_detected":
        return {
            "error": True,
            "message": result.get("message", "Duplicate detected"),
            "code": "duplicate_detected",
            "candidates": result.get("candidates"),
        }

    return {
        "id": result.get("id"),
        "status": "draft",
    }


async def op_draft_publish(
    user: str,
    *,
    service: object,
    draft_id: str,
    refined_content: str | None = None,
) -> dict:
    """Pipeline write: promote draft to published. Only works on pipeline drafts."""

    exp = await service.get_by_id(draft_id)
    if not exp:
        return {
            "error": True,
            "message": f"Experience {draft_id} not found",
            "code": "not_found",
        }
    if exp.get("source") not in ("pipeline", "obsidian"):
        return {
            "error": True,
            "message": f"Can only publish pipeline/obsidian drafts, got source='{exp.get('source')}'",
            "code": "validation_error",
        }
    if exp.get("status") != "draft":
        return {
            "error": True,
            "message": f"Can only publish drafts, got status='{exp.get('status')}'",
            "code": "validation_error",
        }

    update_kwargs = {"experience_id": draft_id, "exp_status": "published", "user": user}
    if refined_content:
        update_kwargs["description"] = refined_content

    result = await service.update(**update_kwargs)

    if result is None:
        return {
            "error": True,
            "message": f"Failed to update experience {draft_id}",
            "code": "internal_error",
        }

    return {
        "id": result.get("id"),
        "status": "published",
    }


async def op_experience_update(
    user: str,
    *,
    service: object,
    experience_id: str,
    title: str | None = None,
    problem: str | None = None,
    solution: str | None = None,
    tags: list[str] | None = None,
    experience_type: str | None = None,
    exp_status: str | None = None,
) -> dict:
    """Update an existing experience in-place. Only provided fields are modified.

    Args:
        user: The user performing the update.
        experience_id: UUID of the experience to update.
        title: New title (optional).
        problem: New problem/description (optional).
        solution: New solution (optional).
        tags: New tags list (optional).
        experience_type: New type (optional).
        exp_status: New status (optional).

    Returns:
        Dict with updated experience data or error info.
    """

    kwargs: dict = {"experience_id": experience_id, "user": user}
    if title is not None:
        kwargs["title"] = title
    if problem is not None:
        kwargs["description"] = problem
    if solution is not None:
        kwargs["solution"] = solution
    if tags is not None:
        kwargs["tags"] = tags
    if experience_type is not None:
        kwargs["experience_type"] = experience_type
    if exp_status is not None:
        kwargs["exp_status"] = exp_status

    result = await service.update(**kwargs)

    if result is None:
        return {
            "error": True,
            "message": f"Experience {experience_id} not found",
            "code": "not_found",
        }

    return {
        "id": result.get("id"),
        "title": result.get("title"),
        "status": result.get("exp_status"),
        "message": "Experience updated successfully.",
    }
