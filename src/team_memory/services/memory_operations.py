"""MCP memory tool orchestration — business logic extracted from server.py.

Callers (MCP server) inject ``user`` and serialize/guard JSON output.
"""

from __future__ import annotations

import logging
import os
import uuid as uuid_mod
from pathlib import PurePosixPath

from pydantic import ValidationError

from team_memory.bootstrap import bootstrap, get_context  # noqa  # noqa: layer-check
from team_memory.schemas import ArchiveCreateRequest
from team_memory.services.archive import ArchiveUploadError
from team_memory.storage.database import get_session
from team_memory.utils.project import resolve_project as _resolve_project

logger = logging.getLogger("team_memory")


def _get_service():
    """Get ExperienceService from the shared AppContext (lazy-init on first call)."""
    try:
        return get_context().service
    except RuntimeError:
        return bootstrap(enable_background=False).service


def _get_archive_service():
    """Get ArchiveService from the shared AppContext (lazy-init on first call)."""
    try:
        return get_context().archive_service
    except RuntimeError:
        return bootstrap(enable_background=False).archive_service


def _get_search_orchestrator():
    """Get SearchOrchestrator from the shared AppContext (lazy-init on first call)."""
    try:
        return get_context().search_orchestrator
    except RuntimeError:
        return bootstrap(enable_background=False).search_orchestrator


def _get_settings():
    """Get Settings from the shared AppContext (lazy-init on first call)."""
    try:
        return get_context().settings
    except RuntimeError:
        return bootstrap(enable_background=False).settings


def _get_db_url() -> str:
    return get_context().db_url


async def _try_extract_and_save_personal_memory(
    conversation: str, user: str | None, settings: object
) -> None:
    """Extract personal preferences from conversation and write to personal memory.

    Only runs for logged-in non-anonymous user. Failure is logged and
    does not block the caller.
    """
    if not user or str(user).strip().lower() == "anonymous":
        logger.info("personal_memory: skipped — user is anonymous or empty")
        return
    try:
        from team_memory.services.llm_parser import parse_personal_memory
        from team_memory.services.personal_memory import PersonalMemoryService

        logger.debug(
            "personal_memory: parsing preferences (conversation_chars=%s user=%s)",
            len(conversation or ""),
            user,
        )
        items = await parse_personal_memory(conversation, llm_config=settings.llm, timeout=25.0)
        if not items:
            logger.info("personal_memory: no rows to save — LLM returned no preference items")
            return
        ctx = get_context()
        pm_svc = PersonalMemoryService(embedding_provider=ctx.embedding, db_url=ctx.db_url)
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
) -> dict:
    """LLM parse + save helper (tm_learn logic)."""
    from team_memory.services.llm_parser import (
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
            source="auto_extract",
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
            "message": (
                f"Found {len(result['candidates'])} similar experiences. "
                "The knowledge may already exist."
            ),
            "duplicate_detected": True,
            "data": {"candidates": result["candidates"]},
        }

    await _try_extract_and_save_personal_memory(content, user, settings)
    return {
        "message": f"Knowledge extracted and saved: {parsed.get('title', 'Untitled')}",
        "data": {
            "id": result.get("id"),
            "title": result.get("title"),
            "tags": result.get("tags", []),
            "status": result.get("exp_status"),
        },
    }


async def _append_user_profile_to_recall_json(
    payload: dict,
    *,
    user: str,
    context_text: str | None,
    include_user_profile: bool,
) -> dict:
    data = {**payload}
    if not include_user_profile:
        data["profile"] = None
        return data
    ctx = get_context()
    settings = _get_settings()
    max_side = getattr(settings.mcp, "profile_max_strings_per_side", None) or 20
    try:
        from team_memory.services.personal_memory import PersonalMemoryService

        pm_svc = PersonalMemoryService(embedding_provider=ctx.embedding, db_url=ctx.db_url)
        data["profile"] = await pm_svc.build_profile_for_user(
            user_id=user,
            current_context=context_text,
            max_per_side=max_side,
        )
    except Exception:
        logger.debug("recall: attach profile failed", exc_info=True)
        data["profile"] = {"static": [], "dynamic": []}
    return data


async def _recall_solve(
    *,
    problem: str,
    file_path: str | None,
    language: str | None,
    framework: str | None,
    tags: list[str] | None,
    max_results: int,
    search_orchestrator: object,
    user: str,
    project: str,
    include_archives: bool,
) -> dict:
    """Solve mode: enhanced query + implicit feedback via search."""
    query_parts = [problem]
    if language:
        query_parts.append(f"language: {language}")
    if framework:
        query_parts.append(f"framework: {framework}")
    if file_path:
        p = PurePosixPath(file_path)
        query_parts.append(f"file: {p.name}")
    enhanced_query = " | ".join(query_parts)

    combined_tags = list(tags) if tags else []
    if language and language.lower() not in combined_tags:
        combined_tags.append(language.lower())
    if framework and framework.lower() not in combined_tags:
        combined_tags.append(framework.lower())

    osearch = await search_orchestrator.search(
        query=enhanced_query,
        tags=combined_tags or None,
        max_results=max_results,
        min_similarity=0.5,
        user_name=user,
        source="mcp",
        grouped=True,
        top_k_children=2,
        project=project,
        include_archives=include_archives,
    )
    results = osearch.results

    if not results:
        return {
            "message": (
                "No matching experiences found. "
                "After solving this problem, call memory_save to share the solution."
            ),
            "results": [],
            "reranked": False,
        }

    best_id = results[0].get("group_id") or results[0].get("id", "")
    return {
        "message": f"Found {len(results)} solution(s).",
        "results": results,
        "reranked": osearch.reranked,
        "feedback_hint": (f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)"),
    }


async def _recall_search(
    *,
    query: str,
    tags: list[str] | None,
    max_results: int,
    search_orchestrator: object,
    user: str,
    project: str,
    include_archives: bool,
) -> dict:
    """Search mode: direct query."""
    osearch = await search_orchestrator.search(
        query=query,
        tags=tags,
        max_results=max_results,
        min_similarity=0.6,
        user_name=user,
        source="mcp",
        grouped=True,
        top_k_children=3,
        project=project,
        include_archives=include_archives,
    )
    results = osearch.results

    if not results:
        return {
            "message": "No matching experiences found.",
            "results": [],
            "reranked": False,
        }

    best_id = results[0].get("group_id") or results[0].get("id", "")
    return {
        "message": f"Found {len(results)} result(s).",
        "results": results,
        "reranked": osearch.reranked,
        "feedback_hint": (f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)"),
    }


async def _recall_suggest(
    *,
    file_path: str | None,
    language: str | None,
    framework: str | None,
    max_results: int,
    search_orchestrator: object,
    user: str,
    project: str,
) -> dict:
    """Suggest mode: build query from file context."""
    query_parts: list[str] = []

    if file_path:
        p = PurePosixPath(file_path)
        parts_lower = [part.lower() for part in p.parts]
        for hint in (
            "test",
            "tests",
            "migration",
            "migrations",
            "api",
            "auth",
            "config",
            "deploy",
            "docker",
            "ci",
            "scripts",
        ):
            if hint in parts_lower:
                query_parts.append(hint)
        if p.suffix:
            query_parts.append(f"file type: {p.suffix}")
        query_parts.append(p.name)

    if language:
        query_parts.append(language)
    if framework:
        query_parts.append(framework)

    if not query_parts:
        return {"message": "No context to build suggestions from.", "results": []}

    query = " ".join(query_parts)
    filter_tags: list[str] = []
    if language:
        filter_tags.append(language.lower())
    if framework:
        filter_tags.append(framework.lower())

    osearch = await search_orchestrator.search(
        query=query,
        tags=filter_tags or None,
        max_results=max_results,
        min_similarity=0.4,
        user_name=user,
        source="mcp",
        grouped=True,
        top_k_children=1,
        project=project,
    )
    results = osearch.results

    if not results:
        return {
            "message": "No relevant experiences for this context.",
            "results": [],
            "reranked": False,
        }

    suggestions = []
    for r in results:
        parent = r.get("parent", r)
        suggestions.append(
            {
                "id": r.get("group_id") or parent.get("id", ""),
                "title": parent.get("title", "Untitled"),
                "tags": parent.get("tags", []),
                "score": r.get("score", r.get("similarity", 0)),
                "confidence": r.get("confidence", "medium"),
            }
        )

    return {
        "message": f"Found {len(suggestions)} suggestion(s) for your context.",
        "results": suggestions,
        "reranked": osearch.reranked,
    }


# --- Public orchestration API (MCP tools delegate here) ---


async def op_save(
    user: str,
    *,
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
    service = _get_service()
    settings = _get_settings()
    db_url = _get_db_url()
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
                    "message": (
                        f"Tag too long (max {settings.mcp.max_tag_length} chars): {tag[:20]}..."
                    ),
                    "code": "validation_error",
                }

    if scope == "archive":
        return {
            "error": True,
            "message": (
                "scope='archive' is no longer supported. "
                "Use the /archive skill or POST /api/v1/archives instead."
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
        )

    if not title or not problem:
        return {
            "error": True,
            "message": (
                "Provide either: (1) title + problem for direct save, or "
                "(2) content for LLM extraction."
            ),
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
            "message": (
                f"Found {len(result['candidates'])} similar experiences. "
                "The knowledge may already exist."
            ),
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


async def op_recall(
    user: str,
    *,
    query: str | None = None,
    problem: str | None = None,
    file_path: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 5,
    project: str | None = None,
    include_archives: bool | None = None,
    include_user_profile: bool = False,
) -> dict:
    """Unified read: route to solve/search/suggest mode based on input."""
    if include_archives is None:
        from team_memory.config import _default_include_archives

        include_archives = _default_include_archives()

    search_orchestrator = _get_search_orchestrator()
    resolved_project = _resolve_project(project)

    ctx_hint: str | None = None
    if problem:
        ctx_hint = problem
    elif query:
        ctx_hint = query
    elif file_path:
        ctx_hint = PurePosixPath(file_path).name

    if problem:
        raw = await _recall_solve(
            problem=problem,
            file_path=file_path,
            language=language,
            framework=framework,
            tags=tags,
            max_results=max_results,
            search_orchestrator=search_orchestrator,
            user=user,
            project=resolved_project,
            include_archives=include_archives,
        )
        return await _append_user_profile_to_recall_json(
            raw,
            user=user,
            context_text=ctx_hint,
            include_user_profile=include_user_profile,
        )

    if query:
        raw = await _recall_search(
            query=query,
            tags=tags,
            max_results=max_results,
            search_orchestrator=search_orchestrator,
            user=user,
            project=resolved_project,
            include_archives=include_archives,
        )
        return await _append_user_profile_to_recall_json(
            raw,
            user=user,
            context_text=ctx_hint,
            include_user_profile=include_user_profile,
        )

    if file_path or language or framework:
        raw = await _recall_suggest(
            file_path=file_path,
            language=language,
            framework=framework,
            max_results=max_results,
            search_orchestrator=search_orchestrator,
            user=user,
            project=resolved_project,
        )
        return await _append_user_profile_to_recall_json(
            raw,
            user=user,
            context_text=ctx_hint,
            include_user_profile=include_user_profile,
        )

    raw = {
        "error": True,
        "message": ("Provide at least one of: problem, query, file_path, language, or framework."),
        "code": "validation_error",
    }
    return await _append_user_profile_to_recall_json(
        raw,
        user=user,
        context_text=ctx_hint,
        include_user_profile=include_user_profile,
    )


async def op_context(
    user: str,
    *,
    file_paths: list[str] | None = None,
    task_description: str | None = None,
    project: str | None = None,
) -> dict:
    """Return user profile + relevant experiences for current context."""
    resolved_project = _resolve_project(project)
    ctx = get_context()

    settings = _get_settings()
    max_side = getattr(settings.mcp, "profile_max_strings_per_side", None) or 20
    result: dict = {
        "user": user,
        "project": resolved_project,
        "profile": {"static": [], "dynamic": []},
        "relevant_experiences": [],
    }

    try:
        from team_memory.services.personal_memory import PersonalMemoryService

        pm_svc = PersonalMemoryService(embedding_provider=ctx.embedding, db_url=ctx.db_url)
        context_text = task_description
        if not context_text and file_paths:
            context_text = " ".join(PurePosixPath(fp).name for fp in file_paths[:5])
        result["profile"] = await pm_svc.build_profile_for_user(
            user_id=user,
            current_context=context_text,
            max_per_side=max_side,
        )
    except Exception:
        logger.debug("Failed to build user profile", exc_info=True)

    query_parts: list[str] = []
    if task_description:
        query_parts.append(task_description)
    if file_paths:
        for fp in file_paths[:3]:
            p = PurePosixPath(fp)
            query_parts.append(p.name)

    if query_parts:
        try:
            search_orch = _get_search_orchestrator()
            query = " ".join(query_parts)
            osearch = await search_orch.search(
                query=query,
                max_results=3,
                min_similarity=0.4,
                user_name=user,
                source="mcp",
                grouped=True,
                top_k_children=1,
                project=resolved_project,
            )
            result["search_reranked"] = osearch.reranked
            for r in osearch.results:
                parent = r.get("parent", r)
                result["relevant_experiences"].append(
                    {
                        "id": r.get("group_id") or parent.get("id", ""),
                        "title": parent.get("title", "Untitled"),
                        "tags": parent.get("tags", []),
                        "confidence": r.get("confidence", "medium"),
                    }
                )
        except Exception:
            logger.debug("Failed to search relevant experiences", exc_info=True)

    return result


async def op_get_archive(
    user: str,
    *,
    archive_id: str,
    project: str | None = None,
) -> dict:
    """Return L2 dict or error-shaped dict."""
    try:
        aid = uuid_mod.UUID(archive_id.strip())
    except (ValueError, TypeError, AttributeError):
        return {"error": True, "message": "Archive not found", "code": "not_found"}

    resolved = _resolve_project(project)
    archive_svc = _get_archive_service()
    try:
        out = await archive_svc.get_archive(aid, viewer=user, project=resolved)
    except Exception as e:
        logger.exception("memory_get_archive failed: %s", e)
        return {"error": True, "message": str(e), "code": "internal_error"}
    if out is None:
        return {"error": True, "message": "Archive not found", "code": "not_found"}
    if "attachments" not in out or out["attachments"] is None:
        out = {**out, "attachments": []}
    if "document_tree_nodes" not in out or out["document_tree_nodes"] is None:
        out = {**out, "document_tree_nodes": []}
    return out


async def op_archive_upsert(
    user: str,
    *,
    title: str,
    solution_doc: str,
    content_type: str = "session_archive",
    value_summary: str | None = None,
    tags: list[str] | None = None,
    overview: str | None = None,
    conversation_summary: str | None = None,
    raw_conversation: str | None = None,
    linked_experience_ids: list[str] | None = None,
    project: str | None = None,
    scope: str = "session",
    scope_ref: str | None = None,
) -> dict:
    """Upsert archive via ArchiveService; success or error dict."""
    settings = _get_settings()
    max_sd = int(getattr(settings.mcp, "max_archive_solution_doc_chars", 64_000))
    if len(solution_doc or "") > max_sd:
        return {
            "error": True,
            "message": (
                f"solution_doc exceeds MCP limit ({max_sd} chars). "
                "Shorten/summarize or upload full text as an attachment via HTTP/cli upload."
            ),
            "code": "validation_error",
        }

    try:
        body = ArchiveCreateRequest(
            title=title,
            solution_doc=solution_doc,
            content_type=content_type,
            value_summary=value_summary,
            tags=tags,
            overview=overview,
            conversation_summary=conversation_summary,
            raw_conversation=raw_conversation,
            linked_experience_ids=linked_experience_ids,
            project=project,
            scope=scope,
            scope_ref=scope_ref,
        )
    except ValidationError as e:
        return {"error": True, "message": str(e), "code": "validation_error"}

    resolved = _resolve_project(body.project)

    linked_uuids: list[uuid_mod.UUID] = []
    if body.linked_experience_ids:
        for s in body.linked_experience_ids:
            try:
                linked_uuids.append(uuid_mod.UUID(s))
            except (ValueError, TypeError):
                pass

    archive_svc = _get_archive_service()
    try:
        result = await archive_svc.archive_upsert(
            title=body.title,
            solution_doc=body.solution_doc,
            created_by=user,
            project=resolved,
            scope=body.scope,
            scope_ref=body.scope_ref,
            overview=body.overview,
            conversation_summary=body.conversation_summary,
            raw_conversation=body.raw_conversation,
            content_type=body.content_type,
            value_summary=body.value_summary,
            tags=body.tags,
            linked_experience_ids=linked_uuids if linked_uuids else None,
        )
    except ArchiveUploadError as e:
        return {"error": True, "message": e.message, "code": e.error_code}
    except Exception as e:
        logger.exception("memory_archive_upsert failed: %s", e)
        return {"error": True, "message": str(e), "code": "internal_error"}

    item = dict(result)
    archive_id = item.get("archive_id")
    if archive_id is not None and not isinstance(archive_id, str):
        archive_id = str(archive_id)
        item = {**item, "archive_id": archive_id}
    action = item.get("action", "created")
    msg = "Updated successfully" if action == "updated" else "Created successfully"
    return {
        "archive_id": archive_id,
        "action": action,
        "message": msg,
        "item": item,
    }


async def op_feedback(
    user: str,
    *,
    experience_id: str,
    rating: int,
    comment: str | None = None,
) -> dict:
    """Submit feedback for an experience."""
    if not (1 <= rating <= 5):
        return {
            "error": True,
            "message": "Rating must be between 1 and 5.",
            "code": "validation_error",
        }

    service = _get_service()

    success = await service.feedback(
        experience_id=experience_id,
        rating=rating,
        feedback_by=user,
        comment=comment,
    )

    if success:
        return {"message": "Feedback recorded. Thank you!"}
    return {
        "error": True,
        "message": "Experience not found.",
        "code": "not_found",
    }
