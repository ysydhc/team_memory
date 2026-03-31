"""Lite MCP Server — 5 aggregated tools for Cursor and other IDE clients.

Registers memory_save, memory_recall, memory_get_archive, memory_context, memory_feedback.
Internally delegates to the same service layer as the full server.py.

Usage:
    python -m team_memory.server_lite        # stdio mode
    team-memory-lite                         # via entry point
"""

from __future__ import annotations

import json
import logging
from pathlib import PurePosixPath
from typing import Any

from fastmcp import FastMCP

from team_memory.bootstrap import bootstrap, get_context
from team_memory.server import (
    _get_archive_service,
    _get_current_user,
    _get_db_url,
    _get_service,
    _get_settings,
    _guard_output,
    _resolve_project,
    _try_extract_and_save_personal_memory,
    track_usage,
)
from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory.lite")

# ============================================================
# MCP Server (Lite)
# ============================================================

mcp = FastMCP(
    "team_memory",
    instructions=(
        "team_memory gives you persistent memory across coding sessions.\n\n"
        "**3 habits:**\n"
        "1. At the start of a new task -> call `memory_context` "
        "(file_paths / task_description) and use `profile.static` + `profile.dynamic` "
        "(string lists) plus `relevant_experiences`.\n"
        "2. Before solving a hard problem -> `memory_recall` (e.g. problem=...)\n"
        "3. After you fix something or decide something -> `memory_save`\n\n"
        "**Optional:** `memory_recall(..., include_user_profile=True)` attaches the same "
        "`profile` object when you only use recall.\n\n"
        "**Tools:**\n"
        "- `memory_context`: User profile `{static, dynamic}` + relevant team knowledge\n"
        "- `memory_recall`: Search team knowledge; optional user profile; "
        "use include_archives=True to mix in archives (L0/L1 previews).\n"
        "- `memory_get_archive`: After recall returns type=archive, fetch full L2 by id.\n"
        "- `memory_save`: Save team knowledge (or parse long `content`)\n"
        "- `memory_feedback`: Rate a helpful result"
    ),
)


# ============================================================
# Tool 1: memory_save (unified write)
# ============================================================


@mcp.tool(
    name="memory_save",
    description=(
        "Save valuable knowledge: solutions, decisions, patterns, pitfalls. "
        "Call this when you: "
        "solve a bug or technical problem; "
        "make an architecture or design decision; "
        "discover a useful pattern or workaround; "
        "complete a task with lessons learned. "
        "Use scope='personal' for preferences, 'project' for team knowledge, "
        "'archive' for session summaries."
    ),
)
@track_usage
async def memory_save(
    # Direct save mode
    title: str | None = None,
    problem: str | None = None,
    solution: str | None = None,
    # Long content parse mode
    content: str | None = None,
    # Common
    tags: list[str] | None = None,
    scope: str = "project",
    experience_type: str | None = None,
    project: str | None = None,
    group_key: str | None = None,
    # Archive-specific
    overview: str | None = None,
    linked_experience_ids: list[str] | None = None,
    conversation_summary: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    archive_record_scope: str | None = None,
    archive_scope_ref: str | None = None,
) -> str:
    """Unified write: route to direct save, LLM parse, or archive based on input.

    Args:
        title: Experience title (required for direct save mode).
        problem: Problem description (required for direct save mode).
        solution: Solution description.
        content: Long text / conversation to parse via LLM (learn mode).
        tags: Tags for categorization.
        scope: "personal" | "project" (default) | "archive".
        experience_type: general/feature/bugfix/tech_design/incident/best_practice/learning.
        project: Project scope.
        group_key: Auto-group key (e.g. sprint name, feature name).
        overview: Archive overview (archive scope only).
        linked_experience_ids: Archive linked experience UUIDs (archive scope only).
        conversation_summary: 可选对话摘要（仅 archive）。
        attachments: Optional list of {kind, path, content_snapshot, git_commit, git_refs, snippet}.
        archive_record_scope: archives.scope when scope=archive (default session).
        archive_scope_ref: archives.scope_ref (e.g. plan id).
    """
    service = _get_service()
    settings = _get_settings()
    db_url = _get_db_url()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    # ---- Route: archive ----
    if scope == "archive":
        if not title or not solution:
            return json.dumps(
                {"error": True, "message": "Archive requires title and solution."},
                ensure_ascii=False,
            )
        return await _save_archive(
            title=title,
            solution_doc=solution,
            user=user,
            project=resolved_project,
            overview=overview,
            linked_experience_ids=linked_experience_ids,
            conversation_summary=conversation_summary,
            attachments=attachments,
            record_scope=archive_record_scope or "session",
            scope_ref=archive_scope_ref,
        )

    # ---- Route: content parse (tm_learn mode) ----
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

    # ---- Route: direct save (tm_save mode) ----
    if not title or not problem:
        return json.dumps(
            {
                "error": True,
                "message": (
                    "Provide either: (1) title + problem for direct save, "
                    "(2) content for LLM extraction, or "
                    "(3) scope='archive' + title + solution."
                ),
            },
            ensure_ascii=False,
        )

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

    if result.get("status") == "duplicate_detected":
        return json.dumps(
            {
                "message": (
                    f"Found {len(result['candidates'])} similar experiences. "
                    "The knowledge may already exist."
                ),
                "duplicate_detected": True,
                "candidates": result["candidates"],
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "message": "Knowledge saved successfully.",
            "experience": {
                "id": result.get("id"),
                "title": result.get("title"),
                "status": result.get("exp_status"),
            },
        },
        ensure_ascii=False,
    )


async def _save_archive(
    *,
    title: str,
    solution_doc: str,
    user: str,
    project: str,
    overview: str | None,
    linked_experience_ids: list[str] | None,
    conversation_summary: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    record_scope: str = "session",
    scope_ref: str | None = None,
) -> str:
    """Archive save helper."""
    import uuid as _uuid

    archive_svc = _get_archive_service()
    linked_uuids = []
    if linked_experience_ids:
        for s in linked_experience_ids:
            try:
                linked_uuids.append(_uuid.UUID(s))
            except (ValueError, TypeError):
                pass

    try:
        archive_id = await archive_svc.archive_save(
            title=title,
            solution_doc=solution_doc,
            created_by=user,
            project=project or None,
            scope=record_scope,
            scope_ref=scope_ref,
            overview=overview,
            conversation_summary=conversation_summary,
            linked_experience_ids=linked_uuids if linked_uuids else None,
            attachments=attachments,
        )
    except Exception as e:
        logger.exception("memory_save archive failed: %s", e)
        return json.dumps({"error": True, "message": str(e)}, ensure_ascii=False)

    return json.dumps(
        {"message": "Archive saved.", "archive_id": str(archive_id)},
        ensure_ascii=False,
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
) -> str:
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
        return json.dumps(
            {"error": True, "message": f"Failed to parse content: {e}"},
            ensure_ascii=False,
        )

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

    if result.get("status") == "duplicate_detected":
        return json.dumps(
            {
                "message": (
                    f"Found {len(result['candidates'])} similar experiences. "
                    "The knowledge may already exist."
                ),
                "duplicate_detected": True,
                "candidates": result["candidates"],
            },
            ensure_ascii=False,
        )

    await _try_extract_and_save_personal_memory(content, user, settings)
    return json.dumps(
        {
            "message": f"Knowledge extracted and saved: {parsed.get('title', 'Untitled')}",
            "experience": {
                "id": result.get("id"),
                "title": result.get("title"),
                "tags": result.get("tags", []),
                "status": result.get("exp_status"),
            },
        },
        ensure_ascii=False,
    )


async def _append_user_profile_to_recall_json(
    payload: str,
    *,
    user: str,
    context_text: str | None,
    include_user_profile: bool,
) -> str:
    if not include_user_profile:
        return payload
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return payload
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
    return json.dumps(data, ensure_ascii=False)


# ============================================================
# Tool 2: memory_recall (unified read)
# ============================================================


@mcp.tool(
    name="memory_recall",
    description=(
        "Search team knowledge base before solving problems. "
        "Call this BEFORE you: "
        "debug an error or exception; "
        "implement a feature in an unfamiliar area; "
        "make a design or architecture decision; "
        "work with code you haven't seen before. "
        "Provide 'problem' for focused solutions, 'query' for exploratory search, "
        "or just 'file_path' for context-based suggestions. "
        "With include_archives=True, results may include type=archive (previews only); "
        "call memory_get_archive(archive_id) for full L2 text."
    ),
)
@track_usage
async def memory_recall(
    query: str | None = None,
    problem: str | None = None,
    file_path: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 5,
    project: str | None = None,
    include_archives: bool = False,
    include_user_profile: bool = False,
) -> str:
    """Unified read: route to solve/search/suggest mode based on input.

    Args:
        query: Explicit search query (search mode).
        problem: Problem description (solve mode — enhanced query + use_count).
        file_path: Current file path (suggest mode — context-based).
        language: Programming language for context.
        framework: Framework for context.
        tags: Filter by tags.
        max_results: Max results to return.
        project: Project scope.
        include_archives: Include archived experiences.
        include_user_profile: If True, add profile {static, dynamic} to the JSON response.
    """
    service = _get_service()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    ctx_hint: str | None = None
    if problem:
        ctx_hint = problem
    elif query:
        ctx_hint = query
    elif file_path:
        ctx_hint = PurePosixPath(file_path).name

    # ---- Route: solve mode ----
    if problem:
        raw = await _recall_solve(
            problem=problem,
            file_path=file_path,
            language=language,
            framework=framework,
            tags=tags,
            max_results=max_results,
            service=service,
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

    # ---- Route: search mode ----
    if query:
        raw = await _recall_search(
            query=query,
            tags=tags,
            max_results=max_results,
            service=service,
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

    # ---- Route: suggest mode ----
    if file_path or language or framework:
        raw = await _recall_suggest(
            file_path=file_path,
            language=language,
            framework=framework,
            max_results=max_results,
            service=service,
            user=user,
            project=resolved_project,
        )
        return await _append_user_profile_to_recall_json(
            raw,
            user=user,
            context_text=ctx_hint,
            include_user_profile=include_user_profile,
        )

    raw = json.dumps(
        {
            "error": True,
            "message": (
                "Provide at least one of: problem, query, file_path, language, or framework."
            ),
        },
        ensure_ascii=False,
    )
    return await _append_user_profile_to_recall_json(
        raw,
        user=user,
        context_text=ctx_hint,
        include_user_profile=include_user_profile,
    )


async def _recall_solve(
    *,
    problem: str,
    file_path: str | None,
    language: str | None,
    framework: str | None,
    tags: list[str] | None,
    max_results: int,
    service: object,
    user: str,
    project: str,
    include_archives: bool,
) -> str:
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

    # service.search() already increments use_count for top results (implicit feedback)
    results = await service.search(
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

    if not results:
        return json.dumps(
            {
                "message": (
                    "No matching experiences found. "
                    "After solving this problem, call memory_save to share the solution."
                ),
                "results": [],
            },
            ensure_ascii=False,
        )

    best_id = results[0].get("group_id") or results[0].get("id", "")
    output = json.dumps(
        {
            "message": f"Found {len(results)} solution(s).",
            "results": results,
            "feedback_hint": (
                f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)"
            ),
        },
        ensure_ascii=False,
    )
    return _guard_output(output)


async def _recall_search(
    *,
    query: str,
    tags: list[str] | None,
    max_results: int,
    service: object,
    user: str,
    project: str,
    include_archives: bool,
) -> str:
    """Search mode: direct query."""
    results = await service.search(
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

    if not results:
        return json.dumps(
            {"message": "No matching experiences found.", "results": []},
            ensure_ascii=False,
        )

    best_id = results[0].get("group_id") or results[0].get("id", "")
    output = json.dumps(
        {
            "message": f"Found {len(results)} result(s).",
            "results": results,
            "feedback_hint": (
                f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)"
            ),
        },
        ensure_ascii=False,
    )
    return _guard_output(output)


async def _recall_suggest(
    *,
    file_path: str | None,
    language: str | None,
    framework: str | None,
    max_results: int,
    service: object,
    user: str,
    project: str,
) -> str:
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
        return json.dumps(
            {"message": "No context to build suggestions from.", "results": []},
            ensure_ascii=False,
        )

    query = " ".join(query_parts)
    filter_tags: list[str] = []
    if language:
        filter_tags.append(language.lower())
    if framework:
        filter_tags.append(framework.lower())

    results = await service.search(
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

    if not results:
        return json.dumps(
            {"message": "No relevant experiences for this context.", "results": []},
            ensure_ascii=False,
        )

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

    output = json.dumps(
        {
            "message": f"Found {len(suggestions)} suggestion(s) for your context.",
            "results": suggestions,
        },
        ensure_ascii=False,
    )
    return _guard_output(output)


# ============================================================
# Tool 3: memory_context (profile + relevant knowledge)
# ============================================================


@mcp.tool(
    name="memory_context",
    description=(
        "Get your profile and relevant team knowledge for current work. "
        "Call this when: "
        "starting a new task or conversation; "
        "switching to a different part of the codebase; "
        "wanting to understand team conventions for a file or module."
    ),
)
@track_usage
async def memory_context(
    file_paths: list[str] | None = None,
    task_description: str | None = None,
    project: str | None = None,
) -> str:
    """Return user profile + relevant experiences for current context.

    Args:
        file_paths: Current file paths for context-based search.
        task_description: Optional task description for search.
        project: Project scope.
    """
    user = await _get_current_user()
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

    # 1. User profile (Supermemory-shaped)
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

    # 2. Search relevant experiences based on file/task context
    query_parts: list[str] = []
    if task_description:
        query_parts.append(task_description)
    if file_paths:
        for fp in file_paths[:3]:
            p = PurePosixPath(fp)
            query_parts.append(p.name)

    if query_parts:
        try:
            service = _get_service()
            query = " ".join(query_parts)
            results = await service.search(
                query=query,
                max_results=3,
                min_similarity=0.4,
                user_name=user,
                source="mcp",
                grouped=True,
                top_k_children=1,
                project=resolved_project,
            )
            for r in results:
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

    return json.dumps(result, ensure_ascii=False)


# ============================================================
# Tool 4: memory_get_archive (L2 read, dual-phase with memory_recall)
# ============================================================


@mcp.tool(
    name="memory_get_archive",
    description=(
        "Load full archive body (L2) by id: solution_doc, overview, "
        "conversation_summary, attachments (with content_snapshot), "
        "document_tree_nodes. Use when memory_recall returned type=archive."
    ),
)
@track_usage
async def memory_get_archive(archive_id: str, project: str | None = None) -> str:
    """Return L2 JSON or 404-style error."""
    import uuid as _uuid

    try:
        aid = _uuid.UUID(archive_id.strip())
    except (ValueError, TypeError, AttributeError):
        return json.dumps(
            {"error": "archive not found", "code": 404},
            ensure_ascii=False,
        )

    user = await _get_current_user()
    resolved = _resolve_project(project)
    archive_svc = _get_archive_service()
    try:
        out = await archive_svc.get_archive(aid, viewer=user, project=resolved)
    except Exception as e:
        logger.exception("memory_get_archive failed: %s", e)
        return json.dumps(
            {"error": True, "message": str(e)},
            ensure_ascii=False,
        )
    if out is None:
        return json.dumps(
            {"error": "archive not found", "code": 404},
            ensure_ascii=False,
        )
    if "attachments" not in out or out["attachments"] is None:
        out = {**out, "attachments": []}
    if "document_tree_nodes" not in out or out["document_tree_nodes"] is None:
        out = {**out, "document_tree_nodes": []}
    return json.dumps(out, ensure_ascii=False)


# ============================================================
# Tool 5: memory_feedback (feedback loop)
# ============================================================


@mcp.tool(
    name="memory_feedback",
    description=(
        "Rate a knowledge result after using it (1=not helpful, 5=very helpful). "
        "Call this when a memory_recall result helped you solve a problem."
    ),
)
@track_usage
async def memory_feedback(
    experience_id: str,
    rating: int,
    comment: str | None = None,
) -> str:
    """Submit feedback for an experience.

    Args:
        experience_id: The ID of the experience to rate.
        rating: Rating from 1 to 5 (5 = most helpful).
        comment: Optional feedback comment.
    """
    if not (1 <= rating <= 5):
        return json.dumps(
            {"error": True, "message": "Rating must be between 1 and 5."},
            ensure_ascii=False,
        )

    service = _get_service()
    user = await _get_current_user()

    success = await service.feedback(
        experience_id=experience_id,
        rating=rating,
        feedback_by=user,
        comment=comment,
    )

    if success:
        return json.dumps({"message": "Feedback recorded. Thank you!"}, ensure_ascii=False)
    else:
        return json.dumps(
            {"error": True, "message": "Experience not found."},
            ensure_ascii=False,
        )


# ============================================================
# Entry point
# ============================================================


def main() -> None:
    """Run the Lite MCP server (stdio mode)."""
    logging.basicConfig(level=logging.INFO)
    bootstrap(enable_background=False)
    logger.info(
        "TeamMemory Lite server started with %d tools",
        len([t for t in dir(mcp) if not t.startswith("_")]),
    )
    mcp.run()


if __name__ == "__main__":
    main()
