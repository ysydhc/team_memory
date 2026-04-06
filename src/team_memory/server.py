"""MCP Server — 5 tools for persistent team memory across coding sessions.

Registers memory_save, memory_recall, memory_get_archive, memory_context, memory_feedback.
Delegates to the shared service layer via AppContext singleton.

Usage:
    python -m team_memory.server             # stdio mode
    team-memory                              # via entry point
"""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import PurePosixPath

from fastmcp import FastMCP

from team_memory import io_logger
from team_memory.bootstrap import bootstrap, get_context
from team_memory.storage.database import get_session
from team_memory.utils.project import resolve_project as _resolve_project

try:
    from team_memory.services.context_trimmer import estimate_tokens
except ImportError:

    def estimate_tokens(text: str) -> int:
        return len(text) // 4


logger = logging.getLogger("team_memory")


# ============================================================
# Shared helpers (formerly in server.py)
# ============================================================


def track_usage(func):
    """Decorator to log MCP tool usage via io_logger."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = func.__name__

        try:
            if io_logger.is_io_enabled():
                io_logger.log_mcp_io(tool_name, "in", kwargs)
        except Exception:
            pass

        try:
            result = await func(*args, **kwargs)
            try:
                if io_logger.is_io_enabled():
                    io_logger.log_mcp_io(tool_name, "out", result)
            except Exception:
                pass
            return result
        except Exception as e:
            try:
                if io_logger.is_io_enabled():
                    io_logger.log_mcp_io(tool_name, "out", {"error": str(e)})
            except Exception:
                pass
            raise

    return wrapper


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


async def _get_current_user() -> str:
    """Resolve authenticated user: HTTP header (remote) → env API Key (stdio) → error."""
    # Step 1: HTTP MCP — read user injected by MCPAuthMiddleware
    try:
        from fastmcp.server.http import _current_http_request

        request = _current_http_request.get()
        if request is not None:
            mcp_user = getattr(request.state, "mcp_user", None)
            if mcp_user:
                return mcp_user
    except Exception:
        pass

    # Step 2: stdio MCP — authenticate via env TEAM_MEMORY_API_KEY
    api_key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if api_key:
        try:
            ctx = get_context()
            if ctx.auth:
                user = await ctx.auth.authenticate({"api_key": api_key})
                if user is not None:
                    return user.name
            logger.warning("MCP auth resolve failed for TEAM_MEMORY_API_KEY")
        except RuntimeError:
            logger.warning("MCP get_context failed, cannot resolve user")
        except Exception as e:
            logger.warning("MCP auth error: %s", type(e).__name__)
            logger.debug("MCP auth error details", exc_info=True)

    # Step 3: no identity — error
    raise RuntimeError(
        "No authenticated user. Set TEAM_MEMORY_API_KEY env var "
        "or provide Authorization header for HTTP MCP."
    )


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


def _guard_output(result_json: str, max_tokens: int | None = None) -> str:
    """Enforce token budget on MCP tool output.

    Progressively trims: low-confidence results → long fields → trailing results.
    """
    settings = _get_settings()
    if max_tokens is None:
        max_tokens = settings.mcp.max_output_tokens

    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return result_json

    results = data.get("results", [])
    if not results:
        return result_json

    truncated = False
    max_solution_chars = settings.mcp.truncate_solution_at

    # Step 1: Remove low-confidence results
    if len(results) > 1:
        high_medium = [r for r in results if r.get("confidence", "high") in ("high", "medium")]
        if high_medium and len(high_medium) < len(results):
            results = high_medium
            truncated = True

    # Step 2: Truncate long fields
    for result in results:
        for field in ("solution", "code_snippets", "description"):
            val = result.get(field)
            if isinstance(val, str) and len(val) > max_solution_chars:
                result[field] = val[:max_solution_chars] + "... [truncated]"
                truncated = True

        parent = result.get("parent")
        if isinstance(parent, dict):
            for field in ("solution", "code_snippets", "description"):
                val = parent.get(field)
                if isinstance(val, str) and len(val) > max_solution_chars:
                    parent[field] = val[:max_solution_chars] + "... [truncated]"
                    truncated = True

        children = result.get("children", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    for field in ("solution", "code_snippets", "description"):
                        val = child.get(field)
                        if isinstance(val, str) and len(val) > max_solution_chars:
                            child[field] = val[:max_solution_chars] + "... [truncated]"
                            truncated = True

    if truncated:
        data["truncated"] = True

    data["results"] = results
    output = json.dumps(data, ensure_ascii=False)

    if estimate_tokens(output) <= max_tokens:
        return output

    # Step 3: Progressively remove trailing results
    data["truncated"] = True
    while estimate_tokens(output) > max_tokens and len(results) > 1:
        results.pop()
        data["results"] = results
        output = json.dumps(data, ensure_ascii=False)

    return output


# ============================================================
# MCP Server (Lite)
# ============================================================

mcp = FastMCP(
    "team_memory",
    instructions=(
        "team_memory gives you persistent memory across coding sessions.\n\n"
        "**3 habits:**\n"
        "1. Starting a task -> `memory_context` "
        "(file_paths / task_description) and use `profile.static` + `profile.dynamic` "
        "(string lists) plus `relevant_experiences`.\n"
        "2. Hit a problem -> `memory_recall`(problem=...) before attempting a fix.\n"
        "3. `memory_save` when ANY of these happen:\n"
        "   - You fixed an unexpected error (save problem + solution)\n"
        "   - You chose between 2+ alternatives (save decision + rationale)\n"
        "   - User corrected your approach (save the correction as feedback)\n"
        "   - User provided credentials, config, or access info (save as reference)\n\n"
        "**Optional:** `memory_recall(..., include_user_profile=True)` attaches the same "
        "`profile` object when you only use recall.\n\n"
        "**Tools:**\n"
        "- `memory_context`: User profile `{static, dynamic}` + relevant team knowledge\n"
        "- `memory_recall`: Search team knowledge; optional user profile; "
        "use include_archives=True to mix in archives (L0/L1 previews).\n"
        "- `memory_get_archive`: After recall returns type=archive, fetch full L2 by id.\n"
        "- `memory_save`: Save team knowledge (or parse long `content`). "
        "scope='archive' is removed — use the /archive skill or POST /api/v1/archives.\n"
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
        "fix an unexpected error "
        "(title=error summary, problem=what failed, solution=what fixed it); "
        "choose between alternatives "
        "(title=decision, problem=context, solution=chosen option + why); "
        "receive user correction on your approach "
        "(title=feedback, problem=what you did wrong, solution=correct approach); "
        "learn access info like credentials or config "
        "(title=reference, problem=what you needed, solution=how to access it). "
        "Use scope='personal' for preferences, 'project' for team knowledge. "
        "For session archiving, use the /archive skill (not this tool)."
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
) -> str:
    """Unified write: route to direct save or LLM parse based on input.

    Args:
        title: Experience title (required for direct save mode).
        problem: Problem description (required for direct save mode).
        solution: Solution description.
        content: Long text / conversation to parse via LLM (learn mode).
        tags: Tags for categorization.
        scope: "personal" | "project" (default).
        experience_type: general/feature/bugfix/tech_design/incident/best_practice/learning.
        project: Project scope.
        group_key: Auto-group key (e.g. sprint name, feature name).
    """
    service = _get_service()
    settings = _get_settings()
    db_url = _get_db_url()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    # ---- Validate content length ----
    if content and len(content) > settings.mcp.max_content_chars:
        return json.dumps(
            {
                "error": True,
                "message": f"Content too long. Max {settings.mcp.max_content_chars} characters.",
                "code": "content_too_long",
            },
            ensure_ascii=False,
        )

    # ---- Validate tags ----
    if tags:
        if len(tags) > settings.mcp.max_tags:
            return json.dumps(
                {
                    "error": True,
                    "message": f"Too many tags. Maximum {settings.mcp.max_tags} allowed.",
                    "code": "validation_error",
                },
                ensure_ascii=False,
            )
        for tag in tags:
            if len(tag) > settings.mcp.max_tag_length:
                return json.dumps(
                    {
                        "error": True,
                        "message": (
                            f"Tag too long (max {settings.mcp.max_tag_length} chars): {tag[:20]}..."
                        ),
                        "code": "validation_error",
                    },
                    ensure_ascii=False,
                )

    # ---- scope=archive removed — use /archive skill or POST /api/v1/archives ----
    if scope == "archive":
        return json.dumps(
            {
                "error": True,
                "message": (
                    "scope='archive' is no longer supported. "
                    "Use the /archive skill or POST /api/v1/archives instead."
                ),
                "code": "scope_removed",
            },
            ensure_ascii=False,
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
                    "Provide either: (1) title + problem for direct save, or "
                    "(2) content for LLM extraction."
                ),
                "code": "validation_error",
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

    if result.get("error"):
        return json.dumps(
            {
                "error": True,
                "message": result.get("message", "Save failed."),
                "code": "internal_error",
            },
            ensure_ascii=False,
        )

    if result.get("status") == "duplicate_detected":
        return json.dumps(
            {
                "message": (
                    f"Found {len(result['candidates'])} similar experiences. "
                    "The knowledge may already exist."
                ),
                "duplicate_detected": True,
                "data": {"candidates": result["candidates"]},
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "message": "Knowledge saved successfully.",
            "data": {
                "id": result.get("id"),
                "title": result.get("title"),
                "status": result.get("exp_status"),
            },
        },
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
            {
                "error": True,
                "message": f"Failed to parse content: {e}",
                "code": "embedding_failed",
            },
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

    if result.get("error"):
        return json.dumps(
            {
                "error": True,
                "message": result.get("message", "Save failed."),
                "code": "internal_error",
            },
            ensure_ascii=False,
        )

    if result.get("status") == "duplicate_detected":
        return json.dumps(
            {
                "message": (
                    f"Found {len(result['candidates'])} similar experiences. "
                    "The knowledge may already exist."
                ),
                "duplicate_detected": True,
                "data": {"candidates": result["candidates"]},
            },
            ensure_ascii=False,
        )

    await _try_extract_and_save_personal_memory(content, user, settings)
    return json.dumps(
        {
            "message": f"Knowledge extracted and saved: {parsed.get('title', 'Untitled')}",
            "data": {
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
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return payload
        data["profile"] = None
        return json.dumps(data, ensure_ascii=False)
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
    include_archives: bool | None = None,
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
        include_archives: Include archived experiences. Defaults to True in dev, False in prod.
        include_user_profile: If True, add profile {static, dynamic} to the JSON response.
    """
    if include_archives is None:
        from team_memory.config import _default_include_archives

        include_archives = _default_include_archives()

    search_orchestrator = _get_search_orchestrator()
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

    # ---- Route: search mode ----
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

    # ---- Route: suggest mode ----
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

    raw = json.dumps(
        {
            "error": True,
            "message": (
                "Provide at least one of: problem, query, file_path, language, or framework."
            ),
            "code": "validation_error",
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
    search_orchestrator: object,
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

    # search_orchestrator.search() increments use_count for top results (implicit feedback)
    results = await search_orchestrator.search(
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
    search_orchestrator: object,
    user: str,
    project: str,
    include_archives: bool,
) -> str:
    """Search mode: direct query."""
    results = await search_orchestrator.search(
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
    search_orchestrator: object,
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

    results = await search_orchestrator.search(
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
            search_orch = _get_search_orchestrator()
            query = " ".join(query_parts)
            results = await search_orch.search(
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
            {"error": True, "message": "Archive not found", "code": "not_found"},
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
            {"error": True, "message": str(e), "code": "internal_error"},
            ensure_ascii=False,
        )
    if out is None:
        return json.dumps(
            {"error": True, "message": "Archive not found", "code": "not_found"},
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
            {
                "error": True,
                "message": "Rating must be between 1 and 5.",
                "code": "validation_error",
            },
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
            {
                "error": True,
                "message": "Experience not found.",
                "code": "not_found",
            },
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
