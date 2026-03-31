"""FastMCP Server entry point for TeamMemory.

Registers all MCP tools (tm_* namespace), resources, and prompts.
Uses the shared AppContext singleton from bootstrap.py for all services.

Tool namespace: All tools use the `tm_` prefix to help LLM clients
identify TeamMemory capabilities (e.g. tm_search, tm_save, tm_solve).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, append=False)

import functools
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml
from fastmcp import FastMCP

from team_memory import io_logger
from team_memory.bootstrap import bootstrap, get_context

try:
    from team_memory.services.context_trimmer import estimate_tokens
except ImportError:

    def estimate_tokens(text: str) -> int:
        return len(text) // 4


from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory")


def track_usage(func):
    """Decorator to auto-track MCP tool usage."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        from team_memory.services.hooks import (
            HookContext,
            HookEvent,
            get_hook_registry,
        )

        registry = get_hook_registry()
        tool_name = func.__name__
        user = "anonymous"
        try:
            user = await _get_current_user()
        except Exception:
            pass
        api_key_name = os.environ.get("TEAM_MEMORY_API_KEY_NAME") or None

        # MCP debug: log input (env TEAM_MEMORY_DEBUG=1 or TEAM_MEMORY_MCP_DEBUG=1)
        try:
            from team_memory import mcp_debug_log

            if mcp_debug_log.is_mcp_debug_enabled():
                await mcp_debug_log.log_mcp_io_async(tool_name, "in", kwargs)
        except Exception:
            pass

        # io_logger: log input when enabled (independent of mcp_debug_log)
        try:
            if io_logger.is_io_enabled():
                io_logger.log_mcp_io(tool_name, "in", kwargs)
        except Exception:
            pass

        pre_ctx = HookContext(
            event=HookEvent.PRE_TOOL_CALL,
            tool_name=tool_name,
            user=user,
            timestamp=datetime.now(timezone.utc),
            api_key_name=api_key_name,
        )
        await registry.fire(pre_ctx)

        success = True
        error_msg = None
        try:
            result = await func(*args, **kwargs)
            try:
                from team_memory import mcp_debug_log

                if mcp_debug_log.is_mcp_debug_enabled():
                    await mcp_debug_log.log_mcp_io_async(tool_name, "out", result)
            except Exception:
                pass
            try:
                if io_logger.is_io_enabled():
                    io_logger.log_mcp_io(tool_name, "out", result)
            except Exception:
                pass
            return result
        except Exception as e:
            success = False
            error_msg = str(e)
            try:
                from team_memory import mcp_debug_log

                if mcp_debug_log.is_mcp_debug_enabled():
                    await mcp_debug_log.log_mcp_io_async(tool_name, "out", {"error": error_msg})
            except Exception:
                pass
            try:
                if io_logger.is_io_enabled():
                    io_logger.log_mcp_io(tool_name, "out", {"error": error_msg})
            except Exception:
                pass
            raise
        finally:
            post_ctx = HookContext(
                event=HookEvent.POST_TOOL_CALL,
                tool_name=tool_name,
                user=user,
                timestamp=datetime.now(timezone.utc),
                metadata={"success": success, "error_message": error_msg},
                api_key_name=api_key_name,
            )
            try:
                await registry.fire(post_ctx)
            except Exception:
                pass

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


def _get_settings():
    """Get Settings from the shared AppContext (lazy-init on first call)."""
    try:
        return get_context().settings
    except RuntimeError:
        return bootstrap(enable_background=False).settings


async def _get_current_user() -> str:
    api_key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if api_key:
        try:
            ctx = get_context()
            if ctx.auth:
                user = await ctx.auth.authenticate({"api_key": api_key})
                if user is not None:
                    return user.name
            logger.warning("MCP auth resolve failed, using fallback user")
        except RuntimeError:
            logger.warning("MCP get_context failed, using fallback user")
        except Exception as e:
            logger.warning("MCP auth error: %s", type(e).__name__, exc_info=False)
    # Fallback: 项目级 mcp.json 的 env 中配置 TEAM_MEMORY_USER，写入的经验归到该用户
    return (os.environ.get("TEAM_MEMORY_USER", "") or "anonymous").strip() or "anonymous"


def _normalize_project_name(project: str | None) -> str:
    """Normalize legacy project aliases to a canonical project name."""
    if not project:
        return ""
    value = project.strip()
    alias_map = {
        "team-memory": "team_memory",
        "team_doc": "team_memory",
    }
    return alias_map.get(value, value)


def _resolve_project(project: str | None = None) -> str:
    """Resolve project from explicit param > env > settings default."""
    normalized = _normalize_project_name(project)
    if normalized:
        return normalized
    env_project = _normalize_project_name(os.environ.get("TEAM_MEMORY_PROJECT", ""))
    if env_project:
        return env_project
    ctx = get_context()
    default_project = _normalize_project_name(ctx.settings.default_project)
    return default_project or "default"


def _get_db_url() -> str:
    return get_context().db_url


async def _try_update_user_expansion_from_search(
    query: str, results: list[dict], user: str | None
) -> None:
    """No-op: UserExpansion removed in MVP simplification."""
    return


async def _try_extract_and_save_personal_memory(
    conversation: str, user: str | None, settings
) -> None:
    """Extract personal preferences from conversation and write to personal memory.

    Only runs for logged-in non-anonymous user. Timeout/failure are logged and
    do not block the caller (tm_learn experience save already done).
    """
    if not user or str(user).strip().lower() == "anonymous":
        logger.info(
            "personal_memory: skipped — user is anonymous or empty "
            "(set MCP env TEAM_MEMORY_API_KEY valid for auth, or TEAM_MEMORY_USER for fallback)"
        )
        return
    try:
        from team_memory.bootstrap import get_context
        from team_memory.services.llm_parser import parse_personal_memory
        from team_memory.services.personal_memory import PersonalMemoryService

        logger.debug(
            "personal_memory: parsing preferences (conversation_chars=%s user=%s)",
            len(conversation or ""),
            user,
        )
        items = await parse_personal_memory(conversation, llm_config=settings.llm, timeout=25.0)
        if not items:
            logger.info(
                "personal_memory: no rows to save — LLM returned no preference items "
                "(content may have no extractable habits, or llm_parser warnings above; "
                "try TEAM_MEMORY_DEBUG=1)"
            )
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
            "personal_memory: extract/save failed (experience save already succeeded): %s",
            e,
            exc_info=os.environ.get("TEAM_MEMORY_DEBUG", "0") == "1",
        )


# ============================================================
# Token Budget Guard (C3)
# ============================================================


def _guard_output(result_json: str, max_tokens: int | None = None) -> str:
    """Enforce token budget on MCP tool output.

    If the JSON output exceeds max_tokens, progressively trims:
    1. Remove low-confidence results
    2. Truncate solution / code_snippets fields
    3. Add a "truncated" flag to the response

    Args:
        result_json: JSON string to check/trim.
        max_tokens: Token budget. None = use config default.

    Returns:
        Possibly trimmed JSON string.
    """
    settings = _get_settings()
    if max_tokens is None:
        max_tokens = settings.mcp.max_output_tokens

    # Parse JSON for field-level processing
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return result_json

    results = data.get("results", [])
    if not results:
        return result_json

    truncated = False
    max_solution_chars = settings.mcp.truncate_solution_at

    # Step 1: Remove low-confidence results (if present)
    if len(results) > 1:
        high_medium = [r for r in results if r.get("confidence", "high") in ("high", "medium")]
        if high_medium and len(high_medium) < len(results):
            results = high_medium
            truncated = True

    # Step 2: Truncate long fields in each result
    for result in results:
        # Truncate in top-level fields
        for field in ("solution", "code_snippets", "description"):
            val = result.get(field)
            if isinstance(val, str) and len(val) > max_solution_chars:
                result[field] = val[:max_solution_chars] + "... [truncated]"
                truncated = True

        # Truncate in nested parent/children (grouped results)
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

    # Apply truncation flag if any changes were made
    if truncated:
        data["truncated"] = True

    # Re-serialize after field-level trimming
    data["results"] = results
    output = json.dumps(data, ensure_ascii=False)

    # Check overall token budget
    if estimate_tokens(output) <= max_tokens:
        return output

    # Step 3: If still over budget, progressively remove trailing results
    data["truncated"] = True
    while estimate_tokens(output) > max_tokens and len(results) > 1:
        results.pop()
        data["results"] = results
        output = json.dumps(data, ensure_ascii=False)

    return output


# ============================================================
# MCP Server
# ============================================================

mcp = FastMCP(
    "team_memory",
    instructions=(
        "team_memory is a team experience database. All tools use the tm_ prefix.\n\n"
        "**Mandatory**: At task start, call tm_preflight(task_description=..., current_files=...). "
        "Use search_depth (skip/light/full) to decide if tm_search/tm_solve is needed.\n\n"
        "**When to use which tool:**\n"
        "- tm_solve(problem=...): First for concrete technical problems. "
        "Returns best solution and marks used; add file_path/language for recall.\n"
        "- tm_search(query=..., tags=..., max_results=5): Exploratory or more results. "
        "Short queries get lower min_similarity automatically.\n"
        "- tm_learn(conversation=..., as_group=...): User pastes long doc or chat. "
        "LLM extracts experience; as_group=True for multi-step.\n"
        "- tm_save(title=..., problem=..., solution=...): Quick-save. "
        "Use tm_save_typed for experience_type.\n"
        "- tm_feedback(experience_id=..., rating=1..5): After a result helped; improves ranking.\n"
        "- tm_task(action=...): create/list/get/update; completed+summary → experience.\n\n"
        "Types: general, feature, bugfix, tech_design, incident, best_practice, learning. "
        "Prefer specific over 'general'. Use project= in multi-project setups."
    ),
)


# ============================================================
# Tools — C1: Workflow-oriented tools
# ============================================================


@mcp.tool(
    name="tm_solve",
    description=(
        "Smart problem solving: search the team experience database, "
        "auto-format the best solution, and mark it as used. "
        "Call this FIRST when encountering a technical problem. "
        "Optional current_file_locations (path, start_line, end_line, etc.) for location boost. "
        "Returns ~500-2000 tokens (focused on top matches)."
    ),
)
@track_usage
async def tm_solve(
    problem: str,
    file_path: str | None = None,
    file_paths: list[str] | None = None,
    language: str | None = None,
    framework: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 3,
    use_pageindex_lite: bool | None = None,
    project: str | None = None,
    current_file_locations: list[dict] | None = None,
    include_archives: bool = False,
) -> str:
    """Solve a problem by searching team experiences with enhanced context.

    Builds an enriched query from the problem description plus optional
    context (file path, language, framework). Automatically increments
    use_count on the best match.

    Args:
        problem: Description of the problem to solve (required).
        file_path: Current file path for context enrichment.
        file_paths: File paths for node boost (if file_path given and file_paths
            is None, treated as file_paths=[file_path]).
        language: Programming language for filtering.
        framework: Framework for filtering.
        tags: Optional tags to filter by.
        max_results: Max solutions to return (default 3, focused).
        current_file_locations: Optional list of dicts with path, start_line, end_line;
            optional file_content, file_mtime, file_content_hash for location_score boost.
    """
    if file_path is not None and file_paths is None:
        file_paths = [file_path]

    service = _get_service()
    db_url = _get_db_url()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    # Build enhanced query from context
    query_parts = [problem]
    if language:
        query_parts.append(f"language: {language}")
    if framework:
        query_parts.append(f"framework: {framework}")
    effective_path = file_path or (file_paths[0] if file_paths else None)
    if effective_path:
        # Extract meaningful parts from file path
        p = PurePosixPath(effective_path)
        query_parts.append(f"file: {p.name}")
    enhanced_query = " | ".join(query_parts)

    # Build combined tags from explicit tags + language/framework
    combined_tags = list(tags) if tags else []
    if language and language.lower() not in combined_tags:
        combined_tags.append(language.lower())
    if framework and framework.lower() not in combined_tags:
        combined_tags.append(framework.lower())

    io_logger.log_internal(
        "solve_query_build",
        {
            "enhanced_query": (enhanced_query or "")[:80],
            "combined_tags": combined_tags[:10],
        },
    )

    results = await service.search(
        query=enhanced_query,
        tags=combined_tags or None,
        max_results=max_results,
        min_similarity=0.5,
        user_name=user,
        source="mcp",
        grouped=True,
        top_k_children=2,
        use_pageindex_lite=use_pageindex_lite,
        project=resolved_project,
        current_file_locations=current_file_locations,
        include_archives=include_archives,
    )

    # Auto-increment use_count + quality score boost on best match (skip when top result is archive)
    if results:
        reflection_start = time.monotonic()
        best = results[0]
        best_id = best.get("group_id") or best.get("id")
        if best_id and best.get("type") != "archive":
            async with get_session(db_url) as session:
                from team_memory.storage.repository import ExperienceRepository

                repo = ExperienceRepository(session)
                try:
                    import uuid as _uuid

                    await repo.increment_use_count(_uuid.UUID(best_id))
                except Exception:
                    logger.debug("Failed to increment use_count", exc_info=True)
        reflection_duration_ms = int((time.monotonic() - reflection_start) * 1000)
        io_logger.log_internal(
            "solve_reflection",
            {
                "best_id": str(best_id) if best_id else None,
                "result_count": len(results),
            },
            duration_ms=reflection_duration_ms,
        )

    if not results:
        return json.dumps(
            {
                "message": (
                    "No matching experiences found for this problem. "
                    "After solving it, consider calling tm_learn or tm_save "
                    "to share the solution with the team."
                ),
                "results": [],
                "suggestion": "tm_save",
            },
            ensure_ascii=False,
        )

    # P0-4: Prefer summary over full content when available
    for result in results:
        parent = result.get("parent", result)
        if parent.get("summary"):
            result["_has_summary"] = True

    best_id = results[0].get("group_id") or results[0].get("id", "")
    feedback_hint = (
        f"如果此方案有帮助，请调用 tm_feedback(experience_id='{best_id}', rating=5) 进行评分"
    )

    output = json.dumps(
        {
            "message": (
                f"Found {len(results)} relevant solution(s). "
                f"Best match score: {results[0].get('score', 'N/A')}. "
                "If helpful, call tm_feedback with the experience ID."
            ),
            "results": results,
            "feedback_hint": feedback_hint,
        },
        ensure_ascii=False,
    )
    return _guard_output(output)


def _get_template_by_id(template_id: str) -> dict | None:
    """Load workflow template by id from config/templates/templates.yaml."""
    try:
        base = Path(__file__).resolve().parent.parent.parent
        path = base / "config" / "templates" / "templates.yaml"
        if not path.exists():
            return None
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for t in data.get("templates", []):
            if (t.get("id") or t.get("experience_type")) == template_id:
                return t
    except Exception:
        pass
    return None


@mcp.tool(
    name="tm_learn",
    description=(
        "Extract and save experience from long conversation or document text. "
        "Use when the user pastes a doc, chat log, or incident report. "
        "Parameters: conversation (required), as_group=True for multi-step content, "
        "tags to add, template=bugfix|feature|tech_design|... to apply workflow template, "
        "save_as_draft=True to save as draft. Returns ~200-500 tokens."
    ),
)
@track_usage
async def tm_learn(
    conversation: str,
    tags: list[str] | None = None,
    as_group: bool = False,
    save_as_draft: bool = True,
    project: str | None = None,
    template: str | None = None,
) -> str:
    """Extract and save an experience from conversation/document text.

    Uses an LLM to parse free-form text into structured fields
    (title, problem, solution, tags, etc.) then saves automatically.
    AI-extracted content defaults to draft mode for review.
    When template is provided (e.g. bugfix, feature, tech_design), applies
    that template's experience_type and suggested_tags.

    Args:
        conversation: The conversation or document text to learn from (required).
        tags: Additional tags to merge with LLM-extracted tags.
        as_group: If True, extract as parent + children experience group.
        save_as_draft: If True (default), save as draft requiring review.
        template: Optional template id to apply (experience_type + suggested_tags).
    """
    from team_memory.services.llm_parser import (
        LLMParseError,
        parse_content,
    )

    service = _get_service()
    settings = _get_settings()
    db_url = _get_db_url()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    status = "draft" if save_as_draft else "published"

    # Parse content with LLM (quality gate and retry from extraction config)
    ext = getattr(settings, "extraction", None)
    quality_min = ext.quality_gate if ext is not None else 2
    retry_once = (ext.max_retries > 0) if ext is not None else True
    try:
        parsed = await parse_content(
            content=conversation,
            llm_config=settings.llm,
            as_group=as_group,
            quality_min_score=quality_min,
            quality_retry_once=retry_once,
        )
    except LLMParseError as e:
        return json.dumps(
            {"message": f"Failed to parse content: {e}", "error": True},
            ensure_ascii=False,
        )

    # Apply template: experience_type + suggested_tags
    tpl_type: str | None = None
    tpl_tags: list[str] = []
    if template:
        tpl = _get_template_by_id(template.strip())
        if tpl:
            tpl_type = tpl.get("experience_type") or tpl.get("id")
            tpl_tags = list(tpl.get("suggested_tags") or [])

    # Merge user-provided tags (and template suggested_tags)
    if as_group:
        parent_tags = parsed["parent"].get("tags", [])
        if tpl_tags:
            parent_tags = list(set(parent_tags + [t.lower() for t in tpl_tags]))
        if tags:
            parent_tags = list(set(parent_tags + [t.lower() for t in tags]))
        parsed["parent"]["tags"] = parent_tags
        if tpl_type and not parsed["parent"].get("experience_type"):
            parsed["parent"]["experience_type"] = tpl_type

        async with get_session(db_url) as session:
            from team_memory.storage.repository import ExperienceRepository

            repo = ExperienceRepository(session)
            parent_data = {
                "title": parsed["parent"].get("title", "Untitled"),
                "description": parsed["parent"].get("problem", ""),
                "solution": parsed["parent"].get("solution"),
                "tags": parent_tags,
                "embedding": None,
                "created_by": user,
                "exp_status": status,
            }
            children_data = [
                {
                    "title": c.get("title", "Untitled"),
                    "description": c.get("problem", ""),
                    "solution": c.get("solution"),
                    "tags": c.get("tags", []),
                    "embedding": None,
                }
                for c in parsed.get("children", [])
            ]
            result = await repo.create_group(
                parent_data=parent_data,
                children_data=children_data,
                created_by=user,
                project=resolved_project,
            )
            await session.commit()

        child_count = len(parsed["children"])
        draft_note = " (已保存为草稿)" if status == "draft" else ""
        await _try_extract_and_save_personal_memory(conversation, user, settings)
        return json.dumps(
            {
                "message": (
                    f"Experience group saved: 1 parent + {child_count} children. "
                    f"Title: {parsed['parent'].get('title', 'N/A')}{draft_note}"
                ),
                "group": result,
                "status": status,
            },
            ensure_ascii=False,
        )
    else:
        extracted_tags = parsed.get("tags", [])
        if tpl_tags:
            extracted_tags = list(set(extracted_tags + [t.lower() for t in tpl_tags]))
        if tags:
            extracted_tags = list(set(extracted_tags + [t.lower() for t in tags]))
        experience_type = parsed.get("experience_type") or tpl_type or "general"

        async with get_session(db_url) as session:
            result = await service.save(
                session=session,
                title=parsed.get("title", "Untitled"),
                problem=parsed.get("problem", ""),
                solution=parsed.get("solution"),
                created_by=user,
                tags=extracted_tags,
                source="auto_extract",
                exp_status=status,
                visibility="project",
                experience_type=experience_type,
                project=resolved_project,
            )

        # Handle dedup detection
        if result.get("status") == "duplicate_detected":
            return json.dumps(
                {
                    "message": (
                        f"发现 {len(result['candidates'])} 条相似经验。"
                        "如需强制保存，请使用 tm_save 并设置 skip_dedup=true。"
                    ),
                    "duplicate_detected": True,
                    "candidates": result["candidates"],
                },
                ensure_ascii=False,
            )

        draft_note = " (已保存为草稿)" if result.get("exp_status") == "draft" else ""
        await _try_extract_and_save_personal_memory(conversation, user, settings)
        return json.dumps(
            {
                "message": (
                    f"Experience saved: {parsed.get('title', 'Untitled')}. "
                    f"ID: {result.get('id', 'N/A')}{draft_note}"
                ),
                "experience": {
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "tags": result.get("tags", []),
                    "status": result.get("exp_status"),
                    "visibility": result.get("visibility"),
                    "created_at": result.get("created_at"),
                },
            },
            ensure_ascii=False,
        )


# ============================================================
# Tools — C2: Context-based suggestions
# ============================================================


@mcp.tool(
    name="tm_suggest",
    description=(
        "Get experience recommendations based on current work context. "
        "Unlike tm_search which needs an explicit query, tm_suggest "
        "builds a query from file path, language, framework, or error message. "
        "Returns ~500-2000 tokens (lightweight format)."
    ),
)
@track_usage
async def tm_suggest(
    file_path: str | None = None,
    file_paths: list[str] | None = None,
    language: str | None = None,
    framework: str | None = None,
    error_message: str | None = None,
    max_results: int = 5,
    use_pageindex_lite: bool | None = None,
    project: str | None = None,
) -> str:
    """Suggest relevant experiences based on current working context.

    At least one context parameter should be provided.

    Args:
        file_path: Current file path (extracts directory/filename hints).
        file_paths: File paths for node boost (if file_path given and file_paths
            is None, treated as file_paths=[file_path]).
        language: Programming language being used.
        framework: Framework being used.
        error_message: Error message encountered (if any).
        max_results: Maximum suggestions to return.
    """
    if file_path is not None and file_paths is None:
        file_paths = [file_path]

    # Build context-based query
    query_parts = []

    if error_message:
        query_parts.append(error_message)

    effective_path = file_path or (file_paths[0] if file_paths else None)
    if effective_path:
        p = PurePosixPath(effective_path)
        # Extract meaningful directory hints
        parts_lower = [part.lower() for part in p.parts]
        context_hints = []
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
                context_hints.append(hint)
        if context_hints:
            query_parts.append(" ".join(context_hints))
        # Add file extension / name
        if p.suffix:
            query_parts.append(f"file type: {p.suffix}")
        query_parts.append(p.name)

    if language:
        query_parts.append(language)
    if framework:
        query_parts.append(framework)

    if not query_parts:
        return json.dumps(
            {
                "message": "No context provided. Please specify at least one of: "
                "file_path, language, framework, or error_message.",
                "results": [],
            },
            ensure_ascii=False,
        )

    query = " ".join(query_parts)

    # Build tags from language/framework
    filter_tags = []
    if language:
        filter_tags.append(language.lower())
    if framework:
        filter_tags.append(framework.lower())

    service = _get_service()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    results = await service.search(
        query=query,
        tags=filter_tags or None,
        max_results=max_results,
        min_similarity=0.4,
        user_name=user,
        source="mcp",
        grouped=True,
        top_k_children=1,
        use_pageindex_lite=use_pageindex_lite,
        project=resolved_project,
    )

    if not results:
        return json.dumps(
            {"message": "No relevant experiences found for this context.", "results": []},
            ensure_ascii=False,
        )

    # Lightweight format: title + tags + score + id only
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
                "children_count": r.get("total_children", 0),
            }
        )

    output = json.dumps(
        {
            "message": f"Found {len(suggestions)} suggestion(s) based on your context.",
            "context_query": query,
            "results": suggestions,
        },
        ensure_ascii=False,
    )
    return _guard_output(output)


# ============================================================
# Tools — Renamed existing tools (C4 namespace)
# ============================================================


@mcp.tool(
    name="tm_search",
    description=(
        "Search the team experience database for relevant solutions. "
        "Call this BEFORE starting to solve a technical problem to check "
        "if the team already has a solution. "
        "Optional current_file_locations (path, start_line, end_line, etc.) for location boost. "
        "Returns ~1000-4000 tokens depending on result count."
    ),
)
@track_usage
async def tm_search(
    query: str,
    tags: list[str] | None = None,
    file_path: str | None = None,
    file_paths: list[str] | None = None,
    max_results: int = 5,
    min_similarity: float = 0.6,
    grouped: bool = True,
    top_k_children: int = 3,
    use_pageindex_lite: bool | None = None,
    project: str | None = None,
    current_file_locations: list[dict] | None = None,
    include_archives: bool = False,
) -> str:
    """Search team experiences by semantic similarity.

    Results are sorted by relevance. Each result includes:
    - confidence: "high"/"medium"/"low" — how relevant this result is
    - reranked: bool — whether server-side reranking was applied
    - score: float — the relevance score

    If reranked=false, the client LLM should use confidence and score
    to judge result quality. Low-confidence results may be less relevant.

    Args:
        query: The search query.
        tags: Optional tags to filter by.
        file_path: Current file path for node boost (if file_paths is None).
        file_paths: File paths for node boost.
        max_results: Maximum number of results (or groups when grouped=True).
        min_similarity: Minimum similarity threshold.
        grouped: Return results grouped by parent-child. Default True.
        top_k_children: Max children per group. Default 3.
        current_file_locations: Optional list of dicts with path, start_line, end_line;
            optional file_content, file_mtime, file_content_hash for location_score boost.
    """
    if file_path is not None and file_paths is None:
        file_paths = [file_path]

    service = _get_service()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    results = await service.search(
        query=query,
        tags=tags,
        max_results=max_results,
        min_similarity=min_similarity,
        user_name=user,
        source="mcp",
        grouped=grouped,
        top_k_children=top_k_children,
        use_pageindex_lite=use_pageindex_lite,
        project=resolved_project,
        current_file_locations=current_file_locations,
        include_archives=include_archives,
    )

    if not results:
        return json.dumps(
            {"message": "No matching experiences found.", "results": []},
            ensure_ascii=False,
        )

    # Task 5: auto-update per-user tag_synonyms from query + result tags
    await _try_update_user_expansion_from_search(query, results, user)

    # Auto-increment use_count on top-1 result (mirrors tm_solve behavior); skip for archives
    best = results[0]
    best_id = best.get("group_id") or best.get("id")
    if best_id and best.get("type") != "archive":
        try:
            import uuid as _uuid

            from team_memory.storage.repository import ExperienceRepository

            db_url = _get_db_url()
            async with get_session(db_url) as session:
                repo = ExperienceRepository(session)
                await repo.increment_use_count(_uuid.UUID(best_id))
        except Exception:
            logger.debug("tm_search: failed to increment use_count", exc_info=True)

    feedback_hint = (
        f"如果搜索结果有帮助，请调用 tm_feedback(experience_id='{best_id}', rating=5) 进行评分"
        if best_id
        else ""
    )

    output = json.dumps(
        {
            "message": f"Found {len(results)} matching experience(s).",
            "results": results,
            "feedback_hint": feedback_hint,
        },
        ensure_ascii=False,
    )
    return _guard_output(output)


@mcp.tool(
    name="tm_save",
    description=(
        "Quick-save a simple experience (title + problem required, solution optional). "
        "Use this for fast knowledge capture — solution can be added later. "
        "Returns ~100-200 tokens."
    ),
)
@track_usage
async def tm_save(
    title: str,
    problem: str,
    solution: str | None = None,
    tags: list[str] | None = None,
    status: str = "published",
    visibility: str = "project",
    skip_dedup: bool = False,
    project: str | None = None,
    group_key: str | None = None,
) -> str:
    """Quick-save a new experience to the team knowledge base.

    Args:
        title: Experience title (required).
        problem: Problem description (required).
        solution: Solution description (optional).
        tags: Tags for the experience.
        status: "published" (default) or "draft".
        visibility: "project" (default), "private", or "global".
        skip_dedup: If True, skip duplicate detection check.
        project: Project scope.
        group_key: Auto-group key (experiences with same key share a parent).
    """
    service = _get_service()
    db_url = _get_db_url()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    async with get_session(db_url) as session:
        result = await service.save(
            session=session,
            title=title,
            problem=problem,
            solution=solution,
            created_by=user,
            tags=tags,
            source="auto_extract",
            exp_status=status,
            visibility=visibility,
            skip_dedup=skip_dedup,
            project=resolved_project,
            group_key=group_key,
        )

    # Handle dedup detection
    if result.get("status") == "duplicate_detected":
        return json.dumps(
            {
                "message": (
                    f"发现 {len(result['candidates'])} 条相似经验。"
                    "如需强制保存，请设置 skip_dedup=true 重新调用。"
                ),
                "duplicate_detected": True,
                "candidates": result["candidates"],
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "message": "Experience saved successfully.",
            "experience": {
                "id": result.get("id"),
                "title": result.get("title"),
                "status": result.get("exp_status"),
                "visibility": result.get("visibility"),
                "completeness_score": result.get("completeness_score"),
                "created_at": result.get("created_at"),
            },
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="tm_archive_save",
    description=(
        "Save an archive entry (session/plan-level summary doc). "
        "Required: title, solution_doc, created_by (or omit to use current user). "
        "Optional: overview, conversation_summary, scope, scope_ref, project, "
        "linked_experience_ids (list of experience UUIDs), "
        "attachments (list of dicts with kind, path?, snippet?, "
        "content_snapshot?, git_commit?, git_refs?). "
        "Returns archive_id."
    ),
)
@track_usage
async def tm_archive_save(
    title: str,
    solution_doc: str,
    created_by: str | None = None,
    project: str | None = None,
    scope: str = "session",
    scope_ref: str | None = None,
    overview: str | None = None,
    conversation_summary: str | None = None,
    linked_experience_ids: list[str] | None = None,
    attachments: list[dict] | None = None,
) -> str:
    """Save an archive to the team knowledge base."""
    archive_svc = _get_archive_service()
    user = (created_by or (await _get_current_user())).strip() or "anonymous"
    resolved_project = _resolve_project(project)

    import uuid as _uuid

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
            project=resolved_project or None,
            scope=scope,
            scope_ref=scope_ref,
            overview=overview,
            conversation_summary=conversation_summary,
            linked_experience_ids=linked_uuids if linked_uuids else None,
            attachments=attachments,
        )
    except Exception as e:
        logger.exception("tm_archive_save failed: %s", e)
        return json.dumps(
            {"error": str(e), "code": 400},
            ensure_ascii=False,
        )

    return json.dumps(
        {"message": "Archive saved successfully.", "archive_id": str(archive_id)},
        ensure_ascii=False,
    )


@mcp.tool(
    name="tm_get_archive",
    description=(
        "Get full archive by id (L2: solution_doc, overview, attachments). "
        "Returns 404 when archive does not exist or id is invalid."
    ),
)
@track_usage
async def tm_get_archive(archive_id: str) -> str:
    """Return L2 archive by id; 404 when not found."""
    import uuid as _uuid

    try:
        aid = _uuid.UUID(archive_id)
    except (ValueError, TypeError):
        return json.dumps(
            {"error": "archive not found", "code": 404},
            ensure_ascii=False,
        )

    archive_svc = _get_archive_service()
    user = await _get_current_user()
    resolved_project = _resolve_project(None)
    out = await archive_svc.get_archive(aid, viewer=user, project=resolved_project)
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


@mcp.tool(
    name="tm_save_typed",
    description=(
        "Save an experience with explicit experience_type. "
        "Types: general, feature, bugfix, tech_design, incident, best_practice, learning. "
        "Returns ~200-400 tokens."
    ),
)
@track_usage
async def tm_save_typed(
    title: str,
    problem: str,
    experience_type: str = "general",
    solution: str | None = None,
    tags: list[str] | None = None,
    status: str = "published",
    visibility: str = "project",
    skip_dedup: bool = False,
    project: str | None = None,
    group_key: str | None = None,
) -> str:
    """Save a typed experience.

    Args:
        title: Experience title (required).
        problem: Problem description (required).
        experience_type: Type — general/feature/bugfix/tech_design/incident/best_practice/learning.
        solution: Solution (optional).
        tags: Tags for the experience.
        status: "published" (default) or "draft".
        visibility: "project" (default), "private", or "global".
        skip_dedup: If True, skip duplicate detection check.
        project: Project scope.
        group_key: Auto-group key.
    """
    service = _get_service()
    db_url = _get_db_url()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    async with get_session(db_url) as session:
        result = await service.save(
            session=session,
            title=title,
            problem=problem,
            solution=solution,
            created_by=user,
            tags=tags,
            source="auto_extract",
            exp_status=status,
            visibility=visibility,
            skip_dedup=skip_dedup,
            experience_type=experience_type,
            project=resolved_project,
            group_key=group_key,
        )

    if result.get("status") == "duplicate_detected":
        return json.dumps(
            {
                "message": f"发现 {len(result['candidates'])} 条相似经验。",
                "duplicate_detected": True,
                "candidates": result["candidates"],
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "message": f"Typed experience ({experience_type}) saved successfully.",
            "experience": {
                "id": result.get("id"),
                "title": result.get("title"),
                "experience_type": result.get("experience_type"),
                "status": result.get("exp_status"),
                "visibility": result.get("visibility"),
                "created_at": result.get("created_at"),
            },
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="tm_save_group",
    description=(
        "Save a group of related experiences (parent + children). "
        "Use this when a solution involves multiple steps or stages. "
        "Returns ~200-500 tokens."
    ),
)
async def tm_save_group(
    parent_title: str,
    parent_problem: str,
    children: list[dict],
    parent_solution: str | None = None,
    parent_tags: list[str] | None = None,
    experience_type: str = "general",
    project: str | None = None,
) -> str:
    """Save a group of experiences (parent with children).

    Args:
        parent_title: Title for the parent experience.
        parent_problem: Problem description for the parent.
        children: List of dicts, each with keys: title, problem, solution, tags.
        parent_solution: Overall solution summary (optional).
        parent_tags: Tags for the parent.
        experience_type: Type for the group (default "general").
        project: Project scope.
    """
    db_url = _get_db_url()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)

    parent_data = {
        "title": parent_title,
        "description": parent_problem,
        "solution": parent_solution or "",
        "tags": parent_tags or [],
        "embedding": None,
        "exp_status": "published",
        "project": resolved_project,
        "experience_type": experience_type,
    }

    children_data = [
        {
            "title": c.get("title", ""),
            "description": c.get("problem", ""),
            "solution": c.get("solution", ""),
            "tags": c.get("tags", []),
            "embedding": None,
            "project": resolved_project,
            "experience_type": experience_type,
        }
        for c in children
    ]

    async with get_session(db_url) as session:
        from team_memory.storage.repository import ExperienceRepository

        repo = ExperienceRepository(session)
        result = await repo.create_group(
            parent_data=parent_data,
            children_data=children_data,
            created_by=user,
        )
        await session.commit()

    msg = f"Experience group ({experience_type}) saved: 1 parent + {len(children_data)} children."
    return json.dumps(
        {
            "message": msg,
            "group": result,
        },
        ensure_ascii=False,
    )


# ============================================================
# Tools — P3-2: Agent Collaboration
# ============================================================


@mcp.tool(
    name="tm_claim",
    description=(
        "Claim an experience/problem so other agents know you're working on it. "
        "Stored as a tag agent_claim|user|ISO8601|message (prior claim tag replaced)."
    ),
)
async def tm_claim(
    experience_id: str,
    message: str | None = None,
) -> str:
    """Mark that you're working on a specific experience or problem.

    Args:
        experience_id: The experience ID to claim.
        message: Optional message describing what you're doing.
    """
    import uuid as _uuid
    from datetime import datetime, timezone

    _ = _get_service()  # ensure service initialized
    db_url = _get_db_url()
    user = await _get_current_user()

    async with get_session(db_url) as session:
        from team_memory.storage.models import Experience

        exp = await session.get(Experience, _uuid.UUID(experience_id))
        if exp is None or exp.is_deleted:
            return json.dumps({"message": "Experience not found.", "error": True})
        now_iso = datetime.now(timezone.utc).isoformat()
        msg = (message or "").replace("\n", " ").strip()[:200]
        claim_tag = f"agent_claim|{user}|{now_iso}|{msg}"
        prev = list(exp.tags or [])
        kept = [t for t in prev if not str(t).startswith("agent_claim|")]
        exp.tags = kept + [claim_tag]
        await session.commit()

    return json.dumps(
        {
            "message": f"Claimed by {user}. Other agents will see this.",
            "claimed_by": user,
        }
    )


@mcp.tool(
    name="tm_notify",
    description=(
        "Notify the team that a new experience has been saved. "
        "Use after tm_save or tm_learn to signal other agents."
    ),
)
async def tm_notify(
    experience_id: str,
    message: str = "New experience available",
) -> str:
    """Notify the team about a new or updated experience.

    Args:
        experience_id: The experience ID to notify about.
        message: Human-readable notification message.
    """
    user = await _get_current_user()

    # Emit via event bus (will trigger webhook if configured)
    service = _get_service()
    if service._event_bus:
        from team_memory.services.event_bus import Events

        await service._event_bus.emit(
            Events.EXPERIENCE_UPDATED,
            {
                "experience_id": experience_id,
                "notified_by": user,
                "message": message,
            },
        )

    return json.dumps(
        {
            "message": f"Notification sent: {message}",
            "notified_by": user,
            "experience_id": experience_id,
        }
    )


@mcp.tool(
    name="tm_feedback",
    description=(
        "Rate an experience after using it (e.g. from tm_search/tm_solve). "
        "Required: experience_id (from search result), rating 1-5 (5=very helpful). "
        "Optional: comment, fitness_score 1-5. Call this when a result helped — "
        "it improves future ranking and completes the feedback loop. Returns ~50 tokens."
    ),
)
@track_usage
async def tm_feedback(
    experience_id: str,
    rating: int,
    fitness_score: int | None = None,
    comment: str | None = None,
    session: object = None,  # MCP context may inject; ignored, not passed to service
) -> str:
    """Submit feedback for an experience.

    Args:
        experience_id: The ID of the experience to rate.
        rating: Rating from 1 to 5 (5 = most helpful).
        fitness_score: Post-use fitness score 1-5 (how well it matched your need).
        comment: Optional feedback comment.
        session: Ignored (MCP context); not passed to service.

    Returns:
        JSON string with the result.
    """
    if not (1 <= rating <= 5):
        return json.dumps({"message": "Rating must be between 1 and 5.", "error": True})
    if fitness_score is not None and not (1 <= fitness_score <= 5):
        return json.dumps({"message": "fitness_score must be between 1 and 5.", "error": True})
    service = _get_service()
    user = await _get_current_user()

    success = await service.feedback(
        experience_id=experience_id,
        rating=rating,
        feedback_by=user,
        comment=comment,
        fitness_score=fitness_score,
    )

    if success:
        return json.dumps({"message": "Feedback recorded. Thank you!"})
    else:
        return json.dumps({"message": "Experience not found.", "error": True})


@mcp.tool(
    name="tm_update",
    description=(
        "Update an existing experience with additional solution details "
        "or new tags. "
        "Returns ~100-300 tokens."
    ),
)
async def tm_update(
    experience_id: str,
    solution_addendum: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update an existing experience record."""
    service = _get_service()

    result = await service.update(
        experience_id=experience_id,
        solution_addendum=solution_addendum,
        tags=tags,
    )

    if result is None:
        return json.dumps({"message": "Experience not found.", "error": True})

    return json.dumps(
        {
            "message": "Experience updated successfully.",
            "experience": result,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="tm_config",
    description=(
        "Read runtime retrieval configuration snapshot (retrieval/search/cache/pageindex-lite)."
    ),
)
async def tm_config() -> str:
    """Return current runtime configuration snapshot for diagnostics."""
    settings = _get_settings()
    return json.dumps(
        {
            "default_project": settings.default_project,
            "retrieval": {
                "max_tokens": settings.retrieval.max_tokens,
                "max_count": settings.retrieval.max_count,
                "trim_strategy": settings.retrieval.trim_strategy,
                "top_k_children": settings.retrieval.top_k_children,
                "min_avg_rating": settings.retrieval.min_avg_rating,
                "rating_weight": settings.retrieval.rating_weight,
                "summary_model": settings.retrieval.summary_model,
            },
            "search": {
                "mode": settings.search.mode,
                "rrf_k": settings.search.rrf_k,
                "vector_weight": settings.search.vector_weight,
                "fts_weight": settings.search.fts_weight,
                "location_weight": settings.search.location_weight,
                "adaptive_filter": settings.search.adaptive_filter,
                "score_gap_threshold": settings.search.score_gap_threshold,
                "min_confidence_ratio": settings.search.min_confidence_ratio,
            },
            "cache": {
                "enabled": settings.cache.enabled,
                "backend": settings.cache.backend,
                "ttl_seconds": settings.cache.ttl_seconds,
                "max_size": settings.cache.max_size,
                "embedding_cache_size": settings.cache.embedding_cache_size,
            },
            "pageindex_lite": {
                "enabled": settings.pageindex_lite.enabled,
                "only_long_docs": settings.pageindex_lite.only_long_docs,
                "min_doc_chars": settings.pageindex_lite.min_doc_chars,
                "max_tree_depth": settings.pageindex_lite.max_tree_depth,
                "max_nodes_per_doc": settings.pageindex_lite.max_nodes_per_doc,
                "max_node_chars": settings.pageindex_lite.max_node_chars,
                "tree_weight": settings.pageindex_lite.tree_weight,
                "min_node_score": settings.pageindex_lite.min_node_score,
                "include_matched_nodes": settings.pageindex_lite.include_matched_nodes,
            },
            "installable_catalog": {
                "sources": list(settings.installable_catalog.sources or []),
                "local_base_dir": settings.installable_catalog.local_base_dir,
                "registry_manifest_url": settings.installable_catalog.registry_manifest_url,
                "target_rules_dir": settings.installable_catalog.target_rules_dir,
                "target_prompts_dir": settings.installable_catalog.target_prompts_dir,
                "request_timeout_seconds": settings.installable_catalog.request_timeout_seconds,
            },
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="tm_status",
    description=(
        "Read runtime status summary for diagnostics "
        "(service/search-pipeline/cache/pageindex-lite)."
    ),
)
async def tm_status() -> str:
    """Return runtime status for troubleshooting."""
    service = _get_service()
    settings = _get_settings()
    pipeline = service._search_pipeline
    cache = pipeline._cache if pipeline is not None else None
    cache_stats = await cache.stats if cache is not None else {}

    return json.dumps(
        {
            "service_initialized": service is not None,
            "search_pipeline_enabled": pipeline is not None,
            "embedding_provider": settings.embedding.provider,
            "cache_enabled": cache_stats.get("enabled", False),
            "cache_size": cache_stats.get("result_cache_size", 0),
            "embedding_cache_size": cache_stats.get("embedding_cache_size", 0),
            "pageindex_lite_enabled": settings.pageindex_lite.enabled,
            "pageindex_lite_only_long_docs": settings.pageindex_lite.only_long_docs,
            "active_project": _resolve_project(None),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="tm_invalidate_search_cache",
    description=(
        "Clear the search result and embedding cache. "
        "Call this when config changed (e.g. query_expansion_enabled) to force fresh retrieval."
    ),
)
async def tm_invalidate_search_cache() -> str:
    """Invalidate search cache so next tm_search runs the full pipeline
    (e.g. with query expansion).
    """
    service = _get_service()
    await service.invalidate_search_cache()
    return json.dumps({"message": "Search cache cleared"}, ensure_ascii=False)


@mcp.tool(
    name="tm_skill_manage",
    description=(
        "Manage skills: list, disable, enable. "
        "action: 'list' | 'disable' | 'enable', "
        "skill_path: path to SKILL.md file (required for disable/enable)."
    ),
)
async def tm_skill_manage(
    action: str = "list",
    skill_path: str | None = None,
) -> str:
    """Manage skills: list, disable, enable."""
    from pathlib import Path

    if action == "list":
        cache_dir = Path.home() / ".team_memory" / "disabled_skills"
        cached_names = set()
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                if f.name.endswith("__SKILL.md"):
                    cached_names.add(f.name.replace("__SKILL.md", ""))
                elif f.is_dir() and f.name.count("__") >= 2:
                    parts = f.name.split("__", 2)
                    if len(parts) == 3:
                        cached_names.add(parts[2])

        skill_dirs = [
            Path.cwd() / ".cursor" / "skills",
            Path.cwd() / ".claude" / "skills",
        ]
        skills = []
        for d in skill_dirs:
            if d.exists():
                for skill_dir in sorted(d.iterdir()):
                    if skill_dir.is_dir():
                        active = (skill_dir / "SKILL.md").exists()
                        disabled = (
                            skill_dir.name in cached_names
                            or (skill_dir / "SKILL.md.disabled").exists()
                        )
                        if active or disabled:
                            skills.append(
                                {
                                    "name": skill_dir.name,
                                    "path": str(skill_dir),
                                    "active": active,
                                }
                            )
        lines = [f"{'✅' if s['active'] else '⏸️'} {s['name']}" for s in skills]
        return f"Skills ({len(skills)}):\n" + "\n".join(lines) if lines else "No skills found."

    if not skill_path:
        return "❌ skill_path required for disable/enable"

    import hashlib
    import shutil

    cache_dir = Path.home() / ".team_memory" / "disabled_skills"

    def _skill_cache_key(base_path: Path, skill_dir_name: str) -> tuple[str, str]:
        """Infer category from base_path and return (category, cache_key). Matches Web format."""
        base_resolved = str(base_path.resolve())
        home = str(Path.home())
        if ".claude" in base_resolved and "skills" in base_resolved:
            category = "user_claude_skills" if base_resolved.startswith(home) else "claude_skills"
        elif ".cursor" in base_resolved and "skills-cursor" in base_resolved:
            category = "user_cursor_skills" if base_resolved.startswith(home) else "cursor_skills"
        else:
            category = "claude_skills"
        safe = hashlib.sha256(base_resolved.encode()).hexdigest()[:12]
        return category, f"{category}__{safe}__{skill_dir_name}"

    p = Path(skill_path).resolve()
    skill_dir = p if p.is_dir() else p.parent
    if not skill_dir.is_dir():
        return f"❌ Not a directory: {skill_dir}"
    skill_dir_name = skill_dir.name
    base_path = skill_dir.parent

    if action == "disable":
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists() or not skill_md.is_file():
            return f"❌ SKILL.md not found in {skill_dir}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _, cache_key = _skill_cache_key(base_path, skill_dir_name)
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            shutil.rmtree(str(cache_path))
        shutil.copytree(str(skill_dir), str(cache_path))
        shutil.rmtree(str(skill_dir))
        return f"⏸️ Disabled: {skill_dir_name} (whole folder cached to {cache_dir})"

    if action == "enable":
        _, cache_key = _skill_cache_key(base_path, skill_dir_name)
        cache_path = cache_dir / cache_key
        if cache_path.exists() and cache_path.is_dir():
            # copytree(src, dst) requires dst to NOT exist; do not mkdir(skill_dir) first
            shutil.copytree(str(cache_path), str(skill_dir))
            shutil.rmtree(str(cache_path))
            return f"✅ Enabled: {skill_dir_name} (folder restored from cache)"
        legacy_file = cache_dir / (skill_dir_name + "__SKILL.md")
        if legacy_file.exists() and legacy_file.is_file():
            skill_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(legacy_file), str(skill_dir / "SKILL.md"))
            os.remove(str(legacy_file))
            return f"✅ Enabled: {skill_dir_name} (SKILL.md restored from legacy cache)"
        old_disabled = skill_dir / "SKILL.md.disabled"
        if old_disabled.exists():
            skill_dir.mkdir(parents=True, exist_ok=True)
            os.rename(str(old_disabled), str(skill_dir / "SKILL.md"))
            return f"✅ Enabled: {skill_dir_name} (legacy rename)"
        return f"❌ Not found in cache or disabled: {skill_dir_name}"

    return f"❌ Unknown action: {action}"


@mcp.tool(
    name="tm_analyze_patterns",
    description=(
        "Analyze user conversation patterns and extract recurring instruction styles. "
        "Identifies frequently used phrases, preferences, and instruction patterns. "
        "Optionally saves them as user_pattern experiences. Returns ~200-500 tokens."
    ),
)
@track_usage
async def tm_analyze_patterns(
    conversation_text: str | None = None,
    auto_save: bool = True,
) -> str:
    """Analyze user conversation patterns and extract recurring instruction styles.

    Identifies frequently used phrases, preferences, and instruction patterns
    from conversation text. Optionally saves them as user_pattern experiences.
    Returns extracted patterns (~200-500 tokens).
    """
    if not conversation_text:
        return "❌ conversation_text is required"

    user = await _get_current_user()
    project = _resolve_project(None)
    settings = _get_settings()

    # Use LLM to extract patterns
    from team_memory.services.llm_provider import create_llm_provider

    llm = create_llm_provider(settings.llm)

    prompt = f"""分析以下用户对话文本，提取用户反复使用的指令模式和偏好习惯。

对话文本：
{conversation_text[:8000]}

请提取：
1. 用户常用的指令短语（如"请一步步思考"、"你有什么问题可以和我进行讨论"）
2. 用户偏好的工作方式（如"先搜索再执行"、"需要多角色审视"）
3. 用户对输出格式的要求（如"给出对比方案"、"在一个回答内说清楚"）

以 JSON 格式返回：
{{
  "patterns": [
    {{
      "pattern": "用户常说的话或行为模式",
      "category": "instruction_style|workflow_preference|output_format",
      "frequency_hint": "high|medium|low",
      "suggested_rule": "建议生成的规则描述"
    }}
  ]
}}"""

    try:
        response = await llm.chat([{"role": "user", "content": prompt}])

        # Parse JSON from response
        import re

        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return f"提取完成但无法解析结果:\n{response[:500]}"

        data = json.loads(json_match.group())
        patterns = data.get("patterns", [])

        if not patterns:
            return "未发现明显的重复模式"

        saved_count = 0
        if auto_save:
            service = _get_service()
            for p in patterns:
                try:
                    await service.save(
                        title=f"用户模式: {p['pattern'][:50]}",
                        problem=p.get("suggested_rule", p["pattern"]),
                        created_by=user,
                        tags=["user-pattern", p.get("category", "general")],
                        source="auto_extract",
                        exp_status="published",
                        visibility="private",
                        experience_type="learning",
                        project=project,
                        skip_dedup=False,
                    )
                    saved_count += 1
                except Exception:
                    pass

        lines = []
        for p in patterns:
            freq = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢",
            }.get(p.get("frequency_hint"), "⚪")
            lines.append(f"{freq} [{p.get('category', '?')}] {p['pattern']}")

        result = f"发现 {len(patterns)} 个模式"
        if saved_count:
            result += f"，已保存 {saved_count} 条"
        result += ":\n" + "\n".join(lines)
        return result

    except Exception as e:
        return f"❌ 模式分析失败: {e}"


# ============================================================
# Resources
# ============================================================


@mcp.resource("experiences://recent")
async def recent_experiences() -> str:
    """Get the 10 most recently added experiences."""
    service = _get_service()
    db_url = _get_db_url()
    project = _resolve_project(None)

    async with get_session(db_url) as session:
        results = await service.get_recent(session=session, limit=10, project=project)

    if not results:
        return "No experiences in the database yet."

    lines = ["# Recent Experiences\n"]
    for exp in results:
        tags_str = ", ".join(exp.get("tags", []))
        lines.append(f"- **{exp['title']}** [{tags_str}] ({exp['created_at']})")

    return "\n".join(lines)


# ============================================================
# Prompts
# ============================================================


@mcp.prompt(
    description=(
        "Summarize the current conversation into a structured experience document. "
        "Use this after solving a problem to create a reusable knowledge entry."
    )
)
def summarize_experience() -> str:
    """Guide the Agent to summarize the current conversation as a structured experience."""
    return (
        "Please summarize the key problem-solving experience from our "
        "conversation into a structured format. Extract and organize the "
        "following fields:\n\n"
        "1. **title**: A concise title describing what was solved\n"
        "2. **problem**: A clear description of the problem encountered\n"
        "3. **root_cause**: The underlying cause of the problem\n"
        "4. **solution**: Step-by-step description of how the problem was solved\n"
        "5. **tags**: Relevant classification tags\n"
        "6. **language**: The primary programming language involved (if any)\n"
        "7. **framework**: The framework involved (if any)\n"
        "8. **code_snippets**: Key code examples that were part of the "
        "solution (if any)\n\n"
        "If the source material is long (design docs / markdown / imported URL), "
        "search first with `tm_search(..., use_pageindex_lite=true)` and preserve "
        "the matched section path information in the final solution summary.\n\n"
        "After organizing this information, call the `tm_save` tool with "
        "these fields to store it in the team knowledge base."
    )


@mcp.prompt(
    description=(
        "Submit a document (text/markdown) as a team experience. "
        "Parse it into structured fields and save to the knowledge base."
    )
)
def submit_doc_experience(document: str) -> str:
    """Guide the Agent to parse a document into a structured experience and save it."""
    return (
        "Please analyze the following document and extract a structured "
        "experience entry from it.\n\n"
        "---\n"
        f"{document}\n"
        "---\n\n"
        "Extract and organize into these fields:\n"
        "1. **title**: A concise descriptive title\n"
        "2. **problem**: The problem or challenge described\n"
        "3. **root_cause**: The root cause (if identifiable)\n"
        "4. **solution**: The solution or approach described\n"
        "5. **tags**: Relevant classification tags\n"
        "6. **language**: Programming language (if applicable)\n"
        "7. **framework**: Framework (if applicable)\n"
        "8. **code_snippets**: Key code examples (if any)\n\n"
        "For long documents, first run `tm_search` with "
        "`use_pageindex_lite=true` to verify whether similar section-level "
        "experience already exists. Reuse section paths when useful.\n\n"
        "After extracting, call the `tm_save` tool to store this in the "
        "team knowledge base."
    )


@mcp.prompt(
    description=(
        "Review an experience for quality, accuracy, and completeness. "
        "Use this to systematically check if an experience is ready for publishing."
    )
)
def review_experience(experience_id: str) -> str:
    """Guide the Agent to review an experience entry for quality."""
    return f"""Please review the experience with ID: {experience_id}

Check the following aspects:
1. **Completeness**: Does it have a clear title, problem description, and solution?
2. **Accuracy**: Is the technical information correct and up-to-date?
3. **Clarity**: Is the writing clear enough for other team members to understand?
4. **Actionability**: Can someone follow the solution to resolve the same problem?
5. **Tags**: Are the tags appropriate and sufficient for discovery?
6. **Root Cause**: Is the root cause identified (if applicable)?

After reviewing, provide:
- A quality score (1-5, where 5 is excellent)
- Specific suggestions for improvement (if any)
- Whether to approve or request changes

If the experience is good, call `tm_feedback` with a rating.
If changes are needed, call `tm_update` with improvements."""


@mcp.prompt(
    description=(
        "Systematic troubleshooting guide. "
        "Use this when encountering an error to search existing solutions "
        "and follow a structured debugging process."
    )
)
def troubleshoot(error_message: str, context: str = "") -> str:
    """Guide the Agent through systematic problem troubleshooting."""
    ctx = f"\nAdditional context: {context}" if context else ""
    return f"""I'm encountering the following error:

```
{error_message}
```
{ctx}

Please help me troubleshoot this systematically:

1. **Search existing solutions**: First, call `tm_solve` with the error
   message to check if the team has already solved this.
   If logs/docs are long, set `use_pageindex_lite=true`.

2. **Analyze the error**: If no existing solution is found:
   - Identify the error type and category
   - List possible root causes (most likely first)
   - Suggest diagnostic steps to narrow down the cause

3. **Apply solution**: Once identified:
   - Provide step-by-step fix instructions
   - Explain WHY the fix works (root cause)
   - Note any potential side effects

4. **Save the experience**: After resolving, call `tm_save` or `tm_learn` to capture:
   - The error message and context
   - The root cause
   - The solution steps
   - Relevant tags for future discovery
   - For long documents, include matched section paths from PageIndex-Lite"""


@mcp.prompt(
    description=(
        "Analyze a long document with section-level retrieval hints. "
        "Use PageIndex-Lite search strategy to avoid context overflow."
    )
)
def analyze_long_document(document_goal: str) -> str:
    """Guide the Agent to process long documents with PageIndex-Lite workflow."""
    return f"""Please analyze this long-document task: {document_goal}

Use this workflow:
1. Call `tm_config` to inspect current retrieval and PageIndex-Lite settings.
2. Call `tm_search` with `use_pageindex_lite=true` and collect matched nodes.
3. Build a concise answer using section paths + node summaries first.
4. If no relevant node is found, fallback to normal semantic results.
5. If the output is useful for future reuse, save it with `tm_save` or `tm_learn`.

Keep the response compact and reference section paths whenever available."""


# ============================================================
# Entry point
# ============================================================


def main():
    """Run the MCP server (stdio mode)."""
    logging.basicConfig(level=logging.INFO)
    bootstrap(enable_background=False)
    mcp.run()


if __name__ == "__main__":
    main()
