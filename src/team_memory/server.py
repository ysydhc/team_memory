"""MCP Server — 8 tools for persistent team memory across coding sessions.

Registers memory_save, memory_recall, memory_get_archive, memory_archive_upsert,
memory_context, memory_feedback, memory_draft_save, memory_draft_publish,
memory_submit_response.
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
from types import SimpleNamespace
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from team_memory import io_logger
from team_memory.bootstrap import bootstrap
from team_memory.bootstrap import get_context as _app_context
from team_memory.config.mcp import MCPConfig
from team_memory.services.memory import memory_operations

if TYPE_CHECKING:
    from team_memory.remote_client import RemoteTMClient

try:
    from team_memory.services.context_trimmer import estimate_tokens
except ImportError:

    def estimate_tokens(text: str) -> int:
        return len(text) // 4


logger = logging.getLogger("team_memory")


# ============================================================
# Remote mode — 模块级远程客户端 / module-level remote client
# ============================================================

_remote_client: "RemoteTMClient | None" = None


def _init_mode() -> None:
    """初始化运行模式: remote 或 local / Initialize running mode: remote or local.

    检测逻辑 / Detection logic:
      1. TM_SERVER_URL 环境变量 / env var
      2. ClientConfig.server.url (config.toml)
      3. 非 localhost URL → remote 模式 / non-localhost → remote mode
      4. TM_REMOTE_MODE=1 强制 remote / force remote mode
    """
    global _remote_client

    from team_memory.remote_client import RemoteTMClient, detect_remote_mode

    is_remote, base_url, api_key = detect_remote_mode()

    if is_remote:
        _remote_client = RemoteTMClient(base_url=base_url, api_key=api_key)
        logger.info(
            "[INIT] Remote mode — MCP ops will delegate to %s / 远程模式",
            base_url,
        )
    else:
        _remote_client = None
        logger.info("[INIT] Local mode — MCP ops use local DB / 本地模式")


# ============================================================
# MCP-specific helpers
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
            ctx = _app_context()
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
        "No authenticated user. Set TEAM_MEMORY_API_KEY env var or provide Authorization header for HTTP MCP."
    )


def _guard_output(
    result_json: str,
    *,
    settings: object | None = None,
    max_tokens: int | None = None,
) -> str:
    """Enforce token budget on MCP tool output.

    Progressively trims: low-confidence results → long fields → trailing results.
    """
    effective = settings if settings is not None else SimpleNamespace(mcp=MCPConfig())
    if max_tokens is None:
        max_tokens = effective.mcp.max_output_tokens

    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return result_json

    results = data.get("results", [])
    if not results:
        return result_json

    truncated = False
    max_solution_chars = effective.mcp.truncate_solution_at

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

    if not truncated and estimate_tokens(result_json) <= max_tokens:
        return result_json

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

if not hasattr(mcp, "get_tools") and hasattr(mcp, "list_tools"):

    async def _compat_get_tools() -> dict:
        return {t.name: t for t in await mcp.list_tools()}

    mcp.get_tools = _compat_get_tools  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Auto-inferred project name (zero-config)
# ---------------------------------------------------------------------------
from team_memory.project_resolver import (  # noqa: E402
    infer_project_name,
    infer_project_with_aliases,
)

_inferred_project: str = infer_project_name()
_inferred_aliases: list[str] = infer_project_with_aliases().get("aliases", [])
logger.info(
    "[INIT] Auto-inferred project: %s (aliases: %s)",
    _inferred_project,
    _inferred_aliases or "none",
)

# Register aliases so recall/search can resolve historical names
if _inferred_aliases:
    from team_memory.project_resolver import register_aliases

    register_aliases(_inferred_project, _inferred_aliases)


def _resolve_project(project: str | None) -> str | None:
    """Use inferred project when caller does not specify one.

    When a project is given, normalize it through the alias registry
    so that historical names (e.g. ``team_doc``) resolve to the canonical
    name (e.g. ``team_memory``).
    """
    if project:
        try:
            from team_memory.utils.project import normalize_project_name

            normalized = normalize_project_name(project)
            if normalized:
                return normalized
        except Exception:
            pass
        return project
    return _inferred_project or None


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
    title: str | None = None,
    problem: str | None = None,
    solution: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    scope: str = "project",
    experience_type: str | None = None,
    project: str | None = None,
    group_key: str | None = None,
) -> str:
    """Unified write: route to direct save or LLM parse based on input."""
    user = await _get_current_user()
    resolved_project = _resolve_project(project)
    truncate_settings = None
    if _remote_client:
        result = await _remote_client.op_save(
            user,
            title=title,
            problem=problem,
            solution=solution,
            content=content,
            tags=tags,
            scope=scope,
            experience_type=experience_type,
            project=resolved_project,
            group_key=group_key,
        )
    else:
        ctx = _app_context()
        truncate_settings = ctx.settings
        result = await memory_operations.op_save(
            user,
            service=ctx.service,
            settings=ctx.settings,
            db_url=ctx.db_url,
            embedding_provider=ctx.embedding,
            title=title,
            problem=problem,
            solution=solution,
            content=content,
            tags=tags,
            scope=scope,
            experience_type=experience_type,
            project=resolved_project,
            group_key=group_key,
        )
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=truncate_settings)


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
    max_results: int = 10,
    project: str | None = None,
    include_archives: bool | None = None,
    include_user_profile: bool = True,
    adaptive: bool = True,
) -> str:
    """Unified read: route to solve/search/suggest mode based on input."""
    user = await _get_current_user()
    resolved_project = _resolve_project(project)
    truncate_settings = None
    if _remote_client:
        result = await _remote_client.op_recall(
            user,
            query=query,
            problem=problem,
            file_path=file_path,
            language=language,
            framework=framework,
            tags=tags,
            max_results=max_results,
            project=resolved_project,
            include_archives=include_archives,
            include_user_profile=include_user_profile,
        )
    else:
        ctx = _app_context()
        truncate_settings = ctx.settings
        result = await memory_operations.op_recall(
            user,
            search_orchestrator=ctx.search_orchestrator,
            db_url=ctx.db_url,
            settings=ctx.settings,
            embedding_provider=ctx.embedding,
            query=query,
            problem=problem,
            file_path=file_path,
            language=language,
            framework=framework,
            tags=tags,
            max_results=max_results,
            project=resolved_project,
            include_archives=include_archives,
            include_user_profile=include_user_profile,
            adaptive=adaptive,
            adaptive_retriever=ctx.adaptive_retriever,
        )
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=truncate_settings)


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
    """Return user profile + relevant experiences for current context."""
    user = await _get_current_user()
    resolved_project = _resolve_project(project)
    truncate_settings = None
    if _remote_client:
        result = await _remote_client.op_context(
            user,
            file_paths=file_paths,
            task_description=task_description,
            project=resolved_project,
        )
    else:
        ctx = _app_context()
        truncate_settings = ctx.settings
        result = await memory_operations.op_context(
            user,
            search_orchestrator=ctx.search_orchestrator,
            settings=ctx.settings,
            embedding_provider=ctx.embedding,
            db_url=ctx.db_url,
            file_paths=file_paths,
            task_description=task_description,
            project=resolved_project,
        )
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=truncate_settings)


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
    user = await _get_current_user()
    ctx = _app_context()
    result = await memory_operations.op_get_archive(
        user,
        archive_service=ctx.archive_service,
        archive_id=archive_id,
        project=_resolve_project(project),
    )
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=ctx.settings)


# ============================================================
# Tool 5: memory_archive_upsert (L2 write, same as POST /api/v1/archives)
# ============================================================


@mcp.tool(
    name="memory_archive_upsert",
    description=(
        "Create or update an archive (title + project dedup), same contract as "
        "POST /api/v1/archives. Use for L0/L1/L2 text fields only; large files via HTTP "
        "POST /api/v1/archives/{archive_id}/attachments/upload or "
        "python -m team_memory.cli upload (after you have archive_id). Does not embed file bytes."
    ),
)
@track_usage
async def memory_archive_upsert(
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
) -> str:
    """Upsert archive via ArchiveService (embedding + DB); JSON success or error."""
    user = await _get_current_user()
    ctx = _app_context()
    result = await memory_operations.op_archive_upsert(
        user,
        archive_service=ctx.archive_service,
        settings=ctx.settings,
        title=title,
        solution_doc=solution_doc,
        content_type=content_type,
        value_summary=value_summary,
        tags=tags,
        overview=overview,
        conversation_summary=conversation_summary,
        raw_conversation=raw_conversation,
        linked_experience_ids=linked_experience_ids,
        project=_resolve_project(project),
        scope=scope,
        scope_ref=scope_ref,
    )
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=ctx.settings)


# ============================================================
# Tool 6: memory_feedback (feedback loop)
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
    """Submit feedback for an experience."""
    user = await _get_current_user()
    ctx = _app_context()
    result = await memory_operations.op_feedback(
        user,
        service=ctx.service,
        experience_id=experience_id,
        rating=rating,
        comment=comment,
    )
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=ctx.settings)


# ============================================================
# Tool 7: memory_draft_save (pipeline write)
# ============================================================


@mcp.tool(
    name="memory_draft_save",
    description=("Pipeline-only: save a draft memory. source is always 'pipeline', exp_status is always 'draft'."),
)
@track_usage
async def memory_draft_save(
    title: str,
    content: str,
    tags: list[str] | None = None,
    project: str | None = None,
    group_key: str | None = None,
    conversation_id: str | None = None,
) -> str:
    """Pipeline write: save a draft memory with forced source='pipeline', exp_status='draft'."""
    user = await _get_current_user()
    ctx = _app_context()
    result = await memory_operations.op_draft_save(
        user,
        service=ctx.service,
        db_url=ctx.db_url,
        title=title,
        content=content,
        tags=tags,
        project=_resolve_project(project),
        group_key=group_key,
        conversation_id=conversation_id,
    )
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=ctx.settings)


# ============================================================
# Tool 8: memory_draft_publish (pipeline write)
# ============================================================


@mcp.tool(
    name="memory_draft_publish",
    description=(
        "Pipeline-only: promote a draft to published. "
        "Only works on experiences with source='pipeline' and exp_status='draft'."
    ),
)
@track_usage
async def memory_draft_publish(
    draft_id: str,
    refined_content: str | None = None,
) -> str:
    """Pipeline write: promote a pipeline draft to published."""
    user = await _get_current_user()
    ctx = _app_context()
    result = await memory_operations.op_draft_publish(
        user,
        service=ctx.service,
        draft_id=draft_id,
        refined_content=refined_content,
    )
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=ctx.settings)


# ============================================================
# Tool 9: memory_submit_response (faithfulness buffer)
# ============================================================


@mcp.tool(
    name="memory_submit_response",
    description=(
        "Submit your response text after using memory_recall results. "
        "This enables faithfulness evaluation — measuring whether your response "
        "was based on the retrieved knowledge. "
        "Call this AFTER you've used memory_recall results to answer a question."
    ),
)
@track_usage
async def memory_submit_response(
    query: str,
    response: str,
    result_ids: list[str] | None = None,
    project: str | None = None,
) -> str:
    """Submit agent response for faithfulness evaluation.

    Args:
        query: The original search query used in memory_recall
        response: Your response text that used the search results
        result_ids: Optional list of experience IDs from the search results
        project: Optional project scope (resolved like other memory_* tools)
    """
    user = await _get_current_user()
    resolved_project = _resolve_project(project)
    truncate_settings = None
    if _remote_client:
        result = await _remote_client.op_submit_response(
            user,
            query=query,
            response=response,
            result_ids=result_ids,
            project=resolved_project,
        )
    else:
        ctx = _app_context()
        truncate_settings = ctx.settings
        result = await memory_operations.op_submit_response(
            user,
            db_url=ctx.db_url,
            query=query,
            response=response,
            result_ids=result_ids,
            project=resolved_project,
        )

    if not isinstance(result, dict):
        result = {"error": True, "code": "internal_error", "message": str(result)}
    elif result.get("status") == "buffered" and result.get("buffer_id"):
        result = {
            **result,
            "message": "Response submitted for faithfulness evaluation.",
        }
    return _guard_output(json.dumps(result, ensure_ascii=False), settings=truncate_settings)


# ============================================================
# Entry point
# ============================================================


def main() -> None:
    """Run the Lite MCP server (stdio mode).

    Two operating modes, selected by environment variables:

    LOCAL mode (default):
        Bootstraps a local AppContext, connects directly to PostgreSQL.
        Requires DB_URL, TEAM_MEMORY_API_KEY, and Ollama/embedding service.

    REMOTE mode (set TEAM_MEMORY_REMOTE_URL):
        No local DB or AppContext. All op_* calls are forwarded via HTTP to
        a remote team_memory_service instance (Docker / server).
        Requires TEAM_MEMORY_REMOTE_URL and TEAM_MEMORY_API_KEY.
        Example:
            export TEAM_MEMORY_REMOTE_URL=http://your-server:9111
            export TEAM_MEMORY_API_KEY=your-key
            python -m team_memory.server
    """
    logging.basicConfig(level=logging.INFO)

    remote_url = os.environ.get("TEAM_MEMORY_REMOTE_URL", "").strip()
    if remote_url:
        # Remote mode: patch memory_operations with HTTP client, skip bootstrap
        from team_memory.remote_client import setup_remote_ops

        api_key = os.environ.get("TEAM_MEMORY_API_KEY", "")
        setup_remote_ops(base_url=remote_url, api_key=api_key)
    else:
        # Local mode: bootstrap AppContext with direct DB connection
        bootstrap(enable_background=False)

    logger.info(
        "TeamMemory MCP server started (mode=%s)",
        "remote" if remote_url else "local",
    )
    mcp.run()


if __name__ == "__main__":
    main()
