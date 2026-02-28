"""FastMCP Server entry point for TeamMemory.

Registers all MCP tools (tm_* namespace), resources, and prompts.
Uses the shared AppContext singleton from bootstrap.py for all services.

Tool namespace: All tools use the `tm_` prefix to help LLM clients
identify TeamMemory capabilities (e.g. tm_search, tm_save, tm_solve).
"""

from __future__ import annotations

import functools
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import PurePosixPath

from fastmcp import FastMCP

from team_memory.bootstrap import bootstrap, get_context
from team_memory.services.context_trimmer import estimate_tokens
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
            user = _get_current_user()
        except Exception:
            pass

        pre_ctx = HookContext(
            event=HookEvent.PRE_TOOL_CALL,
            tool_name=tool_name,
            user=user,
            timestamp=datetime.now(timezone.utc),
        )
        await registry.fire(pre_ctx)

        success = True
        error_msg = None
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            post_ctx = HookContext(
                event=HookEvent.POST_TOOL_CALL,
                tool_name=tool_name,
                user=user,
                timestamp=datetime.now(timezone.utc),
                metadata={"success": success, "error_message": error_msg},
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


def _get_settings():
    """Get Settings from the shared AppContext (lazy-init on first call)."""
    try:
        return get_context().settings
    except RuntimeError:
        return bootstrap(enable_background=False).settings


def _get_current_user() -> str:
    return os.environ.get("TEAM_MEMORY_USER", "anonymous")


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


async def _create_reflection(
    session, task, summary: str, sediment_experience_id: str | None, settings
) -> str | None:
    """P2.C: Create a reflection record after task completion.

    Generates a self-assessment using LLM, then stores it.
    Returns the reflection ID or None if generation fails.
    """
    import uuid as _uuid

    import httpx

    from team_memory.storage.models import ExperienceReflection

    prompt = f"""你是一个技术复盘助手。任务刚刚完成，请基于以下信息进行简要复盘。

任务标题: {task.title}
完成摘要: {summary[:2000]}

请严格以 JSON 格式返回（不要用 markdown 代码块包裹）:
{{
  "success_points": "做得好的地方（1-3点）",
  "failure_points": "遇到的问题或不足（1-3点，没有则为 null）",
  "improvements": "下次可以改进的地方（1-3点）",
  "generalized_strategy": "从这次任务中提炼的通用策略（1-2句话）"
}}"""

    llm_model = settings.llm.model
    llm_base_url = settings.llm.base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{llm_base_url}/api/chat",
                json={
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(raw)
    except Exception:
        logger.warning("Reflection LLM call failed, creating minimal reflection")
        parsed = {
            "success_points": "任务已完成",
            "failure_points": None,
            "improvements": None,
            "generalized_strategy": None,
        }

    exp_uuid = _uuid.UUID(sediment_experience_id) if sediment_experience_id else None
    reflection = ExperienceReflection(
        task_id=task.id,
        experience_id=exp_uuid,
        success_points=parsed.get("success_points"),
        failure_points=parsed.get("failure_points"),
        improvements=parsed.get("improvements"),
        generalized_strategy=parsed.get("generalized_strategy"),
    )
    session.add(reflection)
    await session.flush()
    return str(reflection.id)


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
        high_medium = [
            r for r in results
            if r.get("confidence", "high") in ("high", "medium")
        ]
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

        # Strip code_snippets entirely if config says so
        if not settings.mcp.include_code_snippets:
            result.pop("code_snippets", None)
            if isinstance(parent, dict):
                parent.pop("code_snippets", None)
            for child in children:
                if isinstance(child, dict):
                    child.pop("code_snippets", None)
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
        "team_memory is a team experience database. All tools use the tm_ prefix.\n"
        "Available tools:\n"
        "- tm_solve: Smart problem solving — search + format best solution + mark used\n"
        "- tm_search: General-purpose semantic search across all experiences\n"
        "- tm_suggest: Context-based recommendations from file/language/framework\n"
        "- tm_learn: Extract and save experience from conversation/document text\n"
        "- tm_save: Quick-save a simple experience (title + problem, solution optional)\n"
        "- tm_save_typed: Save a typed experience with full fields (type, severity, git_refs...)\n"
        "- tm_save_group: Save a parent + children experience group\n"
        "- tm_feedback: Rate an experience (1-5) to improve future search\n"
        "- tm_update: Update fields on an existing experience\n"
        "- tm_analyze_patterns: Analyze conversation patterns, extract user instruction styles\n\n"
        "- tm_config: Read current retrieval/runtime configuration snapshot\n"
        "- tm_status: Read runtime health and pipeline status\n\n"
        "Experience types: general, feature, bugfix, tech_design, "
        "incident, best_practice, learning\n"
        "Recommended workflow:\n"
        "1. Before solving a problem: call tm_solve or tm_search\n"
        "2. After solving: call tm_learn or tm_save / tm_save_typed to share knowledge\n"
        "3. If a solution helped: call tm_feedback with rating"
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
        "Returns ~500-2000 tokens (focused on top matches)."
    ),
)
@track_usage
async def tm_solve(
    problem: str,
    file_path: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 3,
    use_pageindex_lite: bool | None = None,
    project: str | None = None,
) -> str:
    """Solve a problem by searching team experiences with enhanced context.

    Builds an enriched query from the problem description plus optional
    context (file path, language, framework). Automatically increments
    use_count on the best match.

    Args:
        problem: Description of the problem to solve (required).
        file_path: Current file path for context enrichment.
        language: Programming language for filtering.
        framework: Framework for filtering.
        tags: Optional tags to filter by.
        max_results: Max solutions to return (default 3, focused).
    """
    service = _get_service()
    db_url = _get_db_url()
    user = _get_current_user()
    resolved_project = _resolve_project(project)

    # Build enhanced query from context
    query_parts = [problem]
    if language:
        query_parts.append(f"language: {language}")
    if framework:
        query_parts.append(f"framework: {framework}")
    if file_path:
        # Extract meaningful parts from file path
        p = PurePosixPath(file_path)
        query_parts.append(f"file: {p.name}")
    enhanced_query = " | ".join(query_parts)

    # Build combined tags from explicit tags + language/framework
    combined_tags = list(tags) if tags else []
    if language and language.lower() not in combined_tags:
        combined_tags.append(language.lower())
    if framework and framework.lower() not in combined_tags:
        combined_tags.append(framework.lower())

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
    )

    # Auto-increment use_count + quality score boost on best match
    if results:
        async with get_session(db_url) as session:
            from team_memory.storage.repository import ExperienceRepository
            repo = ExperienceRepository(session)
            best = results[0]
            best_id = best.get("group_id") or best.get("id")
            if best_id:
                try:
                    import uuid as _uuid
                    await repo.increment_use_count(_uuid.UUID(best_id))
                except Exception:
                    logger.debug("Failed to increment use_count", exc_info=True)
                try:
                    from sqlalchemy import select as sa_select

                    from team_memory.services.scoring import apply_reference_boost
                    from team_memory.storage.models import Experience
                    q = sa_select(Experience).where(
                        Experience.id == _uuid.UUID(best_id)
                    )
                    r = await session.execute(q)
                    exp = r.scalar_one_or_none()
                    if exp:
                        exp.quality_score = apply_reference_boost(
                            exp.quality_score
                        )
                        await session.commit()
                except Exception:
                    logger.debug("Failed to apply reference boost", exc_info=True)

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

    output = json.dumps(
        {
            "message": (
                f"Found {len(results)} relevant solution(s). "
                f"Best match score: {results[0].get('score', 'N/A')}. "
                "If helpful, call tm_feedback with the experience ID."
            ),
            "results": results,
        },
        ensure_ascii=False,
    )
    return _guard_output(output)


@mcp.tool(
    name="tm_learn",
    description=(
        "Learn from a conversation or document: extract structured experience "
        "using LLM and auto-save to the knowledge base. "
        "Call this after solving a problem to capture the knowledge. "
        "By default saves as draft (requires review before publishing). "
        "Returns ~200-500 tokens."
    ),
)
@track_usage
async def tm_learn(
    conversation: str,
    tags: list[str] | None = None,
    as_group: bool = False,
    save_as_draft: bool = True,
    project: str | None = None,
) -> str:
    """Extract and save an experience from conversation/document text.

    Uses an LLM to parse free-form text into structured fields
    (title, problem, solution, tags, etc.) then saves automatically.
    AI-extracted content defaults to draft mode for review.

    Args:
        conversation: The conversation or document text to learn from (required).
        tags: Additional tags to merge with LLM-extracted tags.
        as_group: If True, extract as parent + children experience group.
        save_as_draft: If True (default), save as draft requiring review.
    """
    from team_memory.services.llm_parser import (
        LLMParseError,
        compute_quality_score,
        parse_content,
    )

    service = _get_service()
    settings = _get_settings()
    db_url = _get_db_url()
    user = _get_current_user()
    resolved_project = _resolve_project(project)

    publish_status = "draft" if save_as_draft else "published"

    # Parse content with LLM
    try:
        parsed = await parse_content(
            content=conversation,
            llm_config=settings.llm,
            as_group=as_group,
        )
    except LLMParseError as e:
        return json.dumps(
            {"message": f"Failed to parse content: {e}", "error": True},
            ensure_ascii=False,
        )

    # Merge user-provided tags
    if as_group:
        parent_tags = parsed["parent"].get("tags", [])
        if tags:
            parent_tags = list(set(parent_tags + [t.lower() for t in tags]))
        parsed["parent"]["tags"] = parent_tags

        async with get_session(db_url) as session:
            result = await service.save_group(
                session=session,
                parent=parsed["parent"],
                children=parsed["children"],
                created_by=user,
                project=resolved_project,
            )

        child_count = len(parsed["children"])
        draft_note = (
            " (已保存为草稿，需通过 Web UI 审核后发布)"
            if publish_status == "draft"
            else ""
        )
        return json.dumps(
            {
                "message": (
                    f"Experience group saved: 1 parent + {child_count} children. "
                    f"Title: {parsed['parent'].get('title', 'N/A')}{draft_note}"
                ),
                "group": result,
                "publish_status": publish_status,
            },
            ensure_ascii=False,
        )
    else:
        extracted_tags = parsed.get("tags", [])
        if tags:
            extracted_tags = list(set(extracted_tags + [t.lower() for t in tags]))

        q_score = compute_quality_score(parsed)

        async with get_session(db_url) as session:
            result = await service.save(
                session=session,
                title=parsed.get("title", "Untitled"),
                problem=parsed.get("problem", ""),
                solution=parsed.get("solution"),
                created_by=user,
                tags=extracted_tags,
                code_snippets=parsed.get("code_snippets"),
                language=parsed.get("language"),
                framework=parsed.get("framework"),
                root_cause=parsed.get("root_cause"),
                source="auto_extract",
                publish_status=publish_status,
                experience_type=parsed.get("experience_type", "general"),
                severity=parsed.get("severity"),
                category=parsed.get("category"),
                structured_data=parsed.get("structured_data"),
                git_refs=parsed.get("git_refs"),
                project=resolved_project,
                quality_score=q_score,
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

        draft_note = (
            " (已保存为草稿，需通过 Web UI 审核后发布)"
            if result.get("publish_status") == "draft"
            else ""
        )
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
                    "publish_status": result.get("publish_status"),
                    "created_at": result.get("created_at"),
                },
            },
            ensure_ascii=False,
        )


# ============================================================
# Tools — P2.B: Verbatim artifact extraction
# ============================================================

@mcp.tool(
    name="tm_extract_artifacts",
    description=(
        "Extract verbatim knowledge artifacts from conversations or documents. "
        "Unlike tm_learn which summarizes, this preserves exact quotes with context. "
        "Best for Plan discussions, architecture decisions, requirement documents. "
        "Returns ~300-800 tokens."
    ),
)
async def tm_extract_artifacts(
    text: str,
    experience_id: str | None = None,
    source_ref: str = "",
    artifact_types: list[str] | None = None,
) -> str:
    """Extract verbatim artifacts from text and optionally link to an experience.

    Args:
        text: The conversation or document text to extract from (required).
        experience_id: Optional existing experience ID to link artifacts to.
        source_ref: Source identifier (session ID, file path, etc.).
        artifact_types: Filter types: decision, problem, pattern, constraint, fact.
    """
    settings = _get_settings()
    db_url = _get_db_url()
    allowed_types = artifact_types or ["decision", "problem", "pattern", "constraint", "fact"]

    prompt = f"""从以下文本中提取知识 artifact（逐字引用，不要摘要改写）。

每个 artifact 必须是以下类型之一: {', '.join(allowed_types)}

请严格以 JSON 格式返回（不要用 markdown 代码块包裹）:
{{
  "artifacts": [
    {{
      "type": "decision|problem|pattern|constraint|fact",
      "content": "原文引用（保留原始表述）",
      "context_before": "引用前的上下文（1-2句，帮助理解背景）",
      "context_after": "引用后的上下文（1-2句，帮助理解结果）"
    }}
  ]
}}

注意:
- content 必须是原文中的直接引用，不要改写
- 每个 artifact 应该是一个独立的知识点
- 最多提取 10 个最重要的 artifact
- context_before/context_after 可以为 null

文本:
{text[:8000]}"""

    import httpx
    llm_model = settings.llm.model
    llm_base_url = settings.llm.base_url.rstrip("/")

    artifacts_data = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{llm_base_url}/api/chat",
                json={
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(raw)
            artifacts_data = parsed.get("artifacts", [])
    except Exception as e:
        return json.dumps(
            {"error": True, "message": f"Artifact extraction failed: {e}"},
            ensure_ascii=False,
        )

    if not artifacts_data:
        return json.dumps(
            {"message": "No artifacts extracted from the text.", "artifacts": []},
            ensure_ascii=False,
        )

    # Save to database
    import uuid as _uuid

    from team_memory.storage.models import ExperienceArtifact

    exp_uuid = _uuid.UUID(experience_id) if experience_id else None
    saved = []

    if exp_uuid:
        from team_memory.storage.database import get_session
        async with get_session(db_url) as session:
            for a in artifacts_data:
                art = ExperienceArtifact(
                    experience_id=exp_uuid,
                    artifact_type=a.get("type", "fact"),
                    content=a.get("content", ""),
                    context_before=a.get("context_before"),
                    context_after=a.get("context_after"),
                    source_ref=source_ref,
                )
                session.add(art)
                saved.append(art.to_dict())
            await session.commit()

    return json.dumps(
        {
            "message": f"Extracted {len(artifacts_data)} artifacts"
            + (f", saved to experience {experience_id}" if exp_uuid else ""),
            "artifacts": saved if saved else artifacts_data,
            "count": len(artifacts_data),
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
        language: Programming language being used.
        framework: Framework being used.
        error_message: Error message encountered (if any).
        max_results: Maximum suggestions to return.
    """
    # Build context-based query
    query_parts = []

    if error_message:
        query_parts.append(error_message)

    if file_path:
        p = PurePosixPath(file_path)
        # Extract meaningful directory hints
        parts_lower = [part.lower() for part in p.parts]
        context_hints = []
        for hint in ("test", "tests", "migration", "migrations", "api", "auth",
                      "config", "deploy", "docker", "ci", "scripts"):
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
    user = _get_current_user()
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
        suggestions.append({
            "id": r.get("group_id") or parent.get("id", ""),
            "title": parent.get("title", "Untitled"),
            "tags": parent.get("tags", []),
            "score": r.get("score", r.get("similarity", 0)),
            "confidence": r.get("confidence", "medium"),
            "children_count": r.get("total_children", 0),
        })

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
        "Returns ~1000-4000 tokens depending on result count."
    ),
)
@track_usage
async def tm_search(
    query: str,
    tags: list[str] | None = None,
    max_results: int = 5,
    min_similarity: float = 0.6,
    grouped: bool = True,
    top_k_children: int = 3,
    use_pageindex_lite: bool | None = None,
    project: str | None = None,
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
        max_results: Maximum number of results (or groups when grouped=True).
        min_similarity: Minimum similarity threshold.
        grouped: Return results grouped by parent-child. Default True.
        top_k_children: Max children per group. Default 3.
    """
    service = _get_service()
    user = _get_current_user()
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
    )

    if not results:
        return json.dumps(
            {"message": "No matching experiences found.", "results": []},
            ensure_ascii=False,
        )

    output = json.dumps(
        {
            "message": f"Found {len(results)} matching experience(s).",
            "results": results,
        },
        ensure_ascii=False,
    )
    return _guard_output(output)


@mcp.tool(
    name="tm_save",
    description=(
        "Quick-save a simple experience (title + problem required, solution optional). "
        "Use this for fast knowledge capture — solution can be added later. "
        "For typed experiences with full fields, use tm_save_typed instead. "
        "Returns ~100-200 tokens."
    ),
)
@track_usage
async def tm_save(
    title: str,
    problem: str,
    solution: str | None = None,
    tags: list[str] | None = None,
    code_snippets: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    root_cause: str | None = None,
    publish_status: str = "personal",
    skip_dedup: bool = False,
    project: str | None = None,
) -> str:
    """Quick-save a new experience to the team knowledge base.

    Args:
        title: Experience title (required).
        problem: Problem description (required).
        solution: Solution description (optional — allows incomplete experiences).
        tags: Tags for the experience.
        code_snippets: Key code examples.
        language: Programming language.
        framework: Framework.
        root_cause: Root cause analysis.
        publish_status: "personal" (default), "published", or "draft".
        skip_dedup: If True, skip duplicate detection check.
    """
    service = _get_service()
    db_url = _get_db_url()
    user = _get_current_user()
    resolved_project = _resolve_project(project)

    async with get_session(db_url) as session:
        result = await service.save(
            session=session,
            title=title,
            problem=problem,
            solution=solution,
            created_by=user,
            tags=tags,
            code_snippets=code_snippets,
            language=language,
            framework=framework,
            source="auto_extract",
            root_cause=root_cause,
            publish_status=publish_status,
            skip_dedup=skip_dedup,
            project=resolved_project,
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
                "publish_status": result.get("publish_status"),
                "completeness_score": result.get("completeness_score"),
                "created_at": result.get("created_at"),
            },
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="tm_save_typed",
    description=(
        "Save a typed experience with full fields (experience_type, severity, "
        "category, structured_data, git_refs, related_links, progress_status). "
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
    code_snippets: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    root_cause: str | None = None,
    severity: str | None = None,
    category: str | None = None,
    progress_status: str | None = None,
    structured_data: dict | None = None,
    git_refs: list[dict] | None = None,
    related_links: list[dict] | None = None,
    publish_status: str = "published",
    skip_dedup: bool = False,
    project: str | None = None,
) -> str:
    """Save a typed experience with full fields.

    Args:
        title: Experience title (required).
        problem: Problem description (required).
        experience_type: Type — general/feature/bugfix/tech_design/incident/best_practice/learning.
        solution: Solution (optional — allows incomplete experiences).
        tags: Tags for the experience.
        code_snippets: Key code examples.
        language: Programming language.
        framework: Framework.
        root_cause: Root cause analysis.
        severity: Severity level (P0-P4, for bugfix/incident).
        category: Category (frontend/backend/database/infra/performance/security/other).
        progress_status: Progress status (type-specific, e.g., open/investigating/fixed/verified).
        structured_data: Type-specific data dict (e.g., reproduction_steps, environment for bugfix).
        git_refs: List of git references [{type, url, hash, description}].
        related_links: List of related links [{type, url, title}].
        publish_status: "published" (default) or "draft".
        skip_dedup: If True, skip duplicate detection check.
    """
    service = _get_service()
    db_url = _get_db_url()
    user = _get_current_user()
    resolved_project = _resolve_project(project)

    async with get_session(db_url) as session:
        result = await service.save(
            session=session,
            title=title,
            problem=problem,
            solution=solution,
            created_by=user,
            tags=tags,
            code_snippets=code_snippets,
            language=language,
            framework=framework,
            source="auto_extract",
            root_cause=root_cause,
            publish_status=publish_status,
            skip_dedup=skip_dedup,
            experience_type=experience_type,
            severity=severity,
            category=category,
            progress_status=progress_status,
            structured_data=structured_data,
            git_refs=git_refs,
            related_links=related_links,
            project=resolved_project,
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
                "severity": result.get("severity"),
                "category": result.get("category"),
                "progress_status": result.get("progress_status"),
                "completeness_score": result.get("completeness_score"),
                "publish_status": result.get("publish_status"),
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
        "The parent describes the overall problem/solution, "
        "children describe individual steps. "
        "Returns ~200-500 tokens."
    ),
)
async def tm_save_group(
    parent_title: str,
    parent_problem: str,
    children: list[dict],
    parent_solution: str | None = None,
    parent_tags: list[str] | None = None,
    parent_root_cause: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    experience_type: str = "general",
    severity: str | None = None,
    category: str | None = None,
    project: str | None = None,
) -> str:
    """Save a group of experiences (parent with children).

    Args:
        parent_title: Title for the parent experience.
        parent_problem: Problem description for the parent.
        children: List of dicts, each with keys: title, problem, solution,
                  and optionally: tags, code_snippets, root_cause.
        parent_solution: Overall solution summary (optional).
        parent_tags: Tags for the parent.
        parent_root_cause: Root cause for the parent.
        language: Programming language.
        framework: Framework.
        experience_type: Type for the group (default "general").
        severity: Severity for bugfix/incident groups.
        category: Category classification.

    Returns:
        JSON string with the created group.
    """
    service = _get_service()
    db_url = _get_db_url()
    user = _get_current_user()
    resolved_project = _resolve_project(project)

    parent_data = {
        "title": parent_title,
        "problem": parent_problem,
        "solution": parent_solution or "",
        "tags": parent_tags,
        "root_cause": parent_root_cause,
        "language": language,
        "framework": framework,
        "source": "auto_extract",
        "project": resolved_project,
    }

    children_data = []
    for child in children:
        children_data.append({
            "title": child.get("title", ""),
            "problem": child.get("problem", ""),
            "solution": child.get("solution", ""),
            "tags": child.get("tags"),
            "code_snippets": child.get("code_snippets"),
            "root_cause": child.get("root_cause"),
            "source": "auto_extract",
            "project": resolved_project,
        })

    async with get_session(db_url) as session:
        result = await service.save_group(
            session=session,
            parent=parent_data,
            children=children_data,
            created_by=user,
            project=resolved_project,
        )

    msg = (
        f"Experience group ({experience_type}) saved: "
        f"1 parent + {len(children_data)} children."
    )
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
        "Claims auto-expire after 30 minutes of inactivity."
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
    user = _get_current_user()

    async with get_session(db_url) as session:
        from sqlalchemy import update

        from team_memory.storage.models import Experience

        result = await session.execute(
            update(Experience)
            .where(Experience.id == _uuid.UUID(experience_id))
            .values(
                source_context=json.dumps({
                    "claimed_by": user,
                    "claimed_at": datetime.now(timezone.utc).isoformat(),
                    "message": message or "",
                })
            )
            .returning(Experience.id)
        )
        if not result.first():
            return json.dumps({"message": "Experience not found.", "error": True})
        await session.commit()

    return json.dumps({
        "message": f"Claimed by {user}. Other agents will see this.",
        "claimed_by": user,
    })


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
    user = _get_current_user()

    # Emit via event bus (will trigger webhook if configured)
    service = _get_service()
    if service._event_bus:
        from team_memory.services.event_bus import Events
        await service._event_bus.emit(Events.EXPERIENCE_UPDATED, {
            "experience_id": experience_id,
            "notified_by": user,
            "message": message,
        })

    return json.dumps({
        "message": f"Notification sent: {message}",
        "notified_by": user,
        "experience_id": experience_id,
    })


@mcp.tool(
    name="tm_feedback",
    description=(
        "Provide feedback on a searched experience — "
        "rate 1-5 (5=best) and optionally score fitness (1-5). "
        "This improves future search results. Returns ~50 tokens."
    ),
)
@track_usage
async def tm_feedback(
    experience_id: str,
    rating: int,
    fitness_score: int | None = None,
    comment: str | None = None,
) -> str:
    """Submit feedback for an experience.

    Args:
        experience_id: The ID of the experience to rate.
        rating: Rating from 1 to 5 (5 = most helpful).
        fitness_score: Post-use fitness score 1-5 (how well it matched your need).
        comment: Optional feedback comment.

    Returns:
        JSON string with the result.
    """
    if not (1 <= rating <= 5):
        return json.dumps({"message": "Rating must be between 1 and 5.", "error": True})
    if fitness_score is not None and not (1 <= fitness_score <= 5):
        return json.dumps({"message": "fitness_score must be between 1 and 5.", "error": True})
    service = _get_service()
    db_url = _get_db_url()
    user = _get_current_user()

    success = await service.feedback(
        experience_id=experience_id,
        rating=rating,
        feedback_by=user,
        comment=comment,
        fitness_score=fitness_score,
    )

    if success:
        # Apply quality score boost for high ratings
        try:
            from team_memory.services.scoring import apply_rating_boost
            from team_memory.storage.models import Experience

            async with get_session(db_url) as s2:
                from sqlalchemy import select as sa_select

                q = sa_select(Experience).where(Experience.id == experience_id)
                r = await s2.execute(q)
                exp = r.scalar_one_or_none()
                if exp:
                    exp.quality_score = apply_rating_boost(exp.quality_score, rating)
                    await s2.commit()
        except Exception:
            pass
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
        "Read runtime retrieval configuration snapshot "
        "(retrieval/search/cache/pageindex-lite)."
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
            "reranker_provider": settings.reranker.provider,
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
    name="tm_track",
    description=(
        "Report external MCP/skill usage for analytics tracking. "
        "Call this to report usage of tools from other MCP servers or skills. "
        "Returns ~50 tokens."
    ),
)
async def tm_track(
    tool_name: str,
    tool_type: str = "mcp",
    duration_ms: int | None = None,
    success: bool = True,
    session_id: str | None = None,
    error_message: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Report external MCP/skill usage for analytics tracking."""
    from team_memory.storage.models import ToolUsageLog

    db_url = _get_db_url()
    user = _get_current_user()
    async with get_session(db_url) as session:
        log = ToolUsageLog(
            tool_name=tool_name,
            tool_type=tool_type,
            user=user,
            project=_resolve_project(None),
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            session_id=session_id,
            metadata_extra=metadata or {},
        )
        session.add(log)
        await session.commit()
    return f"✅ Tracked: {tool_name} ({tool_type})"


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
                            skills.append({
                                "name": skill_dir.name,
                                "path": str(skill_dir),
                                "active": active,
                            })
        lines = [f"{'✅' if s['active'] else '⏸️'} {s['name']}" for s in skills]
        return (
            f"Skills ({len(skills)}):\n" + "\n".join(lines)
            if lines
            else "No skills found."
        )

    if not skill_path:
        return "❌ skill_path required for disable/enable"

    import shutil

    cache_dir = Path.home() / ".team_memory" / "disabled_skills"

    p = Path(skill_path)
    if action == "disable":
        src = p / "SKILL.md" if p.is_dir() else p
        if src.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = src.parent.name + "__SKILL.md"
            shutil.copy2(str(src), str(cache_dir / cache_name))
            os.remove(str(src))
            return f"⏸️ Disabled: {src.parent.name} (cached to {cache_dir})"
        return f"❌ Not found: {src}"

    if action == "enable":
        skill_dir = p if p.is_dir() else p.parent
        cache_name = skill_dir.name + "__SKILL.md"
        cached = cache_dir / cache_name
        dst = skill_dir / "SKILL.md"
        if cached.exists():
            shutil.copy2(str(cached), str(dst))
            os.remove(str(cached))
            return f"✅ Enabled: {skill_dir.name} (restored from cache)"
        old_disabled = skill_dir / "SKILL.md.disabled"
        if old_disabled.exists():
            os.rename(str(old_disabled), str(dst))
            return f"✅ Enabled: {skill_dir.name} (legacy rename)"
        return f"❌ Not found in cache or disabled: {skill_dir.name}"

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

    user = _get_current_user()
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
                        publish_status="personal",
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


@mcp.resource("experiences://stats")
async def experience_stats() -> str:
    """Get experience database statistics."""
    service = _get_service()
    db_url = _get_db_url()

    async with get_session(db_url) as session:
        stats = await service.get_stats(session=session)

    lines = [
        "# Experience Database Stats\n",
        f"- **Total experiences**: {stats['total_experiences']}",
        f"- **Added in last 7 days**: {stats['recent_7days']}",
        f"- **Pending reviews**: {stats.get('pending_reviews', 0)}",
        "\n## Top Tags\n",
    ]
    for tag, count in stats.get("tag_distribution", {}).items():
        lines.append(f"- {tag}: {count}")

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
# Task Management Tools — tm_task, tm_doc_sync
# ============================================================


@mcp.tool(
    name="tm_task",
    description=(
        "Unified task CRUD — create, update, list, or get personal tasks. "
        "Completing a task auto-generates an experience (sediment). "
        "Returns ~200 tokens."
    ),
)
@track_usage
async def tm_task(
    action: str,
    task_id: str | None = None,
    group_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    importance: int | None = None,
    project: str | None = None,
    due_date: str | None = None,
    labels: list[str] | None = None,
    experience_id: str | None = None,
    summary: str | None = None,
    with_context: bool = False,
) -> str:
    """Manage personal tasks.

    Args:
        action: One of 'create', 'update', 'list', 'get'.
        task_id: Required for update/get.
        group_id: Optional group filter for list, or group to assign for create.
        title: Task title (required for create).
        description: Optional description.
        status: wait/plan/in_progress/completed/cancelled.
        priority: low/medium/high/urgent.
        importance: 1-5 importance level.
        project: Project name (defaults to env/config).
        due_date: ISO date string for deadline.
        labels: Tag labels.
        experience_id: Link to an existing experience.
        summary: Required when completing (status=completed) for auto-sediment.
        with_context: If True and action=get, include linked experience content.
    """
    import uuid as _uuid
    from datetime import datetime as _dt

    db_url = _get_db_url()
    resolved_project = _resolve_project(project)
    user = _get_current_user()

    from team_memory.storage.repository import TaskRepository

    if action == "create":
        if not title:
            return json.dumps({"error": True, "message": "title is required for create"})
        async with get_session(db_url) as session:
            repo = TaskRepository(session)
            if status == "in_progress":
                ok, cnt = await repo.check_wip(resolved_project, user)
                if not ok:
                    return json.dumps({
                        "error": True,
                        "message": f"WIP limit ({repo.WIP_LIMIT}) exceeded ({cnt} in progress).",
                    })
            parsed_due = None
            if due_date:
                try:
                    parsed_due = _dt.fromisoformat(due_date)
                except ValueError:
                    pass
            task = await repo.create_task(
                title=title,
                user_id=user,
                project=resolved_project,
                group_id=_uuid.UUID(group_id) if group_id else None,
                description=description,
                status=status or "wait",
                priority=priority or "medium",
                importance=importance or 3,
                due_date=parsed_due,
                labels=labels,
                experience_id=_uuid.UUID(experience_id) if experience_id else None,
            )
            await session.commit()
            task_dict = task.to_dict()

        related = []
        if title:
            try:
                service = _get_service()
                search_query = f"{title} {description or ''}"[:200]
                hits = await service.search(
                    query=search_query, max_results=3,
                    min_similarity=0.4, user_name=user, source="task_create",
                    grouped=True, top_k_children=1, project=resolved_project,
                )
                for h in hits:
                    p = h.get("parent", h)
                    related.append({
                        "id": p.get("id", ""),
                        "title": p.get("title", ""),
                        "score": round(h.get("similarity", 0), 3),
                    })
            except Exception:
                pass
        return json.dumps({
            "task": task_dict, "message": "Task created.",
            "related_experiences": related,
        }, ensure_ascii=False)

    elif action == "update":
        if not task_id:
            return json.dumps({"error": True, "message": "task_id is required for update"})
        async with get_session(db_url) as session:
            repo = TaskRepository(session)
            task = await repo.get_task(_uuid.UUID(task_id))
            if not task:
                return json.dumps({"error": True, "message": "Task not found."})
            kwargs = {}
            if title is not None:
                kwargs["title"] = title
            if description is not None:
                kwargs["description"] = description
            if status is not None:
                kwargs["status"] = status
            if priority is not None:
                kwargs["priority"] = priority
            if importance is not None:
                kwargs["importance"] = importance
            if due_date is not None:
                try:
                    kwargs["due_date"] = _dt.fromisoformat(due_date)
                except ValueError:
                    pass
            if labels is not None:
                kwargs["labels"] = labels

            warning = None
            if status == "in_progress" and task.status != "in_progress":
                ok, cnt = await repo.check_wip(task.project, task.user_id)
                if not ok:
                    warning = f"WIP limit ({repo.WIP_LIMIT}) exceeded ({cnt} in progress)"

            updated = await repo.update_task(task.id, **kwargs)

            # Auto-sediment on completion
            sediment_id = None
            reflection_id = None
            if status == "completed" and summary:
                try:
                    from team_memory.services.llm_parser import compute_quality_score
                    service = _get_service()
                    q_score = compute_quality_score({
                        "problem": summary,
                        "solution": summary,
                    })
                    learned = await service.save(
                        session=session,
                        title=f"[任务沉淀] {task.title}",
                        problem=summary,
                        tags=task.labels or ["task-sediment"],
                        created_by=user,
                        project=task.project,
                        source="task_sediment",
                        quality_score=q_score,
                    )
                    if learned and hasattr(learned, "id"):
                        sediment_id = str(learned.id)
                        if updated:
                            updated.sediment_experience_id = learned.id
                except Exception as e:
                    logger.warning("Auto-sediment failed: %s", e)

                # P2.C: Auto-reflection on task completion
                try:
                    reflection_id = await _create_reflection(
                        session=session,
                        task=task,
                        summary=summary,
                        sediment_experience_id=sediment_id,
                        settings=_get_settings(),
                    )
                except Exception as e:
                    logger.warning("Auto-reflection failed: %s", e)

            await session.commit()
        result = {
            "task": updated.to_dict() if updated else {},
            "message": "Task updated.",
        }
        if sediment_id:
            result["sediment_experience_id"] = sediment_id
            result["message"] = f"Task completed. Experience sediment created: {sediment_id}"
        if reflection_id:
            result["reflection_id"] = reflection_id
        if warning:
            result["warning"] = warning
        return json.dumps(result)

    elif action == "list":
        async with get_session(db_url) as session:
            repo = TaskRepository(session)
            tasks = await repo.list_tasks(
                project=resolved_project,
                user_id=user,
                status=status,
                group_id=_uuid.UUID(group_id) if group_id else None,
            )
        return json.dumps({
            "tasks": [t.to_dict() for t in tasks],
            "total": len(tasks),
        })

    elif action == "get":
        if not task_id and not group_id:
            return json.dumps({"error": True, "message": "task_id or group_id required"})
        async with get_session(db_url) as session:
            repo = TaskRepository(session)
            if task_id:
                task = await repo.get_task(_uuid.UUID(task_id))
                if not task:
                    return json.dumps({"error": True, "message": "Task not found."})
                data = task.to_dict()
                if with_context and task.experience_id:
                    from team_memory.storage.repository import ExperienceRepository
                    exp_repo = ExperienceRepository(session)
                    exp = await exp_repo.get_by_id(task.experience_id)
                    if exp:
                        data["experience_context"] = exp.to_dict()
                return json.dumps(data)
            else:
                group = await repo.get_group(_uuid.UUID(group_id))
                if not group:
                    return json.dumps({"error": True, "message": "Group not found."})
                return json.dumps(group.to_dict(include_tasks=True))

    return json.dumps({"error": True, "message": f"Unknown action: {action}"})


@mcp.tool(
    name="tm_task_claim",
    description=(
        "Atomically claim a task (assigns to you + sets in_progress). "
        "Respects WIP limit. Returns ~100 tokens."
    ),
)
async def tm_task_claim(
    task_id: str,
) -> str:
    """Claim a task for yourself.

    Args:
        task_id: The task ID to claim.
    """
    import uuid as _uuid

    db_url = _get_db_url()
    user = _get_current_user()

    from team_memory.storage.repository import TaskRepository

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        try:
            task = await repo.claim_task(
                _uuid.UUID(task_id), user, project=_resolve_project(None),
            )
        except ValueError as e:
            return json.dumps({"error": True, "message": str(e)})
        if not task:
            return json.dumps({"error": True, "message": "Task not found"})
        await session.commit()
    return json.dumps({"task": task.to_dict(), "message": f"Claimed by {user}"})


@mcp.tool(
    name="tm_ready",
    description=(
        "List tasks ready to start (no unresolved blocking dependencies). "
        "Returns ~200 tokens."
    ),
)
async def tm_ready(
    project: str | None = None,
) -> str:
    """Get tasks that are ready to work on.

    Args:
        project: Project filter.
    """
    db_url = _get_db_url()
    user = _get_current_user()
    resolved_project = _resolve_project(project)

    from team_memory.storage.repository import TaskRepository

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        tasks = await repo.get_ready_tasks(resolved_project, user)
    return json.dumps({
        "ready_tasks": [t.to_dict() for t in tasks],
        "total": len(tasks),
    })


@mcp.tool(
    name="tm_message",
    description=(
        "Leave a message/comment on a task. Supports threaded replies. "
        "Returns ~50 tokens."
    ),
)
async def tm_message(
    task_id: str,
    content: str,
    thread_id: str | None = None,
) -> str:
    """Add a message to a task.

    Args:
        task_id: The task to comment on.
        content: Message content.
        thread_id: Optional thread ID for replies.
    """
    import uuid as _uuid

    db_url = _get_db_url()
    user = _get_current_user()

    from team_memory.storage.repository import TaskRepository

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        msg = await repo.add_message(
            task_id=_uuid.UUID(task_id),
            author=user,
            content=content,
            thread_id=_uuid.UUID(thread_id) if thread_id else None,
        )
        await session.commit()
    return json.dumps({"message": msg.to_dict()})


@mcp.tool(
    name="tm_dependency",
    description=(
        "Manage task dependencies — add or remove blocking/related relationships. "
        "Returns ~50 tokens."
    ),
)
async def tm_dependency(
    action: str,
    source_task_id: str,
    target_task_id: str,
    dep_type: str = "blocks",
) -> str:
    """Manage task dependencies.

    Args:
        action: 'add' or 'remove'.
        source_task_id: The blocking/source task.
        target_task_id: The blocked/target task.
        dep_type: blocks, related, or discovered_from.
    """
    import uuid as _uuid

    db_url = _get_db_url()
    user = _get_current_user()

    from team_memory.storage.repository import TaskRepository

    async with get_session(db_url) as session:
        repo = TaskRepository(session)
        if action == "add":
            dep = await repo.add_dependency(
                _uuid.UUID(source_task_id),
                _uuid.UUID(target_task_id),
                dep_type=dep_type,
                created_by=user,
            )
            await session.commit()
            return json.dumps({"dependency": dep.to_dict(), "message": "Added"})
        elif action == "remove":
            ok = await repo.remove_dependency(
                _uuid.UUID(source_task_id), _uuid.UUID(target_task_id),
            )
            await session.commit()
            if not ok:
                return json.dumps({"error": True, "message": "Dependency not found"})
            return json.dumps({"message": "Dependency removed"})
        return json.dumps({"error": True, "message": f"Unknown action: {action}"})


@mcp.tool(
    name="tm_doc_sync",
    description=(
        "Sync a .debug document into team_memory. "
        "Routes to Experience or TaskGroup based on doc_type. "
        "Idempotent via content_hash. Returns ~100 tokens."
    ),
)
async def tm_doc_sync(
    file_path: str,
    content: str,
    doc_type: str = "experience",
    project: str | None = None,
) -> str:
    """Sync a local document into team_memory.

    Args:
        file_path: Source file path (e.g. .debug/01-deploy-guide.md).
        content: Full file content.
        doc_type: 'experience', 'task_group', or 'guide'.
        project: Target project.
    """
    import hashlib

    db_url = _get_db_url()
    resolved_project = _resolve_project(project)
    user = _get_current_user()
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    from team_memory.storage.repository import TaskRepository

    if doc_type == "task_group":
        async with get_session(db_url) as session:
            repo = TaskRepository(session)
            existing = await repo.get_group_by_source(file_path)
            if existing and existing.content_hash == content_hash:
                return json.dumps({
                    "message": "Document unchanged (hash match), skipped.",
                    "group_id": str(existing.id),
                })

            import re
            lines = content.strip().split("\n")
            doc_title = lines[0].lstrip("# ").strip() if lines else file_path

            if existing:
                existing.title = doc_title
                existing.content_hash = content_hash
                existing.description = content[:500]
                group = existing
            else:
                group = await repo.create_group(
                    title=doc_title,
                    user_id=user,
                    project=resolved_project,
                    source_doc=file_path,
                    content_hash=content_hash,
                    description=content[:500],
                )

            task_pattern = re.compile(r"^-\s*\[([xX ])\]\s*(.+)$", re.MULTILINE)
            matches = task_pattern.findall(content)
            created_count = 0
            for checked, task_title in matches:
                st = "completed" if checked.lower() == "x" else "wait"
                await repo.create_task(
                    title=task_title.strip(),
                    user_id=user,
                    project=resolved_project,
                    group_id=group.id,
                    status=st,
                )
                created_count += 1

            await session.commit()
        return json.dumps({
            "message": f"Task group synced: {created_count} tasks.",
            "group_id": str(group.id),
        })

    else:
        service = _get_service()
        db_url = _get_db_url()
        async with get_session(db_url) as session:
            from sqlalchemy import text as _sa_text
            existing_results = await session.execute(
                _sa_text(
                    "SELECT id FROM experiences WHERE source_context = :src LIMIT 1"
                ),
                {"src": file_path},
            )
            existing_row = existing_results.first()
            if existing_row:
                return json.dumps({
                    "message": "Document already synced as experience.",
                    "experience_id": str(existing_row[0]),
                })

        exp_type = "runbook" if doc_type == "guide" else "general"
        lines = content.strip().split("\n")
        title = lines[0].lstrip("# ").strip() if lines else file_path

        result = await service.save(
            title=title,
            problem=content[:2000],
            solution=content[2000:4000] if len(content) > 2000 else None,
            tags=["doc-sync", doc_type],
            created_by=user,
            project=resolved_project,
            source="doc_sync",
            experience_type=exp_type,
        )
        exp_id = str(result.id) if hasattr(result, "id") else str(result.get("id", ""))
        return json.dumps({
            "message": f"Document synced as {doc_type} experience.",
            "experience_id": exp_id,
        })


# ============================================================
# Pre-flight Check — tm_preflight
# ============================================================


COMPLEX_KEYWORDS = {
    "架构", "迁移", "安全", "plan", "重构", "新功能", "设计", "权限",
    "migration", "refactor", "architecture", "security", "redesign",
    "多模块", "数据库", "schema", "部署", "deploy", "taskgroup",
}
MEDIUM_KEYWORDS = {
    "修复", "fix", "bug", "优化", "改进", "enhance", "update",
    "添加", "add", "feature", "功能", "页面", "组件", "api",
}


def _estimate_complexity(
    task_description: str, current_files: list[str] | None = None
) -> str:
    desc_lower = task_description.lower()
    file_count = len(current_files) if current_files else 0
    desc_len = len(task_description)

    if any(kw in desc_lower for kw in COMPLEX_KEYWORDS) or file_count > 2 or desc_len > 100:
        return "complex"
    if any(kw in desc_lower for kw in MEDIUM_KEYWORDS) or file_count >= 1:
        return "medium"
    return "trivial"


@mcp.tool(
    name="tm_preflight",
    description=(
        "Task pre-flight check: analyze complexity and return quick experience "
        "search results. MUST be called at the start of every task. "
        "Returns complexity assessment + top 3 related experiences (~300 tokens)."
    ),
)
async def tm_preflight(
    task_description: str,
    current_files: list[str] | None = None,
    project: str | None = None,
) -> str:
    """Pre-flight check before starting any task.

    Analyzes task complexity and returns quick search results to guide
    whether a full tm_search is needed.

    Args:
        task_description: What you're about to work on.
        current_files: List of files that will be modified.
        project: Project scope.
    """
    complexity = _estimate_complexity(task_description, current_files)
    depth_map = {"trivial": "skip", "medium": "light", "complex": "full"}
    search_depth = depth_map[complexity]

    quick_results = []
    if complexity != "trivial":
        service = _get_service()
        resolved_project = _resolve_project(project)
        user = _get_current_user()
        try:
            results = await service.search(
                query=task_description,
                max_results=3,
                min_similarity=0.4,
                user_name=user,
                source="preflight",
                grouped=True,
                top_k_children=1,
                project=resolved_project,
            )
            for r in results:
                parent = r.get("parent", r)
                quick_results.append({
                    "id": parent.get("id", ""),
                    "title": parent.get("title", ""),
                    "score": round(r.get("similarity", 0), 3),
                    "type": parent.get("experience_type", "general"),
                    "solution_preview": (parent.get("solution") or "")[:200],
                })
        except Exception as exc:
            logger.debug("tm_preflight quick search failed: %s", exc)

    hint_map = {
        "trivial": "任务简单，可直接执行。",
        "medium": "中等复杂度，请参考 quick_results 中的经验。",
        "complex": "复杂任务，请继续调用 tm_search/tm_solve 做全量检索后再执行。",
    }
    return json.dumps({
        "complexity": complexity,
        "search_depth": search_depth,
        "quick_results": quick_results,
        "action_hint": hint_map[complexity],
    }, ensure_ascii=False)


# ============================================================
# Task Prompts — execute_task, resume_project
# ============================================================


@mcp.prompt(
    description=(
        "Get structured instructions to execute a task or task group. "
        "Use this to hand over a task to an AI agent for execution."
    )
)
def execute_task(task_id: str | None = None, group_id: str | None = None) -> str:
    """Generate a prompt for executing a task."""
    target = f"task_id={task_id}" if task_id else f"group_id={group_id}"
    return f"""You are executing a TeamMemory task.

1. First, call `tm_task` with action="get", {target}, with_context=true
2. Read the task description and any linked experience context carefully.
3. Execute the task according to the description and acceptance criteria.
4. When done, call `tm_task` with action="update", task_id=<id>, status="completed",
   summary="<brief summary of what you did and the outcome>"
5. The summary will be automatically saved as a reusable experience.

Important:
- If you encounter issues, update status to "in_progress" with a description of the blocker.
- Reference related experiences from team_memory when applicable.
- Keep your summary concise but include key decisions and outcomes."""


@mcp.prompt(
    description=(
        "Resume a project by getting the current task context. "
        "Call this at the start of a new session to restore project state."
    )
)
def resume_project(project: str | None = None) -> str:
    """Generate a prompt for resuming project context."""
    proj = project or "default"
    return f"""You are resuming work on project: {proj}

To restore context, run these commands:
1. `tm_task` action="list", project="{proj}", status="in_progress"
   → Shows tasks you're currently working on.
2. `tm_task` action="list", project="{proj}", status="wait"
   → Shows tasks waiting to be picked up (sorted by importance*urgency).
3. `tm_task` action="list", project="{proj}", status="completed"
   → Shows recently completed tasks for context.

After reviewing:
- Pick up the highest-priority incomplete task.
- Use `tm_solve` if you need related experience for the task.
- Update task status as you work: wait → plan → in_progress → completed.
- WIP limit is 5 tasks in_progress simultaneously.
- When completing a task, always provide a summary for experience sediment."""


# ============================================================
# Resources — P1-5: Add experiences://stale
# ============================================================

@mcp.resource("experiences://stale")
async def stale_experiences() -> str:
    """Get experiences that may be outdated (unused for N months)."""
    service = _get_service()
    settings = _get_settings()
    months = settings.lifecycle.stale_months

    results = await service.scan_stale(months=months)

    if not results:
        return f"No stale experiences found (threshold: {months} months of inactivity)."

    lines = [f"# Stale Experiences (unused > {months} months)\n"]
    for exp in results:
        tags_str = ", ".join(exp.get("tags", []))
        last_used = exp.get("last_used_at", "never")
        lines.append(
            f"- **{exp['title']}** [{tags_str}] (last used: {last_used})"
        )

    lines.append(f"\nTotal: {len(results)} stale experience(s)")
    return "\n".join(lines)


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
