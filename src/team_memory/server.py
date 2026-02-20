"""FastMCP Server entry point for TeamMemory.

Registers all MCP tools (tm_* namespace), resources, and prompts.
Uses the shared AppContext singleton from bootstrap.py for all services.

Tool namespace: All tools use the `tm_` prefix to help LLM clients
identify TeamMemory capabilities (e.g. tm_search, tm_save, tm_solve).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import PurePosixPath

from fastmcp import FastMCP

from team_memory.bootstrap import bootstrap, get_context
from team_memory.services.context_trimmer import estimate_tokens
from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory")


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


def _resolve_project(project: str | None = None) -> str:
    """Resolve project from explicit param > env > settings default."""
    if project and project.strip():
        return project.strip()
    env_project = os.environ.get("TEAM_MEMORY_PROJECT", "").strip()
    if env_project:
        return env_project
    ctx = get_context()
    default_project = (
        ctx.settings.default_project.strip() if ctx.settings.default_project else ""
    )
    return default_project or "default"


def _get_db_url() -> str:
    return get_context().db_url


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
        "- tm_update: Update fields on an existing experience\n\n"
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

    async with get_session(db_url) as session:
        from team_memory.storage.repository import ExperienceRepository

        results = await service.search(
            session=session,
            query=enhanced_query,
            tags=combined_tags or None,
            max_results=max_results,
            min_similarity=0.5,  # Lower threshold for broader recall
            user_name=user,
            source="mcp",
            grouped=True,
            top_k_children=2,  # Fewer children for focused output
            use_pageindex_lite=use_pageindex_lite,
            project=resolved_project,
        )

        # Auto-increment use_count on best match
        if results:
            repo = ExperienceRepository(session)
            best = results[0]
            best_id = best.get("group_id") or best.get("id")
            if best_id:
                try:
                    import uuid as _uuid
                    await repo.increment_use_count(_uuid.UUID(best_id))
                except Exception:
                    logger.debug("Failed to increment use_count", exc_info=True)

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
    from team_memory.services.llm_parser import LLMParseError, parse_content

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
    db_url = _get_db_url()
    user = _get_current_user()
    resolved_project = _resolve_project(project)

    async with get_session(db_url) as session:
        results = await service.search(
            session=session,
            query=query,
            tags=filter_tags or None,
            max_results=max_results,
            min_similarity=0.4,  # Lower threshold for broader suggestions
            user_name=user,
            source="mcp",
            grouped=True,
            top_k_children=1,  # Minimal children for lightweight output
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
    db_url = _get_db_url()
    user = _get_current_user()
    resolved_project = _resolve_project(project)

    async with get_session(db_url) as session:
        results = await service.search(
            session=session,
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
async def tm_save(
    title: str,
    problem: str,
    solution: str | None = None,
    tags: list[str] | None = None,
    code_snippets: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    root_cause: str | None = None,
    publish_status: str = "published",
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
        "rate 1-5 (5=best). This improves future search results. "
        "Returns ~50 tokens."
    ),
)
async def tm_feedback(
    experience_id: str,
    rating: int,
    comment: str | None = None,
) -> str:
    """Submit feedback for an experience.

    Args:
        experience_id: The ID of the experience to rate.
        rating: Rating from 1 to 5 (5 = most helpful).
        comment: Optional feedback comment.

    Returns:
        JSON string with the result.
    """
    if not (1 <= rating <= 5):
        return json.dumps({"message": "Rating must be between 1 and 5.", "error": True})
    service = _get_service()
    db_url = _get_db_url()
    user = _get_current_user()

    async with get_session(db_url) as session:
        success = await service.feedback(
            session=session,
            experience_id=experience_id,
            rating=rating,
            feedback_by=user,
            comment=comment,
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
    db_url = _get_db_url()

    async with get_session(db_url) as session:
        result = await service.update(
            session=session,
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
