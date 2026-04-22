"""Cursor beforeSubmitPrompt hook script.

Fires before the user's prompt is sent to the model. If the prompt
contains retrieval trigger keywords (e.g. "之前", "上次", "经验"),
the hook queries TeamMemory for relevant past experiences and injects
them as additionalContext.

Input (stdin JSON):
    {"prompt": "user text", "workspace_roots": [...], "conversation_id": "..."}

Output (stdout JSON):
    On trigger: {"additionalContext": "<retrieved context>"}
    No trigger: {}  (empty — no retrieval)
    On error:  {}  (silent failure — don't block user)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure scripts/hooks/ is importable for sibling modules
_HOOKS_DIR = str(Path(__file__).resolve().parent)
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)

from common import call_mcp_tool, get_project_from_path, parse_hook_input  # noqa: E402
from retrieval_trigger import should_retrieve  # noqa: E402

logger = logging.getLogger("cursor_before_prompt")


def main() -> str:
    """Entry point for the beforeSubmitPrompt hook.

    Returns:
        JSON string with additionalContext if triggered, empty dict otherwise.
    """
    try:
        # 1. Parse stdin JSON
        data = parse_hook_input()
        user_prompt = data.get("prompt", "")
        workspace_roots = data.get("workspace_roots", [])

        # 2. Check if prompt contains keyword triggers
        if not should_retrieve(user_prompt):
            return json.dumps({})

        # 3. Resolve project from workspace path
        workspace_root = workspace_roots[0] if workspace_roots else ""
        project = get_project_from_path(workspace_root) if workspace_root else None

        if project is None:
            # Can't query without a project — return empty (don't block user)
            return json.dumps({})

        # 4. Call TM memory_recall MCP tool
        result = call_mcp_tool("memory_recall", {
            "query": user_prompt,
            "max_results": 3,
            "project": project,
        })

        # 5. Format and return context
        context = _format_context(result)
        return json.dumps({"additionalContext": context})

    except Exception:
        # Silent failure — never block the user's prompt
        logger.exception("Error in beforeSubmitPrompt hook")
        return json.dumps({})


def _format_context(mcp_result: dict) -> str:
    """Format MCP memory_recall result into a context string.

    Args:
        mcp_result: The parsed JSON response from the MCP tool.

    Returns:
        A formatted context string, or empty string if no results.
    """
    results = mcp_result.get("results", [])
    if not results:
        return ""

    parts: list[str] = []
    for item in results:
        title = item.get("title", "")
        content = item.get("content", "")
        if title and content:
            parts.append(f"## {title}\n{content}")
        elif content:
            parts.append(content)

    return "\n\n".join(parts)


if __name__ == "__main__":
    output = main()
    print(output)
