"""Cursor sessionStart hook script.

Fires when a new Cursor agent session starts. Retrieves relevant
project context from TeamMemory so the agent has prior experience
available from the very beginning.

Input (stdin JSON):
    {"workspace_roots": [...], "conversation_id": "..."}

Output (stdout JSON):
    On success: {"additionalContext": "<retrieved context>"}
    On error:   {"status": "error", "message": "..."}
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

logger = logging.getLogger("cursor_session_start")


def main() -> str:
    """Entry point for the sessionStart hook.

    Returns:
        JSON string with additionalContext on success, or error status.
    """
    try:
        # 1. Parse stdin JSON
        data = parse_hook_input()
        workspace_roots = data.get("workspace_roots", [])

        # 2. Resolve project from workspace path
        workspace_root = workspace_roots[0] if workspace_roots else ""
        project = get_project_from_path(workspace_root) if workspace_root else None

        if project is None:
            return json.dumps({"additionalContext": ""})

        # 3. Call TM memory_context MCP tool
        result = call_mcp_tool("memory_context", {"project": project})

        # 4. Extract context from result
        context = _format_context(result)
        return json.dumps({"additionalContext": context})

    except Exception as exc:
        logger.exception("Error in sessionStart hook")
        return json.dumps({"status": "error", "message": str(exc)})


def _format_context(mcp_result: dict) -> str:
    """Format MCP memory_context result into a context string.

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
