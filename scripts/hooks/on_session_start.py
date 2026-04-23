"""on_session_start — async sessionStart hook for auto-retrieval.

Fires when a new agent session starts. Retrieves relevant project
context from TeamMemory so the agent has prior experience available
from the very beginning.

Flow:
  1. Parse HookInput from stdin
  2. Extract project from workspace_roots
  3. Call TM memory_context for project memories
  4. Output the memories as additional_context (for Cursor hooks)
     or print for manual use
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

from shared import HookInput, PipelineConfig, TMClient  # noqa: E402

logger = logging.getLogger("on_session_start")


async def process_session_start(
    input_data: HookInput, config: PipelineConfig
) -> None:
    """Process a sessionStart event — fetch project context from TM.

    Args:
        input_data: Parsed hook input with workspace_roots, etc.
        config: Pipeline configuration (tm_url, max_context_chars, etc.).
    """
    # 1. Extract project from workspace_roots
    project: str | None = None
    if input_data.workspace_roots:
        project = input_data.workspace_roots[0].split("/")[-1] or None

    # 2. Create TMClient and fetch context
    tm = TMClient(config.tm_url)

    try:
        context = await tm.get_context(project=project)
    except Exception as exc:
        logger.exception("Error fetching context in sessionStart hook")
        print(json.dumps({"action": "error", "project": project, "message": str(exc)}))
        return

    # 3. Output context or no-context result
    if context and context.get("results"):
        context_str = _format_context(context)
        capped = context_str[: config.max_context_chars]
        print(json.dumps({
            "action": "context_injected",
            "project": project,
            "context_length": len(context_str),
            "additional_context": capped,
        }))
    else:
        print(json.dumps({"action": "no_context", "project": project}))


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
    import asyncio

    raw = sys.stdin.read()
    data = json.loads(raw) if raw.strip() else {}
    hook_input = HookInput.from_dict(data)
    cfg = PipelineConfig()
    asyncio.run(process_session_start(hook_input, cfg))
