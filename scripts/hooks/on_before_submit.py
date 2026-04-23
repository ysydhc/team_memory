"""on_before_submit — async beforeSubmitPrompt hook for auto-retrieval with keyword triggering.

Fires before the user's prompt is sent to the model. If the prompt
contains retrieval trigger keywords (e.g. "之前", "上次", "经验",
"remember", "previously"), the hook queries TeamMemory for relevant
past experiences and injects them as additional_context.

Flow:
  1. Parse HookInput from stdin
  2. Extract user message from prompt
  3. Check if message contains retrieval trigger keywords
  4. If triggered → call TM memory_recall with the user message
  5. Output the retrieved memories as additional_context
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

logger = logging.getLogger("on_before_submit")

RETRIEVAL_KEYWORDS: list[str] = [
    # Chinese
    "之前", "上次", "经验", "踩坑", "之前遇到过", "以前", "历史",
    # English
    "remember", "before", "previously", "earlier",
]


async def process_before_submit(
    input_data: HookInput, config: PipelineConfig
) -> None:
    """Process a beforeSubmitPrompt event — conditionally retrieve memories.

    Args:
        input_data: Parsed hook input with prompt, workspace_roots, etc.
        config: Pipeline configuration (tm_url, max_context_chars, etc.).
    """
    # 1. Extract user message
    user_message = input_data.prompt or ""

    # 2. Check trigger keywords
    if not user_message or not any(kw in user_message for kw in RETRIEVAL_KEYWORDS):
        print(json.dumps({"action": "skip", "reason": "no_trigger_keywords"}))
        return

    # 3. Resolve project
    project: str | None = None
    if input_data.workspace_roots:
        project = input_data.workspace_roots[0].split("/")[-1] or None

    # 4. Call TM memory_recall
    tm = TMClient(config.tm_url)

    try:
        results = await tm.recall(query=user_message, project=project)
    except Exception as exc:
        logger.exception("Error in beforeSubmitPrompt hook")
        print(json.dumps({"action": "error", "message": str(exc)}))
        return

    # 5. Output results
    if results and results.get("results"):
        context_str = _format_context(results)
        capped = context_str[: config.max_context_chars]
        print(json.dumps({
            "action": "retrieved",
            "query": user_message[:100],
            "result_count": len(results["results"]),
            "additional_context": capped,
        }))
    else:
        print(json.dumps({"action": "no_results", "query": user_message[:100]}))


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
    import asyncio

    raw = sys.stdin.read()
    data = json.loads(raw) if raw.strip() else {}
    hook_input = HookInput.from_dict(data)
    cfg = PipelineConfig()
    asyncio.run(process_before_submit(hook_input, cfg))
