"""Cursor afterAgentResponse hook script.

Fires after the Cursor agent replies. Parses the agent response,
detects convergence signals, and updates/creates a draft in the
local DraftBuffer. If convergence is detected, the draft is marked
for publishing (phase 2 will handle the actual MCP call).

Input (stdin JSON):
    conversation_id, prompt, workspace_roots, model, ...

Output (stdout JSON):
    {"status": "ok"|"error", "convergence": bool, "draft_id": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure scripts/hooks/ is importable for sibling modules
_HOOKS_DIR = str(Path(__file__).resolve().parent)
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)

from common import get_project_from_path, load_config, parse_hook_input  # noqa: E402
from convergence_detector import ConvergenceDetector  # noqa: E402
from draft_buffer import DraftBuffer  # noqa: E402

logger = logging.getLogger("cursor_after_response")


async def _run() -> dict:
    """Core async logic for the afterAgentResponse hook."""
    # 1. Parse stdin JSON
    data = parse_hook_input()
    conversation_id = data.get("conversation_id", "")
    prompt = data.get("prompt", "")
    workspace_roots = data.get("workspace_roots", [])

    # 2. Resolve project from workspace path
    workspace_root = workspace_roots[0] if workspace_roots else ""
    project = get_project_from_path(workspace_root) if workspace_root else None

    # 3. Detect convergence
    detector = ConvergenceDetector()
    convergence = detector.detect_convergence(prompt)

    # 4. If no project resolved, return early with ok status
    if project is None:
        logger.warning("No project resolved for workspace: %s", workspace_root)
        return {"status": "ok", "convergence": convergence, "draft_id": ""}

    # 5. Update or create draft
    config = load_config()
    db_path = config.get("draft", {}).get("db_path", "~/.cache/tm-pipeline/drafts.db")
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    buf = DraftBuffer(str(db_path))
    async with buf:
        # Check for existing pending draft for this conversation
        existing = await buf.find_pending_by_conversation(project, conversation_id)

        if existing:
            # Update existing draft
            draft_id = existing[0]["id"]
            old_content = existing[0].get("content", "")
            new_content = f"{old_content}\n{prompt}" if old_content else prompt
            await buf.update_draft(draft_id, new_content)
        else:
            # Create new draft
            draft_id = await buf.create_draft(project, conversation_id, prompt)

        # 6. If convergence detected, mark for publishing
        if convergence:
            await buf.mark_for_publishing(draft_id)

    return {"status": "ok", "convergence": convergence, "draft_id": draft_id}


def main() -> str:
    """Entry point for the afterAgentResponse hook.

    Returns:
        JSON string with status, convergence flag, and draft_id.
    """
    try:
        result = asyncio.run(_run())
    except Exception:
        logger.exception("Unexpected error in afterAgentResponse hook")
        return json.dumps({"status": "error", "convergence": False, "draft_id": ""})
    return json.dumps(result)


if __name__ == "__main__":
    output = main()
    print(output)
