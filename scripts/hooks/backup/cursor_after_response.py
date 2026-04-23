"""Cursor afterAgentResponse hook script — full draft pipeline.

Fires after the Cursor agent replies. Parses the agent response,
accumulates text in DraftBuffer, detects convergence signals, and
saves/updates drafts via DraftRefiner. When convergence is detected,
the draft is refined and published to TeamMemory.

Input (stdin JSON):
    conversation_id, prompt, workspace_roots, model, ...

Output (stdout JSON):
    {"action": "draft_saved"|"published"|"ok"|"error", ...}
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure scripts/hooks/ is importable for sibling modules
_HOOKS_DIR = str(Path(__file__).resolve().parent)
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)

from common import get_project_from_path, load_config, parse_hook_input  # noqa: E402
from convergence_detector import ConvergenceDetector  # noqa: E402
from draft_buffer import DraftBuffer  # noqa: E402
from draft_refiner import DraftRefiner  # noqa: E402
from shared import TMClient  # noqa: E402

logger = logging.getLogger("cursor_after_response")


async def process_response(input_data: dict[str, Any], config: dict) -> dict[str, Any]:
    """Core async logic for the afterAgentResponse hook.

    Flow:
      1. Resolve session_id and project from input data.
      2. Open DraftBuffer and check for existing pending drafts.
      3. Accumulate response text into the draft buffer.
      4. Run ConvergenceDetector on the response text.
      5. If converged and draft exists → refine_and_publish.
      6. If not converged → save_draft (create or update).

    Args:
        input_data: Parsed stdin JSON dict.
        config: Loaded pipeline config dict.

    Returns:
        Result dict with action and details.
    """
    session_id = input_data.get("conversation_id", "") or "unknown"
    response_text = input_data.get("prompt", "") or ""
    workspace_roots = input_data.get("workspace_roots", [])

    # Resolve project from workspace path
    workspace_root = workspace_roots[0] if workspace_roots else ""
    project = get_project_from_path(workspace_root, config=config) if workspace_root else None

    # If no project resolved, return early
    if project is None:
        logger.warning("No project resolved for workspace: %s", workspace_root)
        return {"action": "ok", "convergence": False, "draft_id": ""}

    # Prepare DraftBuffer path
    db_path = config.get("draft", {}).get("db_path", "~/.cache/tm-pipeline/drafts.db")
    db_path = str(Path(db_path).expanduser())
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    tm_url = config.get("tm", {}).get("base_url", "http://localhost:3900")

    buf = DraftBuffer(db_path)
    async with buf:
        # Get existing pending draft for this session
        existing = await buf.get_pending(session_id)

        # Build accumulated text
        if existing:
            accumulated = existing[0].get("content", "") + "\n" + response_text
        else:
            accumulated = response_text

        # Check convergence
        detector = ConvergenceDetector()
        converged = detector.detect_convergence(
            response_text,
            recent_tools=[],
            current_path=project,
            previous_path=None,
        )

        # Create refiner with TM client and buffer
        tm = TMClient(tm_url)
        refiner = DraftRefiner(tm, buf)

        if converged and existing:
            # Converged with existing draft → publish
            # First update buffer with latest accumulated content
            draft_id = existing[0]["id"]
            await buf.update_draft(draft_id, accumulated)

            result = await refiner.refine_and_publish(session_id)
            if result is not None:
                return {
                    "action": "published",
                    "convergence": True,
                    "draft_id": result.get("draft_id", draft_id),
                    "status": result.get("status", "published"),
                }
            else:
                # Fallback: no pending drafts found (edge case)
                return {"action": "ok", "convergence": True, "draft_id": draft_id}

        elif converged and not existing:
            # Converged on first response — save as draft and publish immediately
            title = f"Session {session_id[:8]} draft"
            tm_response = await refiner.save_draft(
                session_id, title, accumulated, project=project,
            )
            # Now try to publish it
            result = await refiner.refine_and_publish(session_id)
            if result is not None:
                return {
                    "action": "published",
                    "convergence": True,
                    "draft_id": result.get("draft_id", tm_response.get("id", "")),
                    "status": result.get("status", "published"),
                }
            return {
                "action": "draft_saved",
                "convergence": True,
                "draft_id": tm_response.get("id", ""),
            }

        else:
            # Not converged → save/update draft
            title = f"Session {session_id[:8]} draft"
            tm_response = await refiner.save_draft(
                session_id, title, accumulated, project=project,
            )
            return {
                "action": "draft_saved",
                "convergence": False,
                "draft_id": tm_response.get("id", ""),
            }


def main() -> str:
    """Entry point for the afterAgentResponse hook.

    Returns:
        JSON string with action, convergence flag, and draft_id.
    """
    try:
        input_data = parse_hook_input()
        config = load_config()
        result = asyncio.run(process_response(input_data, config))
    except Exception:
        logger.exception("Unexpected error in afterAgentResponse hook")
        return json.dumps({"action": "error", "convergence": False, "draft_id": ""})
    return json.dumps(result)


if __name__ == "__main__":
    output = main()
    print(output)
