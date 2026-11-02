"""Antigravity PreInvocation hook — retrieves L1 memory summaries before model invocation.

Input (from Antigravity stdin):
  {
    "conversationId": "...",
    "workspacePaths": ["/path/to/workspace"],
    "transcriptPath": ".../transcript.jsonl",
    "modelName": "..."
  }

Output:
  {"injectSteps": [{"ephemeralMessage": "<L1 progressive context>"}]} or {}
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.output import print_antigravity_ephemeral_message, print_empty
from team_memory.hooks.core.stdin import parse_stdin_json
from team_memory.hooks.core.transcript import (
    parse_antigravity_transcript,
    read_last_user_from_antigravity_transcript,
    resolve_antigravity_transcript_path,
)


def _already_processed_turn(conversation_id: str, prompt: str) -> bool:
    """Return True if this (conversation_id, prompt) was already sent in the current turn."""
    if not conversation_id or not prompt:
        return False
    try:
        safe_id = "".join(c for c in conversation_id if c.isalnum() or c in "-_")[:64]
        state_path = Path(tempfile.gettempdir()) / f"tm-agy-preinv-{safe_id}.txt"
        digest = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()[:24]
        if state_path.is_file() and state_path.read_text(encoding="utf-8").strip() == digest:
            return True
        state_path.write_text(digest, encoding="utf-8")
    except Exception:
        pass
    return False


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    conversation_id = input_data.get("conversationId") or input_data.get("conversation_id") or ""
    workspace_roots = input_data.get("workspacePaths") or input_data.get("workspace_roots") or []
    if isinstance(workspace_roots, str):
        workspace_roots = [workspace_roots]

    transcript_path = resolve_antigravity_transcript_path(
        input_data.get("transcriptPath") or input_data.get("transcript_path") or "",
        conversation_id,
    )
    prompt = input_data.get("prompt") or ""
    from_transcript = False
    if not prompt and transcript_path:
        parsed = parse_antigravity_transcript(transcript_path)
        turns = parsed.get("turns") or []
        # In Antigravity, PreInvocation fires before EVERY planner step in a tool loop.
        # If the latest turn in transcript.jsonl is already an assistant step, we are
        # mid-turn (following up on a tool call), NOT starting a new user prompt.
        if turns and turns[-1].get("role") == "assistant":
            print_empty()
            return
        prompt = str(parsed.get("last_user_input") or "") or read_last_user_from_antigravity_transcript(transcript_path)
        from_transcript = True

    if not prompt:
        print_empty()
        return

    if from_transcript and _already_processed_turn(conversation_id, prompt):
        print_empty()
        return

    payload = {
        "conversation_id": conversation_id,
        "prompt": prompt,
        "workspace_roots": [str(p) for p in workspace_roots if p],
        "source_platform": "antigravity",
    }

    try:
        resp = post_hook("before_prompt", payload)
        result = resp.json()
        l1_context = result.get("l1_context") or result.get("context") or ""
        if l1_context:
            print_antigravity_ephemeral_message(l1_context)
        else:
            print_empty()
    except Exception:
        print_empty()


if __name__ == "__main__":
    main()
