"""Antigravity Stop hook — distills session trajectory when the agent finishes a turn.

Input (from Antigravity stdin):
  {
    "conversationId": "...",
    "workspacePaths": ["/path/to/workspace"],
    "transcriptPath": ".../transcript.jsonl",
    "terminationReason": "NO_TOOL_CALL",
    "fullyIdle": true
  }

Output:
  {"decision": "allow"}
"""

from __future__ import annotations

from team_memory.hooks.core.daemon_http import post_hook
from team_memory.hooks.core.mcp_callback import send_mcp_callback
from team_memory.hooks.core.output import print_antigravity_stop
from team_memory.hooks.core.stdin import parse_stdin_json
from team_memory.hooks.core.transcript import (
    parse_antigravity_transcript,
    resolve_antigravity_transcript_path,
)


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_antigravity_stop("allow")
        return

    conversation_id = input_data.get("conversationId") or input_data.get("conversation_id") or ""
    workspace_roots = input_data.get("workspacePaths") or input_data.get("workspace_roots") or []
    if isinstance(workspace_roots, str):
        workspace_roots = [workspace_roots]

    transcript_path = resolve_antigravity_transcript_path(
        input_data.get("transcriptPath") or input_data.get("transcript_path") or "",
        conversation_id,
    )
    response_text = input_data.get("response_text") or input_data.get("last_assistant_message") or ""
    last_user_input = input_data.get("prompt") or ""
    recent_tools = input_data.get("recent_tools") or []
    mcp_recalls: list[dict] = []

    if transcript_path:
        parsed = parse_antigravity_transcript(transcript_path)
        if not response_text:
            response_text = str(parsed.get("last_assistant_response") or "")
        if not last_user_input:
            last_user_input = str(parsed.get("last_user_input") or "")
        if not recent_tools:
            recent_tools = parsed.get("recent_tools") or []
        mcp_recalls = parsed.get("last_turn_mcp_recalls") or (parsed.get("mcp_recalls") or [])[-3:]

    if not response_text:
        print_antigravity_stop("allow")
        return

    if conversation_id and mcp_recalls:
        for rec in mcp_recalls:
            s_log_id = str(rec.get("search_log_id") or "")
            if s_log_id:
                send_mcp_callback(
                    conversation_id,
                    f"step-{rec.get('step_index', 0)}",
                    s_log_id,
                    "memory_recall",
                    rec.get("result_ids") or [],
                )

    payload = {
        "conversation_id": conversation_id,
        "prompt": last_user_input or response_text,
        "response_text": response_text,
        "workspace_roots": [str(p) for p in workspace_roots if p],
        "recent_tools": recent_tools,
        "source_platform": "antigravity",
        "termination_reason": input_data.get("terminationReason", ""),
    }

    try:
        # Also record user prompt if we parsed it from transcript and before_prompt wasn't used
        if last_user_input:
            payload["user_prompt"] = last_user_input
        post_hook("after_response", payload)
    except Exception:
        pass

    print_antigravity_stop("allow")


if __name__ == "__main__":
    main()
