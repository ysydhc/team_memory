"""OpenAI Codex CLI Stop hook — captures turn trajectory into TM drafts.db pipeline.

Fires when Codex finishes a turn. Reads assistant response from stdin payload or
falls back to parsing ~/.codex/sessions/**/rollout-<session_id>.jsonl.

Input (from Codex CLI stdin):
  {
    "session_id": "...",
    "last_assistant_message": "...",
    "cwd": "/path/to/project",
    "transcript_path": "~/.codex/sessions/.../rollout-....jsonl"
  }

Output: {} (non-blocking)
"""

from __future__ import annotations

from team_memory.hooks.core.daemon_http import post_hook_fire_and_forget
from team_memory.hooks.core.output import print_empty
from team_memory.hooks.core.stdin import parse_stdin_json
from team_memory.hooks.core.transcript import (
    parse_codex_transcript,
    resolve_codex_transcript_path,
)


def _extract_workspaces(input_data: dict) -> list[str]:
    ws = input_data.get("workspace_roots") or input_data.get("workspacePaths") or []
    if isinstance(ws, str):
        ws = [ws]
    single = input_data.get("cwd") or input_data.get("workspace") or input_data.get("workingDirectory")
    if single and single not in ws:
        ws = [single, *ws]
    return [str(p) for p in ws if p]


def main() -> None:
    input_data = parse_stdin_json()
    if not input_data:
        print_empty()
        return

    session_id = str(
        input_data.get("session_id") or input_data.get("thread_id") or input_data.get("conversation_id") or ""
    )
    workspace_roots = _extract_workspaces(input_data)
    response_text = str(
        input_data.get("last_assistant_message")
        or input_data.get("last_agent_message")
        or input_data.get("response_text")
        or ""
    )
    last_user_input = str(input_data.get("prompt") or "")
    recent_tools = input_data.get("recent_tools") or []

    transcript_path = resolve_codex_transcript_path(
        str(input_data.get("transcript_path") or input_data.get("transcriptPath") or ""),
        session_id=session_id,
    )
    if transcript_path:
        parsed = parse_codex_transcript(transcript_path)
        if not response_text:
            response_text = str(parsed.get("last_assistant_response") or "")
        if not last_user_input:
            last_user_input = str(parsed.get("last_user_input") or "")
        if not recent_tools:
            recent_tools = parsed.get("recent_tools") or []
        if not workspace_roots:
            workspace_roots = list(parsed.get("workspace_roots") or [])

    if not response_text:
        print_empty()
        return

    payload: dict = {
        "conversation_id": session_id,
        "response_text": response_text,
        "workspace_roots": workspace_roots,
        "recent_tools": recent_tools,
        "source_platform": "codex",
    }
    if last_user_input:
        payload["user_prompt"] = last_user_input
        payload["prompt"] = last_user_input

    try:
        post_hook_fire_and_forget("after_response", payload)
    except Exception:
        pass

    print_empty()


if __name__ == "__main__":
    main()
