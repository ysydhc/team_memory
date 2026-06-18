"""Transcript file reader — shared utility for reading JSONL transcripts.

Supports the Cursor/Claude Code transcript format:
  {"role": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}
"""

from __future__ import annotations

import json
import os


def read_last_assistant_from_transcript(transcript_path: str) -> str:
    """Read the last assistant message from a JSONL transcript file.

    Args:
        transcript_path: Path to a .jsonl transcript file.

    Returns:
        The text content of the last assistant message, or "" if not found.
    """
    if not transcript_path or not os.path.isfile(transcript_path):
        return ""
    try:
        last_assistant = ""
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("role") == "assistant":
                    msg = entry.get("message", {})
                    content_parts = msg.get("content", [])
                    if isinstance(content_parts, list):
                        texts = [p.get("text", "") for p in content_parts if p.get("type") == "text"]
                        if texts:
                            last_assistant = "\n".join(texts)
                    elif isinstance(content_parts, str):
                        last_assistant = content_parts
        return last_assistant
    except Exception:
        return ""
