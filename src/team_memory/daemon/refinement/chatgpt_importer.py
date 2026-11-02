"""ChatGPT session importer & two-stage value gate for TeamMemory.

Supports:
1. OpenAI official data export (`conversations.json` or `.zip` archive)
   with DAG branch linearization from `current_node` backwards to root.
2. Direct single-session JSON payload from browser userscript / webhook (`/ingest/session`).
3. Two-Stage Value Gate:
   - Stage 1: Zero-cost rule filter (removes short/non-technical/trivial Q&As).
   - Stage 2: Long-conversation topic segmentation & automatic project/tech-stack inference.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("tm_daemon.chatgpt_importer")

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_ERROR_SIGNAL_RE = re.compile(
    r"(Traceback \(most recent call last\)|Exception:|Error:|failed|errno|报错|异常|崩溃|排障|根因|死锁|超时)",
    re.IGNORECASE,
)
_TECH_SIGNAL_RE = re.compile(
    r"\b(python|typescript|javascript|golang|rust|java|sql|postgres|pgvector|redis|docker|kubernetes|"
    r"fastapi|react|vue|nextjs|mcp|agent|llm|embedding|rag|hook|daemon|git|pytest|alembic|asyncio|"
    r"架构|数据库|接口|性能|缓存|索引|并发|部署|配置|重构|设计模式|迁移|提纯)\b",
    re.IGNORECASE,
)
_REPO_OR_PATH_RE = re.compile(r"(?:[a-zA-Z0-9_-]+/){1,4}[a-zA-Z0-9_.-]+\.(?:py|ts|tsx|js|go|rs|yaml|toml|sql|md)")


def _extract_message_text(msg: dict[str, Any]) -> str:
    """Extract readable text from an OpenAI mapping message object."""
    content = msg.get("content") or {}
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, dict):
        return ""

    content_type = content.get("content_type", "text")
    parts = content.get("parts") or []
    text_chunks: list[str] = []

    for part in parts:
        if isinstance(part, str):
            text_chunks.append(part)
        elif isinstance(part, dict):
            # Multimodal text or execution output dict
            if part.get("text"):
                text_chunks.append(str(part["text"]))

    if not text_chunks and content.get("text"):
        text_chunks.append(str(content["text"]))

    joined = "\n".join(c for c in text_chunks if c).strip()
    if content_type == "code" and joined and not joined.startswith("```"):
        lang = content.get("language") or ""
        joined = f"```{lang}\n{joined}\n```"
    elif content_type == "execution_output" and joined:
        joined = f"[Tool Execution Output]:\n{joined[:600]}"
    return joined


def linearize_chatgpt_mapping(
    mapping: dict[str, Any],
    current_node: str | None = None,
) -> list[dict[str, Any]]:
    """Linearize a ChatGPT conversation DAG from `current_node` back to root.

    Only keeps the active branch accepted by the user (ignoring abandoned regenerations).
    """
    if not mapping or not isinstance(mapping, dict):
        return []

    target_id = current_node
    if not target_id or target_id not in mapping:
        # Fallback: pick node with no children or latest create_time
        leaf_candidates = [
            nid for nid, node in mapping.items() if isinstance(node, dict) and not (node.get("children") or [])
        ]
        target_id = leaf_candidates[-1] if leaf_candidates else next(iter(mapping.keys()), None)

    chain: list[dict[str, Any]] = []
    visited: set[str] = set()
    curr = target_id

    while curr and curr in mapping and curr not in visited:
        visited.add(curr)
        node = mapping[curr]
        if not isinstance(node, dict):
            break
        msg = node.get("message")
        if isinstance(msg, dict):
            metadata = msg.get("metadata") or {}
            if not metadata.get("is_visually_hidden_from_conversation"):
                author = (msg.get("author") or {}).get("role", "")
                if author in ("user", "assistant", "tool"):
                    text = _extract_message_text(msg)
                    if text:
                        role = "assistant" if author in ("assistant", "tool") else "user"
                        chain.append(
                            {
                                "role": role,
                                "author_role": author,
                                "content": text,
                                "create_time": msg.get("create_time"),
                            }
                        )
        curr = node.get("parent")

    chain.reverse()

    # Merge consecutive assistant+tool messages into cohesive assistant turns
    merged: list[dict[str, Any]] = []
    for item in chain:
        if merged and merged[-1]["role"] == item["role"] == "assistant":
            merged[-1]["content"] = f"{merged[-1]['content']}\n\n{item['content']}"
        else:
            merged.append(item)
    return merged


def evaluate_chatgpt_session_value(
    messages: list[dict[str, Any]],
    title: str = "",
) -> tuple[bool, float, str]:
    """Two-Stage Value Gate (Stage 1 rule + density scorer).

    Returns:
        (is_valuable, score, reason)
    """
    user_msgs = [m for m in messages if m.get("role") == "user"]
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not user_msgs or not asst_msgs:
        return False, 0.0, "incomplete_dialogue"

    full_text = f"{title}\n" + "\n".join(str(m.get("content") or "") for m in messages)
    total_chars = len(full_text)
    if total_chars < 300:
        return False, 0.1, "too_short"

    code_blocks = len(_CODE_BLOCK_RE.findall(full_text))
    error_hits = len(_ERROR_SIGNAL_RE.findall(full_text))
    tech_hits = len(set(m.lower() for m in _TECH_SIGNAL_RE.findall(full_text)))

    score = 0.0
    score += min(code_blocks * 0.25, 0.5)
    score += min(error_hits * 0.2, 0.4)
    score += min(tech_hits * 0.1, 0.5)
    if len(user_msgs) >= 2 and total_chars >= 800:
        score += 0.15

    if score < 0.25:
        return False, round(score, 2), "non_technical_or_trivial"

    return True, round(min(score, 1.0), 2), "valuable_engineering_session"


def infer_project_from_chatgpt_session(
    messages: list[dict[str, Any]],
    title: str = "",
    known_projects: list[str] | None = None,
    default_project: str = "",
) -> str:
    """Infer target TM project name from conversation content & known projects."""
    if default_project:
        return default_project

    full_text = f"{title}\n" + "\n".join(str(m.get("content") or "")[:1000] for m in messages[:6])
    full_lower = full_text.lower()

    if known_projects:
        for proj in known_projects:
            if proj and proj.lower() in full_lower:
                return proj

    path_match = _REPO_OR_PATH_RE.search(full_text)
    if path_match:
        first_seg = path_match.group(0).split("/")[0]
        if first_seg not in ("src", "app", "lib", "tests", "packages", "internal", "cmd"):
            return first_seg

    return "chatgpt"


def split_into_topic_segments(
    messages: list[dict[str, Any]],
    max_turns_per_segment: int = 8,
    max_chars_per_segment: int = 12000,
) -> list[list[dict[str, Any]]]:
    """Split overly long multi-topic ChatGPT sessions into focused segments for distillation."""
    if not messages:
        return []

    segments: list[list[dict[str, Any]]] = []
    current_seg: list[dict[str, Any]] = []
    current_chars = 0
    user_turn_count = 0

    for msg in messages:
        msg_len = len(str(msg.get("content") or ""))
        is_user = msg.get("role") == "user"
        if (
            is_user
            and current_seg
            and (user_turn_count >= max_turns_per_segment or (current_chars + msg_len) > max_chars_per_segment)
        ):
            segments.append(current_seg)
            current_seg = []
            current_chars = 0
            user_turn_count = 0

        current_seg.append(msg)
        current_chars += msg_len
        if is_user:
            user_turn_count += 1

    if current_seg:
        segments.append(current_seg)

    return segments


def load_chatgpt_export_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Load raw conversation dicts from `conversations.json` or OpenAI `.zip` export."""
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"ChatGPT export file not found: {p}")

    raw_data: Any
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            conv_files = [name for name in zf.namelist() if name.endswith("conversations.json")]
            if not conv_files:
                raise ValueError(f"No conversations.json found inside zip archive: {p}")
            with zf.open(conv_files[0]) as f:
                raw_data = json.loads(f.read().decode("utf-8"))
    else:
        raw_data = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(raw_data, list):
        return [c for c in raw_data if isinstance(c, dict)]
    if isinstance(raw_data, dict):
        if "conversations" in raw_data and isinstance(raw_data["conversations"], list):
            return [c for c in raw_data["conversations"] if isinstance(c, dict)]
        return [raw_data]
    return []


def normalize_chatgpt_conversation(
    conv: dict[str, Any],
    *,
    default_project: str = "",
    known_projects: list[str] | None = None,
) -> dict[str, Any]:
    """Normalize either an OpenAI export conversation dict or a direct messages payload."""
    conv_id = str(conv.get("id") or conv.get("conversation_id") or "")
    title = str(conv.get("title") or "").strip()
    update_time = conv.get("update_time") or conv.get("create_time")

    if "mapping" in conv and isinstance(conv["mapping"], dict):
        messages = linearize_chatgpt_mapping(conv["mapping"], conv.get("current_node"))
    elif "messages" in conv and isinstance(conv["messages"], list):
        messages = [
            {
                "role": "assistant" if m.get("role") in ("assistant", "tool") else "user",
                "content": str(m.get("content") or "").strip(),
            }
            for m in conv["messages"]
            if isinstance(m, dict) and str(m.get("content") or "").strip()
        ]
    else:
        messages = []

    project = infer_project_from_chatgpt_session(
        messages,
        title=title,
        known_projects=known_projects,
        default_project=default_project or str(conv.get("project") or ""),
    )

    return {
        "conversation_id": conv_id,
        "title": title,
        "update_time": update_time,
        "project": project,
        "messages": messages,
    }


async def import_chatgpt_sessions(
    *,
    buf: Any | None = None,
    file_path: str | Path | None = None,
    raw_conversations: list[dict[str, Any]] | None = None,
    project: str = "",
    known_projects: list[str] | None = None,
    since_ts: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Parse, filter, segment, and queue ChatGPT conversations into ConversationBuffer."""
    conversations = raw_conversations if raw_conversations is not None else []
    if file_path is not None:
        conversations = load_chatgpt_export_file(file_path)

    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for idx, conv in enumerate(conversations):
        norm = normalize_chatgpt_conversation(
            conv,
            default_project=project,
            known_projects=known_projects,
        )
        conv_id = norm["conversation_id"] or f"anon-{idx}"
        title = norm["title"]
        update_time = norm["update_time"]
        messages = norm["messages"]
        resolved_proj = norm["project"]

        if since_ts is not None and isinstance(update_time, (int, float)):
            if float(update_time) < since_ts:
                skipped.append({"conversation_id": conv_id, "title": title, "reason": "older_than_since"})
                continue

        valuable, score, reason = evaluate_chatgpt_session_value(messages, title=title)
        if not valuable:
            skipped.append(
                {
                    "conversation_id": conv_id,
                    "title": title,
                    "score": score,
                    "reason": reason,
                }
            )
            continue

        segments = split_into_topic_segments(messages)
        for seg_idx, seg_msgs in enumerate(segments):
            seg_val, seg_score, _ = evaluate_chatgpt_session_value(seg_msgs, title=title)
            if not seg_val and len(segments) > 1:
                continue

            suffix = f"-s{seg_idx + 1}" if len(segments) > 1 else ""
            session_id = f"chatgpt-{conv_id}{suffix}" if not conv_id.startswith("chatgpt-") else f"{conv_id}{suffix}"

            if not dry_run and buf is not None:
                existing = await buf.get_session(session_id)
                if existing and existing.get("status") in (
                    "published",
                    "needs_refinement",
                    "refining",
                    "publishing",
                ):
                    skipped.append(
                        {
                            "conversation_id": session_id,
                            "title": title,
                            "reason": f"already_{existing.get('status')}",
                        }
                    )
                    continue

                await buf.ensure_session(session_id, resolved_proj)
                for m_idx, msg in enumerate(seg_msgs):
                    role = msg.get("role", "user")
                    content = str(msg.get("content") or "")
                    if m_idx == 0 and title and role == "user" and title.lower() not in content.lower():
                        content = f"[Session Title: {title}]\n{content}"
                    await buf.add_message(session_id, role, content, project=resolved_proj)

                await buf.mark_needs_refinement(session_id)

            imported.append(
                {
                    "session_id": session_id,
                    "title": title,
                    "project": resolved_proj,
                    "score": seg_score,
                    "messages_count": len(seg_msgs),
                }
            )

    return {
        "source": "chatgpt",
        "total_conversations": len(conversations),
        "imported_segments": len(imported),
        "skipped": len(skipped),
        "sessions": imported,
        "skipped_details": skipped,
        "dry_run": dry_run,
    }
