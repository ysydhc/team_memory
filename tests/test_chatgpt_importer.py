"""Unit tests for ChatGPT Session Distillation:
- DAG mapping tree linearization (active branch extraction, skipping hidden/system nodes)
- Two-stage value gate (L1 zero-cost heuristic rejecting translation/polish/chitchat, accepting engineering/debugging)
- Topic segmentation for long multi-topic conversations
- Project auto-inference from file paths and known projects
- Export file parsing (.json and .zip) and ConversationBuffer ingestion
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from team_memory.daemon.refinement.chatgpt_importer import (
    evaluate_chatgpt_session_value,
    import_chatgpt_sessions,
    infer_project_from_chatgpt_session,
    linearize_chatgpt_mapping,
    load_chatgpt_export_file,
    split_into_topic_segments,
)

pytestmark = pytest.mark.unit


def _make_chatgpt_conversation(
    *,
    conv_id: str,
    title: str,
    user_text: str,
    assistant_text: str,
    abandoned_assistant_text: str | None = None,
    create_time: float = 1729600000.0,
) -> dict:
    """Build a realistic OpenAI ChatGPT conversations.json entry with optional regenerated branch."""
    mapping = {
        "root": {
            "id": "root",
            "parent": None,
            "children": ["sys-1"],
            "message": None,
        },
        "sys-1": {
            "id": "sys-1",
            "parent": "root",
            "children": ["user-1"],
            "message": {
                "id": "sys-1",
                "author": {"role": "system"},
                "content": {"content_type": "text", "parts": ["You are ChatGPT."]},
                "metadata": {"is_visually_hidden_from_conversation": True},
            },
        },
        "user-1": {
            "id": "user-1",
            "parent": "sys-1",
            "children": ["asst-abandoned", "asst-active"] if abandoned_assistant_text else ["asst-active"],
            "message": {
                "id": "user-1",
                "author": {"role": "user"},
                "create_time": create_time,
                "content": {"content_type": "text", "parts": [user_text]},
                "metadata": {},
            },
        },
        "asst-active": {
            "id": "asst-active",
            "parent": "user-1",
            "children": [],
            "message": {
                "id": "asst-active",
                "author": {"role": "assistant"},
                "create_time": create_time + 10,
                "content": {"content_type": "text", "parts": [assistant_text]},
                "metadata": {"model_slug": "gpt-4o"},
            },
        },
    }
    if abandoned_assistant_text:
        mapping["asst-abandoned"] = {
            "id": "asst-abandoned",
            "parent": "user-1",
            "children": [],
            "message": {
                "id": "asst-abandoned",
                "author": {"role": "assistant"},
                "create_time": create_time + 5,
                "content": {"content_type": "text", "parts": [abandoned_assistant_text]},
                "metadata": {"model_slug": "gpt-4o"},
            },
        }

    return {
        "id": conv_id,
        "title": title,
        "create_time": create_time,
        "current_node": "asst-active",
        "mapping": mapping,
    }


class TestChatGPTDAGLinearizer:
    def test_linearize_follows_current_node_active_branch_only(self) -> None:
        conv = _make_chatgpt_conversation(
            conv_id="c-1",
            title="Debug asyncpg connection pool",
            user_text="Why does asyncpg raise InterfaceError: cannot perform operation: another operation is in progress?",
            assistant_text=(
                "Because a single asyncpg Connection cannot be shared concurrently across coroutines.\n"
                "```python\nasync with pool.acquire() as conn:\n    await conn.fetch('SELECT 1')\n```"
            ),
            abandoned_assistant_text="This is an old abandoned regeneration that should be ignored.",
        )

        turns = linearize_chatgpt_mapping(conv["mapping"], conv["current_node"])
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert "InterfaceError" in turns[0]["content"]
        assert turns[1]["role"] == "assistant"
        assert "pool.acquire()" in turns[1]["content"]
        assert "abandoned" not in turns[1]["content"]


class TestChatGPTValueGate:
    def test_rejects_translation_and_chitchat(self) -> None:
        turns_translate = [
            {"role": "user", "content": "帮我翻译这段英文邮件并润色语气：" + "Hello team, please review the PR. " * 10},
            {"role": "assistant", "content": "各位同事好，请帮忙审阅一下这个 PR。" * 10},
        ]
        passed, score, reason = evaluate_chatgpt_session_value(turns_translate, title="翻译英文邮件")
        assert passed is False
        assert score < 0.25
        assert reason == "non_technical_or_trivial"

    def test_accepts_debugging_and_architecture_discussion(self) -> None:
        turns_tech = [
            {
                "role": "user",
                "content": (
                    "在 `team_doc/src/team_memory/server.py` 中出现 Traceback:\n"
                    "RuntimeError: Task attached to a different loop，应该怎么排查和架构改造？"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "根因是全局 AsyncEngine 绑定了旧的 event loop。解决方案是在每个生命周期或 fixture 内重建连接池：\n"
                    "```python\nengine = create_async_engine(url, poolclass=NullPool)\n```\n"
                    "这样可以避免跨事件循环复用连接导致的报错。"
                ),
            },
        ]
        passed, score, reason = evaluate_chatgpt_session_value(turns_tech, title="Fix AsyncEngine event loop error")
        assert passed is True
        assert score >= 0.25
        assert reason == "valuable_engineering_session"
        assert infer_project_from_chatgpt_session(turns_tech, title="Fix AsyncEngine") == "team_doc"


class TestTopicSegmentationAndImport:
    def test_split_into_topic_segments(self) -> None:
        turns = [
            {"role": "user", "content": "PostgreSQL pgvector HNSW index query latency optimization"},
            {"role": "assistant", "content": "Use HNSW index with m=16 and ef_construction=64 in PostgreSQL pgvector."},
            {"role": "user", "content": "PostgreSQL pgvector cosine distance operator <=> index scan"},
            {"role": "assistant", "content": "Ensure vector_cosine_ops is specified on the HNSW index in PostgreSQL."},
            {"role": "user", "content": "React Tailwind CSS keyframe modal fade-in animation transition jitter"},
            {"role": "assistant", "content": "Use transform opacity GPU acceleration in Tailwind CSS config."},
            {"role": "user", "content": "React framer-motion AnimatePresence exit animation layout shift"},
            {"role": "assistant", "content": "Wrap modal component in AnimatePresence mode='wait' in React."},
        ]
        segments = split_into_topic_segments(turns, max_turns_per_segment=2)
        assert len(segments) == 2
        assert len(segments[0]) == 4
        assert len(segments[1]) == 4

    @pytest.mark.asyncio
    async def test_parse_zip_and_import_into_buffer(self, tmp_path: Path) -> None:
        high_val_conv = _make_chatgpt_conversation(
            conv_id="conv-high-1",
            title="修复 FastAPI SQLAlchemy 连接池泄漏",
            user_text=(
                "我们的 `team_doc/src/team_memory/storage/database.py` "
                "在高并发压测下报错 TimeoutError QueuePool limit overflow，如何排查？"
            ),
            assistant_text=(
                "这是典型的连接未归还或长事务占用连接池问题。建议配置 `pool_pre_ping=True` 并使用 `async with session_maker()`：\n"
                "```python\nasync with async_session() as session:\n    result = await session.execute(stmt)\n```\n"
                "同时排查在持有 session 期间是否进行了慢速外部 LLM HTTP 调用。"
            ),
        )
        low_val_conv = _make_chatgpt_conversation(
            conv_id="conv-low-1",
            title="Translate email",
            user_text="请把下面这段话翻译成英文：祝大家周末愉快",
            assistant_text="Wishing everyone a happy weekend!",
        )

        zip_path = tmp_path / "chatgpt_export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", json.dumps([high_val_conv, low_val_conv], ensure_ascii=False))

        loaded = load_chatgpt_export_file(zip_path)
        assert len(loaded) == 2

        mock_buf = AsyncMock()
        mock_buf.get_session = AsyncMock(return_value=None)
        mock_buf.ensure_session = AsyncMock()
        mock_buf.add_message = AsyncMock()
        mock_buf.mark_needs_refinement = AsyncMock()

        res = await import_chatgpt_sessions(buf=mock_buf, file_path=zip_path)
        assert res["total_conversations"] == 2
        assert res["imported_segments"] == 1
        assert res["skipped"] == 1
        assert res["sessions"][0]["project"] == "team_doc"
        mock_buf.ensure_session.assert_awaited_once_with("chatgpt-conv-high-1", "team_doc")
        mock_buf.mark_needs_refinement.assert_awaited_once_with("chatgpt-conv-high-1")
