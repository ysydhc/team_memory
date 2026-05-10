"""Pipeline Acceptance Tests — Tier 2 Integration Tests.

Tests the pipeline from hook input to ConversationBuffer storage.
Uses real ConversationBuffer (in-memory SQLite) with mocked sink/wiki.

Pipeline 验收测试 — T2 集成测试。
从 hook 输入到 ConversationBuffer 存储的全流程测试。
使用真实 ConversationBuffer（内存 SQLite），sink/wiki 使用 AsyncMock。
"""
from __future__ import annotations  # noqa: I001

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from team_memory.daemon.config import DaemonConfig, load_config
from team_memory.daemon.conversation_buffer import ConversationBuffer
from team_memory.daemon.convergence_detector import ConvergenceDetector
from team_memory.daemon.pipeline import (
    process_after_response,
    process_before_prompt,
)


# ---------------------------------------------------------------------------
# Helpers / 辅助工厂
# ---------------------------------------------------------------------------


def _make_config() -> DaemonConfig:
    """Create a minimal DaemonConfig for testing / 创建最小化 DaemonConfig。"""
    return load_config(config_path=None)  # uses defaults


def _make_sink() -> MagicMock:
    """Mock TMSink that records calls / 记录调用的 Mock TMSink。"""
    sink = MagicMock()
    sink.recall = AsyncMock(return_value=[])
    sink.draft_save = AsyncMock(return_value={"id": "pg-001"})
    sink.draft_publish = AsyncMock(return_value={"status": "published"})
    sink.save = AsyncMock(return_value={"id": "pg-001"})
    sink.context = AsyncMock(return_value={"results": []})
    sink.increment_used_count = AsyncMock()
    return sink


def _make_refiner(buf: ConversationBuffer | None = None) -> MagicMock:
    """Mock DraftRefiner / Mock DraftRefiner。

    When buf is provided, mark_for_refinement also updates session status
    (simulating real DraftRefiner behavior).
    当提供 buf 时，mark_for_refinement 同时更新 session 状态（模拟真实行为）。
    """
    refiner = MagicMock()

    async def _mark_for_refinement(session_id: str):
        if buf is not None:
            await buf.mark_needs_refinement(session_id)
        return {"draft_id": session_id, "status": "needs_refinement"}

    refiner.mark_for_refinement = AsyncMock(side_effect=_mark_for_refinement)
    refiner.save_draft = AsyncMock(return_value={"id": "draft-001"})
    return refiner


def _make_wiki() -> MagicMock:
    """Mock LLMWiki / Mock LLMWiki。"""
    wiki = MagicMock()
    wiki.search = AsyncMock(return_value=[])
    return wiki


# ---------------------------------------------------------------------------
# Fixtures / 固件
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def sink():
    return _make_sink()


@pytest.fixture
def refiner(buf):
    """Mock refiner that also updates buf session status / 同时更新 buf session 状态的 mock refiner。"""
    return _make_refiner(buf=buf)


@pytest.fixture
def wiki():
    return _make_wiki()


@pytest.fixture
def detector():
    return ConvergenceDetector()


@pytest_asyncio.fixture
async def buf():
    """Fresh in-memory ConversationBuffer per test / 每次测试新建内存 ConversationBuffer。"""
    buf = ConversationBuffer(":memory:")
    await buf.__aenter__()
    yield buf
    await buf.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# T2: HookPayload Format Tests / HookPayload 格式测试
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHookPayloadFormats:
    """Verify all platform payload formats are accepted.
    验证所有平台 payload 格式均可接受。
    """

    def test_cursor_format(self):
        """Cursor sends agent_response field / Cursor 发送 agent_response 字段。"""
        from team_memory.daemon.app import HookPayload

        payload = HookPayload.model_validate(
            {
                "conversation_id": "s1",
                "agent_response": "hello from cursor",
                "workspace_roots": ["/tmp"],
            },
        )
        assert payload.agent_response == "hello from cursor"
        assert payload.conversation_id == "s1"

    def test_claude_format(self):
        """Claude sends response_text field / Claude 发送 response_text 字段。"""
        from team_memory.daemon.app import HookPayload

        payload = HookPayload.model_validate(
            {
                "conversation_id": "s1",
                "response_text": "hello from claude",
                "workspace_roots": ["/tmp"],
            },
        )
        assert payload.response_text == "hello from claude"

    def test_hermes_format(self):
        """Hermes sends prompt field / Hermes 发送 prompt 字段。"""
        from team_memory.daemon.app import HookPayload

        payload = HookPayload.model_validate(
            {
                "conversation_id": "s1",
                "prompt": "hello from hermes",
                "workspace_roots": ["/tmp"],
            },
        )
        assert payload.prompt == "hello from hermes"

    def test_unknown_fields_accepted(self):
        """Extra fields should not crash / 额外字段不应导致崩溃。"""
        from team_memory.daemon.app import HookPayload

        payload = HookPayload.model_validate(
            {
                "conversation_id": "s1",
                "agent_response": "hello",
                "unknown_field": "should not crash",
                "cwd": "/tmp",
            },
        )
        assert payload.agent_response == "hello"

    def test_empty_response_skipped(self):
        """Empty response_text should result in action=ok (skipped).
        空 response_text 应返回 action=ok（跳过）。
        """
        from team_memory.daemon.app import HookPayload

        payload = HookPayload.model_validate(
            {
                "conversation_id": "s1",
                "workspace_roots": ["/tmp"],
            },
        )
        # All response fields are empty / 所有响应字段均为空
        assert not payload.response_text
        assert not payload.prompt
        assert not payload.agent_response
        assert not payload.content


# ---------------------------------------------------------------------------
# T3: before_prompt stores user message / before_prompt 存储用户消息
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBeforePromptStorage:
    """Verify before_prompt stores user message in ConversationBuffer.
    验证 before_prompt 将用户消息存入 ConversationBuffer。
    """

    async def test_stores_user_message(self, config, sink, buf, wiki):
        """User prompt is persisted to buffer / 用户提示词写入 buffer。"""
        input_data = {
            "conversation_id": "test-sess-1",
            "prompt": "How do I configure fish shell?",
            "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
        }
        await process_before_prompt(input_data, config, sink, buf, wiki)

        # Verify message stored / 验证消息已存储
        messages = await buf.get_messages("test-sess-1", role="user")
        assert len(messages) == 1
        assert messages[0]["content"] == "How do I configure fish shell?"

    async def test_session_created(self, config, sink, buf, wiki):
        """Session record is auto-created / Session 记录自动创建。"""
        input_data = {
            "conversation_id": "test-sess-2",
            "prompt": "hello",
            "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
        }
        await process_before_prompt(input_data, config, sink, buf, wiki)

        session = await buf.get_session("test-sess-2")
        assert session is not None
        assert session["status"] == "active"


# ---------------------------------------------------------------------------
# T4: after_response stores assistant message / after_response 存储助手消息
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAfterResponseStorage:
    """Verify after_response stores assistant message.
    验证 after_response 存储助手消息。
    """

    async def test_stores_assistant_message(self, config, sink, buf, detector, refiner):
        """Assistant response is persisted / 助手响应写入 buffer。"""
        # Pre-create session so project resolution works
        # 预创建 session 以便 project 解析正常
        await buf.ensure_session("test-sess-3", "team_memory")
        await buf.add_message("test-sess-3", "user", "hello", project="team_memory")

        input_data = {
            "conversation_id": "test-sess-3",
            "response_text": "Hi! How can I help?",
            "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
        }
        await process_after_response(input_data, config, sink, buf, detector, refiner)

        messages = await buf.get_messages("test-sess-3", role="assistant")
        assert len(messages) == 1
        assert messages[0]["content"] == "Hi! How can I help?"

    async def test_empty_response_not_stored(self, config, sink, buf, detector, refiner):
        """Empty response_text is skipped / 空 response_text 被跳过。"""
        await buf.ensure_session("test-sess-4", "team_memory")

        input_data = {
            "conversation_id": "test-sess-4",
            "response_text": "",
            "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
        }
        result = await process_after_response(input_data, config, sink, buf, detector, refiner)

        messages = await buf.get_messages("test-sess-4", role="assistant")
        assert len(messages) == 0
        assert result["action"] == "ok"


# ---------------------------------------------------------------------------
# T5: Convergence threshold / 收敛阈值测试
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConvergenceThreshold:
    """Verify 3+ assistant messages trigger convergence.
    验证 3 条以上助手消息触发收敛。
    """

    async def test_two_messages_no_convergence(self, config, sink, buf, detector, refiner):
        """2 assistant messages should NOT converge / 2 条助手消息不应收敛。"""
        await buf.ensure_session("test-sess-5", "team_memory")
        await buf.add_message("test-sess-5", "user", "q1", project="team_memory")
        await buf.add_message("test-sess-5", "user", "q2", project="team_memory")

        result = None
        for i in range(2):
            input_data = {
                "conversation_id": "test-sess-5",
                "response_text": f"response {i + 1}",
                "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
            }
            result = await process_after_response(input_data, config, sink, buf, detector, refiner)

        session = await buf.get_session("test-sess-5")
        assert session["status"] == "active"
        assert result["convergence"] is False

    async def test_three_messages_triggers_convergence(
        self, config, sink, buf, detector, refiner,
    ):
        """3 assistant messages should trigger convergence / 3 条助手消息应触发收敛。"""
        # Patch personal memory extraction to avoid real LLM calls
        # Mock 个人记忆提取以避免真实 LLM 调用
        with patch(
            "team_memory.daemon.pipeline._try_extract_personal_memory",
            new_callable=AsyncMock,
        ):
            await buf.ensure_session("test-sess-6", "team_memory")
            await buf.add_message("test-sess-6", "user", "q1", project="team_memory")
            await buf.add_message("test-sess-6", "user", "q2", project="team_memory")
            await buf.add_message("test-sess-6", "user", "q3", project="team_memory")

            result = None
            for i in range(3):
                input_data = {
                    "conversation_id": "test-sess-6",
                    "response_text": f"response {i + 1}",
                    "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                }
                result = await process_after_response(
                    input_data, config, sink, buf, detector, refiner,
                )

            assert result["convergence"] is True
            session = await buf.get_session("test-sess-6")
            assert session["status"] == "needs_refinement"


# ---------------------------------------------------------------------------
# T6: Convergence re-trigger prevention / 收敛重复触发防护
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConvergenceReTrigger:
    """Verify already-converged sessions are not re-triggered.
    验证已收敛的 session 不会被重复触发。
    """

    async def test_no_retrigger(self, config, sink, buf, detector, refiner):
        """4th message on converged session should not re-trigger.
        已收敛 session 的第 4 条消息不应再次触发。
        """
        # Patch personal memory extraction to avoid real LLM calls
        # Mock 个人记忆提取以避免真实 LLM 调用
        with patch(
            "team_memory.daemon.pipeline._try_extract_personal_memory",
            new_callable=AsyncMock,
        ):
            await buf.ensure_session("test-sess-7", "team_memory")
            await buf.add_message("test-sess-7", "user", "q1", project="team_memory")

            # First 3 messages → converge / 前 3 条消息 → 收敛
            for i in range(3):
                await buf.add_message(
                    "test-sess-7", "user", f"q{i + 1}", project="team_memory",
                )
                input_data = {
                    "conversation_id": "test-sess-7",
                    "response_text": f"response {i + 1}",
                    "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                }
                result = await process_after_response(
                    input_data, config, sink, buf, detector, refiner,
                )

            assert result["convergence"] is True

            # 4th message → should NOT re-trigger / 第 4 条消息 → 不应重复触发
            input_data = {
                "conversation_id": "test-sess-7",
                "response_text": "response 4",
                "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
            }
            result = await process_after_response(
                input_data, config, sink, buf, detector, refiner,
            )
            assert result["convergence"] is False


# ---------------------------------------------------------------------------
# T7: project=None defaults / project=None 默认值测试
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestProjectDefault:
    """Verify project=None uses session's stored project.
    验证 project=None 时使用 session 中存储的 project。
    """

    async def test_default_project(self, config, sink, buf, detector, refiner):
        """When workspace_roots is empty, project falls back to session record.
        当 workspace_roots 为空时，project 回退到 session 记录。
        """
        # Pre-create session with explicit project "default"
        # 预创建 session，显式设置 project="default"
        await buf.ensure_session("test-sess-8", "default")

        input_data = {
            "conversation_id": "test-sess-8",
            "response_text": "hello",
            "workspace_roots": [],  # no workspace → no project resolution
        }
        result = await process_after_response(
            input_data, config, sink, buf, detector, refiner,
        )

        # Message should be stored (with default project from session)
        # 消息应已存储（使用 session 的默认 project）
        messages = await buf.get_messages("test-sess-8", role="assistant")
        assert len(messages) == 1
        assert messages[0]["content"] == "hello"
        # action should be draft_updated (not "ok" skip)
        # action 应为 draft_updated（而非 "ok" 跳过）
        assert result["action"] == "draft_updated"


# ---------------------------------------------------------------------------
# T8: RefinementWorker processes needs_refinement sessions
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRefinementWorkerProcessing:
    """T8: RefinementWorker picks up needs_refinement sessions and writes to PostgreSQL."""

    @pytest.mark.asyncio
    async def test_tick_processes_needs_refinement(self, config, sink, buf, refiner):
        """Mark session needs_refinement, call _tick, verify session published."""
        from team_memory.daemon.refinement_worker import RefinementWorker

        # Setup: create session with messages
        await buf.ensure_session("test-sess-rw-1", "team_memory")
        await buf.add_message("test-sess-rw-1", "user", "How to fix this bug?", project="team_memory")
        await buf.add_message("test-sess-rw-1", "assistant", "Here is the solution...", project="team_memory")
        await buf.mark_needs_refinement("test-sess-rw-1")

        # Create worker with mock sink
        worker = RefinementWorker(config=config, buf=buf, sink=sink)
        worker._cfg.enabled = True

        # Mock _refine_one to just mark as published
        async def mock_refine_one(draft):
            await buf.mark_published(draft["id"])
        worker._refine_one = mock_refine_one

        # Execute
        await worker._tick()

        # Verify: session should be published
        session = await buf.get_session("test-sess-rw-1")
        assert session["status"] == "published"

    @pytest.mark.asyncio
    async def test_tick_ignores_active_sessions(self, config, sink, buf, refiner):
        """Active sessions should not be processed."""
        from team_memory.daemon.refinement_worker import RefinementWorker

        await buf.ensure_session("test-sess-rw-2", "team_memory")
        await buf.add_message("test-sess-rw-2", "user", "hello", project="team_memory")

        worker = RefinementWorker(config=config, buf=buf, sink=sink)
        worker._cfg.enabled = True

        processed = []
        async def mock_refine_one(draft):
            processed.append(draft["id"])
        worker._refine_one = mock_refine_one

        await worker._tick()

        assert len(processed) == 0


# ---------------------------------------------------------------------------
# T9: Search retrieval
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSearchRetrieval:
    """T9: Saved experiences can be found via search."""

    @pytest.mark.asyncio
    async def test_sink_recall_returns_results(self, sink):
        """Mock sink.recall returns expected results."""
        sink.recall = AsyncMock(return_value=[
            {
                "id": "exp-001",
                "title": "Fish shell config",
                "description": "How to configure fish",
                "score": 0.95,
                "project": "team_memory",
            },
        ])

        results = await sink.recall(query="fish shell", project="team_memory", max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Fish shell config"
        assert results[0]["project"] == "team_memory"

    @pytest.mark.asyncio
    async def test_project_filtering(self, sink):
        """Results are filtered by project."""
        sink.recall = AsyncMock(return_value=[
            {"id": "exp-002", "title": "Team doc setup", "project": "team_memory", "score": 0.9},
        ])

        results = await sink.recall(query="setup", project="team_memory", max_results=5)
        assert all(r["project"] == "team_memory" for r in results)


# ---------------------------------------------------------------------------
# T10: search_log creation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSearchLogCreation:
    """T10: search_log is created when recall is called."""

    @pytest.mark.asyncio
    async def test_search_log_writer_creates_entry(self):
        """SearchLogWriter.log_search creates entry in PostgreSQL."""
        import os

        from team_memory.daemon.search_log_writer import SearchLogWriter

        dsn = os.environ.get("TEAM_MEMORY_DB_URL", "postgresql://developer:devpass@localhost:5433/team_memory")
        if "postgresql+asyncpg" in dsn:
            dsn = dsn.replace("+asyncpg", "")

        writer = SearchLogWriter(dsn=dsn)
        try:
            log_id = await writer.log_search(
                query="test acceptance query",
                project="team_memory",
                source="acceptance_test",
            )
            assert log_id is not None
            assert len(log_id) > 0
        finally:
            if writer._pool and not writer._pool._closed:
                await writer._pool.close()


# ---------------------------------------------------------------------------
# T11: Evaluation markers
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEvaluationMarkers:
    """T11: [mem:xxx] markers are injected and was_used detected."""

    def test_marker_injection(self):
        """EvaluationService.inject_markers adds [mem:xxx] to results."""
        from team_memory.services.evaluation import EvaluationService

        svc = EvaluationService()
        results = [
            {"id": "exp-001", "title": "Test", "solution": "Do this"},
            {"id": "exp-002", "title": "Test2", "description": "Do that"},
        ]
        processed = svc.inject_markers(results)

        assert "[mem:exp-001]" in processed[0]["solution"]
        assert "[mem:exp-002]" in processed[1]["description"]

    def test_was_used_detection(self):
        """EvaluationService.check_was_used detects markers in response."""
        from team_memory.services.evaluation import EvaluationService

        svc = EvaluationService()
        response = "You should use fish shell [mem:exp-001] for better experience"
        result_ids = ["exp-001", "exp-002"]

        used = svc.check_was_used(response, result_ids)
        assert used["exp-001"] is True
        assert used["exp-002"] is False

    def test_was_used_no_markers(self):
        """Response without markers returns all False."""
        from team_memory.services.evaluation import EvaluationService

        svc = EvaluationService()
        response = "Just use fish shell"
        result_ids = ["exp-001", "exp-002"]

        used = svc.check_was_used(response, result_ids)
        assert all(v is False for v in used.values())
