"""Tests for IntentGate — LLM-based intent classification with circuit breaker.

Unit tests cover:
- classify() normal operation / 正常分类
- classify() fallback on LLM failure / LLM 失败时降级
- Circuit breaker: fuse after consecutive failures / 连续失败熔断
- Circuit breaker: auto-recover after cooldown / 冷却后自动恢复
- Alert triggered on fuse / 熔断时触发告警
- Context enrichment from previous message / 上下文补充
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from team_memory.daemon.integration.intent_gate import (
    FALLBACK_RESULT,
    IntentGate,
    IntentResult,
)

pytestmark = pytest.mark.unit


class TestIntentGateClassify:
    """Tests for IntentGate.classify() — normal operation."""

    @pytest.fixture
    def gate(self) -> IntentGate:
        return IntentGate(
            base_url="http://localhost:4000/v1",
            model="glm-4-flash",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_returns_experience(self, gate: IntentGate):
        """Should return 'experience' when LLM says so."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "experience"
            result = await gate.classify("之前这个 bug 怎么修的")
            assert result == IntentResult.EXPERIENCE

    @pytest.mark.asyncio
    async def test_returns_wiki(self, gate: IntentGate):
        """Should return 'wiki' when LLM says so."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "wiki"
            result = await gate.classify("API 规范是什么")
            assert result == IntentResult.WIKI

    @pytest.mark.asyncio
    async def test_returns_none(self, gate: IntentGate):
        """Should return 'none' when LLM says so."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "none"
            result = await gate.classify("把这段代码格式化一下")
            assert result == IntentResult.NONE

    @pytest.mark.asyncio
    async def test_returns_both(self, gate: IntentGate):
        """Should return 'both' when LLM says so."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "both"
            result = await gate.classify("实现用户认证模块，参考之前的规范")
            assert result == IntentResult.BOTH

    @pytest.mark.asyncio
    async def test_invalid_response_fallback(self, gate: IntentGate):
        """Invalid LLM response should fallback to 'both'."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "invalid_response_xyz"
            result = await gate.classify("some query")
            assert result == FALLBACK_RESULT


class TestIntentGateFallback:
    """Tests for IntentGate fallback on LLM failure."""

    @pytest.fixture
    def gate(self) -> IntentGate:
        return IntentGate(
            base_url="http://localhost:4000/v1",
            model="glm-4-flash",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_llm_exception_returns_fallback(self, gate: IntentGate):
        """LLM call failure should return fallback without raising."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("connection refused")
            result = await gate.classify("需要查一下之前的方案")
            assert result == FALLBACK_RESULT

    @pytest.mark.asyncio
    async def test_llm_timeout_returns_fallback(self, gate: IntentGate):
        """LLM timeout should return fallback."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = asyncio.TimeoutError()
            result = await gate.classify("查询一下 xxx")
            assert result == FALLBACK_RESULT


class TestIntentGateCircuitBreaker:
    """Tests for circuit breaker (fuse) behavior."""

    @pytest.fixture
    def gate(self) -> IntentGate:
        return IntentGate(
            base_url="http://localhost:4000/v1",
            model="glm-4-flash",
            api_key="test-key",
            fuse_threshold=3,
            fuse_cooldown_seconds=5,
        )

    @pytest.mark.asyncio
    async def test_fuse_after_consecutive_failures(self, gate: IntentGate):
        """Should enter fused state after N consecutive failures."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("fail")
            # Trigger 3 failures
            for _ in range(3):
                await gate.classify("test")
            assert gate.is_fused

    @pytest.mark.asyncio
    async def test_fused_skips_llm_call(self, gate: IntentGate):
        """When fused, should return fallback without calling LLM."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("fail")
            # Trigger fuse
            for _ in range(3):
                await gate.classify("test")

            # Reset mock to track new calls
            mock_llm.reset_mock()
            mock_llm.side_effect = None
            mock_llm.return_value = "experience"

            result = await gate.classify("should not call LLM")
            assert result == FALLBACK_RESULT
            mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_fuse_recovers_after_cooldown(self, gate: IntentGate):
        """Should recover from fused state after cooldown period."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("fail")
            # Trigger fuse
            for _ in range(3):
                await gate.classify("test")
            assert gate.is_fused

            # Simulate cooldown elapsed
            gate._fuse_triggered_at = time.monotonic() - 6  # past 5s cooldown

            mock_llm.side_effect = None
            mock_llm.return_value = "experience"

            result = await gate.classify("should work now")
            assert result == IntentResult.EXPERIENCE
            assert not gate.is_fused

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, gate: IntentGate):
        """Successful call should reset consecutive failure count."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("fail")
            # 2 failures (below threshold)
            await gate.classify("test1")
            await gate.classify("test2")
            assert gate._consecutive_failures == 2

            # Then a success
            mock_llm.side_effect = None
            mock_llm.return_value = "none"
            await gate.classify("test3")
            assert gate._consecutive_failures == 0


class TestIntentGateAlert:
    """Tests for alert/notification on fuse trigger."""

    @pytest.mark.asyncio
    async def test_alert_on_fuse(self):
        """Should call send_daemon_alert when fuse triggers."""
        gate = IntentGate(
            base_url="http://localhost:4000/v1",
            model="glm-4-flash",
            api_key="test-key",
            fuse_threshold=3,
        )

        with (
            patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch(
                "team_memory.daemon.integration.intent_gate.send_daemon_alert",
                new_callable=AsyncMock,
            ) as mock_alert,
        ):
            mock_llm.side_effect = Exception("fail")
            # Trigger fuse (3 consecutive failures)
            for _ in range(3):
                await gate.classify("test")

            mock_alert.assert_called_once()
            call_args = mock_alert.call_args
            assert "intentgate" in call_args.kwargs.get("title", "").lower()

    @pytest.mark.asyncio
    async def test_warning_at_threshold(self):
        """Should log warning at alert_threshold (3) consecutive failures."""
        gate = IntentGate(
            base_url="http://localhost:4000/v1",
            model="glm-4-flash",
            api_key="test-key",
            alert_threshold=3,
            fuse_threshold=10,
        )

        with (
            patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm,
            patch("team_memory.daemon.integration.intent_gate.logger") as mock_logger,
        ):
            mock_llm.side_effect = Exception("fail")
            for _ in range(3):
                await gate.classify("test")

            # Should have logged an error-level alert
            assert mock_logger.error.called


class TestIntentGateContext:
    """Tests for context enrichment from previous message."""

    @pytest.fixture
    def gate(self) -> IntentGate:
        return IntentGate(
            base_url="http://localhost:4000/v1",
            model="glm-4-flash",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_includes_prev_message_in_prompt(self, gate: IntentGate):
        """Should include previous message in LLM prompt when provided."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "experience"
            await gate.classify("继续", prev_message="帮我实现一个用户认证模块")

            # Verify the prompt sent to LLM includes the prev message
            call_args = mock_llm.call_args[0][0]  # first positional arg (prompt)
            assert "用户认证模块" in call_args

    @pytest.mark.asyncio
    async def test_no_prev_message(self, gate: IntentGate):
        """Should work fine without previous message context."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "none"
            result = await gate.classify("格式化代码")
            assert result == IntentResult.NONE

    @pytest.mark.asyncio
    async def test_prev_message_truncated(self, gate: IntentGate):
        """Long previous messages should be truncated."""
        with patch.object(gate, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "both"
            long_msg = "x" * 500
            await gate.classify("继续", prev_message=long_msg)

            call_args = mock_llm.call_args[0][0]
            # Should be truncated (max 200 chars for prev)
            assert len(call_args) < len(long_msg) + 500
