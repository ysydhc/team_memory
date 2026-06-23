"""Unit tests for the rule-based query noise classifier.

The classifier must be CONSERVATIVE: flag obvious noise, never flag a genuine
(even if short) retrieval query. False positives hide real searches, which is
worse than letting a noise row slip into stats.
"""

from __future__ import annotations

import pytest

from team_memory.daemon.refinement.query_noise import NoiseReason, classify_noise_rule


@pytest.mark.unit
class TestFlaggedAsNoise:
    """Inputs that SHOULD be flagged as rule-level noise."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "",
            "   ",
            "Kill test 1",
            "kill test",
            "Pub kill 2",
            "pub kill 3",
            "/memory_save foo",
            "/feature-design-doc",
            "好的",
            "可以",
            "ok",
            "yes",
            "继续",
            "下一步",
            "出错了,重试",
            "重试",
            "1. 先做A;2. 再做B",
            "1、改配置 2、跑测试",
            "1) clean 2) rebuild",
        ],
    )
    def test_is_noise(self, prompt):
        assert classify_noise_rule(prompt) is NoiseReason.RULE


@pytest.mark.unit
class TestNotFlagged:
    """Genuine retrieval queries that must NOT be flagged (defer to LLM)."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "之前怎么解决的",
            "如何配置 BannerAdLoader",
            "uiautomator dump 为什么有两个",
            "现在这个mcp的执行速度怎么样?",
            "我们先把问题搞清楚,为什么会有两个uiautomator dump?",
            "how to configure the retry handler?",
            "XAdRemoteClickHandler 的逻辑",
            "实现一下广告验证的自动化",
            # Looks like a confirmation but is long enough to carry intent
            "可以,但我想先了解之前的失败原因和排查方法",
            # A single numbered item is an instruction, not multi-step noise
            "1. 请帮我分析召回质量下降的原因",
        ],
    )
    def test_not_noise(self, prompt):
        assert classify_noise_rule(prompt) is None


@pytest.mark.unit
class TestEdgeCases:
    def test_confirmation_with_trailing_space(self):
        assert classify_noise_rule("  好的  ") is NoiseReason.RULE

    def test_long_confirmation_not_flagged(self):
        # >12 chars: even if it starts like a confirmation, defer to LLM
        assert classify_noise_rule("可以的话请帮我把这个问题彻底排查一遍谢谢") is None

    def test_kill_test_case_insensitive(self):
        assert classify_noise_rule("KILL TEST 5") is NoiseReason.RULE
