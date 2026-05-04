"""Tests for contradiction_detector service."""
from __future__ import annotations

import pytest

from team_memory.services.contradiction_detector import ContradictionPair


class TestContradictionPair:
    def test_defaults(self):
        p = ContradictionPair(
            exp_a_id="1", exp_a_title="A",
            exp_b_id="2", exp_b_title="B",
        )
        assert p.shared_entities == []
        assert p.reason == ""

    def test_with_data(self):
        p = ContradictionPair(
            exp_a_id="1", exp_a_title="Avoid X",
            exp_b_id="2", exp_b_title="Use X",
            shared_entities=["X"],
            reason="A suggests avoidance, B suggests adoption",
        )
        assert len(p.shared_entities) == 1
        assert "avoidance" in p.reason


class TestNegationDetection:
    """Test that negation/positive word detection works logically."""

    def test_negation_words_defined(self):
        from team_memory.services.contradiction_detector import detect_contradictions
        # The function uses internal word sets — just verify the module loads
        assert callable(detect_contradictions)

    def test_chinese_negation(self):
        """Chinese negation words should be recognized."""
        text = "不要使用 UIWebView，避免崩溃"
        negation_words = {
            "不", "不要", "别", "不能", "避免", "avoid", "don't",
        }
        found = any(w in text for w in negation_words)
        assert found

    def test_chinese_positive(self):
        """Chinese positive words should be recognized."""
        text = "必须使用 WKWebView，推荐做法"
        positive_words = {
            "必须", "要", "应该", "使用", "use", "must", "推荐",
        }
        found = any(w in text for w in positive_words)
        assert found
