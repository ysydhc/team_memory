"""Tests for scripts/hooks/retrieval_trigger.py — keyword-based retrieval triggering."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

from retrieval_trigger import KEYWORD_TRIGGERS, should_retrieve  # noqa: I001,E402


# ---------------------------------------------------------------------------
# KEYWORD_TRIGGERS constant
# ---------------------------------------------------------------------------

class TestKeywordTriggers:
    """KEYWORD_TRIGGERS should contain the expected trigger words."""

    def test_contains_zhijian(self):
        assert "之前" in KEYWORD_TRIGGERS

    def test_contains_shangci(self):
        assert "上次" in KEYWORD_TRIGGERS

    def test_contains_jingyan(self):
        assert "经验" in KEYWORD_TRIGGERS

    def test_contains_caikeng(self):
        assert "踩坑" in KEYWORD_TRIGGERS

    def test_contains_yudaoguo(self):
        assert "遇到过" in KEYWORD_TRIGGERS

    def test_contains_yiqian(self):
        assert "以前" in KEYWORD_TRIGGERS

    def test_has_six_triggers(self):
        assert len(KEYWORD_TRIGGERS) == 6


# ---------------------------------------------------------------------------
# should_retrieve
# ---------------------------------------------------------------------------

class TestShouldRetrieve:
    """should_retrieve returns True when prompt contains a keyword trigger."""

    def test_trigger_zhijian(self):
        assert should_retrieve("之前遇到的Docker问题") is True

    def test_trigger_shangci(self):
        assert should_retrieve("上次说的方案怎么样") is True

    def test_trigger_jingyan(self):
        assert should_retrieve("分享一些经验") is True

    def test_trigger_caikeng(self):
        assert should_retrieve("踩坑记录") is True

    def test_trigger_yudaoguo(self):
        assert should_retrieve("遇到过类似的问题") is True

    def test_trigger_yiqian(self):
        assert should_retrieve("以前做过的项目") is True

    def test_no_trigger_normal_prompt(self):
        assert should_retrieve("帮我写个函数") is False

    def test_no_trigger_english_prompt(self):
        assert should_retrieve("help me write a function") is False

    def test_no_trigger_empty_string(self):
        assert should_retrieve("") is False

    def test_partial_word_no_false_positive(self):
        """A prompt with no keyword at all should return False."""
        assert should_retrieve("这个函数的逻辑") is False
