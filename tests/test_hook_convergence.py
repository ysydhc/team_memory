"""Tests for scripts/hooks/convergence_detector.py — convergence signal detection."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

from convergence_detector import ConvergenceDetector


# ---------------------------------------------------------------------------
# ConvergenceDetector — EXPLICIT_SIGNALS
# ---------------------------------------------------------------------------

class TestExplicitSignals:
    """ConvergenceDetector.EXPLICIT_SIGNALS contains the expected list."""

    def test_has_chinese_signals(self):
        signals = ConvergenceDetector.EXPLICIT_SIGNALS
        assert "解决了" in signals
        assert "问题修复" in signals
        assert "完成了" in signals
        assert "先这样" in signals
        assert "搞定" in signals
        assert "已确认" in signals

    def test_has_english_signals(self):
        signals = ConvergenceDetector.EXPLICIT_SIGNALS
        assert "test passed" in signals

    def test_signals_are_lowercase_english(self):
        """All English signals should be lowercase for case-insensitive matching."""
        for s in ConvergenceDetector.EXPLICIT_SIGNALS:
            if s.isascii():
                assert s == s.lower(), f"English signal '{s}' should be lowercase"


# ---------------------------------------------------------------------------
# detect_convergence — positive cases
# ---------------------------------------------------------------------------

class TestDetectConvergencePositive:
    """detect_convergence returns True when a signal is present."""

    def test_chinese_signal_resolved(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("这个问题已经解决了，可以关了") is True

    def test_chinese_signal_fixed(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("问题修复了") is True

    def test_chinese_signal_done(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("任务完成了") is True

    def test_chinese_signal_for_now(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("先这样吧") is True

    def test_chinese_signal_got_it(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("搞定！") is True

    def test_chinese_signal_confirmed(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("已确认，没问题") is True

    def test_english_signal_test_passed(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("All test passed successfully") is True

    def test_english_case_insensitive(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("All Test Passed successfully") is True

    def test_signal_at_start_of_text(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("解决了，不用再改了") is True

    def test_signal_at_end_of_text(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("很好，问题已经解决了") is True

    def test_signal_in_middle_of_text(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("经过调试，搞定了，后续再优化") is True


# ---------------------------------------------------------------------------
# detect_convergence — negative cases
# ---------------------------------------------------------------------------

class TestDetectConvergenceNegative:
    """detect_convergence returns False when no signal is present."""

    def test_no_signal_plain_text(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("这是一个普通的回复，没有任何信号") is False

    def test_no_signal_english(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("I'm still working on the issue") is False

    def test_no_signal_partial_match(self):
        """Partial substring should not match."""
        det = ConvergenceDetector()
        assert det.detect_convergence("我们还需要解决") is False

    def test_empty_string(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("") is False

    def test_none_input(self):
        det = ConvergenceDetector()
        assert det.detect_convergence(None) is False

    def test_whitespace_only(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("   ") is False
