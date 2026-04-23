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


# ---------------------------------------------------------------------------
# check_tool_pattern
# ---------------------------------------------------------------------------

class TestCheckToolPatternGitCommit:
    """check_tool_pattern returns True when a git commit is present."""

    def test_git_commit_in_cmd(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "git commit -m 'fix: typo'"}]
        assert det.check_tool_pattern(tools) is True

    def test_git_commit_with_other_tools(self):
        det = ConvergenceDetector()
        tools = [
            {"tool": "terminal", "cmd": "ls -la"},
            {"tool": "terminal", "cmd": "git commit -m 'feat: add X'"},
        ]
        assert det.check_tool_pattern(tools) is True

    def test_git_commit_non_terminal_tool(self):
        """Even a non-terminal tool with git commit should match."""
        det = ConvergenceDetector()
        tools = [{"tool": "editor", "cmd": "git commit -m 'fix'"}]
        assert det.check_tool_pattern(tools) is True


class TestCheckToolPatternTestPass:
    """check_tool_pattern returns True when pytest/test/make verify/make test
    with exit_code==0."""

    def test_pytest_exit_code_zero(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "pytest", "exit_code": 0}]
        assert det.check_tool_pattern(tools) is True

    def test_pytest_nonzero_exit_code(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "pytest", "exit_code": 1}]
        assert det.check_tool_pattern(tools) is False

    def test_test_command_exit_zero(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "python -m test", "exit_code": 0}]
        assert det.check_tool_pattern(tools) is True

    def test_make_verify_exit_zero(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "make verify", "exit_code": 0}]
        assert det.check_tool_pattern(tools) is True

    def test_make_test_exit_zero(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "make test", "exit_code": 0}]
        assert det.check_tool_pattern(tools) is True

    def test_make_verify_nonzero(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "make verify", "exit_code": 2}]
        assert det.check_tool_pattern(tools) is False

    def test_test_cmd_missing_exit_code(self):
        """If exit_code key is absent, should not match."""
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "pytest"}]
        assert det.check_tool_pattern(tools) is False

    def test_test_cmd_exit_code_none(self):
        """If exit_code is None, should not match."""
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "pytest", "exit_code": None}]
        assert det.check_tool_pattern(tools) is False


class TestCheckToolPatternNegative:
    """check_tool_pattern returns False when no completion pattern found."""

    def test_empty_list(self):
        det = ConvergenceDetector()
        assert det.check_tool_pattern([]) is False

    def test_unrelated_commands(self):
        det = ConvergenceDetector()
        tools = [
            {"tool": "terminal", "cmd": "ls -la"},
            {"tool": "terminal", "cmd": "echo hello"},
        ]
        assert det.check_tool_pattern(tools) is False

    def test_non_terminal_without_git_commit(self):
        det = ConvergenceDetector()
        tools = [{"tool": "editor", "cmd": "open file.py"}]
        assert det.check_tool_pattern(tools) is False


# ---------------------------------------------------------------------------
# check_topic_shift
# ---------------------------------------------------------------------------

class TestCheckTopicShift:
    """check_topic_shift detects context switches via file path changes."""

    def test_different_paths(self):
        det = ConvergenceDetector()
        assert det.check_topic_shift("src/a.py", "src/b.py") is True

    def test_same_path(self):
        det = ConvergenceDetector()
        assert det.check_topic_shift("src/a.py", "src/a.py") is False

    def test_current_none(self):
        det = ConvergenceDetector()
        assert det.check_topic_shift(None, "src/a.py") is False

    def test_previous_none(self):
        det = ConvergenceDetector()
        assert det.check_topic_shift("src/a.py", None) is False

    def test_both_none(self):
        det = ConvergenceDetector()
        assert det.check_topic_shift(None, None) is False


# ---------------------------------------------------------------------------
# detect_convergence — updated signature with keyword-only params
# ---------------------------------------------------------------------------

class TestDetectConvergenceWithToolPattern:
    """detect_convergence delegates to check_tool_pattern."""

    def test_tool_pattern_git_commit(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "git commit -m 'done'"}]
        assert det.detect_convergence(None, recent_tools=tools) is True

    def test_tool_pattern_test_pass(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "pytest", "exit_code": 0}]
        assert det.detect_convergence("still working", recent_tools=tools) is True

    def test_tool_pattern_no_match(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "ls"}]
        assert det.detect_convergence("still working", recent_tools=tools) is False

    def test_no_tools_no_signal(self):
        det = ConvergenceDetector()
        assert det.detect_convergence("still working", recent_tools=[]) is False


class TestDetectConvergenceWithTopicShift:
    """detect_convergence delegates to check_topic_shift."""

    def test_topic_shift_detected(self):
        det = ConvergenceDetector()
        assert det.detect_convergence(
            "no signal here",
            current_path="src/new.py",
            previous_path="src/old.py",
        ) is True

    def test_same_path_no_shift(self):
        det = ConvergenceDetector()
        assert det.detect_convergence(
            "no signal here",
            current_path="src/a.py",
            previous_path="src/a.py",
        ) is False


class TestDetectConvergencePriority:
    """Explicit signals checked first, then tool pattern, then topic shift."""

    def test_explicit_signal_overrides_all(self):
        det = ConvergenceDetector()
        assert det.detect_convergence(
            "解决了",
            recent_tools=[],
            current_path=None,
            previous_path=None,
        ) is True

    def test_tool_pattern_after_no_signal(self):
        det = ConvergenceDetector()
        tools = [{"tool": "terminal", "cmd": "git commit -m 'x'"}]
        assert det.detect_convergence(
            "no signal",
            recent_tools=tools,
            current_path="a.py",
            previous_path="a.py",
        ) is True

    def test_topic_shift_after_no_signal_no_tools(self):
        det = ConvergenceDetector()
        assert det.detect_convergence(
            "no signal",
            recent_tools=[],
            current_path="a.py",
            previous_path="b.py",
        ) is True

    def test_all_false(self):
        det = ConvergenceDetector()
        assert det.detect_convergence(
            "no signal",
            recent_tools=[{"tool": "terminal", "cmd": "ls"}],
            current_path="a.py",
            previous_path="a.py",
        ) is False

    def test_backward_compat_positional_text(self):
        """Existing callers using detect_convergence(text) still work."""
        det = ConvergenceDetector()
        assert det.detect_convergence("解决了") is True
        assert det.detect_convergence("no signal") is False
