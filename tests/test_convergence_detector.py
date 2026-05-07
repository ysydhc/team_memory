"""Tests for ConvergenceDetector — task completion signal detection / 任务完成信号检测测试

Covers:
- Explicit signal detection / 显式信号检测
- Tool pattern detection (git commit, test commands) / 工具模式检测
- Topic shift detection / 主题切换检测
- detect_convergence priority / 检测优先级
"""
from __future__ import annotations

import pytest

from daemon.convergence_detector import ConvergenceDetector

pytestmark = pytest.mark.unit


class TestExplicitSignals:
    """Tests for explicit convergence signals / 显式收敛信号测试"""

    def setup_method(self):
        self.det = ConvergenceDetector()

    def test_chinese_resolved(self):
        """解决了 should be detected / 解决了 应被检测"""
        assert self.det.detect_convergence("问题已解决了，修复代码已提交") is True

    def test_chinese_done(self):
        """完成了 should be detected / 完成了 应被检测"""
        assert self.det.detect_convergence("功能已完成") is True

    def test_chinese_fixed(self):
        """修复了 should be detected / 修复了 应被检测"""
        assert self.det.detect_convergence("bug已修复了") is True

    def test_english_done(self):
        """done should be detected / done 应被检测"""
        assert self.det.detect_convergence("All done!") is True

    def test_test_passed(self):
        """test passed should be detected / test passed 应被检测"""
        assert self.det.detect_convergence("All test passed") is True

    def test_committed(self):
        """committed should be detected / committed 应被检测"""
        assert self.det.detect_convergence("Changes committed to main") is True

    def test_no_signal(self):
        """Normal text should NOT trigger / 普通文本不应触发"""
        assert self.det.detect_convergence("Let me check the code") is False

    def test_none_text(self):
        """None text should NOT trigger / None 文本不应触发"""
        assert self.det.detect_convergence(None) is False

    def test_empty_text(self):
        """Empty text should NOT trigger / 空文本不应触发"""
        assert self.det.detect_convergence("") is False


class TestToolPatterns:
    """Tests for tool pattern detection / 工具模式检测测试"""

    def setup_method(self):
        self.det = ConvergenceDetector()

    def test_git_commit(self):
        """git commit tool should trigger / git commit 工具应触发"""
        tools = [{"tool": "terminal", "cmd": "git commit -m fix"}]
        assert self.det.check_tool_pattern(tools) is True

    def test_pytest_success(self):
        """pytest with exit_code=0 should trigger / pytest exit_code=0 应触发"""
        tools = [{"tool": "terminal", "cmd": "pytest tests/", "exit_code": 0}]
        assert self.det.check_tool_pattern(tools) is True

    def test_pytest_failure(self):
        """pytest with exit_code!=0 should NOT trigger / pytest exit_code!=0 不应触发"""
        tools = [{"tool": "terminal", "cmd": "pytest tests/", "exit_code": 1}]
        assert self.det.check_tool_pattern(tools) is False

    def test_make_verify_success(self):
        """make verify with exit_code=0 should trigger / make verify exit_code=0 应触发"""
        tools = [{"tool": "terminal", "cmd": "make verify", "exit_code": 0}]
        assert self.det.check_tool_pattern(tools) is True

    def test_empty_tools(self):
        """Empty tools list should NOT trigger / 空工具列表不应触发"""
        assert self.det.check_tool_pattern([]) is False

    def test_no_matching_tool(self):
        """Non-matching tool should NOT trigger / 不匹配工具不应触发"""
        tools = [{"tool": "terminal", "cmd": "ls -la"}]
        assert self.det.check_tool_pattern(tools) is False

    def test_no_cmd_key(self):
        """Tool without cmd key should NOT trigger / 无 cmd 键的工具不应触发"""
        tools = [{"tool": "terminal"}]
        assert self.det.check_tool_pattern(tools) is False


class TestTopicShift:
    """Tests for topic shift detection / 主题切换检测测试"""

    def setup_method(self):
        self.det = ConvergenceDetector()

    def test_different_paths(self):
        """Different paths should indicate shift / 不同路径应表示切换"""
        assert self.det.check_topic_shift("/a/file.py", "/b/other.py") is True

    def test_same_paths(self):
        """Same paths should NOT indicate shift / 相同路径不应表示切换"""
        assert self.det.check_topic_shift("/a/file.py", "/a/file.py") is False

    def test_none_current(self):
        """None current_path should NOT trigger / None 当前路径不应触发"""
        assert self.det.check_topic_shift(None, "/a/file.py") is False

    def test_none_previous(self):
        """None previous_path should NOT trigger / None 前一路径不应触发"""
        assert self.det.check_topic_shift("/a/file.py", None) is False

    def test_both_none(self):
        """Both None should NOT trigger / 两个 None 不应触发"""
        assert self.det.check_topic_shift(None, None) is False


class TestDetectConvergencePriority:
    """Tests for detect_convergence priority / detect_convergence 优先级测试"""

    def setup_method(self):
        self.det = ConvergenceDetector()

    def test_explicit_signal_takes_priority(self):
        """Explicit signal should be detected first / 显式信号应优先检测"""
        # Both explicit and tool pattern present
        result = self.det.detect_convergence(
            "已解决",
            recent_tools=[{"tool": "terminal", "cmd": "git commit"}],
            current_path="/a.py",
            previous_path="/b.py",
        )
        assert result is True

    def test_tool_pattern_without_explicit(self):
        """Tool pattern should trigger when no explicit signal / 无显式信号时工具模式应触发"""
        result = self.det.detect_convergence(
            "Let me commit",
            recent_tools=[{"tool": "terminal", "cmd": "git commit"}],
        )
        assert result is True

    def test_topic_shift_without_others(self):
        """Topic shift should trigger when no other signals / 无其他信号时主题切换应触发"""
        result = self.det.detect_convergence(
            "Working on it",
            recent_tools=[],
            current_path="/a.py",
            previous_path="/b.py",
        )
        assert result is True

    def test_no_signals_at_all(self):
        """No signals should NOT trigger / 无任何信号不应触发"""
        result = self.det.detect_convergence(
            "Still working on it",
            recent_tools=[],
            current_path="/a.py",
            previous_path="/a.py",
        )
        assert result is False
