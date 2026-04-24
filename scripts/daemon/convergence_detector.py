"""Convergence detector for Agent memory pipeline hook scripts.

Detects convergence signals in agent responses to identify when a task
has reached a natural conclusion (e.g., "解决了", "test passed").
"""
from __future__ import annotations

import re
from typing import Any


class ConvergenceDetector:
    """Detects convergence signals in agent response text.

    Convergence signals are phrases that indicate the agent considers
    the task done or the problem resolved. When detected, the draft
    should be marked for publishing.

    Detection is performed in three stages (first match wins):

    1. **Explicit signals** — keyword substrings in response text.
    2. **Tool patterns** — git commit or successful test execution.
    3. **Topic shift** — the working file path changed.

    Usage::

        det = ConvergenceDetector()
        if det.detect_convergence(response_text, recent_tools=tools,
                                  current_path=cur, previous_path=prev):
            # mark draft for publishing
    """

    EXPLICIT_SIGNALS: list[str] = [
        # 直接完成
        "解决了",
        "已解决",
        "问题修复",
        "修复了",
        "搞定了",
        "完成了",
        "已完成",
        "先这样",
        "搞定",
        "done",
        # 确认类
        "已确认",
        "确认无误",
        "验证通过",
        "test passed",
        "all tests passed",
        "tests pass",
        # 提交类
        "已提交",
        "committed",
    ]

    # Regex patterns for test-like commands that indicate completion
    # only when exit_code == 0.
    _TEST_CMD_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\bpytest\b"),
        re.compile(r"\btest\b"),
        re.compile(r"\bmake\s+verify\b"),
        re.compile(r"\bmake\s+test\b"),
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_tool_pattern(self, recent_tools: list[dict[str, Any]]) -> bool:
        """Return True if *recent_tools* indicate task completion.

        A completion is indicated by:

        * Any tool entry whose ``cmd`` contains ``"git commit"``.
        * A terminal tool whose ``cmd`` matches a test pattern **and**
          whose ``exit_code`` is exactly ``0``.

        Args:
            recent_tools: List of tool dicts, each with at least
                ``"tool"`` and ``"cmd"`` keys.  Terminal test commands
                must also include ``"exit_code"``.

        Returns:
            True if a completion pattern is found, False otherwise.
        """
        if not recent_tools:
            return False

        for entry in recent_tools:
            cmd = entry.get("cmd", "")
            if not cmd:
                continue

            # git commit in any tool → done
            if "git commit" in cmd:
                return True

            # test command with exit_code == 0
            if self._is_test_command(cmd) and entry.get("exit_code") == 0:
                return True

        return False

    def check_topic_shift(
        self,
        current_path: str | None,
        previous_path: str | None,
    ) -> bool:
        """Return True if the file path changed (context switch).

        Both *current_path* and *previous_path* must be non-None and
        differ for a shift to be detected.

        Args:
            current_path: The file currently being worked on.
            previous_path: The file previously being worked on.

        Returns:
            True if paths differ, False otherwise.
        """
        if current_path is None or previous_path is None:
            return False
        return current_path != previous_path

    def detect_convergence(
        self,
        text: str | None,
        *,
        recent_tools: list[dict[str, Any]] | None = None,
        current_path: str | None = None,
        previous_path: str | None = None,
    ) -> bool:
        """Check whether any convergence signal is present.

        Detection order (first True wins):

        1. Explicit signals in *text*.
        2. Tool patterns in *recent_tools*.
        3. Topic shift between *current_path* and *previous_path*.

        The new parameters are keyword-only with defaults, so existing
        callers of ``detect_convergence(text)`` continue to work.

        Args:
            text: The agent response text to analyse.  None is treated
                as empty string.
            recent_tools: Optional list of recent tool invocations.
            current_path: Optional file path the agent is currently
                working on.
            previous_path: Optional file path the agent was previously
                working on.

        Returns:
            True if any convergence signal is found, False otherwise.
        """
        # 1. Explicit signals
        if self._check_explicit_signals(text):
            return True

        # 2. Tool patterns
        if recent_tools and self.check_tool_pattern(recent_tools):
            return True

        # 3. Topic shift
        if self.check_topic_shift(current_path, previous_path):
            return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_explicit_signals(self, text: str | None) -> bool:
        """Check text for explicit convergence signal substrings."""
        if not text:
            return False

        text_lower = text.lower()
        for signal in self.EXPLICIT_SIGNALS:
            if signal.isascii():
                if signal.lower() in text_lower:
                    return True
            else:
                if signal in text:
                    return True
        return False

    @classmethod
    def _is_test_command(cls, cmd: str) -> bool:
        """Return True if *cmd* matches a known test-invocation pattern."""
        for pat in cls._TEST_CMD_PATTERNS:
            if pat.search(cmd):
                return True
        return False
