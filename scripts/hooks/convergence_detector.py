"""Convergence detector for Agent memory pipeline hook scripts.

Detects convergence signals in agent responses to identify when a task
has reached a natural conclusion (e.g., "解决了", "test passed").
"""
from __future__ import annotations


class ConvergenceDetector:
    """Detects convergence signals in agent response text.

    Convergence signals are phrases that indicate the agent considers
    the task done or the problem resolved. When detected, the draft
    should be marked for publishing.

    Usage::

        det = ConvergenceDetector()
        if det.detect_convergence(response_text):
            # mark draft for publishing
    """

    EXPLICIT_SIGNALS: list[str] = [
        "解决了",
        "问题修复",
        "完成了",
        "先这样",
        "搞定",
        "test passed",
        "已确认",
    ]

    def detect_convergence(self, text: str | None) -> bool:
        """Check whether *text* contains any convergence signal.

        Performs case-insensitive matching for English signals and
        exact substring matching for Chinese signals.

        Args:
            text: The agent response text to analyze. None is treated
                  as empty string.

        Returns:
            True if any convergence signal is found, False otherwise.
        """
        if not text:
            return False

        text_lower = text.lower()
        for signal in self.EXPLICIT_SIGNALS:
            if signal.isascii():
                # English signals: case-insensitive
                if signal.lower() in text_lower:
                    return True
            else:
                # Chinese signals: exact substring match
                if signal in text:
                    return True

        return False
