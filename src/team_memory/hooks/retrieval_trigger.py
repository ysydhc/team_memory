"""Keyword-based retrieval triggering for beforeSubmitPrompt hook.

Extracts the trigger-detection logic into a separate module so it can
be unit-tested independently of MCP calls or stdin parsing.
"""

from __future__ import annotations

KEYWORD_TRIGGERS: list[str] = ["之前", "上次", "经验", "踩坑", "遇到过", "以前"]


def should_retrieve(user_prompt: str) -> bool:
    """Return True if the user prompt contains any keyword trigger.

    Args:
        user_prompt: The raw user prompt text.

    Returns:
        True if at least one trigger keyword is found as a substring.
    """
    if not user_prompt:
        return False
    return any(kw in user_prompt for kw in KEYWORD_TRIGGERS)
