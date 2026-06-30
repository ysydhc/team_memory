"""Rule-based noise classification for retrieval queries.

A "noise" query is a user utterance that has no real retrieval intent and
therefore should not count as a search when computing hit-rate / use-rate /
faithfulness. Examples: test commands ("Kill test 1"), confirmations ("好的"),
multi-step instructions ("1.xxx 2.xxx"), and short chit-chat.

These are PURE functions with no I/O so they can be unit-tested in isolation
and run with zero cost before any LLM fallback.

Design principle: be CONSERVATIVE. Only flag high-confidence noise. A false
negative (real query slips through) merely keeps a row in the stats; a false
positive (real query wrongly dropped) hides a genuine retrieval, which is
worse. When in doubt, return None and let the LLM gate decide.
"""

from __future__ import annotations

import re
from enum import Enum


class NoiseReason(str, Enum):
    RULE = "rule"
    LLM = "llm"


# Slash commands and confirmation words that never need retrieval.
_COMMAND_PREFIXES = (
    "/memory_archive",
    "/memory_save",
    "/memory_feedback",
    "/brainstorming",
    "/review",
    "/init",
    "/help",
    "/clear",
    "/loop",
    "/simplify",
    "/security-review",
    "/feature-design-doc",
)

# Short standalone confirmations / fillers (matched only when prompt is short).
_TRIVIAL_PATTERNS = re.compile(
    r"^(ok|okay|好|好的|可以|行|确定|确认|是的|对|对的|继续|嗯+|"
    r"continue|yes|no|不|不是|不用|got it|done|next|下一步|"
    r"出错了[,，]?重试|重试|retry|算了|等等|稍等)$",
    re.IGNORECASE,
)

# System-level retrieval queries (agent framework, not user intent).
_SYSTEM_PROMPT_PREFIXES = (
    "review the conversation above and consider",
    "review the conversation above and update",
    "review the conversation and consider",
    "review the conversation and update",
    "the current conversation does not require",
    "you can only call memory and skill management",
    "protected skills (do not edit these):",
    "nothing to save",
)

# Synthetic test-harness commands seen polluting the logs:
#   "Kill test 1", "Pub kill 2", "Kill test", "pub kill 3"
_TEST_COMMAND_PATTERN = re.compile(
    r"^(kill\s+test|pub\s+kill|kill\s+pub|test\s+kill)(\s+\d+)?$",
    re.IGNORECASE,
)

# Numbered multi-step instruction: starts with "1." / "1、" / "1)" and contains
# a second step marker. These are directives to the agent, not retrieval queries.
_NUMBERED_STEPS = re.compile(r"^\s*1\s*[.、)）]\s*\S")
_SECOND_STEP = re.compile(r"[;；]\s*\d\s*[.、)）]|\b2\s*[.、)）]\s*\S")

# Short-query threshold (chars): below this a query needs an identifier to
# survive, matching the historical stats predicate (char_length < 12).
_SHORT_QUERY_MAX = 12


def classify_noise_rule(prompt: str) -> NoiseReason | None:
    """Return NoiseReason.RULE if the prompt is high-confidence noise, else None.

    Returning None means "rules can't decide" — defer to the LLM intent gate.
    """
    stripped = (prompt or "").strip()
    if not stripped:
        # Empty prompt: not a retrieval; safe to flag.
        return NoiseReason.RULE

    lowered = stripped.lower()

    # 1. Slash commands.
    if any(lowered.startswith(prefix) for prefix in _COMMAND_PREFIXES):
        return NoiseReason.RULE

    # 1b. System-level prompts (agent framework, not user retrieval).
    if any(stripped.lower().startswith(p) for p in _SYSTEM_PROMPT_PREFIXES):
        return NoiseReason.RULE

    # 2. Synthetic test commands (Kill test N / Pub kill N).
    if _TEST_COMMAND_PATTERN.match(stripped):
        return NoiseReason.RULE

    # 3. Numbered multi-step instructions (directive to agent, not a query).
    if _NUMBERED_STEPS.match(stripped) and _SECOND_STEP.search(stripped):
        return NoiseReason.RULE

    # 4. Short confirmations / fillers.
    if len(stripped) <= _SHORT_QUERY_MAX and _TRIVIAL_PATTERNS.match(stripped):
        return NoiseReason.RULE

    # NOTE: we deliberately do NOT flag "short query without identifier" here.
    # Short Chinese questions like "之前怎么解决的" are genuine retrieval intent;
    # dropping them would be a false positive. Such ambiguous cases go to the
    # LLM gate, and the stats layer keeps a char_length fallback only for
    # legacy rows that never passed through this gate.
    return None
