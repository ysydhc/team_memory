"""Intent-to-persona classifier — keyword-based, no LLM needed.

Maps user prompts to one of the persona categories used by personal_memories:
coding, writing, research, debugging, general.
"""
from __future__ import annotations

import re

# Keyword patterns per persona (first match wins, order matters)
# NOTE: Keywords are matched with regex. For short English words that could
# appear as substrings (e.g., "log" in "CHANGELOG"), use \b word boundaries.
_PERSONA_PATTERNS: list[tuple[str, list[str]]] = [
    ("debugging", [
        "报错", "error", "bug", "崩溃", "crash", "异常", "exception",
        "修复", "fix", "排查", "troubleshoot", "日志", r"\blog\b", "超时",
        "timeout", "502", "404", "500", "连不上", "失败", "fail",
    ]),
    ("coding", [
        "代码", "code", "实现", "implement", "函数", "function", "类",
        r"\bclass\b", "API", "接口", "重构", "refactor", "feature", "功能",
        "PR", "commit", "模块", "module", "SDK", "配置", "config",
        "部署", "deploy", "CI", "CD", "测试", "test", "pytest",
    ]),
    ("writing", [
        "文档", "doc", "README", "文章", "article", "写", "write",
        "翻译", "translate", "注释", "comment", "CHANGELOG", "说明",
        "specification",
    ]),
    ("research", [
        "调研", "research", "对比", "compare", "分析", "analyze",
        "选型", "方案", "plan", "架构", "architecture", "设计",
        "design", "评估", "evaluate",
    ]),
]

# Pre-compile a single regex per persona for fast matching
# Keywords that already contain regex metacharacters (like \b) are used as-is;
# plain keywords are escaped.
def _build_pattern(keywords: list[str]) -> re.Pattern[str]:
    parts = []
    for kw in keywords:
        if r"\b" in kw:
            parts.append(kw)  # already a regex fragment
        else:
            parts.append(re.escape(kw))
    return re.compile("|".join(parts), re.IGNORECASE)


_PERSONA_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    (persona, _build_pattern(keywords))
    for persona, keywords in _PERSONA_PATTERNS
]


def classify_persona(prompt: str) -> str:
    """Classify user prompt into a persona category.

    Returns one of: 'coding', 'writing', 'research', 'debugging', 'general'
    First match wins (debugging checked before coding).
    """
    if not prompt:
        return "general"

    for persona, pattern in _PERSONA_REGEXES:
        if pattern.search(prompt):
            return persona

    return "general"
