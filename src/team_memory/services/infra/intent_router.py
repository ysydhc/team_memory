"""Intent router — classifies user queries into intent types.

Provides an abstract base class (IntentRouter) and a default implementation
(DefaultIntentRouter) that always returns "general". Future implementations
can use LLM-based classification or rule-based heuristics.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_TICKET_OR_ERROR_RE = re.compile(
    r"(\b[A-Za-z][A-Za-z0-9_-]*-\d{2,}\b|\bERR_[A-Z0-9_]+\b|\bOC\d{3}\b|"
    r"unrecognized selector|file not found|No module named|Traceback|Exception|NullPointer|crash|ANR)",
    re.IGNORECASE,
)
_CODE_IDENT_RE = re.compile(
    r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9]*|[a-z]+_[a-z0-9_]+|[A-Za-z0-9_]+\.(?:h|m|mm|py|go|kt|java|js|ts|md|yaml|json|sh))\b"
)
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_RECENCY_KEYWORDS = (
    "发版",
    "发布",
    "最新",
    "最近",
    "升级",
    "迁移",
    "重构",
    "巡检",
    "release",
    "bump",
    "latest",
    "recent",
)
_MULTIHOP_KEYWORDS = (
    "架构",
    "拓扑",
    "全链路",
    "对比",
    "状态机",
    "漏斗",
    "适配",
    "协同",
    "图谱",
    "矩阵",
    "端云",
    "门禁",
    "验收",
    "三端",
    "跨端",
    "全流程",
)


def classify_rank_intent(query: str) -> str:
    """Fast (<0.1ms) rule-based query classifier for intent-adaptive ranking.

    Returns one of:
      - 'exact_identifier': ticket IDs, error signatures, or >=2 code/file symbols
      - 'recency_release': version numbers or release/upgrade/latest queries
      - 'multihop_arch': cross-module architecture, topology, funnel, or multi-entity queries
      - 'semantic_qa': general natural-language troubleshooting or concept queries
    """
    q = (query or "").strip()
    if not q:
        return "semantic_qa"
    q_low = q.lower()

    if _TICKET_OR_ERROR_RE.search(q):
        return "exact_identifier"
    code_idents = _CODE_IDENT_RE.findall(q)
    if len(code_idents) >= 2:
        return "exact_identifier"

    if _VERSION_RE.search(q) or any(kw in q_low for kw in _RECENCY_KEYWORDS):
        # If it also strongly asks about architecture/topology/matrix, prefer multihop_arch
        arch_hits = sum(1 for kw in _MULTIHOP_KEYWORDS if kw in q_low)
        if arch_hits >= 2:
            return "multihop_arch"
        return "recency_release"

    if any(kw in q_low for kw in _MULTIHOP_KEYWORDS):
        return "multihop_arch"

    if len(code_idents) == 1 and len(q) <= 32:
        return "exact_identifier"

    return "semantic_qa"


@dataclass
class IntentResult:
    """意图分类结果"""

    intent_type: str  # exact_identifier / recency_release / multihop_arch / semantic_qa / general
    params: dict = field(default_factory=dict)  # search parameter suggestions


class IntentRouter(ABC):
    """意图路由抽象基类"""

    @abstractmethod
    async def classify(self, query: str, context: dict | None = None) -> IntentResult:
        pass


class DefaultIntentRouter(IntentRouter):
    """默认实现：所有查询返回 general，不改变检索策略"""

    async def classify(self, query: str, context: dict | None = None) -> IntentResult:
        return IntentResult(intent_type="general", params={})


class RuleBasedIntentRouter(IntentRouter):
    """规则意图路由器：将查询映射到 4 类排序场景（exact_identifier/recency_release/multihop_arch/semantic_qa）"""

    async def classify(self, query: str, context: dict | None = None) -> IntentResult:
        return IntentResult(intent_type=classify_rank_intent(query), params={})
