"""Intent router — classifies user queries into intent types.

Provides an abstract base class (IntentRouter) and a default implementation
(DefaultIntentRouter) that always returns "general". Future implementations
can use LLM-based classification or rule-based heuristics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class IntentResult:
    """意图分类结果"""

    intent_type: str  # factual / exploratory / temporal / causal / general
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
