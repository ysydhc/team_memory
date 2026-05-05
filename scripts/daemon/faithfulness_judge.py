"""Faithfulness Judge — LLM-based evaluation of response-context faithfulness.

Implements a simplified RAGAS faithfulness metric:
1. Split agent response into claims (statements/assertions)
2. For each claim, check if it can be inferred from the retrieved context
3. Score = supported_claims / total_claims

Uses LiteLLM proxy for LLM calls. Model is configurable.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("daemon.faithfulness_judge")

# Default LLM config — overridable via env vars or constructor
DEFAULT_BASE_URL = os.environ.get("FAITHFULNESS_LLM_BASE_URL", "http://localhost:4000/v1")
DEFAULT_MODEL = os.environ.get("FAITHFULNESS_LLM_MODEL", "glm-4-flash")
DEFAULT_API_KEY = os.environ.get("LITELLM_MASTER_KEY", "")
DEFAULT_TIMEOUT = int(os.environ.get("FAITHFULNESS_LLM_TIMEOUT", "60"))


@dataclass
class FaithfulnessResult:
    """Result of a faithfulness evaluation."""
    score: float  # 0.0 - 1.0
    total_claims: int
    supported_claims: int
    claims_detail: list[dict[str, Any]]  # [{claim, supported, reason}]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "claims_detail": self.claims_detail,
        }


JUDGE_PROMPT = """你是一个 RAG (Retrieval-Augmented Generation) 评估专家。

你的任务是判断一个 AI 回复是否基于给定的检索结果。

## 搜索查询
{query}

## 检索结果
{contexts}

## AI 回复
{response}

## 评估规则
1. 把 AI 回复拆成独立的 claims（陈述/断言）。忽略寒暄、过渡句等非实质性内容。
2. 对每个 claim，判断它是否能从检索结果中推断出来：
   - supported: claim 的核心信息在检索结果中有依据
   - unsupported: claim 的信息在检索结果中找不到依据（可能是通用知识或幻觉）
3. 注意：如果回复只是泛泛而谈没有实质内容，总 claims 数可能为 0，此时 score = 1.0

## 输出格式
严格输出 JSON，不要加 markdown 代码块：
{{
  "claims": [
    {{"claim": "具体陈述内容", "supported": true/false, "reason": "简短理由"}}
  ],
  "score": 0.0到1.0的浮点数
}}"""


class FaithfulnessJudge:
    """LLM-based faithfulness evaluator."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or DEFAULT_MODEL
        self._api_key = api_key or DEFAULT_API_KEY
        self._timeout = timeout or DEFAULT_TIMEOUT

    async def evaluate(
        self,
        *,
        query: str,
        contexts: list[str],
        response: str,
    ) -> FaithfulnessResult:
        """Evaluate faithfulness of a response given retrieved contexts.

        Args:
            query: The original search query
            contexts: List of retrieved context strings (experience titles/solutions)
            response: The agent's response text

        Returns:
            FaithfulnessResult with score and breakdown
        """
        if not response or not response.strip():
            return FaithfulnessResult(
                score=0.0, total_claims=0, supported_claims=0,
                claims_detail=[], raw_response="empty response",
            )

        # Format contexts
        context_text = "\n\n".join(
            f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts) if ctx
        )
        if not context_text:
            context_text = "(无检索结果)"

        # Build prompt
        prompt = JUDGE_PROMPT.format(
            query=query[:500],
            contexts=context_text[:3000],
            response=response[:3000],
        )

        try:
            result_text = await self._call_llm(prompt)
            return self._parse_result(result_text)
        except Exception as e:
            logger.warning("Faithfulness judge failed: %s", e)
            return FaithfulnessResult(
                score=-1.0, total_claims=0, supported_claims=0,
                claims_detail=[], raw_response=str(e),
            )

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM via LiteLLM proxy."""
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _parse_result(self, raw: str) -> FaithfulnessResult:
        """Parse LLM output into FaithfulnessResult."""
        # Try to extract JSON from the response
        text = raw.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return FaithfulnessResult(
                        score=-1.0, total_claims=0, supported_claims=0,
                        claims_detail=[], raw_response=raw[:500],
                    )
            else:
                return FaithfulnessResult(
                    score=-1.0, total_claims=0, supported_claims=0,
                    claims_detail=[], raw_response=raw[:500],
                )

        claims = data.get("claims", [])
        total = len(claims)
        supported = sum(1 for c in claims if c.get("supported", False))
        score = data.get("score", supported / total if total > 0 else 1.0)

        return FaithfulnessResult(
            score=float(score),
            total_claims=total,
            supported_claims=supported,
            claims_detail=claims,
            raw_response=raw[:500],
        )
