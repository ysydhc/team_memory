"""Faithfulness & Per-Experience Attribution Judge — LLM-based evaluation.

Provides two evaluators:
1. ``AttributionJudge`` (recommended for Coding Agents):
   Evaluates each recalled experience independently against the agent's
   response, producing per-experience ``relevance (0.0~1.0)``,
   ``adopted (bool)``, ``evidence_quote (str)``, and ``contradicted (bool)``.
   Enforces a code-level Evidence-Required hard gate: if ``adopted=True``
   but ``evidence_quote`` is empty or shorter than 4 chars, ``adopted`` is
   forced to ``False``.
2. ``FaithfulnessJudge`` (legacy RAGAS claim-splitting evaluator):
   Preserved for backward compatibility with existing callers and tests.

Uses LiteLLM proxy for LLM calls. Model is configurable.
Includes retry with exponential backoff for transient LLM failures and
non-blocking token/latency recording via ``usage_ledger``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from team_memory.services.analysis.usage_ledger import estimate_tokens, record_usage_async

logger = logging.getLogger("daemon.faithfulness_judge")

# Default LLM config — overridable via env vars or constructor
DEFAULT_BASE_URL = os.environ.get("FAITHFULNESS_LLM_BASE_URL", "https://gw.gburouter.com/v1")
DEFAULT_MODEL = os.environ.get("FAITHFULNESS_LLM_MODEL", "gbu-gemini-3.8-flash")
DEFAULT_API_KEY = (
    os.environ.get("FAITHFULNESS_LLM_API_KEY", "")
    or os.environ.get("GATEWAY_API_KEY", "")
    or os.environ.get("GOOGLE_API_KEY", "")
    or os.environ.get("LITELLM_MASTER_KEY", "")
)
DEFAULT_TIMEOUT = int(os.environ.get("FAITHFULNESS_LLM_TIMEOUT", "60"))
DEFAULT_MAX_RETRIES = int(os.environ.get("FAITHFULNESS_LLM_MAX_RETRIES", "3"))
DEFAULT_RETRY_BASE_DELAY = float(os.environ.get("FAITHFULNESS_LLM_RETRY_BASE_DELAY", "1.0"))
MIN_EVIDENCE_QUOTE_LEN = 4


@dataclass
class FaithfulnessResult:
    """Result of a faithfulness evaluation."""

    score: float  # 0.0 - 1.0 (-1.0 on error)
    total_claims: int
    supported_claims: int
    claims_detail: list[dict[str, Any]]  # [{claim, supported, reason}]
    raw_response: str
    retries_used: int = 0  # how many retries were needed (0 = first attempt succeeded)
    used_indices: list[int] = field(default_factory=list)  # 1-based indices into contexts that supported claims

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "claims_detail": self.claims_detail,
            "retries_used": self.retries_used,
            "used_indices": self.used_indices,
        }


# Backward-compatible alias
EvalResult = FaithfulnessResult


@dataclass
class AttributionItem:
    """Per-experience attribution verdict."""

    experience_id: str
    relevance: float  # 0.0 - 1.0
    adopted: bool
    evidence_quote: str
    contradicted: bool = False
    downgraded: bool = False  # True when Evidence-Required gate forced adopted=False

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "relevance": self.relevance,
            "adopted": self.adopted,
            "evidence_quote": self.evidence_quote,
            "contradicted": self.contradicted,
            "downgraded": self.downgraded,
        }


@dataclass
class AttributionResult:
    """Aggregated result of per-experience attribution evaluation.

    Exposes both attribution-specific fields (``items``, ``max_relevance``,
    ``avg_relevance``, ``adopted_ids``, ``contradicted_ids``) and
    ``FaithfulnessResult``-compatible fields (``score``, ``total_claims``,
    ``supported_claims``, ``claims_detail``, ``used_indices``, ``retries_used``).
    """

    score: float  # >= 0.7 if any adopted, < 0.7 if none adopted, -1.0 on error
    max_relevance: float
    avg_relevance: float
    adopted_ids: list[str]
    contradicted_ids: list[str]
    items: list[AttributionItem]
    raw_response: str
    retries_used: int = 0
    used_indices: list[int] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.items)

    @property
    def supported_claims(self) -> int:
        return len(self.adopted_ids)

    @property
    def claims_detail(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.items]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "max_relevance": self.max_relevance,
            "avg_relevance": self.avg_relevance,
            "adopted_ids": self.adopted_ids,
            "contradicted_ids": self.contradicted_ids,
            "used_indices": self.used_indices,
            "items": [item.to_dict() for item in self.items],
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "claims_detail": self.claims_detail,
            "retries_used": self.retries_used,
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
  "used_indices": [1],
  "score": 0.0到1.0的浮点数
}}
其中 `used_indices` 为实际支撑上述 supported claims 的检索结果编号
（从 1 开始，如 [1, 2]；若未引用任何检索结果则为 []）。"""


ATTRIBUTION_JUDGE_PROMPT = """你是一个 Coding Agent 记忆归因评估专家。

注意：Coding Agent 的回复通常包含大量新编写的代码、命令输出和工具操作结果，这属于正常开发输出，绝不能因此扣分。
你的任务是**逐条评估**每一条召回的历史经验：它与当前任务的相关度、
是否被 Agent 实际采纳、以及是否被当前实证推翻（过时/矛盾）。

## 用户查询
{query}

## 召回的历史经验列表
{contexts}

## Coding Agent 实际回复
{response}

## 逐条评估规则
对召回列表中的**每一条**经验输出一个 JSON 对象，包含：
1. `experience_id` (string): 必须与召回列表中的 ID 完全一致。
2. `relevance` (float, 0.0~1.0): 该经验与当前查询/任务的相关性
   （0.0=完全无关，0.5=部分相关，1.0=高度直接相关）。
3. `adopted` (bool): Agent 回复中是否真正采纳、遵循或应用了该经验中的方案、命令、配置、约束或避坑点。
4. `evidence_quote` (string): 若 `adopted` 为 true，必须从 Agent 回复中摘录一段
   直接证明采纳的原文片段（至少 4 个字符）；若 `adopted` 为 false，必须输出空字符串 `""`。
5. `contradicted` (bool): Agent 回复是否明确指出或用实证推翻了该经验（例如：该方案已过时、报错、配置已变更、不再适用）。

## 输出格式
严格输出 JSON 数组，不要加 markdown 代码块或额外解释：
[
  {{
    "experience_id": "经验ID",
    "relevance": 0.85,
    "adopted": true,
    "evidence_quote": "回复中采纳该经验的原文片段",
    "contradicted": false
  }}
]"""


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is worth retrying (transient errors).

    Retries on:
    - httpx.TimeoutException (request timed out)
    - httpx.ConnectError (connection refused / network unreachable)
    - httpx.HTTPStatusError with 429 (rate limit) or 5xx (server error)
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


def _clamp_unit(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


class FaithfulnessJudge:
    """LLM-based faithfulness evaluator with retry support."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        db_url: str | None = None,
        tenant_id: str | None = None,
        usage_caller: str = "faithfulness_judge",
    ) -> None:
        self._base_url = (base_url or os.environ.get("FAITHFULNESS_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or os.environ.get("FAITHFULNESS_LLM_MODEL") or DEFAULT_MODEL
        self._api_key = (
            api_key
            or os.environ.get("FAITHFULNESS_LLM_API_KEY", "")
            or DEFAULT_API_KEY
        )
        self._timeout = timeout or DEFAULT_TIMEOUT
        self._max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
        self._retry_base_delay = retry_base_delay if retry_base_delay is not None else DEFAULT_RETRY_BASE_DELAY
        self._db_url = db_url
        self._tenant_id = tenant_id
        self._usage_caller = usage_caller

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
                score=0.0,
                total_claims=0,
                supported_claims=0,
                claims_detail=[],
                raw_response="empty response",
            )

        # Format contexts
        context_text = "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts) if ctx)
        if not context_text:
            context_text = "(无检索结果)"

        # Build prompt
        prompt = JUDGE_PROMPT.format(
            query=query[:500],
            contexts=context_text[:3000],
            response=response[:3000],
        )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                result_text = await self._call_llm(prompt)
                result = self._parse_result(result_text)
                result.retries_used = attempt
                if attempt > 0:
                    logger.info(
                        "Faithfulness judge succeeded on attempt %d/%d",
                        attempt + 1,
                        self._max_retries + 1,
                    )
                return result
            except Exception as e:
                last_exc = e
                if _is_retryable_error(e) and attempt < self._max_retries:
                    delay = self._retry_base_delay * (2**attempt)
                    logger.warning(
                        "Faithfulness judge LLM call failed (attempt %d/%d, retryable), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable error or all retries exhausted
                    if attempt < self._max_retries:
                        logger.warning(
                            "Faithfulness judge LLM call failed (attempt %d/%d, non-retryable): %s",
                            attempt + 1,
                            self._max_retries + 1,
                            e,
                        )
                    else:
                        logger.warning(
                            "Faithfulness judge failed after %d attempts: %s",
                            self._max_retries + 1,
                            e,
                        )
                    break

        return FaithfulnessResult(
            score=-1.0,
            total_claims=0,
            supported_claims=0,
            claims_detail=[],
            raw_response=str(last_exc),
            retries_used=self._max_retries,
        )

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM via LiteLLM proxy and record usage to usage_ledger."""
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["x-api-key"] = self._api_key

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        t0 = time.monotonic()
        success = False
        content = ""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage") if isinstance(data, dict) else None
                if isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                if prompt_tokens <= 0:
                    prompt_tokens = estimate_tokens(prompt)
                if completion_tokens <= 0:
                    completion_tokens = estimate_tokens(content)
                success = True
                return content
        finally:
            latency_ms = int((time.monotonic() - t0) * 1000)
            if prompt_tokens <= 0:
                prompt_tokens = estimate_tokens(prompt)
            record_usage_async(
                caller=self._usage_caller,
                provider="litellm",
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=success,
                tenant_id=self._tenant_id,
                db_url=self._db_url,
            )

    def _parse_result(self, raw: str) -> FaithfulnessResult:
        """Parse LLM output into FaithfulnessResult."""
        text = _strip_markdown_fences(raw)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return FaithfulnessResult(
                        score=-1.0,
                        total_claims=0,
                        supported_claims=0,
                        claims_detail=[],
                        raw_response=raw[:500],
                    )
            else:
                return FaithfulnessResult(
                    score=-1.0,
                    total_claims=0,
                    supported_claims=0,
                    claims_detail=[],
                    raw_response=raw[:500],
                )

        claims = data.get("claims", [])
        total = len(claims)
        supported = sum(1 for c in claims if c.get("supported", False))
        score = data.get("score", supported / total if total > 0 else 1.0)

        raw_indices = data.get("used_indices", [])
        used_indices: list[int] = []
        if isinstance(raw_indices, list):
            for idx in raw_indices:
                try:
                    val = int(idx)
                    if val >= 1 and val not in used_indices:
                        used_indices.append(val)
                except (TypeError, ValueError):
                    continue

        return FaithfulnessResult(
            score=float(score),
            total_claims=total,
            supported_claims=supported,
            claims_detail=claims,
            raw_response=raw[:500],
            used_indices=used_indices,
        )


class AttributionJudge(FaithfulnessJudge):
    """Per-Experience Attribution Judge tailored for Coding Agent responses.

    Evaluates each recalled experience independently on:
    - ``relevance`` (0.0~1.0)
    - ``adopted`` (bool, gated by Evidence-Required check: ``len(evidence_quote) >= 4``)
    - ``evidence_quote`` (str)
    - ``contradicted`` (bool)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        db_url: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            db_url=db_url,
            tenant_id=tenant_id,
            usage_caller="attribution_judge",
        )

    @staticmethod
    def _normalize_contexts(
        contexts: list[dict[str, Any]] | list[str],
    ) -> list[dict[str, str]]:
        """Normalize structured or plain-string contexts into [{id, title, problem, solution}]."""
        normalized: list[dict[str, str]] = []
        for idx, ctx in enumerate(contexts, start=1):
            if isinstance(ctx, dict):
                eid = str(ctx.get("id") or ctx.get("experience_id") or idx).strip()
                title = str(ctx.get("title") or "").strip()
                problem = str(ctx.get("problem") or ctx.get("description") or ctx.get("overview") or "").strip()
                solution = str(ctx.get("solution") or ctx.get("solution_doc") or "").strip()
                normalized.append(
                    {
                        "id": eid,
                        "title": title,
                        "problem": problem,
                        "solution": solution,
                    }
                )
            elif isinstance(ctx, str) and ctx.strip():
                normalized.append(
                    {
                        "id": str(idx),
                        "title": "",
                        "problem": ctx.strip(),
                        "solution": "",
                    }
                )
        return normalized

    @staticmethod
    def _format_contexts(normalized: list[dict[str, str]]) -> str:
        blocks: list[str] = []
        for idx, item in enumerate(normalized, start=1):
            lines = [f"[{idx}] experience_id: {item['id']}"]
            if item["title"]:
                lines.append(f"    标题: {item['title'][:200]}")
            if item["problem"]:
                lines.append(f"    问题: {item['problem'][:300]}")
            if item["solution"]:
                lines.append(f"    方案: {item['solution'][:400]}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    async def evaluate(  # type: ignore[override]
        self,
        *,
        query: str,
        contexts: list[dict[str, Any]] | list[str],
        response: str,
    ) -> AttributionResult:
        """Evaluate per-experience attribution for a Coding Agent response."""
        if not response or not response.strip():
            return AttributionResult(
                score=0.0,
                max_relevance=0.0,
                avg_relevance=0.0,
                adopted_ids=[],
                contradicted_ids=[],
                items=[],
                raw_response="empty response",
            )

        normalized = self._normalize_contexts(contexts)
        if not normalized:
            return AttributionResult(
                score=0.0,
                max_relevance=0.0,
                avg_relevance=0.0,
                adopted_ids=[],
                contradicted_ids=[],
                items=[],
                raw_response="empty contexts",
            )

        context_text = self._format_contexts(normalized)
        prompt = ATTRIBUTION_JUDGE_PROMPT.format(
            query=query[:600],
            contexts=context_text[:4500],
            response=response[:4000],
        )

        id_to_index = {c["id"]: i + 1 for i, c in enumerate(normalized)}

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                result_text = await self._call_llm(prompt)
                result = self._parse_attribution_result(result_text, id_to_index=id_to_index)
                result.retries_used = attempt
                if attempt > 0:
                    logger.info(
                        "Attribution judge succeeded on attempt %d/%d",
                        attempt + 1,
                        self._max_retries + 1,
                    )
                return result
            except Exception as e:
                last_exc = e
                if _is_retryable_error(e) and attempt < self._max_retries:
                    delay = self._retry_base_delay * (2**attempt)
                    logger.warning(
                        "Attribution judge LLM call failed (attempt %d/%d, retryable), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    continue
                if attempt < self._max_retries:
                    logger.warning(
                        "Attribution judge LLM call failed (attempt %d/%d, non-retryable): %s",
                        attempt + 1,
                        self._max_retries + 1,
                        e,
                    )
                else:
                    logger.warning(
                        "Attribution judge failed after %d attempts: %s",
                        self._max_retries + 1,
                        e,
                    )
                break

        return AttributionResult(
            score=-1.0,
            max_relevance=0.0,
            avg_relevance=0.0,
            adopted_ids=[],
            contradicted_ids=[],
            items=[],
            raw_response=str(last_exc),
            retries_used=self._max_retries,
        )

    def _parse_attribution_result(
        self,
        raw: str,
        *,
        id_to_index: dict[str, int] | None = None,
    ) -> AttributionResult:
        """Parse LLM JSON array into AttributionResult with Evidence-Required hard gate."""
        text = _strip_markdown_fences(raw)
        parsed: Any = None

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            arr_match = re.search(r"\[.*\]", text, re.DOTALL)
            if arr_match:
                try:
                    parsed = json.loads(arr_match.group())
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                obj_match = re.search(r"\{.*\}", text, re.DOTALL)
                if obj_match:
                    try:
                        parsed = json.loads(obj_match.group())
                    except json.JSONDecodeError:
                        parsed = None

        if isinstance(parsed, dict):
            for key in ("items", "attributions", "results", "experiences"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break

        if not isinstance(parsed, list):
            return AttributionResult(
                score=-1.0,
                max_relevance=0.0,
                avg_relevance=0.0,
                adopted_ids=[],
                contradicted_ids=[],
                items=[],
                raw_response=raw[:500],
            )

        items: list[AttributionItem] = []
        seen_ids: set[str] = set()
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            exp_id = str(entry.get("experience_id") or entry.get("id") or "").strip()
            if not exp_id or exp_id in seen_ids:
                continue
            seen_ids.add(exp_id)

            relevance = round(_clamp_unit(entry.get("relevance"), default=0.0), 4)
            evidence_quote = str(entry.get("evidence_quote") or "").strip()
            adopted = _coerce_bool(entry.get("adopted", False))
            contradicted = _coerce_bool(entry.get("contradicted", False))

            # Evidence-Required hard gate: adopted=True MUST have evidence_quote >= 4 chars
            downgraded = False
            if adopted and len(evidence_quote) < MIN_EVIDENCE_QUOTE_LEN:
                logger.debug(
                    "Evidence-Required gate downgraded experience_id=%s (quote_len=%d)",
                    exp_id,
                    len(evidence_quote),
                )
                adopted = False
                downgraded = True

            items.append(
                AttributionItem(
                    experience_id=exp_id,
                    relevance=relevance,
                    adopted=adopted,
                    evidence_quote=evidence_quote[:500],
                    contradicted=contradicted,
                    downgraded=downgraded,
                )
            )

        relevances = [it.relevance for it in items]
        max_relevance = round(max(relevances), 4) if relevances else 0.0
        avg_relevance = round(sum(relevances) / len(relevances), 4) if relevances else 0.0
        adopted_items = [it for it in items if it.adopted]
        adopted_ids = [it.experience_id for it in adopted_items]
        contradicted_ids = [it.experience_id for it in items if it.contradicted]

        used_indices: list[int] = []
        if id_to_index:
            for eid in adopted_ids:
                idx = id_to_index.get(eid)
                if idx is not None and idx not in used_indices:
                    used_indices.append(idx)

        if adopted_items:
            max_adopted_rel = max(it.relevance for it in adopted_items)
            score = round(min(1.0, max(0.8, max_adopted_rel)), 4)
        else:
            score = round(min(0.69, max_relevance), 4)

        return AttributionResult(
            score=score,
            max_relevance=max_relevance,
            avg_relevance=avg_relevance,
            adopted_ids=adopted_ids,
            contradicted_ids=contradicted_ids,
            items=items,
            raw_response=raw[:500],
            used_indices=used_indices,
        )
