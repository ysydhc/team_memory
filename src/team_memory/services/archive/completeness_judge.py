"""Archive completeness judge — decide if an archive is publish-ready.

An archive is "complete" (and can skip human review, going straight to
published) when its solution_doc reads as a finished record:
  1. the problem/goal is clearly described,
  2. the changes/solution are complete (not "TODO" / "待续" / half-done),
  3. the key decisions have landed (no open "we still need to decide ...").

Uses the same LiteLLM proxy + httpx pattern as FaithfulnessJudge / IntentGate.
Fails CLOSED: any LLM/parse error returns not-complete, so an archive stays a
draft rather than being wrongly auto-published.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger("services.archive.completeness_judge")

DEFAULT_BASE_URL = os.environ.get("ARCHIVE_JUDGE_LLM_BASE_URL", "") or os.environ.get(
    "FAITHFULNESS_LLM_BASE_URL", "https://gw.gburouter.com/v1"
)
DEFAULT_MODEL = os.environ.get("ARCHIVE_JUDGE_LLM_MODEL", "") or os.environ.get(
    "FAITHFULNESS_LLM_MODEL", "gbu-gemini-3.8-flash"
)
DEFAULT_API_KEY = (
    os.environ.get("ARCHIVE_JUDGE_LLM_API_KEY", "")
    or os.environ.get("FAITHFULNESS_LLM_API_KEY", "")
    or os.environ.get("GATEWAY_API_KEY", "")
    or os.environ.get("GOOGLE_API_KEY", "")
    or os.environ.get("LITELLM_MASTER_KEY", "")
)
DEFAULT_TIMEOUT = int(os.environ.get("ARCHIVE_JUDGE_LLM_TIMEOUT", "30"))

_PROMPT = """你是一个归档质量审核员。判断下面这份「会话归档」是否已经完整、可以直接发布。

完整的标准（三条都满足才算完整）:
1. 问题/目标描述清晰 —— 能看懂这次归档要解决什么。
2. 修改/解决方案完整 —— 不是半成品（没有 TODO、待续、未完成、"下一步再说"之类）。
3. 关键决策已落地 —— 没有悬而未决的"还需要决定/讨论"。

标题: {title}

概览: {overview}

正文:
{solution_doc}

只返回一个 JSON 对象，不要其它内容:
{{"complete": true 或 false, "reason": "一句话理由"}}"""


@dataclass(frozen=True)
class CompletenessResult:
    complete: bool
    reason: str


class ArchiveCompletenessJudge:
    """LLM judge that decides whether an archive is publish-ready."""

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

    async def judge(
        self, *, title: str, overview: str, solution_doc: str
    ) -> CompletenessResult:
        """Return whether the archive is complete. Fails closed (not complete)."""
        prompt = _PROMPT.format(
            title=(title or "").strip()[:200],
            overview=(overview or "").strip()[:500],
            solution_doc=(solution_doc or "").strip()[:6000],
        )
        try:
            raw = await self._call_llm(prompt)
            return self._parse(raw)
        except Exception as e:
            logger.debug("ArchiveCompletenessJudge failed (fail-closed): %s", e)
            return CompletenessResult(complete=False, reason=f"judge error: {e}")

    @staticmethod
    def _parse(raw: str) -> CompletenessResult:
        import json

        text = raw.strip()
        # Tolerate code-fenced JSON.
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        data = json.loads(text)
        return CompletenessResult(
            complete=bool(data.get("complete", False)),
            reason=str(data.get("reason", ""))[:300],
        )

    async def _call_llm(self, prompt: str) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["x-api-key"] = self._api_key
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            # gemini-3.6-flash reasoning tokens count against max_tokens; give it
            # enough headroom to still emit the JSON verdict.
            "max_tokens": 400,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
