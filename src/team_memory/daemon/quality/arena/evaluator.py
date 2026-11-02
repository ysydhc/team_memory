from __future__ import annotations

import asyncio
import json
import logging
import os
import re

import httpx

from team_memory.daemon.quality.arena.models import EvalResult

logger = logging.getLogger("arena.evaluator")

DEFAULT_BASE_URL = os.environ.get("ARENA_LLM_BASE_URL", "http://localhost:4000/v1")
DEFAULT_MODEL = os.environ.get("ARENA_EVALUATOR_MODEL", "gemini-3.6-flash")
DEFAULT_API_KEY = os.environ.get("ARENA_LLM_API_KEY", "") or os.environ.get("LITELLM_MASTER_KEY", "")


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class ArenaEvaluator:
    """Runs a single evaluator prompt against a task to produce relevance + helpfulness scores."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or DEFAULT_MODEL
        self._api_key = api_key or DEFAULT_API_KEY
        self._timeout = timeout
        self._max_retries = max_retries

    async def evaluate(
        self,
        *,
        prompt_template: str,
        query: str,
        contexts: list[str],
        agent_response: str,
    ) -> EvalResult:
        context_text = "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts) if ctx)
        if not context_text:
            context_text = "(no contexts)"

        prompt = (
            prompt_template.replace("{query}", query[:500])
            .replace("{contexts}", context_text[:3000])
            .replace("{response}", agent_response[:3000])
        )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = await self._call_llm(prompt)
                return self._parse_result(raw)
            except Exception as e:
                last_exc = e
                if _is_retryable_error(e) and attempt < self._max_retries:
                    delay = 1.0 * (2**attempt)
                    logger.warning(
                        "Arena evaluator LLM call failed (attempt %d/%d, retryable), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    if attempt < self._max_retries:
                        logger.warning(
                            "Arena evaluator LLM call failed (attempt %d/%d, non-retryable): %s",
                            attempt + 1,
                            self._max_retries + 1,
                            e,
                        )
                    else:
                        logger.warning(
                            "Arena evaluator failed after %d attempts: %s",
                            self._max_retries + 1,
                            e,
                        )
                    break

        logger.error("Arena evaluator returning sentinel scores (-1): %s", last_exc)
        return EvalResult(relevance=-1.0, helpfulness=-1.0, reasoning=str(last_exc))

    async def _call_llm(self, prompt: str) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1500,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _parse_result(self, raw: str) -> EvalResult:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if data and "relevance" in data and "helpfulness" in data:
            reasoning = data.get("reasoning", "")
            if not isinstance(reasoning, str):
                reasoning = json.dumps(reasoning, ensure_ascii=False)
            return EvalResult(
                relevance=float(data["relevance"]),
                helpfulness=float(data["helpfulness"]),
                reasoning=reasoning,
            )

        # Fallback: regex extraction of floats
        relevance = re.search(r"relevance[\"'\s:]+([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        helpfulness = re.search(r"helpfulness[\"'\s:]+([0-9]*\.?[0-9]+)", text, re.IGNORECASE)

        if relevance and helpfulness:
            return EvalResult(
                relevance=float(relevance.group(1)),
                helpfulness=float(helpfulness.group(1)),
                reasoning=f"regex-extracted from: {raw[:200]}",
            )

        return EvalResult(relevance=-1.0, helpfulness=-1.0, reasoning=f"parse failure: {raw[:200]}")
