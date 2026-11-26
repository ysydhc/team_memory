"""Unified LLM HTTP client — encapsulates all LLM API calls.

Provides a single point for timeout configuration, error handling,
and response extraction when communicating with Ollama-compatible APIs.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import httpx

from team_memory.services.analysis.usage_ledger import estimate_tokens, record_usage_async

if TYPE_CHECKING:
    from team_memory.config import LLMConfig

logger = logging.getLogger("team_memory.llm_client")


class LLMError(Exception):
    """Raised when an LLM call fails."""


class LLMClient:
    """HTTP client for OpenAI-compatible and Ollama-compatible chat completions."""

    def __init__(self, base_url: str, model: str, api_key: str = "", provider: str = "generic"):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._provider = provider

    @classmethod
    def from_config(cls, llm_config: LLMConfig | None) -> LLMClient:
        base_url = "https://gw.gburouter.com/v1"
        model = "gbu-gemini-3.8-flash"
        api_key = ""
        provider = "generic"
        if llm_config:
            base_url = llm_config.base_url or base_url
            model = llm_config.model or model
            api_key = getattr(llm_config, "api_key", "") or ""
            provider = getattr(llm_config, "provider", "") or provider
        return cls(base_url=base_url, model=model, api_key=api_key, provider=provider)

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
        caller: str = "llm_client",
    ) -> str:
        """Send a chat request and return the assistant reply text.

        Automatically uses OpenAI-compatible `/chat/completions` when provider is
        `generic`/`openai`/`litellm` or `base_url` ends with `/v1` or `/openai`,
        and falls back to Ollama `/api/chat` when provider is `ollama`.

        Raises:
            LLMError: On connection failure or HTTP error.
        """
        use_openai = self._provider != "ollama" or self._base_url.endswith(("/v1", "/openai"))
        start = time.monotonic()
        est_prompt = estimate_tokens([system, user])
        prompt_tokens = est_prompt
        completion_tokens = 0
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if use_openai:
                    url = (
                        f"{self._base_url}/chat/completions"
                        if self._base_url.endswith(("/v1", "/openai"))
                        else f"{self._base_url}/v1/chat/completions"
                    )
                    headers: dict[str, str] = {"Content-Type": "application/json"}
                    if self._api_key:
                        headers["Authorization"] = f"Bearer {self._api_key}"
                        headers["x-api-key"] = self._api_key
                    resp = await client.post(
                        url,
                        headers=headers,
                        json={
                            "model": self._model,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            "temperature": temperature,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    text = choices[0].get("message", {}).get("content", "") if choices else ""
                    usage = data.get("usage") if isinstance(data, dict) else None
                    if isinstance(usage, dict):
                        prompt_tokens = int(usage.get("prompt_tokens") or est_prompt)
                        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
                    else:
                        completion_tokens = estimate_tokens(text)
                else:
                    resp = await client.post(
                        f"{self._base_url}/api/chat",
                        json={
                            "model": self._model,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            "stream": False,
                            "options": {"temperature": temperature},
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("message", {}).get("content", "")
                    if isinstance(data, dict):
                        prompt_tokens = int(data.get("prompt_eval_count") or est_prompt)
                        completion_tokens = int(data.get("eval_count") or estimate_tokens(text))
                    else:
                        completion_tokens = estimate_tokens(text)
        except httpx.ConnectError:
            record_usage_async(
                caller=caller,
                provider=self._provider,
                model=self._model,
                prompt_tokens=est_prompt,
                completion_tokens=0,
                latency_ms=int((time.monotonic() - start) * 1000),
                success=False,
            )
            raise LLMError(f"Cannot connect to LLM at {self._base_url}. Make sure the service is running.")
        except httpx.HTTPStatusError as e:
            record_usage_async(
                caller=caller,
                provider=self._provider,
                model=self._model,
                prompt_tokens=est_prompt,
                completion_tokens=0,
                latency_ms=int((time.monotonic() - start) * 1000),
                success=False,
            )
            raise LLMError(f"LLM API error: {e.response.text[:200]}")

        latency_ms = int((time.monotonic() - start) * 1000)
        if not text:
            record_usage_async(
                caller=caller,
                provider=self._provider,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                latency_ms=latency_ms,
                success=False,
            )
            raise LLMError("LLM returned empty response")

        record_usage_async(
            caller=caller,
            provider=self._provider,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            success=True,
        )
        return text

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
        caller: str = "llm_client",
    ) -> dict:
        """``chat()`` followed by JSON extraction."""
        text = await self.chat(system, user, temperature=temperature, timeout=timeout, caller=caller)
        return extract_json(text)


def extract_json(llm_text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    clean = llm_text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        clean = "\n".join(lines).strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(clean[start:end])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"Failed to parse LLM response as JSON: {llm_text[:300]}")
