"""Unified LLM HTTP client — encapsulates all LLM API calls.

Provides a single point for timeout configuration, error handling,
and response extraction when communicating with Ollama-compatible APIs.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from team_memory.config import LLMConfig

logger = logging.getLogger("team_memory.llm_client")


class LLMError(Exception):
    """Raised when an LLM call fails."""


class LLMClient:
    """HTTP client for Ollama-compatible chat completions."""

    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    @classmethod
    def from_config(cls, llm_config: LLMConfig | None) -> LLMClient:
        base_url = "http://localhost:11434"
        model = "gpt-oss:20b-cloud"
        if llm_config:
            base_url = llm_config.base_url
            model = llm_config.model
        return cls(base_url=base_url, model=model)

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> str:
        """Send a chat request and return the assistant reply text.

        Raises:
            LLMError: On connection failure or HTTP error.
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
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
        except httpx.ConnectError:
            raise LLMError(
                f"Cannot connect to LLM at {self._base_url}. "
                "Make sure the service is running."
            )
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LLM API error: {e.response.text[:200]}")

        text = data.get("message", {}).get("content", "")
        if not text:
            raise LLMError("LLM returned empty response")
        return text

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> dict:
        """``chat()`` followed by JSON extraction."""
        text = await self.chat(
            system, user, temperature=temperature, timeout=timeout
        )
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
