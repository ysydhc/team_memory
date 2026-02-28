"""LLM Provider abstraction layer (P3).

Unified interface for calling LLM chat endpoints across multiple providers.
Supports Ollama (default, free), OpenAI, and any OpenAI-compatible API.
Includes token usage tracking and monthly budget control.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from team_memory.config import LLMConfig

logger = logging.getLogger("team_memory.llm_provider")

# In-memory usage tracker (can be replaced with DB-backed version)
_usage_tracker: dict[str, int] = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd_cents": 0,
    "month": datetime.now(timezone.utc).strftime("%Y-%m"),
}


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat request and return the response text."""
        ...

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for given token counts. Returns 0 for free providers."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""
        ...


class OllamaLLMProvider(LLMProvider):
    """Ollama local LLM provider (free, no API key needed)."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> str:
        model = model or self._model
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                _track_usage(0, len(content) // 4, 0)
                return content
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running."
            )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


class OpenAILLMProvider(LLMProvider):
    """OpenAI API provider (paid)."""

    COST_PER_1K_INPUT = 0.0005
    COST_PER_1K_OUTPUT = 0.0015

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "openai"

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> str:
        model = model or self._model
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost_cents = int(self.estimate_cost(in_tok, out_tok) * 100)
        _track_usage(in_tok, out_tok, cost_cents)
        return content

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000 * self.COST_PER_1K_INPUT
            + output_tokens / 1000 * self.COST_PER_1K_OUTPUT
        )


class GenericLLMProvider(LLMProvider):
    """Generic OpenAI-compatible API provider.

    Works with vLLM, LocalAI, Together AI, Groq, etc.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "default",
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._cost_input = cost_per_1k_input
        self._cost_output = cost_per_1k_output

    @property
    def provider_name(self) -> str:
        return "generic"

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> str:
        model = model or self._model
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost_cents = int(self.estimate_cost(in_tok, out_tok) * 100)
        _track_usage(in_tok, out_tok, cost_cents)
        return content

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000 * self._cost_input
            + output_tokens / 1000 * self._cost_output
        )


def _track_usage(input_tokens: int, output_tokens: int, cost_cents: int) -> None:
    """Track token usage in memory. Resets monthly."""
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    if _usage_tracker["month"] != current_month:
        _usage_tracker["total_input_tokens"] = 0
        _usage_tracker["total_output_tokens"] = 0
        _usage_tracker["total_cost_usd_cents"] = 0
        _usage_tracker["month"] = current_month

    _usage_tracker["total_input_tokens"] += input_tokens
    _usage_tracker["total_output_tokens"] += output_tokens
    _usage_tracker["total_cost_usd_cents"] += cost_cents


def get_usage_stats() -> dict:
    """Return current month's usage statistics."""
    return {
        "month": _usage_tracker["month"],
        "input_tokens": _usage_tracker["total_input_tokens"],
        "output_tokens": _usage_tracker["total_output_tokens"],
        "total_cost_usd": _usage_tracker["total_cost_usd_cents"] / 100,
    }


def check_budget(monthly_budget_usd: float) -> bool:
    """Check if current usage is within budget. Returns True if OK."""
    if monthly_budget_usd <= 0:
        return True
    return _usage_tracker["total_cost_usd_cents"] / 100 < monthly_budget_usd


def create_llm_provider(llm_config: "LLMConfig") -> LLMProvider:
    """Factory function to create the appropriate LLM provider from config.

    Currently always returns OllamaLLMProvider.
    Extended provider selection can be configured in config.yaml.
    """
    return OllamaLLMProvider(
        model=llm_config.model,
        base_url=llm_config.base_url,
    )
