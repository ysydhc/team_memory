"""IntentGate — LLM-based intent classification with circuit breaker.

Determines whether a user prompt needs knowledge retrieval and what type.
Uses gemini-3.6-flash via LiteLLM proxy (same channel as FaithfulnessJudge).
Includes circuit breaker (fuse) with automatic recovery and alerting.
"""

from __future__ import annotations

import logging
import os
import time
from enum import Enum

import httpx

logger = logging.getLogger("tm_daemon.intent_gate")

# LLM config — same env vars pattern as faithfulness_judge
DEFAULT_BASE_URL = os.environ.get("INTENT_GATE_LLM_BASE_URL", "") or os.environ.get(
    "FAITHFULNESS_LLM_BASE_URL", "http://localhost:4000/v1"
)
DEFAULT_MODEL = os.environ.get("INTENT_GATE_LLM_MODEL", "") or os.environ.get(
    "FAITHFULNESS_LLM_MODEL", "gemini-3.6-flash"
)
DEFAULT_API_KEY = (
    os.environ.get("INTENT_GATE_LLM_API_KEY", "")
    or os.environ.get("FAITHFULNESS_LLM_API_KEY", "")
    or os.environ.get("LITELLM_MASTER_KEY", "")
)
DEFAULT_TIMEOUT = int(os.environ.get("INTENT_GATE_LLM_TIMEOUT", "10"))

# Circuit breaker defaults
DEFAULT_ALERT_THRESHOLD = 3
DEFAULT_FUSE_THRESHOLD = 10
DEFAULT_FUSE_COOLDOWN = 300  # 5 minutes


class IntentResult(str, Enum):
    NONE = "none"
    EXPERIENCE = "experience"
    WIKI = "wiki"
    BOTH = "both"


FALLBACK_RESULT = IntentResult.BOTH

_VALID_RESULTS = frozenset(r.value for r in IntentResult)

_PROMPT_TEMPLATE = """你是一个意图分类器。根据用户输入判断是否需要从知识库检索。

分类:
- "none": 不需要上下文的操作指令（格式化代码、继续执行、实现上面的方案、确认操作）
- "experience": 需要项目历史经验（之前怎么做的、类似问题、实现某需求）
- "wiki": 需要项目文档/规范（规范、API定义、文档模板）
- "both": 同时需要经验和文档

{prev_context}当前输入: {prompt}

只返回一个词: none / experience / wiki / both"""

_MULTI_TURN_TEMPLATE = """你是一个意图分类器。下面是一段多轮对话的最近几轮（按时间排序）。
请先在心里总结当前这轮用户发言的"真实意图"（结合上下文，而不只看字面），
再判断是否需要从团队记忆库检索。

分类:
- "none": 闲聊/确认/纯操作指令（如"继续"、"重试"、"按这个改"、测试命令、多步骤指令），不需要检索
- "experience": 需要项目历史经验（之前怎么做的、类似问题、实现某需求、排查某错误）
- "wiki": 需要项目文档/规范（规范、API定义、文档模板）
- "both": 同时需要经验和文档

对话上下文:
{history}

当前这轮用户发言: {prompt}

只返回一个词: none / experience / wiki / both"""


async def send_daemon_alert(title: str, body: str, severity: str = "critical") -> None:
    """Proxy to daemon's alert helper — imported lazily to avoid circular deps."""
    try:
        from team_memory.daemon.runtime.app import send_daemon_alert as _send

        await _send(title=title, body=body, severity=severity)
    except Exception:
        logger.debug("Failed to send daemon alert (non-critical)", exc_info=True)


class IntentGate:
    """LLM-based intent gate with circuit breaker."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        alert_threshold: int | None = None,
        fuse_threshold: int | None = None,
        fuse_cooldown_seconds: int | None = None,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or DEFAULT_MODEL
        self._api_key = api_key or DEFAULT_API_KEY
        self._timeout = timeout or DEFAULT_TIMEOUT

        # Circuit breaker state
        self._alert_threshold = alert_threshold if alert_threshold is not None else DEFAULT_ALERT_THRESHOLD
        self._fuse_threshold = fuse_threshold if fuse_threshold is not None else DEFAULT_FUSE_THRESHOLD
        self._fuse_cooldown = fuse_cooldown_seconds if fuse_cooldown_seconds is not None else DEFAULT_FUSE_COOLDOWN
        self._consecutive_failures: int = 0
        self._fuse_triggered_at: float = 0.0

    @property
    def is_fused(self) -> bool:
        """Whether the circuit breaker is currently open (fused)."""
        if self._consecutive_failures < self._fuse_threshold:
            return False
        elapsed = time.monotonic() - self._fuse_triggered_at
        return elapsed < self._fuse_cooldown

    async def classify(self, prompt: str, prev_message: str = "") -> IntentResult:
        """Classify user prompt intent.

        Args:
            prompt: Current user prompt.
            prev_message: Previous user message for context (optional).

        Returns:
            IntentResult indicating what type of retrieval is needed.
        """
        if self.is_fused:
            return FALLBACK_RESULT

        # If was fused but cooldown elapsed, attempt recovery
        was_fused = self._consecutive_failures >= self._fuse_threshold

        try:
            llm_prompt = self._build_prompt(prompt, prev_message)
            raw = await self._call_llm(llm_prompt)
            result = self._parse_result(raw)

            # Success — reset failure counter
            self._consecutive_failures = 0
            if was_fused:
                logger.info("[INTENT_GATE] Circuit breaker recovered")
            return result

        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(
                "[INTENT_GATE] LLM call failed (%d consecutive): %s",
                self._consecutive_failures,
                e,
            )

            # Alert threshold
            if self._consecutive_failures == self._alert_threshold:
                logger.error(
                    "[ALERT] intent_gate consecutive failures reached %d",
                    self._alert_threshold,
                )

            # Fuse threshold — trigger circuit breaker + feishu alert
            if self._consecutive_failures == self._fuse_threshold:
                self._fuse_triggered_at = time.monotonic()
                logger.error(
                    "[ALERT] intent_gate FUSED — %d consecutive failures, cooldown %ds",
                    self._fuse_threshold,
                    self._fuse_cooldown,
                )
                await send_daemon_alert(
                    title="IntentGate 熔断",
                    body=(
                        f"IntentGate 连续失败 {self._fuse_threshold} 次，已熔断 {self._fuse_cooldown}s。"
                        f"\n最后错误: {e}"
                        f"\nModel: {self._model} @ {self._base_url}"
                    ),
                    severity="critical",
                )

            return FALLBACK_RESULT

    async def classify_with_history(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> IntentResult:
        """Classify intent using recent conversation turns for context.

        Unlike :meth:`classify`, this asks the LLM to first summarize the
        *true* intent of the current turn given the surrounding dialogue, then
        decide whether retrieval is needed. Used by the noise gate so that a
        literally-ambiguous utterance (e.g. "再看看那个问题") is judged in
        context rather than on its surface form.

        Args:
            prompt: Current user turn.
            history: Recent turns as ``[{"role": ..., "content": ...}, ...]``
                ordered oldest→newest. The current turn may or may not be
                included; it is passed separately as ``prompt`` regardless.

        Returns:
            IntentResult. Falls back to BOTH on any LLM failure.
        """
        if self.is_fused:
            return FALLBACK_RESULT

        was_fused = self._consecutive_failures >= self._fuse_threshold

        try:
            llm_prompt = self._build_multi_turn_prompt(prompt, history or [])
            raw = await self._call_llm(llm_prompt)
            result = self._parse_result(raw)
            self._consecutive_failures = 0
            if was_fused:
                logger.info("[INTENT_GATE] Circuit breaker recovered")
            return result
        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(
                "[INTENT_GATE] multi-turn LLM call failed (%d consecutive): %s",
                self._consecutive_failures,
                e,
            )
            if self._consecutive_failures == self._fuse_threshold:
                self._fuse_triggered_at = time.monotonic()
            return FALLBACK_RESULT

    def _build_prompt(self, prompt: str, prev_message: str) -> str:
        prev_context = ""
        if prev_message:
            truncated = prev_message[:200]
            prev_context = f"上一条消息: {truncated}\n"

        return _PROMPT_TEMPLATE.format(
            prev_context=prev_context,
            prompt=prompt[:500],
        )

    @staticmethod
    def _format_history(history: list[dict[str, str]], max_turns: int = 6) -> str:
        """Render the last ``max_turns`` messages as a compact transcript."""
        recent = history[-max_turns:] if max_turns > 0 else history
        lines = []
        for msg in recent:
            role = msg.get("role", "user")
            speaker = "用户" if role == "user" else "助手"
            content = (msg.get("content") or "").strip().replace("\n", " ")
            if len(content) > 200:
                content = content[:200] + "…"
            if content:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines) if lines else "（无历史）"

    def _build_multi_turn_prompt(self, prompt: str, history: list[dict[str, str]]) -> str:
        return _MULTI_TURN_TEMPLATE.format(
            history=self._format_history(history),
            prompt=prompt[:500],
        )

    def _parse_result(self, raw: str) -> IntentResult:
        cleaned = raw.strip().lower().strip("\"'.")
        if cleaned in _VALID_RESULTS:
            return IntentResult(cleaned)
        # Try to extract a valid token from the response
        for token in _VALID_RESULTS:
            if token in cleaned:
                return IntentResult(token)
        logger.debug("[INTENT_GATE] Unparseable LLM response: %s", raw[:100])
        return FALLBACK_RESULT

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM via LiteLLM proxy."""
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            # gemini-3.6-flash is a reasoning model: hidden "thinking" tokens are
            # billed against max_tokens, so a tiny budget returns empty content.
            "max_tokens": 300,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"].strip()
