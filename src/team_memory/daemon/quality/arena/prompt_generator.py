from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

import httpx

from team_memory.daemon.quality.arena.models import PromptEntry, RoundRecord

logger = logging.getLogger("arena.prompt_generator")

DEFAULT_BASE_URL = os.environ.get("ARENA_LLM_BASE_URL", "https://gw.gburouter.com/v1")
DEFAULT_GENERATOR_MODEL = os.environ.get("ARENA_GENERATOR_MODEL", "gbu-azure-gpt-5.6-luna")
DEFAULT_API_KEY = (
    os.environ.get("ARENA_LLM_API_KEY", "")
    or os.environ.get("GATEWAY_API_KEY", "")
    or os.environ.get("LITELLM_MASTER_KEY", "")
)

SEED_PROMPT = """你是一个 Prompt 工程专家。你的任务是设计两个不同风格的「检索质量评估 Prompt」。

## 背景
我们有一个 RAG 系统，用户发出 query，系统检索相关内容（contexts），agent 基于检索结果生成 response。
我们需要一个评估 Prompt，交给 gemini-3.6-flash 来打分，判断检索结果的质量。

## 评估维度
1. **relevance** (0.0-1.0): 检索结果与用户 query 的相关程度
2. **helpfulness** (0.0-1.0): 检索结果对 agent 完成任务的实际贡献程度（关键洞察：response 中应能看到 agent 确实使用了检索内容的证据）

## 你要生成的内容
生成两个评估 Prompt 模板，每个模板必须：
- 包含占位符：{query}、{contexts}、{response}
- 指示 gemini-3.6-flash 输出严格 JSON：{"relevance": 0.0-1.0, "helpfulness": 0.0-1.0, "reasoning": "..."}
- 两个 Prompt 采用不同的评估策略（例如：一个更分析性/逐步推理，另一个更整体性/直觉判断）

## 输出格式
严格输出 JSON，不要加 markdown 代码块：
{"prompts": ["<prompt_template_1>", "<prompt_template_2>"]}"""

EVOLVE_PROMPT = """你是一个 Prompt 进化专家。基于 Arena 对战历史，生成一个新的挑战者评估 Prompt。

## 当前冠军 Prompt（胜率最高的评估者）
{champion_text}

## 已退役的 Prompt（被淘汰的评估者，学习它们的失败教训）
{retired_texts}

## 最近几轮的裁判评语（了解 meta-judge 看重什么）
{recent_verdicts}

## 你的任务
生成一个新的挑战者评估 Prompt，它必须：
1. 包含占位符：{query}、{contexts}、{response}
2. 指示 gemini-3.6-flash 输出 JSON：{"relevance": 0.0-1.0, "helpfulness": 0.0-1.0, "reasoning": "..."}
3. 尝试与冠军不同的评估策略
4. 避免已退役 Prompt 的失败模式
5. 针对裁判评语中提到的问题进行优化

## 输出格式
严格输出 JSON，不要加 markdown 代码块：
{"prompt": "<new_prompt_template>"}"""


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class PromptGenerator:
    """Generates evaluator prompts via gpt-5.4."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or DEFAULT_GENERATOR_MODEL
        self._api_key = api_key or DEFAULT_API_KEY
        self._timeout = timeout
        self._max_retries = max_retries

    async def generate_seed_prompts(self) -> list[PromptEntry]:
        """Generate 2 initial evaluator prompts. Called once during arena initialization."""
        raw = await self._call_llm(SEED_PROMPT)
        prompts_text = self._parse_seed_response(raw)

        now = datetime.now(timezone.utc).isoformat()
        entries: list[PromptEntry] = []
        for text in prompts_text:
            entries.append(
                PromptEntry(
                    id=f"prompt-{uuid.uuid4().hex[:8]}",
                    version=1,
                    text=text,
                    elo=1500.0,
                    created_at=now,
                    parent_id=None,
                )
            )
        return entries

    async def generate_evolved_prompt(
        self,
        *,
        champion: PromptEntry,
        retired_prompts: list[PromptEntry],
        recent_rounds: list[RoundRecord],
        parent_id: str,
    ) -> PromptEntry:
        """Generate a new challenger prompt based on arena history."""
        retired_texts = (
            "\n---\n".join(
                f"[已退役] ELO={p.elo:.0f}, 战绩={p.wins}W/{p.losses}L\n{p.text[:500]}" for p in retired_prompts[-5:]
            )
            or "(无退役 Prompt)"
        )

        recent_verdicts = (
            "\n---\n".join(
                f"Round {r.round_id}: verdict={r.meta_verdict}\n理由: {r.meta_reasoning[:300]}"
                for r in recent_rounds[-5:]
            )
            or "(无历史对战记录)"
        )

        prompt = (
            EVOLVE_PROMPT.replace("{champion_text}", champion.text[:1000])
            .replace("{retired_texts}", retired_texts)
            .replace("{recent_verdicts}", recent_verdicts)
        )

        raw = await self._call_llm(prompt)
        new_text = self._parse_evolve_response(raw)

        now = datetime.now(timezone.utc).isoformat()
        return PromptEntry(
            id=f"prompt-{uuid.uuid4().hex[:8]}",
            version=1,
            text=new_text,
            elo=1500.0,
            created_at=now,
            parent_id=parent_id,
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
            "temperature": 0.7,
            "max_tokens": 3000,
        }

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_exc = e
                if _is_retryable_error(e) and attempt < self._max_retries:
                    delay = 1.0 * (2**attempt)
                    logger.warning(
                        "PromptGenerator LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    continue
                break

        raise RuntimeError(f"PromptGenerator LLM call failed after {self._max_retries + 1} attempts: {last_exc}")

    def _parse_seed_response(self, raw: str) -> list[str]:
        text = self._strip_markdown_fences(raw)

        try:
            data = json.loads(text)
            prompts = data.get("prompts", [])
            if len(prompts) >= 2:
                return prompts[:2]
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{\s*"prompts"\s*:\s*\[', text, re.DOTALL)
        if match:
            try:
                bracket_start = match.start()
                data = json.loads(text[bracket_start:])
                prompts = data.get("prompts", [])
                if len(prompts) >= 2:
                    return prompts[:2]
            except json.JSONDecodeError:
                pass

        # Fallback: extract quoted strings that look like prompt templates
        found = re.findall(r'"([^"]{100,})"', text)
        prompt_like = [s for s in found if "{query}" in s and "{contexts}" in s and "{response}" in s]
        if len(prompt_like) >= 2:
            return prompt_like[:2]

        raise ValueError(f"Failed to parse seed prompts from LLM response: {raw[:500]}")

    def _parse_evolve_response(self, raw: str) -> str:
        text = self._strip_markdown_fences(raw)

        try:
            data = json.loads(text)
            prompt = data.get("prompt", "")
            if prompt:
                return prompt
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{\s*"prompt"\s*:\s*"', text, re.DOTALL)
        if match:
            try:
                bracket_start = match.start()
                data = json.loads(text[bracket_start:])
                prompt = data.get("prompt", "")
                if prompt:
                    return prompt
            except json.JSONDecodeError:
                pass

        # Fallback: extract the longest quoted string containing placeholders
        found = re.findall(r'"([^"]{100,})"', text)
        prompt_like = [s for s in found if "{query}" in s and "{contexts}" in s and "{response}" in s]
        if prompt_like:
            return max(prompt_like, key=len)

        raise ValueError(f"Failed to parse evolved prompt from LLM response: {raw[:500]}")

    def _strip_markdown_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            end_idx = len(lines) - 1 if lines[-1].strip().startswith("```") else len(lines)
            text = "\n".join(lines[1:end_idx])
        return text.strip()
