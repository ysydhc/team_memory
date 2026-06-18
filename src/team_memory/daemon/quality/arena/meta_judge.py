from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass

import httpx

from team_memory.daemon.quality.arena.models import EvalResult, TaskEval

logger = logging.getLogger("arena.meta_judge")

DEFAULT_BASE_URL = os.environ.get("ARENA_LLM_BASE_URL", "http://localhost:4000/v1")
DEFAULT_META_MODEL = os.environ.get("ARENA_META_MODEL", "gpt-5.4")
DEFAULT_API_KEY = os.environ.get("ARENA_LLM_API_KEY", "") or os.environ.get("LITELLM_MASTER_KEY", "")

META_JUDGE_PROMPT = """你是一位高级评估质量裁判。你的任务是比较两位评估者对 RAG 检索结果的评估质量，判断哪位评估者做得更好。

## 评估任务数据

以下是被评估的检索任务及其原始数据：

{task_blocks}

## 你的判断标准

请从以下三个维度综合判断哪位评估者表现更好：

1. **相关性判断准确度**：评估者是否准确判断了检索结果与查询的相关程度？分数是否合理反映了实际相关性？
2. **帮助性判断准确度**：评估者是否准确判断了检索结果对 Agent 完成任务的实际帮助程度？是否考虑了 Agent 回复的质量？
3. **推理质量**：评估者的推理是否清晰、逻辑严密、有说服力？是否指出了关键问题或优点？

## 判断原则

- 更好的评估者应该给出与实际情况更匹配的分数
- 更好的评估者应该有更清晰、更具体的推理，而非泛泛而谈
- 如果两位评估者质量相当（差异很小），请判定为平局
- 不要因为评估者给分高或低就偏向某一方，关键是分数是否准确

## 反策略博弈规则（重要）

{anti_collapse_note}

以下情形必须识别并惩罚：

1. **全零策略**：如果一位评估者对所有任务都给出 0.0 分（或接近 0.0），这是退化策略的信号。请逐项验证：
   - 如果某个任务的 context 确实包含与 query 直接相关的信息（如同一文件、同一功能、同一 bug），给 0.0 是错误的
   - 即使多数任务的 context 确实无关，只要有一个任务被错误判为 0 而实际相关性 ≥ 0.3，全零评估者就不应获胜
   - "永远说不相关"和"永远说相关"一样是偷懒策略

2. **区分度要求**：好的评估者应该能区分"完全无关"(0.0-0.2)、"边缘相关"(0.3-0.5)、"部分相关"(0.5-0.7)、"高度相关"(0.8-1.0)。只使用 0.0 或只使用高分的评估者在区分度上有缺陷。

3. **逐任务独立判断**：不要被"多数任务准确"掩盖"关键任务严重错误"。每个任务的准确性应独立计分，然后综合。

## 输出格式

严格输出 JSON，不要加 markdown 代码块：
{"winner": "a_wins" 或 "b_wins" 或 "draw", "reasoning": "你的判断理由（简洁明了）"}"""

TASK_BLOCK_TEMPLATE = """### 任务 {index}

**查询**: {query}

**检索到的上下文**:
{contexts}

**Agent 回复**: {agent_response}

**评估者 A 的评估**:
- 相关性: {eval_a_relevance:.2f}
- 帮助性: {eval_a_helpfulness:.2f}
- 推理: {eval_a_reasoning}

**评估者 B 的评估**:
- 相关性: {eval_b_relevance:.2f}
- 帮助性: {eval_b_helpfulness:.2f}
- 推理: {eval_b_reasoning}"""


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


@dataclass
class MetaVerdict:
    winner: str  # "a_wins" | "b_wins" | "draw"
    reasoning: str


class MetaJudge:
    """GPT-5.4 based judge that compares two evaluations and picks the better one."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 90,
        max_retries: int = 2,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or DEFAULT_META_MODEL
        self._api_key = api_key or DEFAULT_API_KEY
        self._timeout = timeout
        self._max_retries = max_retries

    async def judge(
        self,
        *,
        tasks: list[TaskEval],
        query_contexts: list[tuple[str, list[str], str]],
    ) -> MetaVerdict:
        """Compare two sets of evaluations and decide which evaluator did better.

        Args:
            tasks: Evaluated tasks with eval_a and eval_b filled.
            query_contexts: (query, contexts, agent_response) for each task,
                           aligned by index with tasks.

        Returns:
            MetaVerdict with winner and reasoning.
        """
        prompt = self._build_prompt(tasks, query_contexts)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = await self._call_llm(prompt)
                verdict = self._parse_verdict(raw)
                if attempt > 0:
                    logger.info(
                        "Meta judge succeeded on attempt %d/%d",
                        attempt + 1,
                        self._max_retries + 1,
                    )
                return verdict
            except Exception as e:
                last_exc = e
                if _is_retryable_error(e) and attempt < self._max_retries:
                    delay = 1.0 * (2**attempt)
                    logger.warning(
                        "Meta judge LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.warning(
                        "Meta judge failed after %d attempts: %s",
                        attempt + 1,
                        e,
                    )
                    break

        logger.error("Meta judge returning draw due to failure: %s", last_exc)
        return MetaVerdict(winner="draw", reasoning=f"judge_error: {last_exc}")

    def _build_prompt(
        self,
        tasks: list[TaskEval],
        query_contexts: list[tuple[str, list[str], str]],
    ) -> str:
        blocks: list[str] = []
        for i, (task, qc) in enumerate(zip(tasks, query_contexts)):
            query, contexts, agent_response = qc

            context_text = "\n".join(f"  [{j + 1}] {ctx[:300]}" for j, ctx in enumerate(contexts)) or "  (无检索结果)"

            eval_a = task.eval_a or EvalResult(relevance=0.0, helpfulness=0.0, reasoning="(未完成评估)")
            eval_b = task.eval_b or EvalResult(relevance=0.0, helpfulness=0.0, reasoning="(未完成评估)")

            reasoning_a = eval_a.reasoning if isinstance(eval_a.reasoning, str) else str(eval_a.reasoning)
            reasoning_b = eval_b.reasoning if isinstance(eval_b.reasoning, str) else str(eval_b.reasoning)

            block = (
                TASK_BLOCK_TEMPLATE.replace("{index}", str(i + 1))
                .replace("{query}", query[:200])
                .replace("{contexts}", context_text[:1500])
                .replace("{agent_response}", agent_response[:500])
                .replace("{eval_a_relevance}", f"{eval_a.relevance:.2f}")
                .replace("{eval_a_helpfulness}", f"{eval_a.helpfulness:.2f}")
                .replace("{eval_a_reasoning}", (reasoning_a[:300] or "(无推理)"))
                .replace("{eval_b_relevance}", f"{eval_b.relevance:.2f}")
                .replace("{eval_b_helpfulness}", f"{eval_b.helpfulness:.2f}")
                .replace("{eval_b_reasoning}", (reasoning_b[:300] or "(无推理)"))
            )
            blocks.append(block)

        anti_collapse_note = self._detect_collapse_pattern(tasks)
        task_blocks = "\n\n".join(blocks)
        return META_JUDGE_PROMPT.replace("{task_blocks}", task_blocks).replace(
            "{anti_collapse_note}", anti_collapse_note
        )

    @staticmethod
    def _detect_collapse_pattern(tasks: list[TaskEval]) -> str:
        """Detect if either evaluator shows collapse-to-zero or collapse-to-max pattern."""
        if not tasks:
            return ""

        a_scores: list[float] = []
        b_scores: list[float] = []
        for t in tasks:
            if t.eval_a:
                a_scores.extend([t.eval_a.relevance, t.eval_a.helpfulness])
            if t.eval_b:
                b_scores.extend([t.eval_b.relevance, t.eval_b.helpfulness])

        warnings: list[str] = []
        threshold = 0.05

        if a_scores and all(s <= threshold for s in a_scores):
            warnings.append(
                "⚠️ 注意：评估者 A 对所有任务都给出了接近 0 的分数。"
                "请仔细验证每个任务的 context 是否真的完全无关，"
                "还是评估者 A 在使用「全部否定」的懒惰策略。"
            )
        elif a_scores and all(s >= 0.8 for s in a_scores):
            warnings.append(
                "⚠️ 注意：评估者 A 对所有任务都给出了 ≥0.8 的高分。请验证每个任务的 context 是否真的都高度相关。"
            )

        if b_scores and all(s <= threshold for s in b_scores):
            warnings.append(
                "⚠️ 注意：评估者 B 对所有任务都给出了接近 0 的分数。"
                "请仔细验证每个任务的 context 是否真的完全无关，"
                "还是评估者 B 在使用「全部否定」的懒惰策略。"
            )
        elif b_scores and all(s >= 0.8 for s in b_scores):
            warnings.append(
                "⚠️ 注意：评估者 B 对所有任务都给出了 ≥0.8 的高分。请验证每个任务的 context 是否真的都高度相关。"
            )

        if not warnings:
            return "本轮未检测到明显的策略退化信号。"

        return "\n".join(warnings)

    async def _call_llm(self, prompt: str) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _parse_verdict(self, raw: str) -> MetaVerdict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
            return self._extract_verdict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return self._extract_verdict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        logger.warning("Meta judge parse failed, defaulting to draw. Raw: %s", raw[:200])
        return MetaVerdict(winner="draw", reasoning="parse_error: could not extract JSON verdict")

    def _extract_verdict(self, data: dict) -> MetaVerdict:
        winner = data["winner"]
        valid_winners = {"a_wins", "b_wins", "draw"}
        if winner not in valid_winners:
            raise ValueError(f"Invalid winner value: {winner}")
        reasoning = data.get("reasoning", "")
        return MetaVerdict(winner=winner, reasoning=reasoning)
