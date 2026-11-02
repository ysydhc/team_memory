"""Personal memory extractor — extracts user preferences from conversations.

Triggered at convergence and session_end. Reads user messages from
ConversationBuffer, calls LLM to extract personal facts, and writes
them to the personal_memories table via PersonalMemoryService.

LLM call uses configurable endpoint (LiteLLM/OpenAI-compatible).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from team_memory.daemon.integration.intent_classifier import classify_persona
from team_memory.daemon.runtime.config import DaemonConfig, RefinementSettings

logger = logging.getLogger("tm_daemon.personal_memory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_PERSONAS = frozenset({"coding", "writing", "research", "debugging", "general"})

# Keywords indicating a single-session / single-feature ephemeral directive rather than a durable cross-project trait
_EPHEMERAL_TASK_PATTERNS = (
    "高德与滴滴",
    "下拉菜单",
    "卡片互相遮挡",
    "sandbox 页面",
    "sandbox页面",
    "302 跳转",
    "302跳转",
    "原始链接",
    "前端同学",
    "其他前端成员",
    "进入phase",
    "p1, p2",
)


def _is_ephemeral_task_memory(content: str) -> bool:
    """Return True if content looks like a one-off feature/bug instruction rather than a durable user trait."""
    text = (content or "").strip()
    if len(text) < 8:
        return True
    low = text.lower()
    if any(pat in low for pat in _EPHEMERAL_TASK_PATTERNS):
        return True
    if re.search(r"设置为\s*\d+\s*px", low):
        return True
    return False


# ---------------------------------------------------------------------------
# LLM Prompt
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
你是一个对话分析助手。从下面的用户发言中提取**跨项目、长期稳定**的个人工程画像与偏好。

重点提取:
1. 技术偏好 (编程语言、框架、工具链、编辑器、CLI/自动化测试倾向)
2. 工作习惯 (沟通风格、先调研后改码、决策评估方式如要求基准与打分)
3. 工程纪律 (Git 提交规范、真机验证要求、代码整洁与分层边界偏好)
4. 表达与文档偏好 (图表形式、信息密度、汇报结构)

严格过滤铁律（必须遵守）:
- 【去业务化】：严禁提取单次任务、单个页面或特定业务场景下的临时要求
  （例如："修复某页面下拉菜单"、"坚持使用原始302跳转链接"、"同时支持高德与滴滴"、"解决卡片遮挡"等一律不得提取！）。
- 如果某条发言体现了通用工程态度，必须剥离具体业务名词，抽象为跨所有项目通用的准则
  （例如将"拒绝改302链接规避问题"抽象为"排查问题时坚持修复根因，拒绝通过修改测试输入规避问题"）。
- 只提取明确表达或高置信度体现的持久习惯，如果没有可提取的持久跨项目信息，返回空数组 `"memories": []`。

每条记忆必须标注:
- `persona` 字段:
  - coding: 编程/架构/部署/测试相关的偏好
  - writing: 写作/文档/图表/结构化表达相关的偏好
  - research: 调研/对比/基准打分/方案分析相关的偏好
  - debugging: 排障/日志分析/真机验证相关的偏好
  - general: 跨领域通用沟通与协作习惯
- `priority` 字段 (50-85 整数):
  - 75-85: 用户明确强调的长期铁律或核心工程习惯
  - 55-70: 稳定的常规偏好

输出 JSON 格式:
```json
{{
  "memories": [
    {{"category": "preference", "persona": "coding", "priority": 75, "content": "具体跨项目偏好描述"}},
    {{"category": "habit", "persona": "research", "priority": 80, "content": "具体跨项目习惯描述"}}
  ]
}}
```

用户发言:
{user_messages}
"""


class PersonalMemoryExtractor:
    """Extracts personal facts from user messages and saves to TM.

    Args:
        config: DaemonConfig with LLM and TM settings.
    """

    def __init__(self, config: DaemonConfig) -> None:
        self._config = config
        self._llm: RefinementSettings = config.refinement
        self._user = config.tm.user

    async def extract_and_save(
        self,
        session_id: str,
        user_messages: list[dict[str, Any]],
    ) -> int:
        """Extract personal memories from user messages and save them.

        Args:
            session_id: Conversation session identifier.
            user_messages: List of message dicts with 'content' field.

        Returns:
            Number of memories saved.
        """
        if not user_messages:
            return 0

        # Concatenate user messages
        texts = [m.get("content", "") for m in user_messages if m.get("content")]
        if not texts:
            return 0

        combined = "\n---\n".join(texts)

        # Skip very short conversations (likely just commands)
        if len(combined) < 50:
            logger.debug("User messages too short (%d chars), skipping extraction", len(combined))
            return 0

        # Call LLM to extract
        try:
            memories = await self._call_llm(combined)
        except Exception:
            logger.exception("Failed to extract personal memories for session=%s", session_id)
            return 0

        if not memories:
            return 0

        # Save to personal_memories via TM service
        saved = 0
        for mem in memories:
            category = mem.get("category", "preference")
            content = str(mem.get("content", "") or "").strip()
            if not content or _is_ephemeral_task_memory(content):
                continue

            # Resolve persona: use LLM-extracted value, validate, fallback to classifier
            persona = mem.get("persona", "")
            if persona not in _VALID_PERSONAS:
                persona = classify_persona(content)

            raw_pri = mem.get("priority", 60)
            try:
                priority = max(50, min(90, int(raw_pri)))
            except (TypeError, ValueError):
                priority = 60

            try:
                try:
                    await self._save_memory(category, content, persona=persona, priority=priority)
                except TypeError:
                    await self._save_memory(category, content, persona=persona)
                saved += 1
            except Exception:
                logger.warning("Failed to save personal memory: %s", content[:50], exc_info=True)

        if saved:
            logger.info(
                "[PROFILE] Extracted %d personal memories from session=%s (user=%s)",
                saved,
                session_id[:8],
                self._user,
            )
        return saved

    async def _call_llm(self, user_text: str) -> list[dict[str, str]]:
        """Call LLM to extract personal facts.

        Uses the same LiteLLM/OpenAI endpoint as refinement.
        """
        prompt = _EXTRACT_PROMPT.format(user_messages=user_text[:3000])

        payload = {
            "model": self._llm.model,
            "messages": [
                {"role": "system", "content": "You are a personal information extractor. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
        }

        headers = {"Content-Type": "application/json"}
        if self._llm.api_key_env:
            import os

            api_key = os.environ.get(self._llm.api_key_env, "")
            if api_key:
                headers["x-api-key"] = api_key

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self._llm.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        # Parse LLM response
        content = data["choices"][0]["message"]["content"]
        # Try to extract JSON from response (may be wrapped in ```json...```)
        content = content.strip()
        if content.startswith("```"):
            # Strip markdown code block
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        parsed = json.loads(content)
        return parsed.get("memories", [])

    async def _save_memory(
        self,
        category: str,
        content: str,
        *,
        persona: str = "general",
        priority: int = 60,
    ) -> None:
        """Save a personal memory to the TM database.

        Maps extractor category to TM scope/profile_kind:
          preference/habit → scope=generic, profile_kind=static
          context/project  → scope=context,  profile_kind=dynamic
        """
        from team_memory.bootstrap import get_context
        from team_memory.services.memory.personal_memory import PersonalMemoryService

        ctx = get_context()
        svc = PersonalMemoryService(
            embedding_provider=ctx.embedding,
            db_url=ctx.db_url,
        )

        # Map category to scope
        if category in ("context", "project"):
            scope = "context"
        else:
            scope = "generic"

        await svc.write(
            user_id=self._user,
            content=content,
            scope=scope,
            context_hint=category,
            persona=persona,
            priority=priority,
        )
