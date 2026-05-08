"""Personal memory extractor — extracts user preferences from conversations.

Triggered at convergence and session_end. Reads user messages from
ConversationBuffer, calls LLM to extract personal facts, and writes
them to the personal_memories table via PersonalMemoryService.

LLM call uses configurable endpoint (LiteLLM/OpenAI-compatible).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from daemon.config import DaemonConfig, RefinementSettings
from daemon.intent_classifier import classify_persona

logger = logging.getLogger("tm_daemon.personal_memory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_PERSONAS = frozenset({"coding", "writing", "research", "debugging", "general"})

# ---------------------------------------------------------------------------
# LLM Prompt
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
你是一个对话分析助手。从下面的用户发言中提取与该用户个人相关的信息。

重点提取:
1. 技术偏好 (编程语言、框架、工具链、编辑器)
2. 工作习惯 (沟通风格、决策方式、时间偏好)
3. 项目上下文 (负责的项目、团队角色、工作环境)
4. 个人特征 (语言习惯、关注点、常见需求模式)

注意:
- 只提取明确表达或可直接推断的事实，不要猜测
- 忽略一次性的任务指令，关注可复用的持久信息
- 如果没有可提取的个人信息，返回空数组
- 每条记忆应独立且具体，避免泛泛而谈

每条记忆必须标注 persona 字段，表示该偏好属于哪个场景:
- coding: 编程/调试/部署/测试相关的偏好 (如 "偏好 pytest", "不喜欢 Java")
- writing: 写作/文档/翻译相关的偏好 (如 "中英双语输出", "注释用中文")
- research: 调研/对比/分析相关的偏好 (如 "先比较再决策")
- debugging: 排障/日志分析相关的偏好 (如 "先看日志再改代码")
- general: 通用偏好 (如 "不喜欢太长的回复")

输出 JSON 格式:
```json
{{
  "memories": [
    {{"category": "preference", "persona": "coding", "content": "具体偏好描述"}},
    {{"category": "habit", "persona": "general", "content": "具体习惯描述"}},
    {{"category": "context", "persona": "research", "content": "具体上下文描述"}}
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
            content = mem.get("content", "")
            if not content:
                continue

            # Resolve persona: use LLM-extracted value, validate, fallback to classifier
            persona = mem.get("persona", "")
            if persona not in _VALID_PERSONAS:
                persona = classify_persona(content)

            try:
                await self._save_memory(category, content, persona=persona)
                saved += 1
            except Exception:
                logger.warning("Failed to save personal memory: %s", content[:50], exc_info=True)

        if saved:
            logger.info(
                "[PROFILE] Extracted %d personal memories from session=%s (user=%s)",
                saved, session_id[:8], self._user,
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
                headers["Authorization"] = f"Bearer {api_key}"

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

    async def _save_memory(self, category: str, content: str, *, persona: str = "general") -> None:
        """Save a personal memory to the TM database.

        Maps extractor category to TM scope/profile_kind:
          preference/habit → scope=generic, profile_kind=static
          context/project  → scope=context,  profile_kind=dynamic
        """
        from team_memory.bootstrap import get_context
        from team_memory.services.personal_memory import PersonalMemoryService

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
        )
