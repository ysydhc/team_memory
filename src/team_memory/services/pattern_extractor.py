"""Pattern extraction from conversations — extract user behavior patterns.

Used during archive save to build PersonalMemory from raw conversations,
enabling the system to "understand users better over time".
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from team_memory.services.personal_memory import PersonalMemoryService

logger = logging.getLogger("team_memory.pattern_extractor")

PATTERN_EXTRACTION_PROMPT = """分析以下用户对话文本，提取用户反复使用的指令模式和偏好习惯。

对话文本：
{conversation}

请提取：
1. 用户常用的指令短语（如"请一步步思考"、"你有什么问题可以和我进行讨论"）
2. 用户偏好的工作方式（如"先搜索再执行"、"需要多角色审视"）
3. 用户对输出格式的要求（如"给出对比方案"、"在一个回答内说清楚"）

以 JSON 格式返回：
{{
  "patterns": [
    {{
      "pattern": "用户常说的话或行为模式",
      "category": "instruction_style|workflow_preference|output_format",
      "frequency_hint": "high|medium|low",
      "suggested_rule": "建议生成的规则描述"
    }}
  ]
}}"""

# Map pattern categories to PersonalMemory profile_kind
_CATEGORY_TO_PROFILE_KIND: dict[str, str] = {
    "instruction_style": "static",
    "workflow_preference": "static",
    "output_format": "dynamic",
}

MAX_CONVERSATION_CHARS = 8000


class PatternExtractor:
    """Extract user behavior patterns from conversations via LLM."""

    async def extract_patterns(self, conversation: str, llm_config: Any) -> list[dict[str, str]]:
        """Extract patterns from conversation text.

        Returns list of dicts with: pattern, category, frequency_hint, suggested_rule.
        """
        from team_memory.services.llm_provider import create_llm_provider

        llm = create_llm_provider(llm_config)
        prompt = PATTERN_EXTRACTION_PROMPT.format(
            conversation=conversation[:MAX_CONVERSATION_CHARS]
        )

        try:
            response = await llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            logger.warning("Pattern extraction LLM call failed: %s", e, exc_info=True)
            return []

        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.debug("Pattern extraction: no JSON in LLM response")
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.debug("Pattern extraction: invalid JSON in LLM response")
            return []

        return data.get("patterns", [])

    async def extract_and_save(
        self,
        conversation: str,
        user_id: str,
        llm_config: Any,
        pm_service: PersonalMemoryService,
    ) -> int:
        """Extract patterns and write to PersonalMemory. Returns count saved."""
        patterns = await self.extract_patterns(conversation, llm_config)
        if not patterns:
            return 0

        saved = 0
        for p in patterns:
            category = p.get("category", "instruction_style")
            profile_kind = _CATEGORY_TO_PROFILE_KIND.get(category, "static")
            try:
                await pm_service.write(
                    user_id=user_id,
                    content=p.get("suggested_rule") or p["pattern"],
                    scope=category,
                    context_hint=p["pattern"][:200],
                    profile_kind=profile_kind,
                )
                saved += 1
            except Exception:
                logger.debug("Failed to save pattern: %s", p.get("pattern", ""), exc_info=True)

        logger.info(
            "pattern_extractor: saved %d/%d patterns for user=%s",
            saved,
            len(patterns),
            user_id,
        )
        return saved
