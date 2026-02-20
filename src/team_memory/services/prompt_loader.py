"""Prompt template loading, variable substitution, and AI behavior injection.

Extracted from llm_parser.py to provide a reusable prompt management service
for all AI-related operations (parsing, enrichment, review, suggestions).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team_memory.config import AIBehaviorConfig
    from team_memory.schemas import SchemaRegistry

logger = logging.getLogger("team_memory.prompt_loader")

_BUILTIN_PROMPTS: dict[str, str] = {}


def _register_builtin(name: str, text: str) -> str:
    _BUILTIN_PROMPTS[name] = text
    return text


PARSE_SINGLE = _register_builtin(
    "parse_single",
    """你是一个技术文档分析助手。用户会提供一段技术文档
（可能是 Markdown 或纯文本），你需要从中提取结构化的开发经验信息。

请严格以 JSON 格式返回以下字段（不要包含其他内容，不要用 markdown 代码块包裹）:
{
  "title": "简洁的经验标题（一句话概括）",
  "problem": "问题描述（遇到了什么问题，上下文是什么）",
  "root_cause": "根因分析（为什么会出现这个问题，没有则为 null）",
  "solution": "解决方案（如何解决的，关键步骤，还没有解决则为 null）",
  "tags": ["标签1", "标签2", "标签3"],
  "language": "编程语言（如 python/javascript/go 等，没有则为 null）",
  "framework": "框架名称（如 fastapi/react/spring 等，没有则为 null）",
  "code_snippets": "文档中的关键代码片段（没有则为 null）",
  "experience_type": "经验类型",
  "type_confidence": 0.85,
  "severity": "严重等级（仅需要严重等级的类型才填，其他为 null）",
  "category": "分类",
  "structured_data": {},
  "git_refs": []
}

experience_type 必须是以下之一:
{{experience_types}}

type_confidence: 0.0-1.0，表示你对类型判断的置信度

severity: 取值 {{severity_levels}}

category: 分类，取值 {{categories}}，没有则为 null

structured_data: 根据 experience_type 提取的结构化数据（只填入能从文档中确认的字段）

git_refs: 从文档中提取的 git 引用，格式为
[{"type": "commit"|"pr"|"branch", "hash": "...", "url": "...", "description": "..."}]

注意:
- title 要简洁，不超过 50 个字
- problem 和 solution 要详细，保留关键技术细节
- solution 可以为 null（例如问题已确认但尚未解决）
- root_cause 分析问题的根本原因
- tags 提取 3-8 个相关的技术关键词，小写英文
- code_snippets 只保留最关键的代码
- structured_data 中只填入能从文档中确认的字段，没有的填 null
""",
)

PARSE_GROUP = _register_builtin(
    "parse_group",
    """你是一个技术文档分析助手。用户会提供一段较长的技术文档或对话记录，
其中可能包含多个步骤或多个相关经验。你需要将其拆分为一个"父经验"（整体概述）
和多个"子经验"（具体步骤/子问题）。

请严格以 JSON 格式返回以下结构（不要包含其他内容，不要用 markdown 代码块包裹）:
{
  "parent": {
    "title": "整体经验标题（一句话概括全部内容）",
    "problem": "整体问题描述",
    "root_cause": "整体根因分析（没有则为 null）",
    "solution": "整体解决方案摘要",
    "tags": ["标签1", "标签2"],
    "language": "主要编程语言（没有则为 null）",
    "framework": "主要框架（没有则为 null）",
    "code_snippets": null
  },
  "children": [
    {
      "title": "子步骤1的标题",
      "problem": "子步骤1的问题描述",
      "root_cause": null,
      "solution": "子步骤1的解决方案",
      "tags": ["标签"],
      "code_snippets": "关键代码片段或 null"
    }
  ]
}

注意:
- parent.title 概括全局，children 各自描述一个独立步骤
- 子经验数量不要超过 5 个，合并关系紧密的步骤
- 每个子经验都应有独立的 problem + solution
- tags 用小写英文
- 如果内容不适合拆分为多步骤，返回 children 为空数组 []
""",
)

SUGGEST_TYPE = _register_builtin(
    "suggest_type",
    """你是一个经验分类助手。根据以下内容，判断最适合的经验类型。

可选类型:
{{experience_types}}

请严格以 JSON 格式返回（不要包含其他内容）:
{
  "type": "bugfix",
  "confidence": 0.85,
  "reason": "内容提到了报错信息和修复步骤",
  "fallback_types": ["incident", "general"]
}

- confidence: 0.0-1.0，表示你对类型判断的置信度
- reason: 简短说明判断依据（一句话，中文）
- fallback_types: 备选类型（最多2个，按可能性排序）
""",
)

SUMMARY = _register_builtin(
    "summary",
    """你是一个技术文档摘要助手。用户会提供一条开发经验的详细内容，
你需要生成一段简洁的摘要。

要求:
- 摘要不超过 150 字（中文）或 300 字符（英文）
- 保留最核心的问题和解决思路
- 去除冗余细节和代码片段
- 确保摘要能让读者快速判断此经验是否与自己的问题相关
- 直接返回摘要文本，不要用 JSON 格式，不要包含 "摘要:" 等前缀
""",
)


class PromptLoader:
    """Loads, substitutes variables, and injects AI behavior into prompts."""

    def __init__(
        self,
        schema_registry: SchemaRegistry | None = None,
        ai_behavior: AIBehaviorConfig | None = None,
        prompt_dir: str | None = None,
    ):
        self._registry = schema_registry
        self._behavior = ai_behavior
        self._prompt_dir = prompt_dir

    def load(self, name: str, **extra_vars: str) -> str:
        """Load a prompt by name with variable substitution and AI behavior.

        Loading order:
          1. File at ``{prompt_dir}/{name}.md``
          2. Built-in default
          3. Variable substitution
          4. AI behavior constraints appended
        """
        prompt_text = self._load_from_file(name)
        if prompt_text is None:
            prompt_text = _BUILTIN_PROMPTS.get(name, "")

        if "{{" in prompt_text:
            prompt_text = self._substitute(prompt_text, extra_vars)

        if self._behavior:
            block = build_behavior_block(self._behavior)
            if block:
                prompt_text = prompt_text.rstrip() + "\n\n" + block

        return prompt_text

    def available_prompts(self) -> list[str]:
        """List all available prompt names (file + built-in)."""
        names = set(_BUILTIN_PROMPTS.keys())
        if self._prompt_dir:
            p = Path(self._prompt_dir)
            if p.is_dir():
                names |= {f.stem for f in p.glob("*.md")}
        return sorted(names)

    def _load_from_file(self, name: str) -> str | None:
        if not self._prompt_dir:
            return None
        file_path = Path(self._prompt_dir) / f"{name}.md"
        if not file_path.exists():
            return None
        try:
            text = file_path.read_text(encoding="utf-8")
            logger.debug("Loaded prompt '%s' from file: %s", name, file_path)
            return text
        except Exception as e:
            logger.warning("Failed to load prompt file %s: %s", file_path, e)
            return None

    def _substitute(self, text: str, extra: dict[str, str]) -> str:
        if self._registry:
            text = text.replace("{{experience_types}}", self._registry.types_for_prompt())
            text = text.replace("{{categories}}", self._registry.categories_for_prompt())
            text = text.replace(
                "{{severity_levels}}", self._registry.severity_for_prompt()
            )
        for key, val in extra.items():
            text = text.replace(f"{{{{{key}}}}}", val)
        return text


def build_behavior_block(ai_behavior: AIBehaviorConfig) -> str:
    """Build an AI behavior constraints section to append to prompts."""
    lines: list[str] = ["## 团队定制要求"]

    lang_map = {
        "zh-CN": "中文",
        "en": "English",
        "ja": "日本語",
        "ko": "한국어",
        "zh-TW": "繁體中文",
    }
    lines.append(
        f"- 输出语言：{lang_map.get(ai_behavior.output_language, ai_behavior.output_language)}"
    )

    detail_map = {"detailed": "详细", "concise": "简洁"}
    lines.append(
        f"- 详细程度：{detail_map.get(ai_behavior.detail_level, ai_behavior.detail_level)}"
    )

    if ai_behavior.focus_areas:
        focus_map = {
            "root_cause": "根因分析",
            "solution": "解决方案",
            "code_snippets": "关键代码",
            "reproduction_steps": "复现步骤",
            "performance": "性能数据",
            "architecture": "架构设计",
        }
        labels = [focus_map.get(f, f) for f in ai_behavior.focus_areas]
        lines.append(f"- 重点关注：{'、'.join(labels)}")

    if ai_behavior.custom_instructions:
        lines.append(f"- 团队指令：{ai_behavior.custom_instructions}")

    return "\n".join(lines)
