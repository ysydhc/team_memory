"""LLM-based content parser — extracts structured experience fields from text.

Shared module used by both the Web API (parse-document/parse-url endpoints)
and the MCP server (tm_learn tool). Calls an Ollama-compatible LLM to parse
free-form text into structured experience fields.

Prompt loading order (progressive complexity):
  1. Custom file in prompt_dir (if configured)
  2. Built-in default constant
  3. ai_behavior config appended as constraints
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from team_memory.config import AIBehaviorConfig, LLMConfig

logger = logging.getLogger("team_memory.llm_parser")

# ============================================================
# Built-in Default Prompts (used when no custom file exists)
# ============================================================

_BUILTIN_PARSE_SINGLE = """你是一个技术文档分析助手。用户会提供一段技术文档
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
"""

_BUILTIN_PARSE_GROUP = """你是一个技术文档分析助手。用户会提供一段较长的技术文档或对话记录，
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
"""

_BUILTIN_SUGGEST_TYPE = """你是一个经验分类助手。根据以下内容，判断最适合的经验类型。

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
"""

_BUILTIN_SUMMARY = """你是一个技术文档摘要助手。用户会提供一条开发经验的详细内容，
你需要生成一段简洁的摘要。

要求:
- 摘要不超过 150 字（中文）或 300 字符（英文）
- 保留最核心的问题和解决思路
- 去除冗余细节和代码片段
- 确保摘要能让读者快速判断此经验是否与自己的问题相关
- 直接返回摘要文本，不要用 JSON 格式，不要包含 "摘要:" 等前缀
"""

# Map prompt names to built-in defaults
_BUILTIN_PROMPTS: dict[str, str] = {
    "parse_single": _BUILTIN_PARSE_SINGLE,
    "parse_group": _BUILTIN_PARSE_GROUP,
    "suggest_type": _BUILTIN_SUGGEST_TYPE,
    "summary": _BUILTIN_SUMMARY,
}

# Backward-compatible aliases
PARSE_SINGLE_PROMPT = _BUILTIN_PARSE_SINGLE
PARSE_GROUP_PROMPT = _BUILTIN_PARSE_GROUP


# ============================================================
# Prompt Loader
# ============================================================


def load_prompt(
    name: str,
    llm_config: "LLMConfig | None" = None,
    ai_behavior: "AIBehaviorConfig | None" = None,
) -> str:
    """Load a prompt by name with variable substitution and ai_behavior injection.

    Loading order:
      1. File at ``{prompt_dir}/{name}.md`` (if prompt_dir configured and file exists)
      2. Built-in default constant
      3. Variable substitution: ``{{experience_types}}``, ``{{categories}}``, etc.
      4. ai_behavior constraints appended at the end

    Args:
        name: Prompt name (e.g. "parse_single", "parse_group", "suggest_type", "summary").
        llm_config: LLM config (for prompt_dir path).
        ai_behavior: AI behavior preferences to inject.

    Returns:
        Final prompt string ready for LLM consumption.
    """
    prompt_text: str | None = None

    # Step 1: Try loading from file
    prompt_dir = llm_config.prompt_dir if llm_config else None
    if prompt_dir:
        file_path = Path(prompt_dir) / f"{name}.md"
        if file_path.exists():
            try:
                prompt_text = file_path.read_text(encoding="utf-8")
                logger.debug("Loaded prompt '%s' from file: %s", name, file_path)
            except Exception as e:
                logger.warning("Failed to load prompt file %s: %s", file_path, e)

    # Step 2: Fall back to built-in
    if prompt_text is None:
        prompt_text = _BUILTIN_PROMPTS.get(name, "")

    # Step 3: Variable substitution from SchemaRegistry
    if "{{" in prompt_text:
        prompt_text = _substitute_variables(prompt_text)

    # Step 4: Append ai_behavior constraints
    if ai_behavior is None:
        try:
            from team_memory.config import get_settings
            settings = get_settings()
            ai_behavior = settings.ai_behavior
        except Exception:
            pass

    if ai_behavior:
        behavior_block = _build_behavior_block(ai_behavior)
        if behavior_block:
            prompt_text = prompt_text.rstrip() + "\n\n" + behavior_block

    return prompt_text


def _substitute_variables(prompt_text: str) -> str:
    """Replace ``{{variable}}`` placeholders with SchemaRegistry values."""
    try:
        from team_memory.schemas import get_schema_registry
        registry = get_schema_registry()
    except Exception:
        return prompt_text

    replacements = {
        "{{experience_types}}": registry.types_for_prompt(),
        "{{categories}}": registry.categories_for_prompt(),
        "{{severity_levels}}": registry.severity_for_prompt(),
    }
    for placeholder, value in replacements.items():
        prompt_text = prompt_text.replace(placeholder, value)
    return prompt_text


def _build_behavior_block(ai_behavior: "AIBehaviorConfig") -> str:
    """Build an ai_behavior constraints section to append to prompts."""
    lines: list[str] = []
    lines.append("## 团队定制要求")

    lang_map = {
        "zh-CN": "中文", "en": "English", "ja": "日本語",
        "ko": "한국어", "zh-TW": "繁體中文",
    }
    lang_display = lang_map.get(ai_behavior.output_language, ai_behavior.output_language)
    lines.append(f"- 输出语言：{lang_display}")

    detail_map = {"detailed": "详细", "concise": "简洁"}
    detail = detail_map.get(ai_behavior.detail_level, ai_behavior.detail_level)
    lines.append(f"- 详细程度：{detail}")

    if ai_behavior.focus_areas:
        focus_map = {
            "root_cause": "根因分析", "solution": "解决方案",
            "code_snippets": "关键代码", "reproduction_steps": "复现步骤",
            "performance": "性能数据", "architecture": "架构设计",
        }
        focus_labels = [focus_map.get(f, f) for f in ai_behavior.focus_areas]
        lines.append(f"- 重点关注：{'、'.join(focus_labels)}")

    if ai_behavior.custom_instructions:
        lines.append(f"- 团队指令：{ai_behavior.custom_instructions}")

    return "\n".join(lines)


# ============================================================
# Parser
# ============================================================


class LLMParseError(Exception):
    """Raised when LLM parsing fails."""

    pass


async def parse_content(
    content: str,
    llm_config: "LLMConfig | None" = None,
    as_group: bool = False,
    max_input_chars: int = 8000,
) -> dict:
    """Parse text content into structured experience fields using an LLM.

    Args:
        content: Free-form text (document, conversation, etc.)
        llm_config: LLM configuration. If None, uses default Ollama config.
        as_group: If True, parse as parent + children group.
        max_input_chars: Maximum input characters sent to LLM.

    Returns:
        For as_group=False: dict with title, problem, solution, tags, etc.
        For as_group=True: dict with "parent" and "children" keys.

    Raises:
        LLMParseError: If LLM connection fails or response is unparseable.
    """
    llm_model = "gpt-oss:20b-cloud"
    llm_base_url = "http://localhost:11434"
    if llm_config is not None:
        llm_model = llm_config.model
        llm_base_url = llm_config.base_url

    prompt_name = "parse_group" if as_group else "parse_single"
    system_prompt = load_prompt(prompt_name, llm_config=llm_config)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{llm_base_url.rstrip('/')}/api/chat",
                json={
                    "model": llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content[:max_input_chars]},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        raise LLMParseError(
            f"Cannot connect to Ollama at {llm_base_url}. Make sure Ollama is running."
        )
    except httpx.HTTPStatusError as e:
        raise LLMParseError(f"Ollama API error: {e.response.text[:200]}")

    # Extract the LLM response text
    llm_text = data.get("message", {}).get("content", "")
    if not llm_text:
        raise LLMParseError("LLM returned empty response")

    # Parse JSON from LLM response
    parsed = _extract_json(llm_text)

    if as_group:
        return _normalize_group(parsed)
    else:
        return _normalize_single(parsed)


def _extract_json(llm_text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    clean_text = llm_text.strip()

    # Strip ```json ... ``` wrappers
    if clean_text.startswith("```"):
        lines = clean_text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        clean_text = "\n".join(lines).strip()

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Try extracting the first JSON object
        start = clean_text.find("{")
        end = clean_text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(clean_text[start:end])
            except json.JSONDecodeError:
                pass
        raise LLMParseError(
            f"Failed to parse LLM response as JSON. Raw: {llm_text[:300]}"
        )


def _normalize_single(parsed: dict) -> dict:
    """Normalize a single experience parse result."""
    result = {
        "title": str(parsed.get("title", "")).strip(),
        "problem": str(parsed.get("problem", "")).strip(),
        "root_cause": str(parsed.get("root_cause", "")).strip() or None,
        "solution": str(parsed.get("solution", "")).strip() or None,
        "tags": parsed.get("tags", []),
        "language": parsed.get("language") or None,
        "framework": parsed.get("framework") or None,
        "code_snippets": parsed.get("code_snippets") or None,
        # v3: Type system fields
        "experience_type": str(parsed.get("experience_type", "general")).strip(),
        "type_confidence": float(parsed.get("type_confidence", 0.5)),
        "severity": parsed.get("severity") or None,
        "category": parsed.get("category") or None,
        "structured_data": parsed.get("structured_data") or None,
        "git_refs": parsed.get("git_refs") or None,
    }

    if not isinstance(result["tags"], list):
        result["tags"] = []
    result["tags"] = [str(t).strip().lower() for t in result["tags"] if t]

    # Validate experience_type via SchemaRegistry (dynamic)
    from team_memory.schemas import get_schema_registry
    registry = get_schema_registry()
    if not registry.is_valid_type(result["experience_type"]):
        result["experience_type"] = "general"
        result["type_confidence"] = 0.3

    # Validate severity
    if result["severity"] and not registry.is_valid_severity(result["severity"]):
        result["severity"] = None

    # Validate category
    if result["category"] and not registry.is_valid_category(result["category"]):
        result["category"] = None

    # Validate structured_data
    if result["structured_data"] and isinstance(result["structured_data"], dict):
        from team_memory.schemas import validate_structured_data
        try:
            result["structured_data"] = validate_structured_data(
                result["experience_type"], result["structured_data"]
            )
        except Exception:
            result["structured_data"] = None

    # Validate git_refs
    if result["git_refs"] and isinstance(result["git_refs"], list):
        from team_memory.schemas import validate_git_refs
        try:
            result["git_refs"] = validate_git_refs(result["git_refs"])
        except Exception:
            result["git_refs"] = None

    return result


def _normalize_group(parsed: dict) -> dict:
    """Normalize a group experience parse result (parent + children)."""
    parent_raw = parsed.get("parent", parsed)
    parent = _normalize_single(parent_raw)

    children = []
    for child_raw in parsed.get("children", []):
        children.append(_normalize_single(child_raw))

    return {"parent": parent, "children": children}


# ============================================================
# Type Suggestion (lightweight classification)
# ============================================================

# Legacy constant for backward compat — actual prompt is loaded via load_prompt()
SUGGEST_TYPE_PROMPT = _BUILTIN_SUGGEST_TYPE


async def suggest_experience_type(
    title: str,
    problem: str = "",
    llm_config: "LLMConfig | None" = None,
) -> dict:
    """Suggest experience type based on title and problem description.

    Lightweight LLM call for real-time type recommendation.

    Args:
        title: Experience title.
        problem: Problem description (optional).
        llm_config: LLM configuration.

    Returns:
        Dict with suggested_type, confidence, reason, fallback_types.
    """
    llm_model = "gpt-oss:20b-cloud"
    llm_base_url = "http://localhost:11434"
    if llm_config is not None:
        llm_model = llm_config.model
        llm_base_url = llm_config.base_url

    user_content = f"标题：{title}"
    if problem:
        user_content += f"\n问题描述：{problem}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{llm_base_url.rstrip('/')}/api/chat",
                json={
                    "model": llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": load_prompt("suggest_type", llm_config=llm_config),
                        },
                        {"role": "user", "content": user_content[:2000]},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.ConnectError, httpx.HTTPStatusError) as e:
        logger.warning("Type suggestion LLM call failed: %s", e)
        return {
            "suggested_type": "general",
            "confidence": 0.0,
            "reason": "LLM 不可用，使用默认类型",
            "fallback_types": [],
        }

    llm_text = data.get("message", {}).get("content", "")
    if not llm_text:
        return {
            "suggested_type": "general",
            "confidence": 0.0,
            "reason": "LLM 返回为空",
            "fallback_types": [],
        }

    try:
        parsed = _extract_json(llm_text)
    except LLMParseError:
        return {
            "suggested_type": "general",
            "confidence": 0.0,
            "reason": "无法解析 LLM 返回结果",
            "fallback_types": [],
        }

    # Validate and normalize via SchemaRegistry
    from team_memory.schemas import get_schema_registry
    registry = get_schema_registry()
    suggested = str(parsed.get("type", "general")).strip()
    if not registry.is_valid_type(suggested):
        suggested = "general"

    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason", "")).strip()
    fallbacks = parsed.get("fallback_types", [])
    if not isinstance(fallbacks, list):
        fallbacks = []
    fallbacks = [
        str(f).strip() for f in fallbacks
        if registry.is_valid_type(str(f).strip())
    ][:2]

    return {
        "suggested_type": suggested,
        "confidence": confidence,
        "reason": reason,
        "fallback_types": fallbacks,
    }


# ============================================================
# Summary Generation (P0-4: Memory Compaction)
# ============================================================

# Legacy constant — actual prompt loaded via load_prompt()
SUMMARY_PROMPT = _BUILTIN_SUMMARY


async def generate_summary(
    content: str,
    llm_config: "LLMConfig | None" = None,
    max_input_chars: int = 4000,
) -> str:
    """Generate a concise summary for an experience using LLM.

    Args:
        content: Full experience text (title + problem + solution).
        llm_config: LLM configuration.
        max_input_chars: Maximum input characters.

    Returns:
        Summary text string.

    Raises:
        LLMParseError: If LLM connection fails.
    """
    llm_model = "gpt-oss:20b-cloud"
    llm_base_url = "http://localhost:11434"
    if llm_config is not None:
        if hasattr(llm_config, "model"):
            llm_model = llm_config.model
        if hasattr(llm_config, "base_url"):
            llm_base_url = llm_config.base_url

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{llm_base_url.rstrip('/')}/api/chat",
                json={
                    "model": llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": load_prompt("summary", llm_config=llm_config),
                        },
                        {"role": "user", "content": content[:max_input_chars]},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        raise LLMParseError(
            f"Cannot connect to Ollama at {llm_base_url}. Make sure Ollama is running."
        )
    except httpx.HTTPStatusError as e:
        raise LLMParseError(f"Ollama API error: {e.response.text[:200]}")

    summary = data.get("message", {}).get("content", "").strip()
    if not summary:
        raise LLMParseError("LLM returned empty summary")

    return summary
