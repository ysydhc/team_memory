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
    from team_memory.config import LLMConfig

logger = logging.getLogger("team_memory.llm_parser")

# ============================================================
# Built-in Default Prompts (used when no custom file exists)
# ============================================================

_BUILTIN_PARSE_SINGLE = """你是一个技术文档分析助手。用户会提供一段技术文档
（可能是 Markdown 或纯文本），你需要从中提取结构化的开发经验信息。

请严格按照以下 5 个维度深度提取知识，然后以 JSON 格式返回
（不要包含其他内容，不要用 markdown 代码块包裹）:

**提取维度**:
1. decisions — 决策点：做了什么选择、有哪些备选方案、为什么选择当前方案
2. pitfalls — 陷阱与问题：意外发生的问题、根因分析、修复方案
3. patterns — 可复用模式：架构模式、代码模板、最佳实践
4. verification — 验证清单：如何验证方案正确性、测试步骤、通过标准
5. constraints — 约束条件：环境依赖、版本要求、配置前提

JSON 格式:
{
  "title": "简洁的经验标题（一句话概括）",
  "problem": "问题描述（遇到了什么问题，上下文是什么）",
  "root_cause": "根因分析（为什么会出现这个问题，没有则为 null）",
  "solution": "解决方案（如何解决的，关键步骤，还没有解决则为 null）",
  "decisions": "决策说明：选了什么方案、备选有哪些、选择理由（没有则为 null）",
  "pitfalls": "遇到的陷阱或意外问题，以及如何修复（没有则为 null）",
  "patterns": "可复用的模式或最佳实践（没有则为 null）",
  "verification": "验证步骤和通过标准（没有则为 null）",
  "constraints": "环境/依赖/版本约束（没有则为 null）",
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
若文档涉及代码/文件变更且存在 git 信息，在 description 中可写文件路径或变更说明，
便于按 commit+路径回溯；若无法确定 git，不要只写「见某文件路径」，
应把关键改动或片段直接写入 solution/code_snippets，使经验自包含、不依赖外部文件是否存在。

注意:
- title 要简洁，不超过 50 个字，且只描述一个主题（不要将多个不相关主题混在一个 title 里）
- problem 和 solution 要详细，保留关键技术细节，且一一对应
- solution 可以为 null（例如问题已确认但尚未解决）
- root_cause 分析问题的根本原因
- decisions/pitfalls/patterns/verification/constraints 五个维度尽量填写，是评估经验质量的关键
- tags 提取 3-8 个相关的技术关键词，小写英文
- code_snippets 只保留最关键的代码
- structured_data 中只填入能从文档中确认的字段，没有的填 null

**示例 (Few-shot)**:

示例1 — 故障修复类:
输入: "生产环境 API 返回 502，排查发现是 Nginx 超时 60s，后端处理耗时 2 分钟。"
      "把 proxy_read_timeout 调到 300 解决。"
输出要点: title 只写故障与修复；problem 含现象与上下文；solution 含配置；tags 含 nginx, timeout。

示例2 — 方案决策类:
输入: "选型用 PostgreSQL 存审计日志，对比过 MySQL 和 MongoDB，选 PG 因 JSONB 和分区表好用。"
输出要点: title 概括选型结论；decisions 写备选与理由；problem 写需求；tags 含 postgresql, audit。

示例3 — 多主题必须拆分:
若文档同时包含「UI 按钮样式修改」和「API Key 存储方式」，应拆成两条经验分别提取，
不要合并到一个 title 或一条经验里。
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
- 仅当文档明确包含多个连续步骤、或「先 A 再 B 再 C」的流程时，才拆成 parent + children；
  否则用单条经验（返回 children: []）。
- parent.title 概括全局，children 各自描述一个独立步骤；每个子经验都应有独立的 problem + solution。
- 子经验数量不要超过 5 个，合并关系紧密的步骤。
- tags 用小写英文。
- 若内容只是单点问题或单一主题，返回 children 为空数组 []，parent 中填完整 problem/solution。
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

_BUILTIN_PARSE_PERSONAL_MEMORY = """你是一个对话分析助手。用户会提供一段对话或文档，你需要从中提取
「该用户的个人偏好与工作习惯」，供后续 Agent 理解该用户风格时使用。

请严格以 JSON 格式返回一个数组（不要包含其他内容，不要用 markdown 代码块包裹）:
[
  {
    "content": "一句概括的偏好或习惯（例：喜欢简洁回复、plan 要收口、Web 端 UI 要简洁）",
    "profile_kind": "static 或 dynamic",
    "scope": "generic 或 context（可选，与 profile_kind 对应：generic≈static，context≈dynamic）",
    "context_hint": "仅当 profile_kind 为 dynamic 或 scope 为 context 时可填；否则为 null"
  }
]

规则:
- 每条仅一句话，content 不超过 80 字
- profile_kind: 长期稳定的偏好/习惯用 "static"；
  当前任务、近期语境用 "dynamic"。无法判断则用 "static"。
- 若只提供 scope 不提供 profile_kind：generic→static，context→dynamic（兼容旧格式）。
- context_hint: 动态/场景条目可简短说明适用场景，否则 null
- 若对话中无法提取任何偏好或习惯，返回空数组 []
- 不要编造：只提取对话中明确体现或可合理推断的内容
- 不要把团队公共知识写成个人条目
"""

# Map prompt names to built-in defaults
_BUILTIN_PROMPTS: dict[str, str] = {
    "parse_single": _BUILTIN_PARSE_SINGLE,
    "parse_group": _BUILTIN_PARSE_GROUP,
    "suggest_type": _BUILTIN_SUGGEST_TYPE,
    "summary": _BUILTIN_SUMMARY,
    "parse_personal_memory": _BUILTIN_PARSE_PERSONAL_MEMORY,
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
) -> str:
    """Load a prompt by name.

    Loading order:
      1. File at ``{prompt_dir}/{name}.md`` (if prompt_dir configured and file exists)
      2. Built-in default constant
      3. Variable substitution: ``{{experience_types}}``

    Args:
        name: Prompt name (e.g. "parse_single", "parse_group", "suggest_type", "summary").
        llm_config: LLM config (for prompt_dir path).

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

    # Step 3: Variable substitution
    if "{{" in prompt_text:
        prompt_text = _substitute_variables(prompt_text)

    return prompt_text


def _substitute_variables(prompt_text: str) -> str:
    """Replace ``{{variable}}`` placeholders with static values."""
    from team_memory.schemas import EXPERIENCE_TYPES

    replacements = {
        "{{experience_types}}": ", ".join(EXPERIENCE_TYPES),
    }
    for placeholder, value in replacements.items():
        prompt_text = prompt_text.replace(placeholder, value)
    return prompt_text



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
    quality_min_score: int = 2,
    quality_retry_once: bool = True,
) -> dict:
    """Parse text content into structured experience fields using an LLM.

    Args:
        content: Free-form text (document, conversation, etc.)
        llm_config: LLM configuration. If None, uses default Ollama config.
        as_group: If True, parse as parent + children group.
        max_input_chars: Maximum input characters sent to LLM.
        quality_min_score: Minimum quality score (0-5); below this may retry.
        quality_retry_once: If True, retry once when quality score < quality_min_score.

    Returns:
        For as_group=False: dict with title, problem, solution, tags, etc.
        For as_group=True: dict with "parent" and "children" keys.

    Raises:
        LLMParseError: If LLM connection fails or response is unparseable.
    """
    llm_model = "gpt-oss:120b-cloud"
    llm_base_url = "http://localhost:11434"
    if llm_config is not None:
        llm_model = llm_config.model
        llm_base_url = llm_config.base_url

    prompt_name = "parse_group" if as_group else "parse_single"
    system_prompt = load_prompt(prompt_name, llm_config=llm_config)
    user_content = content[:max_input_chars]

    while True:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                resp = await client.post(
                    f"{llm_base_url.rstrip('/')}/api/chat",
                    json={
                        "model": llm_model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.3},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            raise LLMParseError(
                f"Cannot connect to Ollama at {llm_base_url}. "
                "Make sure Ollama is running."
            )
        except httpx.HTTPStatusError as e:
            raise LLMParseError(f"Ollama API error: {e.response.text[:200]}")

        llm_text = data.get("message", {}).get("content", "")
        if not llm_text:
            raise LLMParseError("LLM returned empty response")

        parsed = _extract_json(llm_text)

        if as_group:
            result = _normalize_group(parsed)
        else:
            result = _normalize_single(parsed)
        break

    return result


async def parse_personal_memory(
    content: str,
    llm_config: "LLMConfig | None" = None,
    timeout: float = 25.0,
    max_input_chars: int = 8000,
) -> list[dict]:
    """Extract personal preference/habit items from conversation for personal memory.

    Used by tm_learn after experience extraction. Timeout/failure must not block
    the main flow: on any exception we log and return [].

    Returns:
        List of {"content": str, "scope": "generic"|"context", "profile_kind": "static"|"dynamic",
        "context_hint": str|None}.
    """
    llm_model = "gpt-oss:120b-cloud"
    llm_base_url = "http://localhost:11434"
    if llm_config is not None:
        llm_model = llm_config.model
        llm_base_url = llm_config.base_url

    system_prompt = load_prompt("parse_personal_memory", llm_config=llm_config)
    user_content = content[:max_input_chars]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            resp = await client.post(
                f"{llm_base_url.rstrip('/')}/api/chat",
                json={
                    "model": llm_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        logger.warning(
            "Personal memory extraction failed (no block): %s", e, exc_info=False
        )
        return []
    except Exception as e:
        logger.warning(
            "Personal memory extraction error (no block): %s", e, exc_info=True
        )
        return []

    llm_text = data.get("message", {}).get("content", "")
    if not llm_text or not llm_text.strip():
        return []

    clean_text = llm_text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        clean_text = "\n".join(lines).strip()
    try:
        raw = json.loads(clean_text)
    except json.JSONDecodeError:
        start = clean_text.find("[")
        end = clean_text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                raw = json.loads(clean_text[start:end])
            except json.JSONDecodeError:
                logger.warning("Personal memory JSON parse failed (array expected)")
                return []
        else:
            logger.warning("Personal memory JSON parse failed (array expected)")
            return []

    if not isinstance(raw, list):
        raw = [raw] if isinstance(raw, dict) else []

    result: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content_str = (item.get("content") or "").strip()
        if not content_str:
            continue
        scope = (item.get("scope") or "generic").strip().lower()
        if scope not in ("generic", "context"):
            scope = "generic"
        pk = item.get("profile_kind")
        if isinstance(pk, str):
            profile_kind = pk.strip().lower()
        else:
            profile_kind = ""
        if profile_kind not in ("static", "dynamic"):
            profile_kind = "dynamic" if scope == "context" else "static"
        if profile_kind == "dynamic":
            scope = "context"
        else:
            scope = "generic"
        context_hint = item.get("context_hint")
        if context_hint is not None:
            context_hint = str(context_hint).strip() or None
        result.append({
            "content": content_str[:500],
            "scope": scope,
            "profile_kind": profile_kind,
            "context_hint": context_hint,
        })

    return result


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
    from team_memory.schemas import EXPERIENCE_TYPES

    result = {
        "title": str(parsed.get("title", "")).strip(),
        "problem": str(parsed.get("problem", "")).strip(),
        "solution": str(parsed.get("solution", "")).strip() or None,
        "tags": parsed.get("tags", []),
        "experience_type": str(parsed.get("experience_type", "general")).strip(),
    }

    if not isinstance(result["tags"], list):
        result["tags"] = []
    result["tags"] = [str(t).strip().lower() for t in result["tags"] if t]

    # Validate experience_type against known types
    if result["experience_type"] not in EXPERIENCE_TYPES:
        result["experience_type"] = "general"

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
    llm_model = "gpt-oss:120b-cloud"
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
                            "content": load_prompt(
                                "suggest_type", llm_config=llm_config
                            ),
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

    # Validate and normalize
    from team_memory.schemas import EXPERIENCE_TYPES

    suggested = str(parsed.get("type", "general")).strip()
    if suggested not in EXPERIENCE_TYPES:
        suggested = "general"

    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason", "")).strip()
    fallbacks = parsed.get("fallback_types", [])
    if not isinstance(fallbacks, list):
        fallbacks = []
    fallbacks = [
        str(f).strip() for f in fallbacks if str(f).strip() in EXPERIENCE_TYPES
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
    llm_model = "gpt-oss:120b-cloud"
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
