"""Schema registry and schema generation routes."""

from __future__ import annotations

import json as json_mod

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException

from team_memory.auth.provider import User
from team_memory.web import app as app_module
from team_memory.web.app import (
    SchemaGenerateRequest,
    get_current_user,
)

router = APIRouter(tags=["schema"])


@router.get("/schema")
async def get_schema(user: User = Depends(get_current_user)):
    """Return the full effective schema (types, categories, severity)."""
    from team_memory.schemas import get_schema_registry

    registry = get_schema_registry()
    return registry.to_dict()


@router.post("/schema/generate")
async def generate_schema(
    req: SchemaGenerateRequest,
    user: User = Depends(get_current_user),
):
    """AI-assisted schema generation from a sample document."""
    _settings = app_module._settings
    if not _settings:
        raise HTTPException(status_code=500, detail="Settings not initialized")

    from team_memory.schemas import get_schema_registry

    registry = get_schema_registry()
    existing_types = [t["id"] for t in registry.to_dict()["experience_types"]]

    prompt = f"""你是一个经验库 Schema 配置分析助手。用户会提供一份文档模板或示例，
你需要从中分析出适合的经验类型定义。

当前已有的类型: {", ".join(existing_types)}

请严格以 JSON 格式返回（不要包含其他内容，不要用 markdown 代码块包裹）:
{{
  "experience_types": [
    {{
      "id": "类型ID（小写英文下划线）",
      "label": "类型显示名（中文）",
      "severity": true/false,
      "progress_states": ["状态1", "状态2"],
      "structured_fields": [
        {{"name": "字段ID", "type": "text|list|bool", "label": "字段显示名"}}
      ]
    }}
  ],
  "categories": [
    {{"id": "分类ID", "label": "分类名"}}
  ],
  "analysis_summary": "一句话说明你的分析结论"
}}

注意:
- 如果文档中的类型与已有类型相似，使用已有的 id 并扩充字段
- id 用小写英文加下划线，label 用中文
- structured_fields 只提取文档中明确出现的字段
- progress_states 按工作流顺序排列
- 如果文档不适合提取经验类型，返回空数组并说明原因
"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{_settings.llm.base_url.rstrip('/')}/api/chat",
                json={
                    "model": _settings.llm.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": req.content[:8000]},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM 调用失败: {e}")

    llm_text = data.get("message", {}).get("content", "")
    if not llm_text:
        raise HTTPException(status_code=502, detail="LLM 返回为空")

    clean = llm_text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        clean = "\n".join(lines).strip()

    try:
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json_mod.loads(clean[start:end])
        else:
            parsed = json_mod.loads(clean)
    except json_mod.JSONDecodeError:
        raise HTTPException(status_code=502, detail="无法解析 LLM 返回的 JSON")

    yaml_data = {"custom_schema": {}}
    if parsed.get("experience_types"):
        yaml_data["custom_schema"]["experience_types"] = parsed["experience_types"]
    if parsed.get("categories"):
        yaml_data["custom_schema"]["categories"] = parsed["categories"]

    yaml_preview = yaml.dump(yaml_data, allow_unicode=True, default_flow_style=False)

    return {
        "yaml_preview": yaml_preview,
        "types_found": parsed.get("experience_types", []),
        "categories_found": parsed.get("categories", []),
        "analysis_summary": parsed.get("analysis_summary", ""),
    }


@router.put("/config/schema")
async def update_schema_config(
    req: dict,
    user: User = Depends(get_current_user),
):
    """Update the runtime schema configuration (non-persistent)."""
    from team_memory.config import CustomSchemaConfig
    from team_memory.schemas import init_schema_registry

    try:
        new_config = CustomSchemaConfig.model_validate(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    if app_module._settings:
        app_module._settings.custom_schema = new_config
    registry = init_schema_registry(new_config)
    return {
        "message": "Schema 已更新（运行时生效，重启后需要写入 config.yaml 才能持久化）",
        "schema": registry.to_dict(),
    }


@router.get("/schema/presets")
async def list_schema_presets(user: User = Depends(get_current_user)):
    """List available schema preset packs."""
    from team_memory.config import CustomSchemaConfig
    from team_memory.schema_presets import list_presets
    from team_memory.schemas import SchemaRegistry

    result = []
    for name in list_presets():
        r = SchemaRegistry(CustomSchemaConfig(preset=name))
        data = r.to_dict()
        result.append({
            "name": name,
            "type_count": len(data["experience_types"]),
            "category_count": len(data["categories"]),
            "types": [t["id"] for t in data["experience_types"]],
        })
    return result
