# MCP 工具开发规范

## 工具实现位置

本项目的 MCP 工具**全部内联**于 `src/team_memory/server.py`，不采用独立 `tools/` 目录。新增工具时：

1. 在本文档的「已注册工具列表」中登记
2. 在 `server.py` 中按现有模式添加 `@mcp.tool` + `@track_usage` 装饰的 `async def tm_xxx(...)` 函数
3. 在 `tests/test_server.py` 或对应测试文件中补充测试
4. `make verify` 全绿后才算完成

## 工具函数签名标准

```python
@mcp.tool(
    name="tm_xxx",
    description=(
        "一句话描述工具用途。AI 依赖此描述决定何时调用。"
        "Returns ~N tokens."  # 建议注明输出规模
    ),
)
@track_usage
async def tm_xxx(
    param1: str,
    param2: int = 5,
    project: str | None = None,
) -> str:
    """Docstring 补充参数说明，供 AI 理解。

    Args:
        param1: 参数含义
        param2: 默认值说明
    """
    service = _get_service()
    user = await _get_current_user()
    resolved_project = _resolve_project(project)
    # 调用 service 层，禁止直接操作 DB
    result = await service.do_something(...)
    return json.dumps({"results": result}, ensure_ascii=False)
```

- 返回值：`str`（JSON 序列化后的字符串），供 MCP 协议传输
- 正常结果：`{"results": [...]}` 或 `{"message": "...", "data": {...}}`
- 错误结果：捕获异常后返回 `{"error": "描述", "code": 400/404/500}`，**禁止**让异常透传给 AI 客户端

## description 字段要求

**description 是给 AI 看的，不是给人看的**

- 写清楚：什么情况下应该调用这个工具
- 写清楚：关键参数的含义和可选值
- 建议注明：`Returns ~N tokens`，便于 Token 预算控制
- 避免：技术术语堆砌，AI 可能理解偏差

## 工具设计原则

**错误返回统一格式**
```python
# 正确——捕获异常并返回结构化错误
try:
    ...
except ValueError as e:
    return json.dumps({"error": str(e), "code": 400}, ensure_ascii=False)
except Exception as e:
    logger.error("tm_xxx failed", exc_info=True)
    return json.dumps({"error": "内部错误，请检查服务日志", "code": 500}, ensure_ascii=False)

# 错误——不要让异常直接抛出
raise HTTPException(...)  # 在 MCP 工具层禁止
```

**幂等性**
- 查询类工具（tm_search、tm_solve、tm_suggest）：天然幂等
- 写入类工具（tm_save、tm_learn）：相同输入多次调用应有去重检测，避免重复经验

**Token 预算意识**
- 返回结果集时控制单条经验长度（使用 summary 字段代替完整 solution）
- 列表类结果默认最多 5～10 条，超出时提示用户细化查询
- 使用 `_guard_output(output, max_tokens=...)` 裁剪超长输出

## 已注册工具列表

| 工具名 | 功能 | 状态 |
|--------|------|------|
| `tm_solve` | 智能问题求解，优先命中历史方案 | ✅ |
| `tm_search` | 语义搜索经验库 | ✅ |
| `tm_suggest` | 根据当前文件上下文主动推荐 | ✅ |
| `tm_learn` | 从对话自动提取并保存经验 | ✅ |
| `tm_save` | 快速保存单条经验 | ✅ |
| `tm_save_typed` | 保存带类型和结构化字段的经验 | ✅ |
| `tm_save_group` | 保存关联经验组 | ✅ |
| `tm_feedback` | 对搜索结果评分（影响排序） | ✅ |
| `tm_update` | 更新已有经验 | ✅ |
| `tm_claim` | 认领经验，声明正在处理 | ✅ |
| `tm_notify` | Webhook 通知团队 | ✅ |
| `tm_config` | 查看运行时配置快照 | ✅ |
| `tm_status` | 查看系统健康状态 | ✅ |
| `tm_doc_sync` | 幂等同步本地文档到经验库 | ✅ |
| `tm_extract_artifacts` | 提取结构化知识 artifact | ✅ |
| `tm_invalidate_search_cache` | 清空搜索缓存 | ✅ |
| `tm_track` | 上报外部 MCP/skill 使用 | ✅ |
| `tm_skill_manage` | 管理 skills：list/disable/enable | ✅ |
| `tm_analyze_patterns` | 分析对话模式提取指令风格 | ✅ |
| `tm_task` | 任务管理：create/list/get/update | ✅ |
| `tm_task_claim` | 原子认领任务 | ✅ |
| `tm_ready` | 列出可启动任务 | ✅ |
| `tm_message` | 在任务上留言 | ✅ |
| `tm_workflow_next_step` | 工作流步骤预言 | ✅ |
| `tm_dependency` | 管理任务依赖 | ✅ |
| `tm_preflight` | 任务预检，返回经验推荐 | ✅ |

> **新增工具时，先在此表登记，再开始编码。**
