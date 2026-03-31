# 测试规范

## 测试目录结构

```
tests/
├── conftest.py          ← 全局 fixtures（数据库、客户端、mock 配置）
├── test_server.py       ← 完整 MCP（`tm_*`）测试；默认产品入口为 Lite，见 [mcp-lite-default](../design-docs/ops/mcp-lite-default.md)
├── test_server_lite.py ← Lite MCP（`memory_*`）测试
├── test_integration.py  ← 集成测试（真实 DB）
├── test_service.py      ← 服务层单元测试
├── test_auth.py         ← 认证测试
├── test_config.py       ← 配置测试
├── test_web.py          ← Web 端到端测试
├── test_harness_*.py    ← Harness 相关（import 检查、doc gardening）
└── scripts/smoke/       ← 端到端冒烟测试（scripts/smoke/smoke_web_dashboard.py）
```

## 每类测试的要求

### MCP 工具测试（最重要）

**产品默认**为 Lite（`memory_*`），新功能优先在 **`test_server_lite.py`** 覆盖。

`test_server.py`（完整 `tm_*`，遗留）覆盖：
- 命名空间：工具使用 `tm_` 前缀
- 工具注册：预期工具均已注册
- description：每个工具有描述，且包含 token 提示
- 正常路径：有结果返回
- 空结果路径：没有匹配经验时的返回格式
- 错误路径：Service 抛异常时返回 error 格式

```python
# tests/test_server.py 示例结构
import pytest
from team_memory.server import mcp

@pytest.mark.asyncio
async def test_all_tools_have_tm_prefix():
    tools = await mcp.get_tools()
    for name in tools:
        assert name.startswith("tm_"), f"Tool '{name}' does not use tm_ prefix"

@pytest.mark.asyncio
async def test_search_returns_results(mock_search_service):
    result = await tm_search(...)
    assert "results" in result
```

### Service 层测试
- 使用 mock 替代 Storage 层，不依赖真实数据库
- 测试业务规则（评分衰减、去重检测、权限校验）

### 集成测试
- 使用测试数据库（`TEAM_MEMORY_TEST_DB_URL` 或 testcontainers；`TEAM_MEMORY_ENV=test` 与 development 配置相同）
- 每个测试用 fixture 清空状态，保证隔离

## 禁止在测试中做的事

- 禁止测试调用真实外部 API（Ollama、OpenAI）——用 mock
- 禁止测试依赖执行顺序——每个测试必须独立
- 禁止 `time.sleep()` 等待——用 mock 或 fixture 控制时间

## 覆盖率要求

- MCP 工具层：≥ 90%
- Services 层：≥ 80%
- 新增代码必须有对应测试，否则 `make verify` 不通过

## 运行测试

```bash
make test                    # 全量测试
pytest tests/test_server.py -v       # 只跑 MCP 工具测试
pytest -k "test_search" -v   # 按名称过滤
pytest --cov=team_memory     # 带覆盖率报告
```
