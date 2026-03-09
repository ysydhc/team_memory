# TeamMemory 架构文档

> 与 [project-extension](design-docs/harness/project-extension.md) 互补：本文描述整体分层与领域模块，project-extension 定义 import 约束与 CI 校验。

## 整体分层

```
┌─────────────────────────────────────────┐
│  MCP Server 层  (server.py)             │  ← AI 客户端调用入口，tm_* 工具
│  Web Routes 层  (web/routes/)           │  ← HTTP API 入口
├─────────────────────────────────────────┤
│  Services 层    (services/)             │  ← 业务逻辑
│  Auth 层        (auth/)                 │  ← 认证与 RBAC
│  Embedding/Reranker (embedding/, reranker/) │  ← 向量与重排
│  Architecture   (architecture/)        │  ← GitNexus 等架构提供方
├─────────────────────────────────────────┤
│  Storage 层     (storage/)              │  ← 数据访问（repository + database）
├─────────────────────────────────────────┤
│  Models 层      (storage/models.py, schemas.py)    │  ← ORM 在 storage/；schemas 在包根
├─────────────────────────────────────────┤
│  Infrastructure (PostgreSQL + pgvector) │  ← 存储
└─────────────────────────────────────────┘
```

## 依赖方向规则

**合法方向（单向向下）：**
```
Server / Web → Services / Auth / Embedding / Reranker / Architecture
    → Storage → Models
```

**禁止的跨层调用：**
- Server / Web 直接访问 Storage 或 Models ✗
- Services 直接执行 SQL ✗
- Storage 调用 Services ✗
- 任何层直接 import 上层模块 ✗

详见 [project-extension](design-docs/harness/project-extension.md) 的 L0～L3 分层表与依赖矩阵。

## 核心领域模块

### Experience（经验）
- 核心实体，包含 title / problem / solution / tags / score / status
- 支持多种类型：general / feature / bugfix / tech_design / incident / best_practice / learning
- 状态流转：draft → review → published / rejected

### Search Pipeline（检索管线）
```
查询输入
  → 同义词扩展（tag_synonyms / user_expansion）
  → 向量搜索（pgvector cosine similarity）
  → 全文检索（PostgreSQL FTS + jieba）
  → RRF 融合排序
  → Reranker（可选 LLM 精排）
  → Token 预算裁剪
  → 返回 Top-N
```

### MCP 工具注册
- 所有工具在 `server.py` 中通过 FastMCP 注册
- 工具实现内联于 `server.py`，命名空间 `tm_*`
- 工具只能调用 Services 层，禁止直接操作数据库

### 配置系统
分层加载优先级（低 → 高）：
```
config.yaml → config.local.yaml → config.{env}.yaml → 环境变量
```
- `config.minimal.yaml` 仅在 `config.yaml` 不存在时作为 fallback，或设置 `TEAM_MEMORY_ENABLE_MINIMAL_OVERLAY=1` 时作为显式 overlay
- 所有配置通过 `team_memory/config.py` 中的 Pydantic Settings 读取，禁止在业务代码中直接读环境变量

## 关键约束

1. **异步优先**：所有数据库操作使用 `async/await`，禁止在异步上下文中使用同步 ORM 操作
2. **外部边界验证**：所有来自外部的数据（MCP 参数、HTTP body、配置值）必须经过 Pydantic model 验证
3. **结构化日志**：使用项目统一 logger，禁止裸 `print()`
4. **敏感信息**：API Key、数据库密码禁止出现在代码、日志、提交记录中
