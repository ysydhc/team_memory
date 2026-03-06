# Harness Brownfield 评估

> 评估日期：2025-03-06  
> 目的：为 Harness Phase 0-1 迁移提供基线，梳理现有代码库与文档现状。

---

## 1. 架构概览

### 1.1 src 目录结构

```
src/team_memory/
├── __init__.py
├── config.py
├── server.py
├── bootstrap.py
├── schemas.py
├── extensions.py
├── schema_presets.py
├── workflow_oracle.py
├── mcp_debug_log.py
├── auth/           # OAuth、权限、admin 初始化
├── embedding/      # Ollama、OpenAI、local provider
├── reranker/       # cross_encoder、jina、ollama_llm、noop
├── storage/        # models、repository、database、audit
├── services/      # experience、search_pipeline、hooks、cache、llm 等
├── web/            # app、routes、metrics、middleware、architecture_models
└── architecture/   # base、factory、gitnexus_provider
```

### 1.2 主要模块

| 模块 | 职责 |
|------|------|
| server | MCP 服务入口、API 路由 |
| storage/repository | 经验 CRUD、检索、FTS |
| services/experience | 经验业务逻辑 |
| services/search_pipeline | 检索管道 |
| web/app | Web 应用、静态资源 |
| workflow_oracle | 工作流 YAML 加载与 step 解析 |

---

## 2. 测试覆盖

### 2.1 覆盖率摘要（pytest --cov=src）

- **总覆盖率**：47%
- **测试通过**：467 passed, 18 skipped
- **总行数**：约 10180 行（src）

### 2.2 高覆盖模块（≥80%）

- storage/audit: 100%
- storage/models: 89%
- web/middleware: 91%
- services/hooks: 97%
- services/event_bus: 99%

### 2.3 低覆盖模块（<30%）

- services/pageindex_lite: 21%
- storage/repository: 22%
- web/routes/import_export: 24%
- web/routes/lifecycle: 25%
- web/routes/tasks: 25%

---

## 3. 文档现状

### 3.1 docs/

| 路径 | 职责 |
|------|------|
| GETTING-STARTED.md | 完整入门指南 |
| EXTENDED.md | 扩展功能（任务、Skills、工作流） |
| plans/ | 计划文档（本评估所在目录） |
| res/ | 可安装 rules/prompts |
| templates/ | 模板文件 |

### 3.2 .cursor/plans/

| 路径 | 职责 |
|------|------|
| workflows/*.yaml | 工作流定义（不迁移） |
| code-arch-viz/*.md | 设计文档 → 迁至 docs/design-docs |
| *.md（根目录及 workflows 下） | 执行计划 → 迁至 docs/exec-plans |

### 3.3 .debug/

- 用于 Agent 自动生成的临时分析、计划
- .debug/docs/analysis、.debug/docs/plan
- 已执行完的计划 → .debug/docs/deprecated/plan

---

## 4. 已知技术债

- repository.py 覆盖率 22%，检索与存储逻辑复杂，测试补充成本高
- server.py 覆盖率 48%，大量路由与 MCP 工具未充分测试
- 部分 services（llm_provider、llm_client、ai_orchestrator 等）覆盖率为 0，依赖外部服务

---

## 5. 迁移影响评估

- **workflow_oracle** 硬编码读取 `.cursor/plans/workflows/{id}.yaml`，YAML 不迁移
- **architecture_models.py、architecture/base.py** 引用 `.cursor/plans/code-arch-viz/`，迁移后需更新为 `docs/design-docs/code-arch-viz/`
- **repository.py** 引用 `.debug/docs/plans/`，与 docs/exec-plans 职责不同，保持不变
