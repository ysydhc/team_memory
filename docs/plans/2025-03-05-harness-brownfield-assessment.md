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

## 3. 文档现状（docs/、.cursor/、.debug/ 结构）

### 3.1 docs/

```
docs/
├── GETTING-STARTED.md    # 完整入门指南
├── EXTENDED.md           # 扩展功能（任务、Skills、工作流）
├── plans/                # 计划文档（本评估所在目录）
├── res/                  # 可安装 rules/prompts 来源
│   ├── manifest.json
│   ├── rules/
│   └── prompts/
└── templates/            # 模板文件
```

### 3.2 .cursor/

```
.cursor/
├── plans/                # 计划与工作流
│   ├── workflows/        # YAML 工作流定义（不迁移）
│   │   ├── *.yaml
│   │   ├── steps/        # 步骤 $ref
│   │   ├── actions/      # 动作 $ref
│   │   ├── checklists/
│   │   ├── archive/
│   │   └── expired/
│   ├── code-arch-viz/    # 设计文档 → 迁至 docs/design-docs
│   └── *.md              # 执行计划 → 迁至 docs/exec-plans
├── rules/                # 项目规则（.mdc）
├── prompts/              # 项目 prompts（.md）
├── skills/               # 项目 skills
└── agents/               # Agent 配置
```

### 3.3 .debug/

```
.debug/
├── docs/                 # Agent 自动生成文档
│   ├── analysis/         # 分析文档
│   ├── plan/             # 计划文档
│   ├── governance/       # 治理文档
│   ├── architecture-viz/ # 架构可视化
│   ├── topics/           # 主题文档（misc、docs-standards、workflow 等）
│   └── deprecated/       # 已归档（analysis、plan、old）
├── knowledge-pack/       # 本地可安装 catalog（rules、prompts、skills）
├── tm/                   # TM 相关（architecture、ops、tech-concepts）
└── mcp_logs/             # MCP 调试日志（可选）
```

---

## 4. 引用关系

### 4.1 代码对 docs/、.cursor/、.debug/ 的引用

| 模块 | 引用路径 | 说明 |
|------|----------|------|
| config.py | `docs/res` | InstallableCatalogConfig.local_base_dir，可安装 rules/prompts 来源 |
| config.py | `.cursor/rules`、`.cursor/prompts`、`.cursor/skills` | 安装目标目录 |
| workflow_oracle.py | `.cursor/plans/workflows/{id}.yaml` | 硬编码，YAML 不迁移 |
| architecture/base.py | `.cursor/plans/code-arch-viz/code-arch-viz-provider-interface.md` | 契约对齐，迁移后需更新 |
| web/architecture_models.py | `.cursor/plans/code-arch-viz/` | 同上 |
| storage/repository.py | `.debug/docs/plans/` | 与 docs/exec-plans 职责不同，保持不变 |
| web/routes/import_export.py | `.cursor/rules`、`.cursor/prompts` | 扫描已安装项、安装写入 |
| web/routes/analytics.py | `.cursor/rules`、`.cursor/prompts`、`.cursor/skills-cursor` | 扫描统计 |
| server.py | `.cursor/skills` | tm_suggest 等扫描 |
| mcp_debug_log.py | `.debug/mcp_logs` | 日志输出目录 |
| services/installable_catalog.py | `.debug/knowledge-pack` | 本地 catalog 来源 |
| web/static/index.html | `.cursor/skills/`、`.cursor/rules/`、`.cursor/prompts/` | 前端扫描路径展示 |

### 4.2 文档间引用

- docs/res 的 manifest.json 指向 rules/、prompts/ 下的可安装项
- .cursor/rules 中 tm-doc-maintenance 等与 harness 文档结构有交集，Phase 1 后需同步

---

## 5. 已知技术债

- repository.py 覆盖率 22%，检索与存储逻辑复杂，测试补充成本高
- server.py 覆盖率 48%，大量路由与 MCP 工具未充分测试
- 部分 services（llm_provider、llm_client、ai_orchestrator 等）覆盖率为 0，依赖外部服务

---

## 6. 迁移风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| workflow_oracle 硬编码 | YAML 不迁移，路径不变，无影响 | 保持 `.cursor/plans/workflows/` |
| code-arch-viz 迁移 | architecture/base.py、architecture_models.py 需改路径 | 迁移后批量替换为 `docs/design-docs/code-arch-viz/` |
| .cursor/plans/*.md 迁移 | 若有代码引用执行计划路径，需排查 | 当前仅 repository 引用 .debug/docs/plans/，职责分离 |
| docs/res 与 .cursor 联动 | config 中 local_base_dir、target_* 已指向正确路径 | 无需变更 |
| .debug 与 docs 职责混淆 | .debug 为 Agent 临时产出，docs 为正式文档 | 明确不迁移 .debug 到 docs |

---

## 7. 建议迁移顺序

1. **Phase 0（基线）**：完成本 Brownfield 评估，不做结构变更。
2. **Phase 1 第一步**：创建 `docs/design-docs/`，将 `.cursor/plans/code-arch-viz/` 迁入，同步更新 `architecture/base.py`、`web/architecture_models.py` 中的路径引用。
3. **Phase 1 第二步**：创建 `docs/exec-plans/`，将 `.cursor/plans/` 下非 workflows 的 `*.md` 迁入；确认无代码硬编码引用后执行。
4. **Phase 1 第三步**：更新 `.cursor/rules/` 中 tm-doc-maintenance 等与 docs 结构相关的规则，确保与 harness 文档布局一致。
5. **不迁移**：`.cursor/plans/workflows/`（YAML）、`.debug/`（Agent 临时产出）。
