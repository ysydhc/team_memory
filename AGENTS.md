# TeamMemory Agent 导航地图

> 项目概览：基于 MCP 的团队经验库，让 AI 拥有跨会话记忆。**Harness 原则**：人类掌舵、Agent 执行、尽量不手写。

---

## 核心命令

```bash
make setup          # 首次安装（Docker + 依赖 + 数据库迁移）
make dev            # 启动全部服务
make web            # 仅启动 Web (http://localhost:9111)
make mcp            # 仅启动 MCP Server（供 Cursor / Claude Desktop 使用）
make test           # 运行测试
make lint           # ruff 代码检查
make harness-check  # Harness 门禁：import 检查 + ruff + lint-js
make verify         # 标准验收：lint + 全量测试（提交前必跑）
make health         # 健康检查
```

---

## 双轨结构（R4）

| 轨道 | 路径 | 用途 |
|------|------|------|
| 设计/执行计划 | [docs/design-docs](docs/design-docs/) · [docs/exec-plans](docs/exec-plans/) | 设计文档、执行计划、归档 |

---

## 架构导航（代码层）

| 模块 | 路径 | 说明 |
|------|------|------|
| MCP 工具层 | `src/team_memory/server.py` | FastMCP 工具注册与实现（tm_* 命名空间） |
| Web API 层 | `src/team_memory/web/` | FastAPI 路由与 handler |
| 数据访问层 | `src/team_memory/storage/` | SQLAlchemy async ORM（repository + database） |
| 服务层 | `src/team_memory/services/` | 业务逻辑，禁止在此直接操作 DB |
| 认证 | `src/team_memory/auth/` | API Key 认证、RBAC（admin/editor/viewer） |
| 嵌入/重排 | `src/team_memory/embedding/` · `src/team_memory/reranker/` | 向量嵌入与检索重排 |
| 模型定义 | `src/team_memory/schemas.py` · `src/team_memory/storage/models.py` | Pydantic schemas + SQLAlchemy models |
| 配置 | `src/team_memory/config.py` | Pydantic Settings，分层加载 |
| 数据库迁移 | `alembic/versions/` | 每次改 model 必须生成迁移文件 |

完整架构说明见 [docs/architecture.md](docs/architecture.md)，分层约束见 [project-extension](docs/design-docs/harness/project-extension.md)。

---

## GitNexus（代码理解）

项目索引：**team_doc**。任务开始先读 `gitnexus://repo/team_doc/context` 检查索引新鲜度。

| 任务 | Skill 文件 |
|------|------------|
| 理解架构 / "How does X work?" | [.claude/skills/gitnexus/exploring/SKILL.md](.claude/skills/gitnexus/exploring/SKILL.md) |
| 影响分析 / "What breaks if I change X?" | [.claude/skills/gitnexus/impact-analysis/SKILL.md](.claude/skills/gitnexus/impact-analysis/SKILL.md) |
| 调试 / "Why is X failing?" | [.claude/skills/gitnexus/debugging/SKILL.md](.claude/skills/gitnexus/debugging/SKILL.md) |
| 重构 / 重命名 / 拆分 | [.claude/skills/gitnexus/refactoring/SKILL.md](.claude/skills/gitnexus/refactoring/SKILL.md) |

工具：`query` · `context` · `impact` · `detect_changes` · `rename` · `cypher`。Schema 见 `gitnexus://repo/team_doc/schema`。

---

## 知识库导航

| 入口 | 路径 | 说明 |
|------|------|------|
| 设计文档 | [docs/design-docs](docs/design-docs/) | 架构、方案、设计决策 |
| Harness 规范 | [docs/design-docs/harness/harness-spec](docs/design-docs/harness/harness-spec.md) | 核心原则、Plan 执行、反馈回路、流程图 |
| 项目扩展 | [project-extension](docs/design-docs/harness/project-extension.md) | 架构分层、质量门禁、收尾清单、tm 边界 |
| 文档维护规范 | [docs/design-docs/harness/doc-maintenance-guide](docs/design-docs/harness/doc-maintenance-guide.md) | 规范（同步、归档、结构变更）+ 扫描设计（rule_id、白名单、输出格式） |
| doc-admin-check | `.cursor/agents/doc-admin-check.md` | 文档健康巡检（只读、报告） |
| doc-admin-organize | [.cursor/agents/doc-admin-organize.md](.cursor/agents/doc-admin-organize.md) | 文档整理（按规范实际整理） |
| 日志格式 | [docs/design-docs/logging-format](docs/design-docs/logging-format.md) | JSON 行日志规范、生产/CI 切换 |
| tm 经验提取构建 | [docs/design-docs/extraction/extraction-build-guide](docs/design-docs/extraction/extraction-build-guide.md) | 状态文件、触发条件、防抖、分步实现 |
| 执行计划 | [docs/exec-plans](docs/exec-plans/) | 计划、任务、归档 |
| 入门 | [docs/getting-started](docs/getting-started.md) | 部署、接入、开发 |
| 扩展 | [docs/extended](docs/extended.md) | 任务管理、Skills、工作流 |

---

## 关键规范入口

| 主题 | 文件 |
|------|------|
| 架构约束与分层 | [docs/architecture.md](docs/architecture.md) · [project-extension](docs/design-docs/harness/project-extension.md) |
| MCP 工具开发规范 | [docs/mcp-patterns.md](docs/mcp-patterns.md) |
| Python 代码约定 | [docs/conventions.md](docs/conventions.md) |
| 测试规范 | [docs/testing.md](docs/testing.md) |
| 安全约定 | [docs/security.md](docs/security.md) |

---

## 规则

规则位于 [.cursor/rules/](.cursor/rules/)：

- **tm-extraction-retrieval** — 经验提取与检索
- **extraction-state-update** — 每次回复必须执行的 extraction-state.json 读写清单
- **harness-engineering** · **harness-plan-execution** — Harness 原则、Plan 执行
- **global** — 全局规则（任务开始、编码、提交前）
- **mcp-tools** — MCP 工具层专属规则（修改 server.py 时生效）

---

## 任务完成标准（Definition of Done）

以下全部通过，才算完成：

- [ ] `make lint` 零报错
- [ ] `make test` 全绿（新功能须有对应测试）
- [ ] 若改动 model，存在对应 Alembic 迁移文件
- [ ] 若新增 MCP 工具，已在 [docs/mcp-patterns.md](docs/mcp-patterns.md) 工具列表中登记
- [ ] README 或相关文档已同步更新
- [ ] 无硬编码密钥、无裸 `print()` 调试语句
- [ ] Web 改动须 `make lint-js` 通过

---

## 遇到不确定的情况

1. 先查 `docs/` 下对应规范文件
2. 再查现有代码中的同类实现作为参考
3. 仍不确定时，**停下来问人，不要猜测实现**
