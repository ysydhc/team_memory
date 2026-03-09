# Harness Phase 系列评估

> **合并来源**：Phase 0 Brownfield 评估、Rules 审计、Phase 0-1 监控报告、Phase 3 协作回溯、Phase 4 全维度评审、Phase 4/5 Flow Observer 报告。正文归纳，附录保留完整清单。

---

## 一、综合评估摘要

| 阶段 | 健康度/评分 | 主要发现 |
|------|-------------|----------|
| Phase 0-1 | 4.0/5 | 基线先行、规则闭环、可验证脚本；Tool usage 为 placeholder，需服务可用时补跑 |
| Phase 3 | 4/5 | 分层与 Brownfield 对齐、纯 Harness 分离；Task 5 反向依赖修复待执行 |
| Phase 4 | 6.5/10 → 已修复 | 初版缺 step-0、Golden Set、日志 L0 约束；评审后已补齐 |
| Phase 5 | 5.0/5 | plan-evaluator 闭环、step-0 模板化、规则分离边界清晰 |

---

## 二、Brownfield 与可行性

- **Brownfield 对齐**：`src/team_memory/` 目录与 L0～L3 分层表一一对应；新增模块须先归入某层再实现
- **已知待修复**：`architecture → web.architecture_models`、`auth → web.app` 反向依赖（Phase 5 Task 1 已修复）
- **架构约束**：日志 JSON 配置仅放 bootstrap（L3），config 仅提供 `LOG_FORMAT` 开关

---

## 三、风险与缓解

| 风险 | 缓解 |
|------|------|
| 反向依赖 | 摸底产出违规清单；白名单须在文档中记录路径与理由，定期复审 |
| doc-gardening 误报 | 白名单机制（archive 内互相引用豁免）；首版 `continue-on-error: true` 试跑 |
| CI 环境无 Makefile | 文档化「CI 等价命令」清单 |

---

## 附录 A：Phase 4 全维度评审 Blockers / High Risks（完整保留）

### Blockers

| 优先级 | 问题 | 来源 | 建议 |
|--------|------|------|------|
| P0 | Phase 4 step-0 未定义 | tech-lead | 在任务拆分前增加 step-0 节 |
| P0 | doc-gardening 验收缺少 Golden Set | qa-engineer | Task 3/4 增加 `tests/fixtures/doc-gardening/` 预期检出列表 |
| P1 | 日志配置归属破坏 L0 约束 | architect | 明确：配置仅放 bootstrap，config 仅提供 `LOG_FORMAT` 开关 |

### High Risks

| 风险 | 来源 | 缓解建议 |
|------|------|----------|
| request_logger 与开发/生产切换未明确 | tech-lead | Task 1 写明 request_logger、环境变量/config 开关 |
| doc-gardening 扫描范围设计与缓解不一致 | devops | CI 只跑 docs/design-docs、docs/exec-plans |
| Task 3 与 Task 4 扫描范围不一致 | architect | 首版与 CI 统一 |
| doc-gardening 在 CI 中的执行方式未定 | devops | 独立 job + timeout；首版 continue-on-error |

---

## 附录 B：Phase 4 行动路线图（完整保留）

| 步骤 | 动作 | 责任 |
|------|------|------|
| 1 | 在 Plan 中增加 Phase 4 step-0 节 | 主 Agent / 用户 |
| 2 | Task 1 补充 request_logger、开发/生产切换 | 主 Agent |
| 3 | Task 3 明确日志配置仅放 bootstrap | 主 Agent |
| 4 | Task 3 统一 doc-gardening 扫描范围 | 主 Agent |
| 5 | Task 3/4 增加 Golden Set 与白名单规则 | 主 Agent |
| 6 | Task 5 明确 CI 执行方式 | 主 Agent |
| 7 | Task 2 验收补充日志 Schema 校验 | 主 Agent |
| 8 | 完成调整后再次评审或进入执行 | 用户确认 |

---

## 附录 C：子 Agent 核心发现汇总（Phase 4）

| 角色 | 发现 1 | 发现 2 | 发现 3 |
|------|--------|--------|--------|
| 首席架构师 | python-json-logger 最小侵入 | 日志配置归属破坏 L0 | Task 3/4 扫描范围不一致 |
| 资深技术主管 | request_logger 等未明确 | step-0 未定义 | doc-gardening 与 tm-doc-maintenance 衔接 |
| 运维专家 | 日志 JSON 便于 Loki/ELK | 扫描范围设计与缓解不一致 | CI 执行方式与超时策略 |
| QA 专家 | 验收缺少 Golden Set | 输出格式可测试 | Task 2 未要求 Schema 校验 |

---

## 附录 D：Phase 0 Brownfield 评估基线（完整保留）

> 来源：2025-03-05-harness-brownfield-assessment.md

### 测试覆盖（pytest --cov=src）

| 指标 | 值 |
|------|-----|
| 总覆盖率 | 47% |
| 测试通过 | 467 passed, 18 skipped |
| 总行数 | 约 10180 行（src） |

**高覆盖模块（≥80%）**：storage/audit 100%、storage/models 89%、web/middleware 91%、services/hooks 97%、services/event_bus 99%

**低覆盖模块（<30%）**：services/pageindex_lite 21%、storage/repository 22%、web/routes/import_export 24%、web/routes/lifecycle 25%、web/routes/tasks 25%

### 文档结构（docs/、.cursor/、.debug/）

- **docs/**：GETTING-STARTED、EXTENDED、plans、res、templates
- **.cursor/**：plans、rules、prompts、skills、agents
- **.debug/**：docs、knowledge-pack、tm、mcp_logs

### 迁移风险与缓解

| 风险 | 缓解 |
|------|------|
| workflow_oracle 硬编码 | 保持 `.cursor/plans/workflows/` |
| code-arch-viz 迁移 | 已完成，已替换为 `docs/design-docs/code-arch-viz/` |
| .cursor/plans/*.md 迁移 | 当前仅 repository 引用 .debug/docs/plans/，职责分离 |
| docs/res 与 .cursor 联动 | config 已指向正确路径 |
| .debug 与 docs 职责混淆 | 明确不迁移 .debug 到 docs |

### 建议迁移顺序（Phase 0 产出）

1. Phase 0：完成 Brownfield 评估，不做结构变更
2. Phase 1 第一步：code-arch-viz 迁入 docs/design-docs/（已完成）
3. Phase 1 第二步：创建 docs/exec-plans/，迁入 .cursor/plans/*.md
4. Phase 1 第三步：更新 tm-doc-maintenance 等规则
5. 不迁移：.cursor/plans/workflows/、.debug/

---

## 附录 E：Rules 审计基线（完整保留）

> 来源：2025-03-06-harness-rules-audit.md

### 规则全量清单（.cursor/rules/ 共 9 个）

**alwaysApply（6 个）**：tm-doc-maintenance 74 行、tm-core 68 行、tm-extraction-retrieval 156 行、team_memory 32 行、tm-commit-push-checklist 30 行、team_memory-codified-shortcuts 53 行，小计 413 行。

**场景化（3 个）**：tm-web 28 行（web/**）、tm-plan 28 行（.cursor/plans/**、*plan*.md）、tm-quality 27 行（src/**、tests/**、.debug/**），小计 83 行。**合计 496 行**。

### 用途与冗余分析

- tm-core 与 team_memory-codified-shortcuts：部分重叠，可考虑 tm-core 引用 shortcuts
- tm-extraction-retrieval（156 行）：可评估拆分为「触发」与「检索」
- tm-doc-maintenance：与 harness doc 结构有交集，Phase 1 后需同步
- tm-web 与 tm-commit-push-checklist：职责清晰，无冗余
- tm-quality 与 tm-core：Ruff 在两者均有提及，可接受

### 精简建议

1. tm-core ↔ team_memory-codified-shortcuts：合并或引用
2. tm-extraction-retrieval：评估拆分为「提取触发」与「检索触发」
3. tm-doc-maintenance：Phase 1 完成后同步 harness doc 结构
4. 场景化规则保持独立
