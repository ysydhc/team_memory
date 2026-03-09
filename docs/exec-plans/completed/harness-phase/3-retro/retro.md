# Harness Phase 系列复盘

> **合并来源**：Phase 0 Brownfield、Rules 审计、Phase 2 实施计划、Phase 0-1 监控报告、Phase 3 协作回溯、Phase 4/5 Flow Observer 报告。正文归纳，复盘必需清单完整保留。

---

## 一、时间线与节点

| 时间 | 节点 | 产出 |
|------|------|------|
| 2025-03-05 | Phase 0 Brownfield 评估 | 架构概览、测试覆盖、文档结构、迁移风险 |
| 2025-03-06 | Phase 0 Rules 审计 | 规则全量、用途与冗余分析、精简建议 |
| 2025-03-06 | Phase 0-1 监控报告 | 协作打分、可复用工作流、改进建议 |
| 2025-03-06～07 | Phase 2 工作流增强 | 先规划再执行、plan-self-review、human-decision-points |
| 2025-03-06 | Phase 3 协作回溯 | 第二轮 plan 评估、关键决策、可复用经验 |
| 2025-03-07 | Phase 4 计划分析与生成 | step-0、Golden Set、CI 策略补齐 |
| 2025-03-07 | Phase 4 执行完成 | Task 1～6 全部完成 |
| 2025-03-07 | Phase 5 计划优化→执行→提交 | P1 修复、6 Task 完成、Phase 6 预留 |

---

## 二、问题追溯（健康度各维度）

### 2.1 Phase 0-1

| 维度 | 得分 | 说明 |
|------|------|------|
| 主 Agent 与子 Agent 协作 | 4/5 | 产出物齐全，结构符合 Phase 0-1 范围 |
| 人类决策点 | 4/5 | Phase 0 基线、Phase 1 迁移顺序在 Brownfield 中明确 |
| 异常处理 | 3.5/5 | harness_tool_usage 为 placeholder，服务不可用时降级 |
| 文档与规则同步 | 4.5/5 | harness-engineering 已新增 |

### 2.2 Phase 3

| 维度 | 得分 | 说明 |
|------|------|------|
| 用户指令清晰度 | 4.5/5 | 「第二轮 plan」「Harness Phase 3 架构约束 + harness 纯化」表述明确 |
| AI 判断准确性 | 4/5 | 架构分层、依赖方向、豁免规则符合 Harness 风格 |
| 执行质量 | 4/5 | 分层文档完整、Plan 任务边界清晰 |

### 2.3 Phase 4（初版 → 修复后）

| 维度 | 初版 | 修复后 | 说明 |
|------|------|--------|------|
| 架构合理性 | 7/10 | — | 日志配置归属已明确 |
| 实现可执行性 | 6/10 | — | step-0、Golden Set、CI 策略已补齐 |
| 质量保障 | 6/10 | — | Schema 校验、Golden Set 已纳入 |

### 2.4 Phase 5

| 维度 | 得分 | 说明 |
|------|------|------|
| 用户指令清晰度 | 5/5 | 「启动监督」「当前任务结束」表述明确 |
| 计划修订完整性 | 5/5 | P1 自检项覆盖、P2 execute/step-0、P3 Phase 6 定义均已采纳 |
| 执行流程规范性 | 5/5 | step-0 门控、每 Task 完成均有 execute 更新 |

---

## 三、Blockers / High Risks 明细与修复对应

| 问题 | 来源 | 修复 |
|------|------|------|
| Phase 4 step-0 未定义 | tech-lead | ✅ 已增加 step-0 节 |
| doc-gardening 验收缺少 Golden Set | qa-engineer | ✅ Task 4 增加 tests/fixtures/doc-gardening/ |
| 日志配置归属破坏 L0 | architect | ✅ 配置仅放 bootstrap，config 仅提供开关 |
| request_logger、开发/生产切换未明确 | tech-lead | ✅ Task 1 已补充 |
| Task 2 未要求 Schema 校验 | qa-engineer | ✅ 已补充 pytest 或样本断言 |

---

## 四、流程衔接

| 衔接点 | 说明 |
|--------|------|
| Phase 0-1 → Phase 2 | Brownfield、Rules 审计产出；Phase 2 增强先规划再执行、自审、人类决策点 |
| Phase 2 → Phase 3 | Phase 2 产出 plan-self-review、human-decision-points；Phase 3 启动架构约束 |
| Phase 3 → Phase 4 | Phase 3 预留节定义可观测性、doc-gardening、经验库可选 |
| Phase 4 → Phase 5 | Phase 4 flow-observer 建议 step-0 模板、Plan 初版自检、Phase 6 预留写法 |
| Phase 5 → Phase 6 | Phase 5 完成收尾；Phase 6 为经验库策略、tm 叠加等 |

---

## 五、操作时机与自检

| 时机 | 动作 |
|------|------|
| Plan 开始 | 创建 execute、notify |
| step-0 通过 | 门控通过后方可进入 Task 1 |
| 每 Task 完成 | 更新 execute、记录产出 |
| Plan 完成 | notify、索引更新 |

**自检与追问**：Phase 5 plan-evaluator 建议的 step-0 含 harness_ref_verify、ruff/pytest、docs 结构；execute 路径、notify 时机已纳入 Plan。

---

## 六、改进建议（完整保留）

### 6.1 Phase 0-1

1. 监控埋点：在 Plan 执行关键节点写入 tm_message 或统一审计格式
2. Tool usage 基线补全：服务可用时执行 harness_tool_usage_baseline.sh
3. 引用扫描收口：迁移完成后执行 harness_ref_verify.sh
4. Subagent 审计格式：tm_message 中增加 `[subagent] task-<id>: <摘要>` 前缀
5. Phase 1 收尾清单：tm-doc-maintenance 同步 harness 文档结构

### 6.2 Phase 2

1. plan-self-review-checklist 与 human-decision-points 可被 Plan 或 rules 引用
2. tm-commit-push-checklist 功能验证与 tm-web 衔接

### 6.3 Phase 3

1. Phase 3 执行前确认摸底运行已执行，违规清单已产出
2. Subagent 审计：回复中记录 `[subagent] task-N: <摘要>`
3. 人类决策点：Task 6～7 迁移、Task 5 反向依赖修复前增加「需用户确认」
4. CI 等价命令：docs 或 .debug 中写出无 Makefile 时的等价命令清单
5. 规则与 feedback-loop 衔接：增加交叉引用

### 6.4 Phase 4/5

1. harness-workflow-execution：step-0 示例增加 Phase 4 模板
2. writing-plans / plan-self-review：新增「Plan 初版自检」：step-0、Golden Set、CI 策略
3. Phase 6 预留：Plan 末尾增加「Phase N+1 预留」节

---

## 七、可复用实践（完整保留）

### Subagent-Driven Development 实践要点

| 要点 | 说明 |
|------|------|
| 任务级调度 | 每个任务由独立子 Agent 执行 |
| 两阶段评审 | 先 spec 合规，再代码质量 |
| 上下文前置 | 主 Agent 提供完整任务文本与上下文 |
| 问题优先回答 | 子 Agent 提问时，主 Agent 先回答再让其继续 |
| 评审闭环 | 评审发现问题 → 实现者修复 → 再次评审 |

### Phase 4 计划生成流程（已验证有效）

1. 从 Phase 3 预留提取范围
2. 编写初版 Plan
3. 启动虚拟评审委员会
4. 产出《全维度评审报告》
5. 按行动路线图逐条更新 Plan
6. 用户确认后进入执行

### Phase 5 「Flow Observer + plan-evaluator」双轨监督

1. 用户启动 flow-observer，明确监督范围
2. 主 Agent 根据 plan-evaluator 报告修订 Plan
3. 主 Agent 执行 Plan，flow-observer 持续观察
4. 用户说「当前任务结束」→ flow-observer 输出报告
