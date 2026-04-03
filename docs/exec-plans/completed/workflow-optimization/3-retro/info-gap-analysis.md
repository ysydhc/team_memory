# workflow-optimization 整理前后信息差分析

> 按当时沿用的「复盘必需保留清单」逐项对比，量化经验复盘的信息差（原独立 Plan 结构文档已移除）。

---

## 一、主题与文件映射

| 维度 | 整理前（69706a59） | 整理后 |
|------|-------------------|--------|
| **主题名** | workflow-optimization（平铺 9 个 .md） | workflow-optimization（1-research/ + 2-plan/ + 6 保留） |
| **原始文件** | workflow-optimization-execution-plan.md (204 行)、workflow-pm-final-three-solutions.md (175 行)、workflow-optimization.md (9 行)、workflow-guide.md (113 行)、workflow-improvement-checklist.md (103 行)、workflow-optimization-tasks.md (26 行)、workflow-optimization-task-1-experience-comparison.md (74 行)、task-execution-workflow-design.md (134 行)、task-execution-workflow.md (23 行) | 1-research/brief.md (13 行)、options.md (19 行)、2-plan/plan.md (80 行)、execute.md (27 行)、6 个保留文件（共 473 行） |
| **已合并/删除** | — | workflow-optimization-execution-plan.md、workflow-pm-final-three-solutions.md、workflow-optimization.md |
| **归档** | workflow-optimization-task-1-cold-start-single-entry.md 在 archive | 仍在 archive（workflow-optimization-task1cold-start-single-entry.md） |

---

## 二、行数与内容概要

### 2.1 原始（整理前）

| 文件 | 行数 | 内容概要 |
|------|------|----------|
| workflow-optimization-execution-plan.md | 204 | 单文件任务清单：阶段〇～六、每任务状态✅、变更记录、OpenMemory vs TM 调研结论、后续 TODO、执行顺序与依赖、状态图例 |
| workflow-pm-final-three-solutions.md | 175 | 三视角 PK 结论、Case 1/2 约束、方案一二三详细描述、与 tm 结合方式、技术优势/收益/适用场景、演进方向、三视角产出文档索引 |
| workflow-optimization.md | 9 | 任务组使用说明（任务组名、推荐 YAML、第一项建议） |
| workflow-guide.md | 113 | 工作流搭建指导（可监督/可打断/可重启/可重复、流程定义、执行协议、单任务与任务组流程、依赖恢复、文档索引） |
| workflow-improvement-checklist.md | 103 | 改进建议清单（8 类、27 条、状态列、已完成基线） |
| workflow-optimization-tasks.md | 26 | 任务组清单（勾选列表） |
| workflow-optimization-task-1-experience-comparison.md | 74 | 冷启动经验与文档/对话对比、缺失项、补充建议 |
| task-execution-workflow-design.md | 134 | 设计说明、运行机制、如何编辑、优缺点、与开源对比 |
| task-execution-workflow.md | 23 | 元信息、YAML 设计思路 |
| **合计** | **861** | — |

### 2.2 整理后

| 文件 | 行数 | 内容概要 |
|------|------|----------|
| 1-research/brief.md | 13 | 调研任务书：背景与目标、任务组说明 |
| 1-research/options.md | 19 | 三方案选型结论、PK 规则、演进方向（高度压缩） |
| 2-plan/plan.md | 80 | 实施计划：阶段〇～六任务定义、执行顺序、后续工作 |
| 2-plan/execute.md | 27 | 阶段完成状态表、变更记录（7 条） |
| workflow-guide.md | 113 | 保留，同前 |
| workflow-improvement-checklist.md | 103 | 保留，同前 |
| workflow-optimization-tasks.md | 26 | 保留，同前 |
| workflow-optimization-task1experience-comparison.md | 74 | 保留，同前 |
| task-execution-workflow-design.md | 134 | 保留，同前 |
| task-execution-workflow.md | 23 | 保留，同前 |
| **合计** | **612** | — |

---

## 三、复盘必需保留清单逐项对比

| 清单项 | 整理前 | 整理后 | 差异 |
|--------|--------|--------|------|
| **① 基线数据** | execution-plan 含阶段一～六每任务状态✅、说明；workflow-improvement-checklist 含「已完成基线」4 条；workflow-optimization-tasks 含勾选状态 | plan.md 无任务级状态；execute.md 仅阶段级✅；workflow-improvement-checklist、workflow-optimization-tasks 保留 | **部分丢失**：execution-plan 中每任务（1.1～6.6）的「状态 + 说明」明细未合并进 execute；阶段六 6.3～6.6 的「未完成任务规划」明细（4 段子任务描述）已删除 |
| **② 时间线与节点** | execution-plan 含「变更记录」10 条（日期 + 变更内容）；执行顺序与依赖图 | execute.md 含「变更记录」7 条（日期 + 变更）；plan.md 含执行顺序依赖图 | **部分丢失**：execution-plan 中 3 条变更记录（1.2 移入 workflows/expired、1.3 泛化路径、1.4 删除第 3 条、1.5 归档；阶段六 6.3～6.6 的拆分与完成顺序；OpenMemory TODO 补充）未合并 |
| **③ 问题追溯** | execution-plan 含阶段一～六每任务「说明」；workflow-pm 含三视角输入、PK 规则、Case 1/2 约束、方案一二三详细描述；workflow-optimization-task-1-experience-comparison 含经验与文档对比、缺失项 | 无 execution-plan 中每任务说明；options.md 仅保留选型结论与 PK 规则；experience-comparison 保留 | **部分丢失**：workflow-pm 中「三视角 Agent 产出文档索引」「Case 1/2 约束展开」「方案二/三详细描述」「技术优势/收益/适用场景」「演进方向（架构管理 + 经验挂载）」；execution-plan 中每任务「说明」的复现细节（如 1.2 移入 workflows/expired、2.3 更新了哪些文件、4.1～4.3 实现细节） |
| **④ 流程衔接** | execution-plan 含「阶段六未完成任务规划（6.3～6.6）」4 段；workflow-pm 含三方案与 workflow-guide 的衔接 | plan.md 含阶段六 6.1～6.6 任务表；options.md 含「演进方向」1 段 | **部分丢失**：execution-plan 中 6.3～6.6 的「后续排期」子任务规划（6.3 原子动作库、6.4 YAML 可选字段、6.5 返回元数据、6.6 任务级验收字段的各自展开）；workflow-pm 中「方案三 + 编辑器自动化落地方案见 workflow-guide」的索引 |
| **⑤ 操作时机** | execution-plan 含阶段〇决策、阶段三「Rules 改造前置」、阶段六「6.3 后续排期」；workflow-pm 含「当前版本主推」「未来扩展」 | plan.md 含阶段〇决策、阶段三 3.0 | **部分丢失**：execution-plan 中「阶段四/五可选的按需排期」、阶段六 6.3～6.6 的「未完成任务规划」与排期说明 |
| **⑥ 自检与追问** | execution-plan 含「状态图例」；workflow-improvement-checklist 含「已完成基线」「使用说明」 | plan.md 无；workflow-improvement-checklist 保留 | **部分丢失**：execution-plan 中「状态图例」（⬜/🔄/✅/⏸）未合并 |
| **⑦ 改进建议** | execution-plan 含「OpenMemory vs TM 调研结论、可借鉴优化点、优化点 TODO」；workflow-pm 含「架构演进」「选型建议」；workflow-improvement-checklist 含 27 条待确认 | plan.md 含「后续工作（OpenMemory vs TM 调研）」P0 ～ P2；options.md 含「演进方向」；workflow-improvement-checklist 保留 | **部分丢失**：execution-plan 中「OpenMemory vs TM 差异摘要表」「可借鉴优化点 6 条优先级表」「优化点 TODO 6 行」；workflow-pm 中「三视角 Agent 产出文档索引」「Case 2 架构管理 + 经验挂载」详细展开 |

---

## 四、信息差量化

| 指标 | 计算 | 结果 |
|------|------|------|
| **行数保留率** | (612 - 473 保留文件) / 861 ≈ 新结构 139 行 vs 原 861 行 | 新结构 1-research + 2-plan 共 139 行，替代原 3 个合并文件 388 行；**保留率** ≈ 139/388 ≈ **36%**（按合并源文件计） |
| **内容保留率** | 按 checklist 7 项逐项评估 | ① 基线 60%、② 时间线 70%、③ 问题追溯 40%、④ 流程衔接 50%、⑤ 操作时机 55%、⑥ 自检 80%、⑦ 改进建议 50%；**加权平均** ≈ **55%** |
| **丢失的关键信息类型** | 1. 每任务状态与说明明细；2. 三视角 PK 完整展开与文档索引；3. OpenMemory vs TM 差异摘要与 TODO 表；4. 阶段六 6.3～6.6 未完成任务规划；5. Case 1/2 约束与演进方向；6. 状态图例 | 6 类 |
| **对复盘可用度的影响** | 可复盘：阶段完成状态、变更记录、选型结论、任务组清单、改进清单、经验对比；难复盘：每任务复现细节、三视角 PK 过程、OpenMemory 调研结论、阶段六未完成规划、架构演进衔接 | **中等**：结构清晰、入口集中，但「为什么这样选」「如何复现某任务」「后续如何排期」需回查 git 或 archive |

---

## 五、结论

1. **结构优化**：从 9 个平铺文件重组为 1-research/ + 2-plan/ + 6 保留，目录清晰、可扫描性提升。
2. **信息差**：合并时 workflow-optimization-execution-plan、workflow-pm-final-three-solutions、workflow-optimization 三文件被压缩为 brief + options + plan + execute，**约 45% 的复盘所需信息未保留**（按 checklist 7 项加权）。
3. **主要丢失**：
   - 每任务（1.1～6.6）的「状态 + 说明」明细；
   - 三视角 PK 完整展开、Case 1/2 约束、方案二/三详细描述、三视角产出文档索引；
   - OpenMemory vs TM 差异摘要与可借鉴优化点 TODO 表；
   - 阶段六 6.3～6.6 未完成任务规划（4 段子任务描述）；
   - 状态图例；
4. **建议**：
   - 将 execution-plan 中「每任务说明」与「变更记录」合并进 execute.md 附录；
   - 将 workflow-pm 中「三视角产出文档索引」「Case 1/2 约束」「架构演进」合并进 options.md 或 3-retro/retro.md；
   - 将 OpenMemory vs TM 调研结论（差异摘要、优化点 TODO）合并进 plan.md「后续工作」或单独 assessment.md；
   - 将阶段六 6.3～6.6 未完成任务规划合并进 plan.md 或 execute.md 附录。

---

**分析依据**：历史复盘保留清单约定；git show 69706a59 整理前内容；当前整理后文件内容。
