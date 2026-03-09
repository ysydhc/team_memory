# Plan 文档结构规范

> **目的**：在文件数量与信息完整性之间取得平衡，便于 Agent 扫描与人工维护。
> **Agent 读取**：结构规范 → 第一章；扫描规则 → 第二章。

---

# 第一章：结构规范

## 一、Plan 全生命周期

```
调研分析 ──→ 计划定稿 ──→ 执行 ──→ 复盘
```

## 二、推荐目录结构

### 2.1 完整版（含调研）

```
docs/exec-plans/{wait|executing|completed}/{主题}/
├── 1-research/
│   ├── brief.md              # 调研任务书、问题定义、范围
│   ├── assessment.md         # 评估（brownfield、可行性、风险）
│   ├── options.md             # 方案对比（多方案合并为一）
│   └── decisions.md          # 关键决策记录（可选）
├── 2-plan/
│   ├── plan.md                # 计划本体
│   └── execute.md             # 执行记录
└── 3-retro/
    └── retro.md               # 复盘（可选）
```

### 2.2 轻量版（无正式调研）

```
{主题}/
├── plan.md
├── execute.md
└── retro.md   # 可选
```

**轻量版判定**：根级 `plan.md` + execute 类文件，且单主题 `.md` 文件数 ≤ 5；超过 5 个则不属于轻量版，应采用完整版结构。

## 三、文件数量建议

| 阶段 | 最少 | 典型 | 上限 |
|------|------|------|------|
| 调研 | 1 | 2–3 | 4 |
| 计划 | 2 | 2 | 2 |
| 复盘 | 0 | 1 | 1 |
| **合计** | **3** | **5–6** | **7–8** |

单主题 `.md` 文件数建议 ≤ 15；超过时考虑合并或拆分 Phase。

## 四、调研阶段合并原则

**合并主原则**：尽可能合并文件；合并时不得删减复盘所需信息。

- 多份评估 → 合并为 1 个 `assessment.md`；正文可归纳，**附录须保留完整清单**（测试覆盖、Rules 全量、用途与冗余分析、精简建议）；**禁止**删减清单明细。
- 多方案对比 → 合并为 1 个 `options.md`
- **区分「无价值草稿」与「有价值的思考/决策过程」**：前者可丢弃，后者必保留。判断标准：是否对复盘、复现、后续决策有参考价值。

## 五、合并时的信息保留原则

Plan 文档合并时，**优先保留完整信息**，正文可归纳，**禁止**删减复盘必需清单中的明细。

**必须保留**（支持复盘与复现）：

| 类型 | 说明 |
|------|------|
| 思考过程 | 为什么这样拆 Task、为什么选该方案 |
| 决策逻辑 | 取舍依据、备选方案 |
| 方案调优 | 评审/讨论后的修改及原因 |
| 重要决定 | 架构约束、白名单规则、豁免约定 |
| 可复用规则 | 项目可复用的实践与清单（如 Subagent 实践要点、Plan 修订自检清单） |
| 观察范围与边界 | 报告/结论的适用条件与局限 |
| 问题与修复对应 | 发现的问题与采纳的修复（Blockers、High Risks、行动路线图） |
| 衔接与依赖 | 与上一阶段/其他文档的衔接 |
| 待改进项与优先级 | 未完成项及 P1/P2/P3 分级 |
| 评审角色与发现 | 谁提出了什么（如 architect / qa 的发现） |

**复盘额外保留**：流程衔接、操作时机、自检与追问、改进建议。

**合并表述**：正文可归纳、重组，保留核心语义；结构化数据用表格；**禁止**堆砌原文。

**复盘必需保留清单**（合并时逐项核对）：① 基线数据（测试覆盖、文档结构、Rules 全量、用途与冗余分析、精简建议）；② 时间线与节点；③ 问题追溯（健康度各维度、Blockers/High Risks 明细、行动路线图、子 Agent 发现汇总）；**禁止**仅保留「已修复」「综合评分」而删明细；④ 流程衔接；⑤ 操作时机；⑥ 自检与追问；⑦ 改进建议。

## 六、信息完整性检查清单

- [ ] 问题与范围清晰（brief 或 plan）
- [ ] 执行动作有记录（execute）
- [ ] 产出可追溯
- [ ] 决策点有记录（若适用）

---

# 第二章：扫描规则

> Agent 或脚本根据本章规则扫描 `docs/exec-plans/` 下全部 `{主题}` 目录，校验是否符合规范。

## 一、扫描范围

| 范围 | 说明 |
|------|------|
| `docs/exec-plans/wait/{主题}/` | 待执行 |
| `docs/exec-plans/executing/{主题}/` | 执行中 |
| `docs/exec-plans/completed/{主题}/` | 已完成 |

**排除**：`docs/exec-plans/README.md`、`docs/exec-plans/completed/archive/`（历史归档，豁免结构校验）。

## 二、rule_id 定义

| rule_id | 说明 | 示例 |
|---------|------|------|
| `DOC_PLAN_EXECUTING_MISSING_EXECUTE` | executing 下主题缺少 execute 类文件 | `executing/foo/` 下无 `*execute*.md` |
| `DOC_PLAN_COMPLETED_MISSING_EXECUTE` | completed 下主题缺少 execute 类文件 | `completed/foo/` 下无 `*execute*.md` |
| `DOC_PLAN_RESEARCH_MISSING_BRIEF` | 有 1-research/ 但缺 brief.md | `1-research/` 存在但无 `brief.md` |
| `DOC_PLAN_PLAN_MISSING_EXECUTE` | 有 2-plan/ 但缺 execute 类文件 | `2-plan/` 存在但无 `*execute*.md` |
| `DOC_PLAN_LEGACY_STRUCTURE` | 主题未采用规范结构（无 1-research/、2-plan/，且无 plan+execute 组合） | 平铺结构、缺 execute 记录 |
| `DOC_PLAN_EXCESS_FILES` | 单主题 .md 文件数超过阈值 | 主题下 .md 数 > 25 |

**execute 类文件**：文件名含 `execute` 的 `.md`（如 `execute.md`、`xxx-execute.md`、`xxxExecute.md`）。

**规范结构**：完整版（1-research/、2-plan/、3-retro/）或轻量版（根级 `plan.md` + execute 类文件，且文件数 ≤ 5）。`DOC_PLAN_LEGACY_STRUCTURE` 为建议迁移提示，历史 plan 可白名单豁免。

## 三、输出格式

与 `harness_doc_gardening` 一致：

```
path: rule_id: message
```

目录级违规无行号，输出 `path: rule_id: message`；文件级违规可带行号。

**示例**：

```
docs/exec-plans/executing/my-plan: DOC_PLAN_EXECUTING_MISSING_EXECUTE: No execute file found
docs/exec-plans/completed/foo: DOC_PLAN_COMPLETED_MISSING_EXECUTE: No execute file found
docs/exec-plans/completed/baz: DOC_PLAN_LEGACY_STRUCTURE: Not using 1-research/2-plan/3-retro or plan+execute structure
docs/exec-plans/completed/foo/1-research: DOC_PLAN_RESEARCH_MISSING_BRIEF: 1-research/ exists but brief.md missing
docs/exec-plans/completed/bar: DOC_PLAN_EXCESS_FILES: 28 .md files (threshold 25)
```

**Exit code**：`0` = 无问题，非 `0` = 有问题。

## 四、白名单

**路径**：`scripts/plan-structure-whitelist.txt`（若不存在则无白名单）。

**格式**：与 doc-gardening 一致，支持 `path`、`path:rule_id`。

**豁免场景**：历史 plan 未采用 1-research/2-plan/3-retro 结构，可加入白名单豁免。

## 五、脚本说明

实现脚本 `scripts/harness_plan_structure_check.py` 应：

1. 遍历 `docs/exec-plans/wait`、`executing`、`completed` 下各 `{主题}` 目录；
2. 排除 `archive` 及 README；
3. 按第二节 rule_id 校验；
4. 应用白名单；
5. 按第三节格式输出，exit 0/非 0 表示通过/失败。

## 六、相关文档

| 文档 | 说明 |
|------|------|
| [doc-maintenance-guide](doc-maintenance-guide.md) | 文档维护规范、Plan 生命周期 |
| [harness-spec](harness-spec.md) | Plan 执行流程 |
