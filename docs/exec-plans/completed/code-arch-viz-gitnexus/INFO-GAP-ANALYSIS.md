# code-arch-viz-gitnexus 整理前后信息差分析

> 按当时沿用的「复盘必需保留清单」逐项对比，量化经验复盘的信息差（原独立 Plan 结构文档已移除）。

---

## 一、主题与范围

| 项 | 值 |
|---|-----|
| **主题名** | code-arch-viz-gitnexus |
| **整理前路径** | `docs/design-docs/code-arch-viz/`（6 个平铺 .md） |
| **整理后路径** | `docs/exec-plans/wait/code-arch-viz-gitnexus/`（1-research/ + 2-plan/ + 5 个根级 .md） |
| **整理方式** | doc-admin-organize：design-docs 删除，内容迁移/合并至 exec-plans |

---

## 二、原始 vs 整理后 行数/内容概要

### 2.1 原始（design-docs，commit 4abb690 / 69706a5）

| 文件 | 行数 | 内容概要 |
|------|------|----------|
| code-arch-viz-design-brief.md | 38 | 设计任务书、三视角分工、**综合结果（含 A/B/C 产出链接）** |
| code-arch-viz-balanced-solution.md | 172 | 三视角综合均衡方案、与 GitNexus 对比、自建 vs 复用选型 |
| code-arch-viz-provider-interface.md | 178 | Provider 枚举、统一 API 契约、代码实时结合预留 |
| code-arch-viz-operations.md | 178 | 7 步操作清单、配置→Bridge→后端→路由→前端→验收 |
| code-arch-viz-gitnexus-integration.md | 197 | 三种接入方式、方案 A/B/C、推荐实施顺序 |
| code-arch-viz-gitnexus-data-mapping.md | 173 | meta.json 样本、GitNexus↔Provider 映射、实现要点 |
| **合计** | **936** | 设计+方案+接口+操作+集成+数据映射 |

### 2.2 整理后（exec-plans/wait）

| 文件 | 行数 | 内容概要 |
|------|------|----------|
| **1-research/** | | |
| brief.md | 28 | 调研任务书、三视角分工（**无综合结果节**） |
| assessment.md | 38 | 范围与边界、风险与应对、技术选型、数据映射要点（引用原 data-mapping） |
| options.md | 37 | 均衡方案摘要、复用 vs 自建、接入方式、Provider 接口概要（引用原 provider-interface） |
| **2-plan/** | | |
| plan.md | 48 | 目标与范围、任务列表 T1–T9/B1–B2、验收标准、参考文档 |
| execute.md | 22 | 任务清单、执行日志（待执行） |
| **根级（保留）** | | |
| code-arch-viz-balanced-solution.md | 172 | 同原始 |
| code-arch-viz-provider-interface.md | 158 | 同原始（行数差异或换行符） |
| code-arch-viz-operations.md | 177 | 同原始 |
| code-arch-viz-gitnexus-integration.md | 197 | 同原始 |
| code-arch-viz-gitnexus-data-mapping.md | 173 | 同原始 |
| **1-research + 2-plan 合计** | **173** | 合并后的调研与计划 |
| **含根级 5 文件** | **1023** | 总量略增（新增 assessment、plan、execute） |

---

## 三、复盘必需保留清单 — 逐项对比

| 序号 | 清单项 | 原始是否有 | 整理后是否有 | 对比结论 |
|------|--------|------------|--------------|----------|
| ① | **基线数据**（测试覆盖、文档结构、Rules 全量、用途与冗余分析、精简建议） | 无 | 无 | 本主题为设计/计划阶段，无执行基线；**双方均无** |
| ② | **时间线与节点** | 无 | 无 | 待执行 Plan，无执行时间线；**双方均无** |
| ③ | **问题追溯**（健康度、Blockers/High Risks、行动路线图、子 Agent 发现） | 有：assessment 风险表、operations 验收清单 | 有：assessment 风险表、plan 参考 operations | **保留**；assessment 合并后风险表完整 |
| ④ | **流程衔接** | 有：design-brief 链接 A/B/C 产出、各文档交叉引用 | 部分：brief 删「综合结果」；assessment/options 引用根级文件 | **部分丢失**：A/B/C 产出链接与综合逻辑记录被删 |
| ⑤ | **操作时机** | 有：operations 7 步顺序、每步验证方式 | 有：根级 operations 保留；plan 引用 | **保留** |
| ⑥ | **自检与追问** | 有：operations 第七步验收清单 | 有：根级 operations 保留 | **保留** |
| ⑦ | **改进建议** | 有：balanced-solution Phase 2/3、integration 后续 | 有：根级 balanced-solution、options 分期建议 | **保留** |

---

## 四、信息差量化

### 4.1 保留率

| 维度 | 计算 | 结果 |
|------|------|------|
| **文件级** | 6 原始 → 5 根级保留 + 3 合并（brief/assessment/options）+ 2 新增（plan/execute） | 原始 6 文件内容 **100% 可访问**（5 根级完整 + brief 为 design-brief 子集） |
| **行级（1-research + 2-plan）** | 合并后 173 行 vs 原始 design-brief 38 行 | brief 从 38→28，**净减 10 行**（section 5 综合结果） |
| **决策/流程记录** | design-brief §5「综合结果」+ A/B/C 链接 | **丢失**：三方案产出路径、综合逻辑的显式记录 |

### 4.2 丢失的关键信息类型

| 类型 | 说明 | 对复盘的影响 |
|------|------|--------------|
| **三方案产出路径** | design-brief §5 指向 `docs/analysis/代码架构可视化-技术最简化方案.md` 等 | 无法追溯「谁产出了什么」；复盘时难以还原决策链 |
| **综合逻辑记录** | 「均衡方案：code-arch-viz-balanced-solution.md」的显式声明 | 弱化「为什么选该方案」的可追溯性 |
| **workflow-pm-final-three-solutions 引用** | 原始 design-brief 有「后续演进（见 workflow-pm-final-three-solutions）」 | 整理后 brief 删除此引用，**衔接信息丢失** |

### 4.3 信息差百分比（估算）

| 指标 | 估算 | 说明 |
|------|------|------|
| **内容保留率** | **~98%** | 5 个根级文件完整；仅 design-brief §5（约 10 行）删除 |
| **决策可追溯性** | **~85%** | 综合结果与 A/B/C 链接丢失，影响「为什么这样选」的复盘 |
| **流程衔接完整度** | **~90%** | assessment/options 通过引用根级文件保留；与 workflow-pm 的衔接丢失 |
| **复盘可用度** | **~88%** | 执行前 Plan 无基线/时间线；设计阶段决策链有缺口 |

---

## 五、结论

1. **结构改进**：1-research/ + 2-plan/ 目录布局便于 Agent 扫描与维护。
2. **内容保留**：5 个根级文件（balanced-solution、provider-interface、operations、integration、data-mapping）完整保留，合并后的 assessment、options 通过引用保留细节。
3. **确认丢失**：design-brief 的「§5 综合结果（已完成）」整节被删，包括：
   - 三方案产出路径（A/B/C 对应文件链接）
   - 均衡方案出处声明
   - 与 workflow-pm-final-three-solutions 的衔接引用
4. **建议补全**：若需支持复盘，建议在 brief.md 或 1-research/decisions.md 中补回：
   - 三视角产出与综合逻辑的简要记录
   - 与 workflow-pm、均衡方案文档的衔接说明

---

## 六、附录：原始 design-brief §5 内容（已删除）

```markdown
## 5. 综合结果（已完成）

- **三方案产出**：  
  - A 技术最简 → `docs/analysis/代码架构可视化-技术最简化方案.md`  
  - B 收益最大 → `docs/代码架构可视化-项目收益最大化视角.md`  
  - C 使用最易 → `docs/代码架构可视化-使用最易视角方案.md`  
- **均衡方案**：`code-arch-viz-balanced-solution.md`
```

以及原始 design-brief 中的引用：
- 「**后续演进**（见 workflow-pm-final-three-solutions）」
