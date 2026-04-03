# Cytoscape 工作流可视化 — 实施计划

> **执行模式**：Subagent-Driven（每个 Task 须派发 implementer，主 Agent 验收；必须按 Task 1→5 顺序执行，每 Task 完成后才派发下一 Task）  
> **修订**：已根据 plan-evaluator、multi-agent 评审报告完成 P0/P1 修复与 4 项必须补充

**Goal:** 在 Web 端提供工作流只读可视化，用户拖入 YAML 工作流文件（支持单文件或文件夹以解析 $ref），使用 Cytoscape 渲染节点+边图，支持缩放平移。

**Architecture:** 纯前端实现，js-yaml 解析、step 级 $ref 递归解析（含循环检测与深度上限），Cytoscape 渲染；解析规范以 workflow_oracle 为准，前端扩展递归语义。

**Tech Stack:** js-yaml 4.x、Cytoscape 3.x（CDN）、原生 JS、现有 Web 静态资源。

---

## 一、目标与范围

**目标**：在 Web 端提供工作流只读可视化，降低理解门槛（相比 YAML/MD 直观）。

**范围**：
- 新增入口：顶部导航「工作流可视化」（后调整为设置-高级配置 Modal）
- 拖入 workflow 文件（单文件或文件夹）
- 解析 YAML 与 step 级 `$ref`，展示完整流程
- 节点+边图，Cytoscape 渲染，支持缩放、平移

**不包含**：编辑、保存、n8n/Dify 格式导入（后续可扩展）。

---

## 二、解析规范与 workflow_oracle 引用

**解析规范**：以 [workflow_oracle.py](../../../../../src/team_memory/workflow_oracle.py) 的 `_resolve_step_ref` 语义为准。前端实现需与之一致，并在此基础上扩展递归解析。

| 项目 | 说明 |
|------|------|
| **引用路径** | `src/team_memory/workflow_oracle.py`（项目根相对） |
| **核心函数** | `_resolve_step_ref(step, base_dir)`：当 step 为 `{ $ref: "path/to/step.yaml" }` 时，加载该文件并返回完整 step 对象；路径相对于 base_dir |
| **workflow_oracle 当前行为** | 仅做**单层** step 级 $ref 解析，无递归、无循环检测、无深度限制 |
| **前端扩展语义** | 支持**递归**解析：若被引用 step 本身含 `$ref`，继续解析直至无 $ref 或达到深度上限；**必须**实现循环检测与深度上限（10 层），与 workflow_oracle 单层行为在「单层 $ref」场景下结果一致 |

---

## 三、$ref 解析策略

| 场景 | 策略 | 说明 |
|------|------|------|
| **单文件拖入** | 仅解析内联 steps | 无 $ref 或 $ref 无法解析时，展示主文件内 steps；若有未解析 $ref 则提示「请拖入包含工作流的文件夹以解析 $ref」 |
| **文件夹拖入** | 递归收集 + 解析 $ref | 使用 `DataTransferItem.getAsEntry()` 递归收集 `.yaml/.yml`，构建 path→content 映射；主文件识别：含 `meta` 且含 `steps` 的首个 YAML；$ref 相对主文件目录解析 |

**$ref 解析约束**（Task 3 必须实现）：
- **循环检测**：解析时维护「当前解析链」路径集合；若某 $ref 指向的路径已在链中，立即报错「检测到循环引用：{路径}」
- **递归深度上限**：$ref 嵌套深度不超过 10 层，超限报错「$ref 引用层级过深（超过 10 层）」

---

## 四、数据格式与 allowed_next/when → 边的映射

| 字段 | 规则 | 边生成示例 |
|------|------|------------|
| `allowed_next: [id1, id2]` | 多值 → 为每个 id 生成一条边 | `(当前 step, id1)`、`(当前 step, id2)`，label 为空 |
| `allowed_next: []` 或缺失 | 空值 → 不生成出边 | 无 |
| `when: [{ condition: "yes", next: id1 }, ...]` | 每个 next 生成一条边，label 为 condition | when 优先于 allowed_next |

---

## 五、文件变更与 Task 映射

| 操作 | 路径 | 所属 Task |
|------|------|-----------|
| 新增 | `src/team_memory/web/static/js/workflow-viewer.js` | Task 3、4、5 |
| 修改 | `src/team_memory/web/static/index.html` | Task 2 |
| 修改 | `src/team_memory/web/static/js/app.js` | Task 2 |
| 修改 | `src/team_memory/web/static/js/pages.js` | Task 2 |
| 新增 | （历史）拟增工作流 YAML；`docs/design-docs/harness/` 已删除，Task 未按该路径落地 | Task 1 |

---

## 六、实现步骤（Task 1～5）

**执行顺序**：必须按 Task 1 → 2 → 3 → 4 → 5 顺序执行，每 Task 完成后才派发下一 Task。

- **Task 1**：创建 Harness 示例 YAML（自包含，无 $ref）
- **Task 2**：新增 workflow-viewer 页面与路由（nav、page 容器、drop zone、CDN）
- **Task 3**：实现 YAML 解析与 step $ref 解析（循环检测、深度上限 10）
- **Task 4**：实现图构建与 Cytoscape 渲染（workflowToGraph、renderGraph）
- **Task 5**：拖放、点击选择与错误处理（错误文案见原实施计划）

---

## 七、验收标准（AC-1～AC-10）

| 用例 | 操作 | 预期结果 |
|------|------|----------|
| AC-1 | 点击「工作流可视化」 | 进入 workflow-viewer 页面，显示 drop zone |
| AC-2 | 拖入单文件无 $ref | 正确展示节点与边，可缩放平移 |
| AC-3 | 拖入含 $ref 的单文件 | 显示「请拖入包含工作流的文件夹以解析 $ref」 |
| AC-4 | 拖入包含主 YAML + steps/ 的文件夹 | 解析 $ref 并展示全流程 |
| AC-9 | 拖入含循环 $ref 的 YAML | 显示「检测到循环引用：{路径}」 |
| AC-10 | 拖入 $ref 嵌套超过 10 层 | 显示「$ref 引用层级过深（超过 10 层）」 |

---

## 八、风险与缓解

| 风险 | 缓解 |
|------|------|
| 循环 $ref | 解析时维护访问路径集合，检测到循环立即报错 |
| $ref 递归过深 | 深度上限 10 层，超限报错 |
| `getAsEntry` 浏览器兼容性 | 检测不可用时降级为仅支持单文件 |
| CDN 不可用 | 提示「请刷新页面」 |
