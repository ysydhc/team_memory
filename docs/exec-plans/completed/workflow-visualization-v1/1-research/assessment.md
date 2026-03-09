# Cytoscape 工作流可视化 — 评估

> 合并 plan-evaluator、plan-detail-evaluation、multi-agent 评审报告结论。**附录保留完整清单**。

## 一、总体评价

目标清晰、技术路线合理、任务拆分基本可行，可支撑 Subagent-Driven 执行。需补充 workflow_oracle 引用、$ref 示例、allowed_next 映射、文件-Task 映射后正式启动。

## 二、致命缺陷清单 (Blockers)

| 优先级 | 缺陷 | 来源 | 必须解决 |
|--------|------|------|----------|
| **P0** | **循环 $ref 检测缺失**：step A $ref B、B $ref A 会导致无限递归，workflow_oracle 也未检测。前端需增加访问路径集合 + 深度上限，超限时提示。 | architect, qa-engineer | 是 |
| **P1** | **前端与 workflow_oracle 解析语义对齐**：两套解析逻辑（js-yaml vs Python）若不一致，MCP 与可视化结果会不同。建议在 Plan 中明确「以 workflow_oracle 为准」。 | architect | 建议 |
| **P1** | **主文件多候选时的选择规则**：Plan 已写「按路径字典序取首个」，但多 workflow 根文件并存时选择可能不符合预期。建议明确「仅支持单 workflow 文件夹」或要求用户指定。 | tech-lead | 建议 |

## 三、高风险区域 (High Risks)

| 风险 | 说明 | 预案建议 |
|------|------|----------|
| 大 workflow 与恶意 YAML | 超大文件、深层 $ref、循环引用可能导致浏览器卡顿或崩溃 | 单文件大小上限（如 512KB）、$ref 递归深度上限（如 10 层）、解析超时（如 3s）降级 |
| getAsEntry 浏览器兼容性 | Safari 等对 `getAsEntry` 支持有限，文件夹拖入可能不可用 | Plan 已写降级为单文件；补充「支持的浏览器列表」或检测逻辑 |
| CDN 与部署 | CDN 不可用、部署回滚、harness-check 是否覆盖未说明 | 补充：CDN 失败提示、部署步骤、回滚方式 |
| 可观测性 | 纯前端实现，错误仅在控制台，无法被现有日志体系采集 | 可选：前端错误上报、解析失败次数、渲染耗时等指标 |

## 四、行动路线图（启动前必须补充）

1. **补充循环 $ref 检测**：在 Task 3 中增加「访问路径集合 + 深度上限（如 10）」，超限时抛出并提示「检测到循环引用或引用层级过深」
2. **明确解析规范**：在 Plan 中增加「解析规范以 `workflow_oracle.py` 为准，前端实现需与 `_resolve_step_ref` 语义一致」
3. **补充防护边界**（可选）：单文件 512KB 上限、解析超时 3s、递归深度 10 层 — 写入 Task 5 或风险缓解
4. **workflow_oracle 引用**：在 Plan 中增加 `src/team_memory/workflow_oracle.py` 路径
5. **$ref 示例**：用 1–2 个 YAML 示例说明 step 级 $ref 结构，以及循环、深度超限的测试用例路径
6. **allowed_next → 边的映射**：说明多值、空值、when 分支的边生成规则，并补充示例
7. **文件-Task 映射**：为 index.html、app.js、pages.js 的修改标注所属 Task

## 五、多角色评审洞察

| 来源 | 洞察 |
|------|------|
| product-manager | 建议补充「用户故事」或「典型使用场景」以支撑 ROI；补充「成功指标」便于迭代决策 |
| ux-designer | drop zone 双输入（拖入 + 点击）需明确主次与引导；错误展示形式（toast/内联/全屏）；深色主题需与现有 Web 体系对齐（复用 `:root` 变量） |
| qa-engineer | 需增加「拖入含循环 $ref 的 YAML → 显示解析失败并提示」用例；allowed_next 空/多值边界用例 |
