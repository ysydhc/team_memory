# 冷启动任务 — 经验数据与文档/对话对比

> 对比对象：TM 中存入的经验（tm_save_typed）、workflow-optimization-task-1-cold-start-single-entry.md、本对话中的执行与改动。

---

## 一、当前经验中已保存的内容（写入时的 payload）

| 字段 | 内容摘要 |
|------|----------|
| **title** | 工作流冷启动只在一个入口（仅 step-0） |
| **problem** | 任务执行工作流中冷启动有两处入口：step-0（认领前）与 step-1（认领后「或冷启动」），导致语义两套、状态/审计记录不清、行为不统一。 |
| **solution** | 收口为仅 step-0。step-1 改为：认领后若 description 为空或不足须补做 step-0 的冷启动动作并用 tm_message 记为 step-0 完成后再进 step-2，不得在 step-1 内以顺带冷启动跳过 step-0。修改三处：task-execution-workflow.yaml（step-0 注释、step-1 action）、task-execution-workflow-design.md（冷启动单入口约定）、tm-core.mdc（冷启动仅在 step-0 一句）。workflow-optimization-workflow.yaml 同步 step-0/step-1 表述。 |
| **root_cause** | 流程与 Rules 中未约定冷启动唯一入口，step-1 的「或冷启动」形成第二套语义。 |
| **tags** | workflow, cold-start, step-0, task-execution-workflow |
| **experience_type** | best_practice |

---

## 二、任务文档中有、经验中未显式体现的内容

| 来源 | 内容 | 对后续改动的价值 |
|------|------|------------------|
| **一、1.1 现象** | 两处入口的原文引用（step-1「先执行 step-0 或冷启动补全 description」） | 高：复现问题或排查时能对照原文 |
| **一、1.2 导致的问题** | 三点展开（语义两套、记录不清、可重复性差） | 中：理解「为什么要收口」 |
| **一、1.3 目标** | 「补做 step-0 并记为 step-0」与「退回 step-0」两种处理方式 | 中：实现时二选一有据可依 |
| **二、2.2 具体改动** | 三处改动的逐条文案（YAML step-1 的完整替换句、设计文档约定句、Rules 一句） | 高：复现改动时可直接粘贴或对照 |
| **二、2.3 验收标准** | 四条（YAML 无「或冷启动」、设计文档有约定、Rules 有收口、阅读得出冷启动=step-0） | 高：验收或回归时 checklist |
| **三、执行建议** | 使用 workflow-optimization-workflow.yaml、认领任务组中对应任务 | 中：同类流程优化任务可复用 |
| **四、改动效果** | 语义唯一、状态与审计一致、两套 YAML 一致、可检索 | 高：说明改完后的收益，便于推广或写方案 |

---

## 三、对话与执行中有、经验中未体现的内容

- **实际改动的文件与位置**：  
  `task-execution-workflow.yaml`（step-0 注释、step-1 action、step-3 末尾「任务结束后总结经验」）、`workflow-optimization-workflow.yaml`（同上 + step-1 补做 step-0 一句）、`task-execution-workflow-design.md`（「一」中新增「冷启动单入口」）、`.cursor/rules/tm-core.mdc`（「按任务执行工作流时」下新增一句）。
- **顺带落地的通用逻辑**：  
  在两条工作流的 step-3 中增加了「任务结束后将对话与文件中有价值内容做语义压缩、通过 tm_save/tm_learn 写入经验库」的通用描述——经验中未提，但对后续「任务结束总结经验」的复现有用。
- **经验 ID**：  
  e5227822-8326-4868-b1b9-3d809f5bcb5a（可用来做 tm_update 或 tm_feedback）。

---

## 四、结论：经验是否很好反映整体改造中有价值的数据

- **已较好反映的**  
  - **问题与根因**：两处入口、语义两套、记录不清、行为不统一；根因是未约定唯一入口。  
  - **解决方案要点**：收口到 step-0、step-1 的语义（补做 step-0 并记为 step-0）、修改三处（YAML / 设计文档 / Rules）及 workflow-optimization 同步。  
  - **类型与标签**：best_practice + workflow/cold-start/step-0 便于检索。

- **缺失或可加强的**  
  1. **改动效果**：未写入「改完后的收益」（语义唯一、状态与审计一致、两套 YAML 一致、可检索），对后续写方案或推广价值大。  
  2. **验收标准**：未写入四条验收标准，复现或回归时缺少 checklist。  
  3. **具体改动文案**：未保存 step-1 的完整替换句、设计文档约定句、Rules 一句，复现时需再打开文档。  
  4. **文件路径**：未写 `.tm_cursor/plans/workflows/`、`.cursor/rules/tm-core.mdc` 等路径，快速定位略弱。  
  5. **可检索性**：当前为 personal；若希望团队检索到，需发布为 published 或由管理员发布。

---

## 五、建议的补充动作

1. **用 tm_update 追加 solution**（或 solution_addendum）：  
   把「四、改动效果」四条 + 「二、2.3 验收标准」四条压缩成一段， appended 到现有 solution，使一条经验里同时有「怎么做」和「改完效果 + 如何验收」。  
2. **可选**：在经验中增加 **code_snippets** 或 **related_links**：  
   存放「step-1 替换句」「设计文档冷启动单入口句」「Rules 一句」的原文，或指向 `workflow-optimization-task-1-cold-start-single-entry.md` 的路径，便于复现。  
3. **检索与发布**：  
   若 TM 检索不到该条，检查是否仅对 personal 可见或 embedding 未就绪；需要团队共享时对该条做「发布到团队」。

---

**说明**：TM 检索「工作流冷启动只在一个入口 step-0」/「冷启动 step-0 工作流」未返回结果，可能因该条为 personal 或尚未入索引。上述对比基于**写入时使用的 payload** 与**任务文档、对话与真实改动**的逐项对照。

**后续已做**：已用 `tm_update(experience_id=e5227822-8326-4868-b1b9-3d809f5bcb5a)` 追加 **改动效果**、**验收标准**、**涉及文件路径** 至 solution，并保留 tags。当前该条经验已包含「怎么做 + 改完效果 + 如何验收 + 改哪些文件」，更贴近文档与对话中的完整价值。
