# 档案馆（Archive）能力设计：经验附件与方案可追溯

> 目标：把 AI 对话沉淀、生成文档、代码/配置变更随经验一起入库，供后续设计方案与 Plan 时检索历史方案与代码演进，提升可行性与一致性。
> 角色：档案馆是 Experience 的**附件**，当关联经验全部发布后档案馆对团队可见；支持个人与团队记忆沉淀。
> 设计原则：**字段选择与检索返回均以不丢失命中率为目标**——参与向量/检索的文本与首屏展示（L0/L1）一致，避免「检索命中了但首屏信息不足导致 Agent 误判为不相关而跳过」。

---

## 一、需求共识（已确认）

| 维度 | 结论 |
|------|------|
| 档案馆与经验关系 | **方案 B**：档案馆为**独立实体**，可关联多条经验；同时语义上视为「经验的附件」——当所有关联经验发布后，档案馆内容对团队可见。 |
| 归档粒度 | 以**一次 Cursor 会话**为主；支持在对话中指定「某个 Plan」或其它范围（如某任务、某时间段）。 |
| 沉淀内容 | 会话 → 沉淀为**方案/计划文档**，内含：AI 方案、关键决策、预估收益、落盘代码（摘要或指针）。 |
| 已有文档 | 需要；未落盘的重要内容（ADR、会议纪要等）→ 优先落盘到 repo，档案馆存**内容快照 + 文件路径 + 版本/commit**。 |
| 代码变更 | 核心改动必存；改动过多时提取关键片段 + **指向 git commit id**，不全文冗余。 |
| 触发方式 | **手动**为主；通过**专用 Agent** 插入流程可自动完成（如 Plan 完成时自动归档）。 |
| 发布策略 | 见 [1.1 发布与可见性](#11-发布与可见性)（默认即写即用，团队可见由关联经验发布推导）。 |

### 1.1 发布与可见性

记忆系统以「即写即用」为宜，若强制「先存草稿再点击发布」会拖慢 AI 时代节奏。建议：

- **默认行为**：写入后**对创建者立即可见、可检索**（个人 scope）；无需额外「发布」操作即可在本人检索与后续会话中使用。
- **团队可见**：当且仅当**所有关联经验均已发布**时，该 Archive 对团队可见、参与 `tm_search`/`tm_solve` 的团队结果；否则仅创建者可见。
- **可选 draft**：若团队需要审核流程，可保留「draft → published」状态，但**不作为默认卡点**；多数场景下由「是否关联已发布经验」推导团队可见性即可。

这样既符合记忆系统即写即用，又保留团队边界与可选审核。

---

## 二、业界可借鉴点（调研小结）

公开产品里很少有「会话 + 代码演进 + 方案文档」一体化的标准形态，但以下模式可对齐：

| 来源 | 模式 | 可借鉴点 |
|------|------|----------|
| **Mem0 / 项目内 agent-memory-evaluation** | 短期（会话）→ 长期（持久化）；Retrieve → Inject → Store | 归档即「Store」；检索时需能「Inject」到 Agent 上下文；与 tm_preflight / tm_search 融合而非另起炉灶。 |
| **LangGraph Checkpoint** | 会话级状态持久化 | 归档单元可视为「会话级快照」；支持按 session / plan 维度。 |
| **本库 ExperienceArtifact / ExperienceReflection** | 经验附属的原文片段、任务反思 | 档案馆是**更粗粒度**的附属：整段方案+决策+代码演进，而非单条 artifact；可复用「经验 + 附属」的检索增强思路（命中经验时附带 Archive 摘要）。 |
| **ADR / 设计决策记录** | 决策 + 上下文 + 结论 | 档案馆内的「方案/计划」文档在结构上可对齐 ADR：背景、选项、决策、结果、关联 commit。 |

结论：采用**方案 B（档案馆独立实体、多对多关联经验）**与业界「会话→知识」「经验+附属」思路一致；档案馆作为「经验的附件」在团队可见性上由「关联经验是否全部发布」推导，避免未审核方案对团队暴露。

### 2.1 与 OpenViking 对齐

[Agent 记忆项目调研](agent-memory-projects-survey.md) 中 OpenViking（字节火山引擎）采用 **L0/L1/L2 分层上下文**，与本设计对齐如下：

| OpenViking | 本设计（Experience + Archive） |
|------------|----------------------------------|
| **L0（摘要）** ~100 token，一句话概括、快速筛选 | 检索第一阶段返回 **L0**：id、title、summary（或 description 前段）、score、experience_type、tags 等；Archive 同构：id、title、solution_preview（前 300–500 字）、linked_experience_ids。字段选择以**不丢失命中率为目标**，L0 的 summary 覆盖或来自参与 embedding 的文本。 |
| **L1（概览）** ~500–2000 token，核心信息、规划决策 | **写入时生成 overview**（中粒度梗概）；检索可返回 **L1**：在 L0 基础上增加 overview、solution_preview/root_cause 前段、code_snippets 有无等，便于判断是否拉全文。 |
| **L2（全文）** 按需加载 | **双阶段 MCP**：第一阶段 `tm_search`/`tm_solve` 返回 L0 或 L1 列表；第二阶段按需 `tm_get_experience(id)` / `tm_get_archive(id)` 拉 L2 全文。 |

据此，本设计采用**写入时生成 overview + 双阶段 MCP（先列表后详情）**，与 OpenViking 分层加载、节省 Token 的思路一致。

---

## 三、概念模型与术语

- **Archive（档案馆条目）**：一条归档记录，对应一次会话（或指定范围）的沉淀结果。
  - 必含：**方案/计划文档**（结构化或 Markdown），内含 AI 方案、关键决策、预估收益、落盘代码摘要或 git 引用。
  - 可含：对话摘要、生成文档快照、文件路径+版本/commit、核心代码片段或 commit id 列表。
- **Experience（经验）**：现有模型不变；一条经验可被多条 Archive 引用，一条 Archive 可关联多条经验。
- **发布规则**：默认写入即对创建者可见、可检索。Archive 对**团队**的可见性由关联的 Experience 决定：仅当所有关联经验均为「已发布」时，该 Archive 才参与团队检索；否则仅创建者可见。可选 draft 状态仅用于需要显式审核的场景（见 1.1）。

---

## 四、数据模型建议（与现有表结构兼容）

### 4.1 新增表：`archives`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| title | String(500) | 归档标题（如「xxx 方案归档」「Plan: 某需求」） |
| scope | String(20) | 归档范围：session \| plan \| task \| custom |
| scope_ref | String(200) | 可选，如 plan_id、task_id、session_id |
| solution_doc | Text | 方案/计划文档正文（Markdown）：AI 方案、关键决策、收益、落盘代码摘要 |
| **overview** | Text | **写入时生成**的中粒度梗概（L1，约 500–2000 字），与 OpenViking L1 对齐；参与检索展示与「不丢命中率」一致。 |
| conversation_summary | Text | 可选，对话摘要（或仅存指针，见下） |
| project | String(100) | 与 experience 一致，用于隔离 |
| created_by | String(100) | 创建者 |
| visibility | String(20) | 继承自关联经验：当全部发布时为 project/global，否则仅创建者 |
| status | String(20) | draft \| published（由关联经验推导或显式标记；默认即写即用下创建者始终可查） |
| created_at, updated_at | DateTime | 时间戳 |

**Embedding 与不丢命中率**：archives 的向量由 **title + overview（或 solution_doc 前段，当 overview 为空时）+ conversation_summary** 拼接后生成，与 Experience 的 embedding 构造方式在同一语义空间，便于 RRF 融合。参与 embedding 的文本应在 L0/L1 返回中有对应展示（如 summary / solution_preview），避免检索命中了但首屏信息不足被误判为不相关。

### 4.2 新增表：`archive_experience_links`（多对多）

| 字段 | 类型 | 说明 |
|------|------|------|
| archive_id | UUID → archives.id | CASCADE |
| experience_id | UUID → experiences.id | CASCADE |
| created_at | DateTime | 可选 |

唯一约束：`(archive_id, experience_id)`。

### 4.3 新增表：`archive_attachments`（文档/代码快照与引用）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| archive_id | UUID → archives.id | CASCADE |
| kind | String(30) | doc_snapshot \| file_ref \| code_snippet \| git_commit |
| path | String(1000) | 可选，文件路径或 doc 路径 |
| content_snapshot | Text | 可选，内容快照（大文本可压缩或外链） |
| git_commit | String(64) | 可选，commit sha |
| git_refs | JSONB | 可选，与 experience 的 git_refs 同格式 |
| snippet | Text | 可选，关键代码片段 |
| created_at | DateTime | 可选 |

- **已有文档**：kind=doc_snapshot 或 file_ref；path + content_snapshot 或 path + git_commit。
- **代码**：改动少存 snippet；改动多存关键片段 + git_commit，其余用 commit 指向。
- **未落盘重要内容**：先建议落盘到 repo，再在此存 path + commit；或短期存 content_snapshot，后续替换为 file_ref。

### 4.4 检索与可见性

- 对**创建者**：本人创建的 Archive 始终可被本人检索（即写即用）。
- 对**团队**：仅当 `archives.status = 'published'` 且由「关联经验全部已发布」推导为可见时，Archive 参与团队检索。
- 检索入口：**不新增单独「档案馆搜索」工具**，而是**扩展现有 tm_search / tm_solve**（见下），按 **L0/L1 先列表、按需 tm_get_archive(id) 拉 L2** 的双阶段方式，通过参数控制是否包含档案馆结果。

---

## 五、检索与 MCP 接口策略（双阶段、与 OpenViking 对齐）

采用**写入时生成 overview + 双阶段 MCP（先列表后详情）**：第一阶段返回 L0/L1 列表供快速筛选，第二阶段按需拉 L2 全文，字段选择与展示以**不丢失命中率为目标**，与 OpenViking L0/L1/L2 对齐。

### 5.1 双阶段检索

- **第一阶段（列表）**：`tm_search`、`tm_solve` 增加可选 `include_archives: bool = False`。当 `include_archives=True` 时，后端在同一语义检索中同时查 experiences 与 archives（archives 使用 title、overview、solution_doc 前段、conversation_summary 参与向量，**以不丢命中率为目标**选参与 embedding 的字段），返回 **L0 或 L1** 混合列表（见 5.3），每条带 `type: "experience"` 或 `type: "archive"`，便于 Agent 做「留/弃」判断。
- **第二阶段（详情）**：Agent 对需要全文的条目调用 `tm_get_experience(id)` 或 `tm_get_archive(id)` 拉取 **L2 全文**。
- **不新增** `tm_archive_search`，避免工具膨胀与「搜经验 vs 搜档案馆」歧义；一个检索动作即可拿到经验 + 历史方案的混合结果。

### 5.2 MCP 表面

| 动作 | 方式 | 说明 |
|------|------|------|
| 创建/更新档案馆 | 新增 `tm_archive_save` | 入参：title、scope、scope_ref、solution_doc、overview（或由服务端写入时生成）、conversation_summary、attachments、linked_experience_ids；内部写 archives（含 overview）+ archive_experience_links + archive_attachments。 |
| 检索（列表 L0/L1） | 扩展 `tm_search` / `tm_solve` | 增加可选 `include_archives: bool = False`；返回混合结果，区分 type=experience / archive，结构见 5.3。 |
| 详情（L2） | 新增 `tm_get_archive`（与现有 `tm_get_experience` 对称） | 按 archive id 返回全文（solution_doc、overview、attachments 等）；MVP 可实现为「按 id 查单条」，后续与 experience 的 L2 形态统一。 |
| 任务预检 | 可选扩展 `tm_preflight` | 返回中增加 `archive_summaries`（与 quick_results 并列）；**列为后续迭代**，不阻塞档案馆 MVP。 |

新增 **1 个写入工具（tm_archive_save）**、**1 个详情工具（tm_get_archive）**，检索仅扩参，保持语义清晰。

### 5.3 混合结果结构（不丢命中率）

第一阶段返回的每条结果在**同一语义下**统一形状，便于 Agent 快速对上目标、不因首屏信息不足误判为不相关。

| 类型 | L0 返回（约 ~100 token/条） | L1 返回（约 500–2000 token/条） |
|------|-----------------------------|----------------------------------|
| **Experience** | id、group_id、title、**summary**（无则 description 前 150 字）、score、experience_type、tags | L0 + **overview**、solution_preview（或 solution 前 400 字）、root_cause 前 100 字、code_snippets 有无或短预览 |
| **Archive** | id、title、**solution_preview**（solution_doc 前 300–500 字或 overview）、score、linked_experience_ids、attachment_count | L0 + **overview**（写入时生成）、solution_doc 前段延长、attachments 摘要 |

- **不丢命中率**：L0 的 summary / solution_preview 必须覆盖或来自**参与 embedding 的文本**（如 description 前段、overview、solution_doc 前段），这样检索命中的条目在首屏即有足够信息供 Agent 判断是否相关、是否拉 L2。
- **摘要长度**：Experience 的 summary 与 Archive 的 solution_preview 约定为前 300–500 字（或首段），由实现统一；L1 的 overview 为写入时生成、约 500–2000 字。
- **详情**：Experience 用现有 `tm_get_experience(id)`；Archive 用 `tm_get_archive(id)` 返回 solution_doc 全文、overview、attachments 列表及内容。
- **附件**：MVP 阶段 **archive_attachments 不参与向量检索**，仅在对某条 archive 调用 `tm_get_archive(id)` 时返回附件列表与内容；后续可考虑将附件摘要拼入 embedding。

---

## 六、流程与 Agent 角色

### 6.1 手动归档（MVP）

1. 用户在一次会话中通过自然语言指令触发，例如：「把这次对话和方案归档到团队经验库」「归档当前 Plan 到档案馆」。
2. Agent（或 Cursor 规则）解析意图后：
   - 整理当前会话中的**方案/计划**（AI 方案、关键决策、收益、已落盘代码）；
   - 生成 `solution_doc`；
   - 收集生成文档、修改过的文件、关键 commit；
   - 调用 `tm_archive_save`，并传入本会话中已保存的 experience id（或由用户指定）作为 `linked_experience_ids`。
3. 写入后对创建者立即可见、可检索（即写即用）。若关联经验尚未全部发布，Archive 对团队为 draft；当这些经验全部发布后，由发布逻辑或经验状态回写将对应 Archive 置为 published，团队检索即可见到。

### 6.2 半自动/自动归档（专用 Agent）— 后续迭代

- **本阶段**：仅实现 MCP 与存储能力，**由调用方（或上层 Agent）传入 solution_doc、overview、attachments**；不实现「从会话自动生成 solution_doc / overview」。
- **后续迭代**：设计「档案馆 Agent」或在工作流中插入步骤——用户说「归档」/「保存方案」或 Plan 完成时触发；从会话/Plan 产出 solution_doc 与 overview，写入 archives + archive_attachments，建立 archive_experience_links。自动归档可先以 draft 写入，由用户确认后再关联经验或对团队发布。

### 6.3 落盘与附件策略

- **已有文档**：档案馆中同时存内容快照 + 文件路径 + 版本/commit（见 archive_attachments）。
- **临时生成的重要文档**：在 Agent 流程中提示「建议将本内容落盘到 repo」；落盘后写入 path + commit，必要时补 content_snapshot 作为历史快照。
- **代码**：核心改动存 snippet；改动多则提取关键片段 + 存 git commit id，检索时通过 commit 再查详情。

---

## 七、对你思路的优化与完善

### 7.1 已对齐且建议保留

- 档案馆作为**经验的附件**、随经验发布而发布：逻辑清晰，与现有 visibility/exp_status 一致。
- 方案/计划中必须包含：AI 方案、关键决策、预估收益、落盘代码（摘要或指针）：这直接对应 `solution_doc` + `archive_attachments`。
- 手动触发 + Agent 自动完成：兼顾可控与自动化。
- 核心代码改动 + 多则用 commit 指向：控制存储与噪音。

### 7.2 建议补充与优化

1. **检索入口统一**：用「扩展 tm_search/tm_solve + include_archives」而非新开 tm_archive_search，减少工具数量与歧义（见第五节）。
2. **发布语义**：明确「档案馆对团队可见 = 所有关联经验已发布」的推导规则，并在数据模型或状态机中写死。**经验状态变更时需回写关联 Archive 的 status**（经验从 published 改为 draft 时，对应 archive 应同步为 draft，避免团队仍搜到「部分关联未发布」的 archive），建议纳入 MVP 或紧接 MVP 的 Task。
3. **对话体量**：会话全文可不必全部进 DB；存 **conversation_summary** + solution_doc 即可；若需全文，可存外部引用（如 Cursor 导出的会话 id）或压缩后放 content_snapshot。
4. **与 ExperienceArtifact 的分工**：ExperienceArtifact 保留为「单条经验的细粒度原文」；Archive 为「会话/方案级粗粒度沉淀」。检索时：先命中 experience，再可选附带其 linked archives 的 solution_doc 摘要，避免重复建设两套检索链。
5. **个人 vs 团队**：档案馆与 experience 共用 project/created_by；可见性完全由「关联经验是否全部发布」驱动，自然支持「个人草稿 → 团队可见」。
6. **真实可用的 Agent 记忆体系**：
   - **个人**：通过 tm_learn / tm_save + 个人 scope 的经验，以及「仅自己可见」的 draft archives 实现个人开发记忆。
   - **团队**：经验发布 + 关联 archives 发布后，tm_search / tm_solve（include_archives=True）即可在写方案、写 Plan 时召回历史方案与代码演进，形成团队可复用的记忆层。

---

## 八、实施顺序建议（高 level）

1. **数据模型**：新增 `archives`、`archive_experience_links`、`archive_attachments` 三张表及迁移。
2. **写入**：实现 `tm_archive_save`（含链接经验、写附件）；可选先不支持「自动从会话生成」，由调用方传入 solution_doc 与 attachments。
3. **发布规则**：实现「关联经验全部发布 → Archive 可见」的逻辑（或 status 更新）。
4. **检索**：在 SearchPipeline 或等价层支持「检索 experiences + archives」并打 type 标签；tm_search / tm_solve 增加 `include_archives` 参数。
5. **档案馆 Agent**：在流程中插入「归档」步骤，从会话/Plan 产出 solution_doc 与 attachments，调用 tm_archive_save。
6. **可选**：tm_preflight 返回中增加近期或相关的 archive 摘要，便于「先看历史再设计」。

---

## 九、预期收益与量级

以下为**方向性量级**，非承诺值；实际效果依赖采纳率、检索质量、项目类型与团队习惯。用于说明「值得做」与「验收时看什么」。

### 9.0 估算依据说明

**这些区间并非来自本项目或本功能的专项实验或内部数据**，而是基于类比与推理给出的量级，用于对齐期望和设定验收方向；真实收益应以第十节的基线 + 上线后测量为准。

| 依据类型 | 说明 |
|----------|------|
| **类比对象** | 与「有设计文档/ADR/历史方案可查」vs「从零讨论」的差异类比；知识库/记忆增强类能力（如 Mem0、文档检索注入上下文）在业界与你们项目内 [agent-memory-evaluation](agent-memory-evaluation.md) 的讨论中，普遍认为能减少重复探索、加速上下文恢复，但多数未给出精确百分比。 |
| **量级推理** | 方案阶段：若 20%–40% 的时间花在「查历史、对齐决策、重复解释」，可检索到高质量历史方案时，节省其中一部分（如一半）是合理区间，故取 10%–25% 作为「方案讨论与迭代时间」的节省区间；返工：类似地，部分返工来自「不知道之前怎么定的、重复踩坑」，可追溯方案与代码演进后，预期可减少其中一部分，取 5%–15% 作为保守区间。 |
| **为何是区间** | 不同团队、需求复杂度、归档与检索质量差异大；用区间而非单点，避免过度承诺；上限相当于「用得好」的情景，下限相当于「有一定采纳但检索一般」的情景。 |
| **如何得到真实数字** | 上线前做基线（方案迭代次数/耗时、返工率等），上线后按第十节指标观测；用「有 Archive 引用 vs 无」或「使用档案馆的周期 vs 未使用」对比，才能得到本团队的可信估计。 |

### 9.1 开发效率

| 维度 | 预期区间 | 前提与说明 |
|------|----------|------------|
| **方案/Plan 设计阶段** | 节省约 **10%–25%** 的方案讨论与迭代时间 | Agent 在做方案前能稳定召回相关历史 Archive；团队在写 Plan 时养成「先查历史方案」的习惯。若归档少或检索不准，收益接近 0。 |
| **决策与对齐** | 减少重复解释、重复讨论 | 历史方案与关键决策可被检索到，新成员或新会话可直接引用，定性收益为主。 |
| **上下文恢复** | 明显缩短「重新理解上次做到哪」的时间 | 复杂需求跨会话时，通过 Archive 快速恢复方案与代码演进，尤其利好个人与交接场景。 |

### 9.2 返工与重复踩坑

| 维度 | 预期区间 | 前提与说明 |
|------|----------|------------|
| **返工率** | 预期降低约 **5%–15%**（同一需求/模块的二次改动、方案推翻重来） | 依赖：历史方案与代码演进被检索到并在本次方案/实现中被参考；归档覆盖到「易踩坑」类会话。 |
| **重复踩坑** | 同类问题重复探索减少 | 检索到相似历史方案后，可复用决策与落盘方式，减少试错；量级依赖相似案例的积累与召回质量。 |

### 9.3 前提与局限

- **采纳率**：若无人归档或极少调用 `include_archives`，收益无法体现；需配合规则/提示与档案馆 Agent 降低使用门槛。
- **冷启动**：上线初期 Archive 少，收益有限；可先以「个人草稿」积累，再逐步发布为团队可见。
- **检索质量**：solution_doc 与附件的向量/FTS 质量直接影响「能否被正确召回」；需随使用迭代 prompt 与索引策略。
- **量级为区间**：上述百分比的估算依据见 9.0；实际应以本团队基线测量为准（见第十节）。

---

## 十、验证标准与效果评估

若缺少可观测、可验收的标准，设计容易停留在「好看但无效果」。以下给出可落地的验证体系，用于上线后评估是否真正带来价值，并在不达预期时驱动迭代或收缩。

### 10.1 指标定义

#### 投入侧（使用与覆盖）

| 指标 | 定义 | 用途 |
|------|------|------|
| **归档量** | 每月/每项目新增 `archives` 条数 | 衡量「有没有人在用」；长期为 0 则视为未生效。 |
| **归档率** | 有 tm_save / Plan 完成的会话中，产生至少 1 条 Archive 的比例（可按周/月） | 衡量归档是否成为流程一部分。 |
| **include_archives 调用占比** | tm_search / tm_solve 调用中 `include_archives=True` 的比例 | 衡量检索侧是否在用档案馆。 |
| **发布 Archive 占比** | 状态为 published 的 Archive 占全部 Archive 的比例 | 衡量从个人草稿到团队可见的转化。 |

#### 结果侧（效果）

| 指标 | 定义 | 用途 |
|------|------|------|
| **方案会话中的 Archive 引用率** | 在「设计方案 / 写 Plan」类会话中，Agent 返回结果里包含至少 1 条 type=archive 的比例（可采样或全量） | 衡量档案馆是否被纳入方案设计上下文。 |
| **返工率（可选）** | 同一需求/模块在首次合入后，N 周内再次发生改动（如 revert、大改、补丁）的比例；或由人工标记「本次为返工」的 task 占比 | 衡量是否减少重复劳动与方案推翻。 |
| **主观反馈** | 轻量问卷：做方案时是否用过历史方案、是否减少重复解释/踩坑（1–5 分或是否） | 补充客观指标，发现「有用但未体现在数据上」的价值。 |

### 10.2 基线与对比方式

- **基线**：上线前 4 周（或选定对照周期）内，对上述结果侧指标做一次测量；若无历史数据，可用「上线后第 1 个月 vs 第 3 个月」做前后对比。
- **对比组（若可行）**：同一团队中部分人/部分项目先用档案馆，对比「使用 vs 未使用」在返工率、方案迭代次数等方面的差异。
- **归因**：结果侧指标变化可能受需求复杂度、人员变动等干扰；分析时需结合投入侧指标（归档量、引用率）做交叉看，避免把无关因素归因到档案馆。

### 10.3 成功标准（验收门禁）

建议在「上线后 N 个月」做一次正式验收，满足以下条件视为**设计有效、非空壳**：

| 层级 | 条件 | 说明 |
|------|------|------|
| **最低** | 投入侧不为空：归档量 > 0，且 include_archives 调用占比 > 0 | 有人归档、有人在检索时看档案馆；不满足则判定为未用起来，需复盘动因（发现性、体验、规则）。 |
| **合格** | 方案会话中 Archive 引用率 ≥ 一定阈值（如 20%–30%，可按实际调整） | Agent 在做方案/Plan 时确实会召回并展示历史方案；阈值可按采样量调整。 |
| **理想** | 返工率较基线下降或持平，且无显著负面反馈 | 说明没有增加负担，且可能带来返工减少；若返工率上升需排查是否与档案馆无关（如需求波动）。 |

未达到「最低」时，不应视为「功能成功」，而应：改进发现性（规则/提示中明确何时归档、何时开 include_archives）、简化归档流程（Agent 自动归档 draft）、或收缩范围（先做个人侧验证）。

### 10.4 持续观测与迭代

- **埋点**：在 tm_archive_save、tm_search/tm_solve（include_archives=True）及检索返回中打点，便于统计归档量、调用占比、引用率（需能区分「返回中含 archive」）。
- **复盘节奏**：每季度或每半年回顾一次投入侧与结果侧指标；若引用率长期偏低，可优化检索策略（向量/FTS、摘要质量）或归档模板（solution_doc 结构）。
- **避免「只建不用」**：将「归档量 / 引用率」纳入团队或项目的轻量复盘（如周报、迭代回顾），避免功能上线后无人问津。

### 10.5 小结

- **预期收益**（第九节）给出开发效率与返工的大致量级和前提，用于对齐期望。
- **验证标准**（本节）把「有效」定义为可观测的指标与验收门禁，确保设计不只「好看」而是可被测量、可被迭代；不达标时能明确是「没用起来」还是「用起来但效果不足」，从而决定是推广、改体验还是收缩范围。

---

## 十一、总结

- **方案 B**（档案馆独立实体、多对多关联经验）与业界「会话→知识」「经验+附属」思路一致；与 **OpenViking** 对齐：L0/L1/L2 分层、**写入时生成 overview**、**双阶段 MCP（先列表 L0/L1，后按需 tm_get_archive 拉 L2）**；字段选择与检索返回以**不丢失命中率为目标**。  
- 档案馆内容 = **方案/计划文档**（必选）+ **overview（写入时生成）** + 对话摘要 + 文档/代码快照与 git 引用；代码多时用 commit 指向。  
- **发布**：默认即写即用（对创建者立即可见）；团队可见由「关联经验全部发布」推导；经验状态变更时回写关联 archive 的 status。  
- **检索**：扩展 `tm_search`/`tm_solve` 的 `include_archives` 返回 L0/L1 混合结果，新增 `tm_get_archive(id)` 拉 L2 全文；不新增单独档案馆搜索工具。  
- **触发**：本阶段由调用方传入 solution_doc/overview；档案馆 Agent 与自动从会话生成属后续迭代。  
- **收益与验证**：开发效率与返工预期有约 10%–25% / 5%–15% 的量级空间，强依赖采纳与检索质量；通过投入侧与结果侧指标及成功标准做验收。
