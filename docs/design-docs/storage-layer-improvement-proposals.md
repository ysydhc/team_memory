# TM 存储分层改进方案（带置信度）

> 基于 Backend Architect、Data Engineer、AI Engineer 三路 subagent 调研，综合产出。
> 日期：2025-03-11

---

## 一、现状摘要

| 维度 | 现状 | 置信度 |
|------|------|--------|
| Experience 职责 | 团队经验 + 个人偏好 + 任务沉淀混在同一表 | 高 |
| PersonalMemory 边界 | 规则写 Experience，实现写 PersonalMemory，路径不一致 | 中 |
| Artifact / Reflection | 有写入无检索，价值未发挥 | 高 |
| 检索入口 | tm_preflight 只拉 Experience，不拉 PersonalMemory | 高 |
| Schema 一致性 | preset 间 type/category 不统一，structured_data 校验不全 | 高 |
| 写入/检索对齐 | PersonalMemory 写入后无 MCP 检索入口 | 高 |

---

## 二、改进方案（按置信度排序）

### 方案 A：tm_preflight 整合 PersonalMemory（推荐优先）

**内容**：在 `tm_preflight` 返回中增加 `personal_memories` 字段，当 `current_user` 已认证时，同时拉取 generic + context 匹配的个人记忆，与 `quick_results` 一并返回。

**置信度**：**0.92**（高）

| 优点 | 缺点 |
|------|------|
| 改动小，仅扩展 tm_preflight 返回结构 | 需确保 current_user 在 MCP 调用中可传递 |
| 规则「任务开始必调 tm_preflight」已存在，无需新增调用 | 返回体略增大 |
| 一次调用获得经验 + 个人记忆，Agent 体验提升明显 | - |
| 与 Mem0 的「Retrieve → Inject」理念一致 | - |

**实施要点**：在 `server.py` 的 `tm_preflight` 中，若 `current_user` 非空，调用 `PersonalMemoryService.pull()` 并合并到返回。

---

### 方案 B：统一「记忆」与「经验」的写入规则

**内容**：更新 `tm-extraction-retrieval.mdc`，明确「记忆」（偏好、习惯）→ PersonalMemory，「经验」（bugfix、tech_design 等）→ Experience；`tags=user_preference` 的写入路由到 PersonalMemory，不再写入 Experience。

**置信度**：**0.88**（高）

| 优点 | 缺点 |
|------|------|
| 规则与实现一致，消除语义混淆 | 需修改 tm_save/tm_learn 的写入逻辑，增加路由判断 |
| 个人偏好不再污染 Experience 检索 | 历史 `user_preference` 数据需迁移或兼容 |
| 与 PersonalMemory 的 mem0 风格覆盖逻辑天然契合 | - |

**实施要点**：在 `tm_save`、`tm_learn` 解析结果中，若 `tags` 含 `user_preference`，调用 `PersonalMemoryService.write()` 而非 `ExperienceService.save()`。

---

### 方案 C：Schema 校验与 preset 孤儿 type 处理

**内容**：（1）在 `ExperienceService.save()` 入口对 `experience_type`、`category`、`severity` 做 SchemaRegistry 校验，非法值拒绝或 fallback 到 `general`；（2）为 data-engineering、devops、general 中带 `structured_fields` 的 type 补充 Pydantic schema 或动态校验；（3）定义 preset 切换时的「孤儿 type」映射表（如 `feature` → `general`），或在 API 层将未知 type 展示为「其他」。

**置信度**：**0.85**（高）

| 优点 | 缺点 |
|------|------|
| 防止脏数据入库，提升数据质量 | 需为多 preset 补充 schema，工作量中等 |
| 预设切换后历史数据可兼容展示 | 动态校验需与 SchemaRegistry 深度集成 |
| 逐步废弃 legacy 常量，统一从 SchemaRegistry 读取 | - |

**实施要点**：分阶段：P0 入口校验；P1 补充 schema；P2 孤儿 type 映射。

---

### 方案 D：Artifact / Reflection 纳入检索（可选阶段）

**内容**：在 search_pipeline 中增加可选阶段：命中 Experience 时，附带其 `ExperienceArtifact` 的 content 或 `ExperienceReflection` 的 `generalized_strategy`，作为补充上下文注入；或为 Artifact 建独立向量索引，在混合检索中参与 RRF 融合。

**置信度**：**0.72**（中高）

| 优点 | 缺点 |
|------|------|
| 发挥 P2 设计价值，verbatim 引用与反思可提升召回质量 | 实现复杂度较高，需评估 Artifact 数据量与索引成本 |
| 与「可复现、可追溯」的团队知识理念一致 | 当前 Artifact/Reflection 写入量可能较少，ROI 待验证 |

**实施要点**：先做「命中 Experience 时附带 Artifact/Reflection」的轻量实现，再视数据量决定是否建独立索引。

---

### 方案 E：Experience 职责拆分（中长期）

**内容**：将「个人偏好」从 Experience 完全剥离，或新增 `UserPreference` 表；任务沉淀保留在 Experience，通过 `experience_type` 或 `source` 区分。个人经验（scope=personal）与团队经验（scope=team）在 schema 上已有区分，可保留。

**置信度**：**0.68**（中）

| 优点 | 缺点 |
|------|------|
| 表职责清晰，检索逻辑简化 | 改动大，涉及迁移、API 兼容、前端展示 |
| 与方案 B 配合可彻底解决边界问题 | 需评估是否 YAGNI，当前问题是否必须通过拆表解决 |

**实施要点**：建议在方案 B 落地并观察一段时间后，再评估是否推进；若 `user_preference` 写入已路由到 PersonalMemory，Experience 表压力会显著减轻。

---

## 三、推荐实施顺序

1. **方案 A**（tm_preflight 整合 PersonalMemory）— 见效快，风险低
2. **方案 B**（统一写入规则）— 与 A 配合形成闭环
3. **方案 C**（Schema 校验）— 提升数据质量，可并行推进
4. **方案 D**（Artifact/Reflection 检索）— 视数据量与需求决定优先级
5. **方案 E**（Experience 拆表）— 中长期，待 A/B 落地后评估

---

## 四、附录：置信度说明

- **0.90+**：基于代码与配置的静态分析，结论可靠  
- **0.80–0.89**：基于调用链与设计文档，需少量运行时验证  
- **0.70–0.79**：基于业界实践与推理，实施效果待验证  
- **0.60–0.69**：涉及较大改动或产品语义，需进一步确认  

---

## 五、OpenMem / MemOS 调研与 TM 可借鉴点

> 字节（记忆张量）联合上海交大、上海算创院等开源的 **OpenMem** 生态，核心产品 **MemOS**（Memory Operating System）是面向 LLM/Agent 的记忆操作系统。GitHub: [MemTensor/MemOS](https://github.com/MemTensor/MemOS)（6.5k+ stars）

### 5.1 OpenMem 核心架构

| 组件 | 说明 |
|------|------|
| **MemOS** | 记忆操作系统，将记忆视为 LLM 一级资源，统一表示、调度、演化 |
| **MemCube** | 统一记忆单元：元数据头 + 语义负载 + 行为指标，支持形态间转换 |
| **三层记忆** | 明文（Plaintext）→ 激活（Activation/KV Cache）→ 参数（Parameter） |
| **三层架构** | 接口层（Memory API）→ 操作层（调度与生命周期）→ 基础设施层（持久化） |
| **核心模块** | MemReader（语义理解）、MemScheduler（调度）、MemLifecycle（生命周期）、MemOperator（索引与图） |

### 5.2 记忆生命周期与状态

MemOS 定义记忆单元可流经：**生成 → 激活 → 合并 → 归档 → 冻结**。支持动态转换：使用频率低可降级，稳定内容可提升为 KV cache，频繁使用的明文可提炼为参数化权重。

### 5.3 生态与工具

| 项目 | 说明 |
|------|------|
| **Text2Mem** | 自然语言 → 标准化 JSON 指令，统一 encode/storage/retrieval/merge/promote/demote |
| **LightMem** | 轻量记忆：感官记忆（预压缩）→ 短时（主题组织）→ 长时（睡眠更新），Token 消费降低 117 倍 |
| **MemOS Skills** | 对话碎片 → 结构化技能，跨任务复用；智能切片、聚类提取、技能转换 |

### 5.4 TM 可借鉴点（按优先级）

| 借鉴点 | 来源 | 落地建议 | 置信度 |
|--------|------|----------|--------|
| **记忆生命周期** | MemOS MemLifecycle | 为 Experience 增加「激活/归档/冻结」等状态，低质量或过期经验可自动降级 | 0.85 |
| **MemCube 式统一封装** | MemOS MemCube | 将 Experience、PersonalMemory、Artifact 抽象为统一「记忆单元」接口，便于调度与融合 | 0.78 |
| **tm_preflight 整合个人记忆** | MemOS 统一 Memory API | 与方案 A 一致：一次调用返回经验 + 个人记忆 | 0.92 |
| **Text2Mem 式操作语言** | Text2Mem | 将 tm_save/tm_learn/tm_search 等 MCP 工具抽象为统一 JSON 指令，便于跨后端适配 | 0.70 |
| **LightMem 语义压缩** | LightMem | 长对话/长文档提取时先做语义压缩与噪声过滤，减少冗余写入 | 0.72 |
| **Skills 跨任务复用** | MemOS Skills | 将任务沉淀（task → experience）升级为「技能」抽象，支持技能版本与跨任务迁移 | 0.68 |

### 5.5 与 TM 的差异

| 维度 | MemOS | TM |
|------|-------|-----|
| 记忆形态 | 明文 + 激活 + 参数 | 仅明文（Experience、PersonalMemory） |
| 调度 | 系统级调度、生命周期 | 检索为主，无显式调度 |
| 存储 | 图数据库（Neo4j）等 | PostgreSQL + pgvector |
| 目标 | 通用 AGI 记忆基础设施 | 团队经验库 + Agent 上下文 |

TM 当前聚焦「明文记忆」层，无需引入 KV cache 或参数记忆；可优先借鉴 MemOS 的**生命周期管理**和**统一封装**思路，而非全量复制。
