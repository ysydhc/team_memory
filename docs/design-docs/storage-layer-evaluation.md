# team_memory 存储分层评估报告

**评估对象**：`src/team_memory/storage/models.py`、`schemas.py`、检索路径（search_pipeline、repository）  
**日期**：2025-03-11

---

## 一、当前问题

### 1. Experience 表职责过重

Experience 同时承载多种语义：

- **团队经验**：bugfix、feature、tech_design、incident、best_practice、learning
- **个人经验**：`scope=personal` / `visibility=private`，仍为 problem-solution 结构
- **任务沉淀**：`PersonalTask.sediment_experience_id`、`experience_id` 关联
- **个人偏好**：规则要求 `tags=user_preference` 的 tm_save → 实际写入 Experience，而非 PersonalMemory

同一张表混用「可复用的团队知识」与「个人偏好/习惯」，导致：

- 检索时需通过 scope/visibility 过滤，逻辑复杂
- 个人偏好与团队经验在 schema 上无区分（无 `user_preference` 等标记）
- 字段冗余：`publish_status`、`review_status` 与 `visibility`、`exp_status` 并存，存在历史包袱

### 2. PersonalMemory 与 Experience 边界不清

| 维度 | PersonalMemory | Experience (scope=personal) |
|------|----------------|----------------------------|
| 用途 | 偏好/习惯，Agent 上下文 | 个人经验卡片，可 publish 到 team |
| 结构 | content + scope(generic/context) | title/description/solution/root_cause |
| 更新 | mem0 风格，语义相似则覆盖 | 常规 CRUD |
| 写入 | tm_learn 解析 → parse_personal_memory | tm_save/tm_learn + tags=user_preference |

问题在于：规则文档将「记忆」定义为 `tm_save` + `tags=user_preference`，但该路径写入的是 Experience，而非 PersonalMemory。PersonalMemory 主要由 tm_learn 的 `parse_personal_memory` 写入。两条写入路径语义重叠，易混淆。

### 3. ExperienceArtifact / ExperienceReflection 使用率低

- **ExperienceArtifact**：存储 verbatim 引用（decision、problem、pattern 等），有 `tm_extract_artifacts` 写入，但 **检索 pipeline 未使用**，search_pipeline 仅查 `experiences` 表。
- **ExperienceReflection**：存储任务后反思（success_points、failure_points、generalized_strategy），有 `tm_save_reflection` 写入，同样 **未纳入检索**。

二者均为 P2 设计（P2.B / P2.C），写入路径存在，但检索路径缺失，价值未发挥。

### 4. 检索路径与存储分层不一致

- **search_pipeline**：只依赖 `ExperienceRepository`，对 `experiences` 做 vector + FTS 检索。
- **PersonalMemory**：独立 `PersonalMemoryRepository`，通过 `/api/v1/personal-memory` 的 pull 接口获取，**不经过 search_pipeline**。
- **tm_preflight**：仅调用 `service.search()`，拉取 Experience，**不拉取 PersonalMemory**。规则要求「任务开始必调 tm_preflight」，但个人记忆需单独获取，检索入口不统一。

---

## 二、改进建议

1. **Experience 职责拆分（中优先级）**
   - 将「个人偏好」明确从 Experience 剥离：`tags=user_preference` 的写入应路由到 PersonalMemory，或新增 `UserPreference` 表。
   - 任务沉淀可保留在 Experience，通过 `experience_type` 或 `source` 区分，避免与团队经验混查。

2. **统一 PersonalMemory 与 Experience 的写入规则**
   - 更新 tm-extraction-retrieval 规则：明确「记忆」= PersonalMemory（或等价存储），「经验」= Experience；避免 `user_preference` 写入 Experience。

3. **Artifact / Reflection 纳入检索（低优先级）**
   - 在 search_pipeline 中增加可选阶段：命中 Experience 时，附带其 Artifact 的 content 或 Reflection 的 generalized_strategy，作为补充上下文；或为 Artifact 建独立向量索引，在混合检索中参与融合。

4. **tm_preflight 与个人记忆整合**
   - 在 tm_preflight 返回中增加 `personal_memories` 字段（当 user 已认证时），或提供 `tm_preflight_with_memory` 变体，使任务启动时一次性获取经验 + 个人记忆。

---

## 三、置信度

| 评估项 | 置信度 | 说明 |
|--------|--------|------|
| Experience 职责过重 | **高** | 代码与 schema 可直接验证，字段与语义混杂明显 |
| PersonalMemory 与 Experience 边界 | **中** | 规则与实现存在不一致，需结合产品语义进一步确认 |
| Artifact / Reflection 使用率 | **高** | grep 与检索路径分析可确认未参与检索 |
| 检索路径与存储分层 | **高** | search_pipeline、repository、tm_preflight 调用链清晰 |

**综合置信度**：**高**。建议优先落地「规则与写入路径统一」和「tm_preflight 整合个人记忆」，再视需求推进 Experience 拆分与 Artifact/Reflection 检索增强。
