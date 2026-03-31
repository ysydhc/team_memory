# TeamMemory Agent 记忆设计评估报告

> 从 AI Agent 记忆视角评估 team_memory 的 tm_preflight、tm_search、tm_solve、tm_learn 等 MCP 工具设计。  
> 评估日期：2025-03-11

---

## 一、现状

### 1.1 经验 / 个人记忆 / 任务的融合策略

**检索时未融合**。当前实现：

- **tm_search / tm_solve / tm_preflight**：仅调用 `SearchPipeline`，检索 `Experience` 表（向量 + FTS 混合）。
- **PersonalMemory**：有独立 Web API `GET /personal-memory?current_context=...`，支持 `generic` + `context` 语义召回，但**无 MCP 工具**，Agent 无法在检索时直接拉取。
- **PersonalTask**：仅与 `experience_id` 关联（completed 后沉淀为 experience），检索时**不参与**。

结论：Experience 与 PersonalMemory 在 Agent 检索路径上**完全分离**，Agent 无法在一次调用中同时获得「团队经验」与「个人偏好」。

### 1.2 短期记忆与长期记忆的衔接

**衔接不完整**：

- **短期**：`.cursor/extraction-state.json` 记录会话轮数、`last_trigger_at` 等，供 Agent 决策是否触发提取；Agent 的上下文由 Cursor/Claude 自身维护。
- **长期**：TM 持久化 DB（Experience、PersonalMemory、PersonalTask）。
- **缺口**：TM 不主动向 Agent 注入短期记忆；Agent 读取 extraction-state 后决定是否调用 tm_save/tm_learn，但**无统一「会话 → 记忆」的注入机制**。Mem0 式的「Retrieve → Inject → Store」流程中，TM 未实现「Inject」环节的自动注入。

### 1.3 提取写入路径与检索路径的对应关系

| 写入路径 | 检索路径 | 对齐情况 |
|----------|----------|----------|
| tm_save / tm_learn → Experience | tm_search / tm_solve / tm_preflight | ✅ 对齐 |
| tm_learn → PersonalMemory（异步） | 无 MCP 工具 | ❌ 不对齐 |
| tm_task completed → 沉淀为 experience | tm_search 可检索 | ✅ 对齐 |

PersonalMemory 写入后，Agent 无法通过 MCP 检索到；仅 Web 端或外部 API 可调用 pull。

---

## 二、业界可借鉴点

| 系统 | 分层模式 | 可借鉴点 |
|------|----------|----------|
| **Mem0** | 短期（会话） + 长期（持久化） | 检索 → 注入上下文 → 存储；统一记忆接口 |
| **Mem0 + LangGraph** | 检索 + 注入 + 存储 | 在每次 LLM 调用前自动检索并注入记忆，减少 Agent 显式调用 |
| **LangGraph Checkpoint** | 图状态持久化 | 会话级 checkpoint 可恢复；TM 可考虑将 extraction-state 与任务状态关联 |

---

## 三、改进方案

1. **新增 `tm_pull_memory` MCP 工具**  
   - 在 tm_preflight 或任务开始时，由 Agent 显式调用；或  
   - 在 tm_preflight 内部实现：返回 `quick_results` 的同时，附带 `personal_memories`（generic + context 匹配的 current_context=task_description）。

2. **检索融合**  
   - 在 tm_preflight 中同时拉取 PersonalMemory 并注入返回结果；  
   - 或新增 `tm_context` 工具：一次调用返回 Experience + PersonalMemory + 当前任务摘要，供 Agent 作为「上下文包」使用。

3. **短期 ↔ 长期衔接**  
   - 在 MCP instructions 中明确：任务开始先调用 tm_preflight，并将返回的 `quick_results` 与 `personal_memories` 一并注入到后续 prompt；  
   - 或由 Cursor/Claude 的规则层在每次会话开始时自动调用 tm_preflight + pull，将结果注入 system prompt。

4. **写入路径对齐**  
   - 在 tm_learn 中已写入 PersonalMemory 的前提下，确保 tm_preflight 或 tm_pull_memory 能检索到，形成闭环。

---

## 四、置信度

| 维度 | 置信度 | 说明 |
|------|--------|------|
| 融合策略现状 | 高 | 代码与 grep 明确；SearchPipeline 仅查 Experience |
| 短期/长期衔接 | 中 | 依赖 extraction-state 与规则设计，需结合 Agent 端实现 |
| 写入/检索对齐 | 高 | PersonalMemory 无 MCP 检索入口，事实明确 |
| 改进方案可行性 | 中 | 需评估 MCP 工具新增与调用链改动 |

---

## 五、附录：关键代码路径

- 检索：`server.py` → `service.search()` → `SearchPipeline` → `ExperienceRepository`
- PersonalMemory 写入：`tm_learn` → `_try_extract_and_save_personal_memory` → `PersonalMemoryService.write`
- PersonalMemory 拉取：`web/routes/personal_memory.py` → `pull_personal_memory`（仅 Web API）
