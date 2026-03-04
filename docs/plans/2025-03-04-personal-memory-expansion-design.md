# 个人记忆与个人扩写 — 设计文档

> **For Claude:** 本文档描述个人记忆与个人扩写功能的实现设计与当前行为，供维护与扩展参考。

**目标**：(1) 提高检索命中率：per-user 扩写（tag_synonyms）在检索前做词表替换 + LLM 扩写；(2) 让 Agent 快速理解个人风格：个人记忆独立存储、从对话自动提炼、拉取时按「通用 + 当前上下文语义」返回。

---

## 一、个人记忆

### 1.1 存储模型

- **表**：`personal_memories`
- **字段**：id, user_id, content, scope, context_hint, embedding, created_at, updated_at
- **scope**：`generic`（始终返回）或 `context`（仅当 current_context 语义相似时返回）
- **embedding**：768 维，与 TM 经验一致，用于语义去重/覆盖与拉取过滤

### 1.2 写入规则

- **语义覆盖**：写入时与同 user 已有条算相似度；超过阈值（0.88）则 **update** 该条，否则 **insert**。无人工确认。
- **阈值**：`PERSONAL_MEMORY_OVERWRITE_THRESHOLD = 0.88`（与经验检索的 min_similarity 区分，仅用于个人记忆覆盖）
- **自动提炼**：`tm_learn` 成功后调用 `parse_personal_memory`，从对话中提取个人偏好与工作习惯；超时/失败**不阻塞**经验主流程

### 1.3 拉取规则

- **接口**：`GET /api/v1/personal-memory?current_context=...` 或 MCP
- **已登录**：返回 scope=generic 全部 + scope=context 且与 current_context 语义相似度达标的条
- **匿名**：直接返回空列表 `[]`，不返回 generic
- **无 current_context**：仅返回 generic

### 1.4 Web 端

- **入口**：设置 → 个人记忆
- **能力**：查看列表（按 scope 筛选）、编辑单条（content/scope/context_hint）、删除单条
- **API**：POST /personal-memory、GET /personal-memory/list、GET/PUT/DELETE /personal-memory/{id}
- **权限**：仅本人可操作，需登录

---

## 二、个人扩写

### 2.1 存储模型

- **表**：`user_expansion_configs`
- **字段**：id, user_id, tag_synonyms (JSONB), updated_at
- **格式**：与 config.tag_synonyms 一致，如 `{ "PG": "PostgreSQL", "JS": "JavaScript" }`

### 2.2 检索前扩写

- **位置**：SearchPipeline 中，按 current_user 取扩写数据 → 词表替换 → 现有 LLM 扩写 → 用扩写后 query 检索
- **匿名**：不加载 per-user 扩写，仅用全局 tag_synonyms
- **Cache key**：含 current_user，避免跨用户错用缓存

### 2.3 自动维护

- **入口**：MCP `tm_search` 返回后，根据 query + 结果中的 tag 推断映射，更新对应用户的 tag_synonyms
- **规则**：prefix/contains 优先，否则用「长度 + 首字母」启发式
- **条件**：仅已登录且非 anonymous 时写入

### 2.4 Web 端

- **入口**：设置 → 个人扩写
- **能力**：查看当前 tag_synonyms、编辑（增删改同义词）
- **API**：GET /user-expansion-config（匿名返回空）、PUT /user-expansion-config（需登录）
- **权限**：仅本人可操作

---

## 三、与经验的边界

- **经验**：团队共享、需审核/发布、可见性由 publish_status 控制
- **个人记忆**：按 user 隔离、仅本人可读写、无审核
- **个人扩写**：按 user 隔离、仅本人可读写、检索时合并到 query
- **自动提炼**：与 `parse_content` 同维度，共用对话输入；输出写个人记忆存储，不写经验表

---

## 四、匿名降级

| 能力       | 匿名行为                         |
|------------|----------------------------------|
| 个人记忆拉取 | 返回 `[]`                        |
| 个人扩写 GET | 返回 `{ tag_synonyms: {} }`      |
| 个人扩写 PUT | 401                              |
| 检索前扩写   | 仅用全局 tag_synonyms            |
| 自动维护    | 不写入                           |

---

## 五、关键代码位置

| 功能         | 文件 |
|--------------|------|
| 个人记忆模型   | `storage/models.py` PersonalMemory |
| 个人记忆仓库   | `storage/repository.py` PersonalMemoryRepository |
| 个人记忆服务   | `services/personal_memory.py` |
| 个人记忆路由   | `web/routes/personal_memory.py` |
| 自动提炼      | `services/llm_parser.py` parse_personal_memory |
| 扩写模型/仓库  | `storage/models.py` UserExpansionConfig, `repository.py` UserExpansionRepository |
| 扩写路由      | `web/routes/user_expansion.py` |
| 检索前扩写    | `services/search_pipeline.py` |
| 自动维护      | `server.py` _try_update_user_expansion_from_search |
