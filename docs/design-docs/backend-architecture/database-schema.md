# 数据库 Schema 与表结构

> 技术架构 | 表结构说明  
> **MVP 以 `src/team_memory/storage/models.py` 与 `migrations/versions/001_initial_mvp.py` 为准。**  
> 运维操作：[database-operations](../ops/database-operations.md) | 技术概念：[设计文档索引](../README.md#techconcepts)

## 表关系总览

```
┌──────────────────────┐       ┌──────────────────────┐
│    experiences        │       │ experience_feedbacks  │
│ ─────────────────── │       │ ──────────────────── │
│ id (UUID, PK)        │◄──┐  │ id (Integer, PK)     │
│ parent_id (UUID, FK) │───┘  │ experience_id (FK) ──│──► experiences.id
│ title                │      │ rating (1-5)          │
│ embedding (vector)   │      │ feedback_by           │
│ fts (tsvector)       │      └──────────────────────┘
│ ...                  │
└──────────────────────┘

experiences.parent_id ──► experiences.id (自引用，父子层级)
```

另：MVP 含 `archives`、`archive_experience_links`、`archive_attachments`、`document_tree_nodes`（挂 `archive_id`）、`personal_memories`。已从旧版删除的表（如 `tool_usage_logs`、`experience_links`）见 `migrations/versions/002_mvp_cleanup.py`。

## 核心表（MVP）

| 表 | 说明 |
|----|------|
| experiences | 经验主表：embedding、fts、父子层级、project、visibility、exp_status、tags、group_key 等 |
| experience_feedbacks | 反馈：rating、fitness_score、comment |
| archives | 档案馆摘要与嵌入 |
| archive_experience_links / archive_attachments | 档案与经验/附件关联 |
| document_tree_nodes | 档案绑定的章节树（PageIndex-Lite） |
| personal_memories | 个人记忆 |
| api_keys | API Key 认证 |

## experiences 主要字段（MVP）

| 字段 | 类型 | 说明 |
|------|------|------|
| title / description / solution | Text | 问题与方案；`description` 即 problem |
| tags | String[] | 含可选认领标签 `agent_claim\|...` |
| group_key | String | 同组经验 |
| experience_type | String(30) | 类型标签 |
| embedding | vector(768) | 与默认 Ollama 维度一致 |
| fts | tsvector | 触发器维护全文检索 |
| project / visibility / exp_status | String | 隔离与状态 |
| is_deleted / deleted_at | bool + timestamptz | 软删除 |
| use_count | int | 隐式召回计数 |

平均评分不冗余存储：需要时对 `experience_feedbacks` 做聚合。

## experience_feedbacks 主要字段

| 字段 | 类型 | 说明 |
|------|------|------|
| rating | Integer | 1-5，5 为最佳 |
| fitness_score | Integer | 可选，1-5 适用度 |
| feedback_by | String(100) | 反馈人 |


## personal_memories（用户画像源）

| 字段 | 类型 | 说明 |
|------|------|------|
| profile_kind | String(16) | `static`（长期偏好） / `dynamic`（近期语境）；迁移 `004_profile_kind` |
| scope | String(20) | `generic` / `context`，与 kind 同步（HTTP 兼容） |
| content | Text | 一条可读事实 |
| embedding | vector(768) | 语义合并与 dynamic 上下文匹配 |


## 设计决策

- **为什么选择 PostgreSQL**：见 [database-design-decisions](database-design-decisions.md)
- **UUID 主键**：跨系统传递不冲突、分布式友好
- **ARRAY 类型**：标签少时一次读取、无需 JOIN
- **软删除**：误删可恢复、数据审计
