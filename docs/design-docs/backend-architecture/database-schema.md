# 数据库 Schema 与表结构

> 技术架构 | 表结构说明
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

## 核心表

| 表 | 说明 |
|----|------|
| experiences | 经验主表，含 embedding、fts、父子层级、经验类型、scope、project |
| experience_feedbacks | 反馈评分（rating 1-5、fitness_score） |
| api_keys | API Key 认证 |
| query_logs | 搜索日志 |

## experiences 主要字段（与 models.py 对齐）

| 字段 | 类型 | 说明 |
|------|------|------|
| experience_type | String(30) | general/feature/bugfix/tech_design/incident/best_practice/learning |
| scope | String(20) | global/team/personal |
| project | String(100) | 项目隔离 |
| structured_data | JSONB | 类型相关结构化数据 |
| git_refs | JSONB | Git 引用 |
| embedding_dim | Integer | 向量维度（支持混合维度） |
| summary | Text | LLM 摘要（记忆压缩） |
| visibility | String(20) | private/project/global |
| exp_status | String(20) | draft/review/published/rejected |
| publish_status | String(20) | personal/draft/pending_team/published/rejected |
| quality_score | Integer | 质量打分（0-300） |

## experience_feedbacks 主要字段

| 字段 | 类型 | 说明 |
|------|------|------|
| rating | Integer | 1-5，5 为最佳 |
| fitness_score | Integer | 1-5 使用后适用度 |
| feedback_by | String(100) | 反馈人 |

## 设计决策

- **为什么选择 PostgreSQL**：见 [database-design-decisions](database-design-decisions.md)
- **UUID 主键**：跨系统传递不冲突、分布式友好
- **ARRAY 类型**：标签少时一次读取、无需 JOIN
- **软删除**：误删可恢复、数据审计
