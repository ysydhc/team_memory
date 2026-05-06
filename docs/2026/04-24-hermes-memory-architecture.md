# Hermes 记忆体系架构设计

> 2026-04-24 · 基于三大方向调研：llm-wiki / TeamMemory / GraphRAG

## 一、三大方向对比

| 维度 | llm-wiki (Karpathy) | TeamMemory (TM) | GraphRAG (微软) |
|------|---------------------|-----------------|-----------------|
| **定位** | 结构化知识编译器 | 会话经验 RAG | 图结构知识引擎 |
| **存什么** | 学到的技术知识 | 踩坑/调试/项目经验 | 实体 + 关系 + 社区 |
| **擅长** | 从零散调研编译出体系化知识 | 语义检索"之前遇到过X吗" | 多跳推理、跨文档关联 |
| **不擅长** | 不能实时检索、不存过程 | 不存结构化知识、无关联推理 | 贵、慢、维护重 |
| **成本/条** | ~$0（本地编译） | ~$0.001（嵌入） | ~$4（LLM抽取+社区） |
| **查询延迟** | N/A（静态文档） | <500ms | 2-5s |
| **更新方式** | 人工/LLM 重编译 | 实时写入 draft→published | 增量重建（慢） |
| **规模** | 按文档 | 按会话（单用户日增 10-50 条） | 按语料库（千+文档） |

### 核心结论

- **GraphRAG 不适合直接用于 Agent 记忆**：$4/条 + 2-5s 延迟 + 维护重，对企业级语料库才有意义
- **GraphRAG 的思想值得借鉴**：实体/关系抽取 + 实体中心检索，可以用轻量方式在 PostgreSQL 内实现
- **三者互补而非互斥**：llm-wiki 编译知识 → TM 存经验和检索 → 图层补关联

## 二、Hermes 记忆体系分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 自有记忆（L0/L1）                    │
│  Hermes: memory + session_search                             │
│  Cursor: .cursor/rules/*.mdc                                 │
│  Claude Code: CLAUDE.md + MEMORY.md                          │
│  ← 不在本系统范围，各 Agent 自行管理                          │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                L2 · TM Experience（痕迹）                     │
│  PostgreSQL + pgvector + TM Daemon                           │
│  · 会话经验：调试过程、踩坑记录、方案选择                      │
│  · 自动捕获：Hooks → draft → 收敛检测 → published            │
│  · 检索：语义向量 + 全文检索 → RRF 融合 → 重排               │
│  · 评估：[mem:xxx] 标记 → was_used 自动判定                  │
│  · 提升：use_count≥3 → 自动提升为 L3                         │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│         L2.5 · 实体关系层（轻量图，新增）                      │
│  PostgreSQL 表：entities + relationships                      │
│  · 从会话/草稿中抽取实体和关系（小模型 or structured output）  │
│  · 存储在 PostgreSQL，用 recursive CTE 做图遍历               │
│  · 不做社区检测（规模不够），不做全图重建                      │
│  · 查询：实体中心 → 1-2 跳关系 → 上下文增强                   │
│  · 成本：~$0.0001/条（本地模型抽取）                          │
│  · 延迟：<100ms（SQL 查询）                                   │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│           L3 · Markdown 知识库（人类知识）                    │
│  Obsidian + Git + TM 索引                                    │
│  · 结构化知识：架构设计、技术体系、行业知识                     │
│  · llm-wiki 编译：零散调研 → LLM 编译 → 结构化 Markdown      │
│  · Git 驱动索引：staged/committed → TM 建向量索引             │
│  · Obsidian 浏览：人类可读、可编辑                            │
│  · Room 约束：ad_learning / ai_learning / team_doc           │
└─────────────────────────────────────────────────────────────┘
```

## 三、三层如何协作

### 场景 1：用户问"之前 crash 上报问题怎么解决的？"

```
1. TM Daemon before_prompt → L2 语义检索
2. 命中 Experience：[mem:3c609b27] "crash 上报方案"
3. Agent 回复中引用 [mem:3c609b27]
4. 评估系统自动标记 was_used=True
```

### 场景 2：用户问"LiteLLM 和 Clash 之间有什么关系？"

```
1. TM Daemon before_prompt → L2 语义检索
   命中多条 Experience 但没有直接关联
2. L2.5 实体查询：
   entities: "LiteLLM", "Clash", "TUN模式"
   relationships: "LiteLLM" -[需要代理]-> "Clash"
                   "Clash" -[配置方式]-> "TUN模式"
3. 合并：向量检索结果 + 实体关系图 → 更完整的回答
```

### 场景 3：用户说"帮我把最近的调研整理成体系"

```
1. TM 检索最近 N 条 L2 Experience（同一 group_key）
2. llm-wiki 编译：零散经验 → 结构化知识树
3. 输出 Markdown → git add → Obsidian 可看 → TM 建 L3 索引
4. 原始 L2 Experience 标记 promoted=True
```

## 四、L2.5 实体关系层设计（新增）

### 为什么不直接用 GraphRAG

| GraphRAG 特性 | 我们需要？ | 替代方案 |
|---------------|-----------|---------|
| Leiden 社区检测 | ❌ 规模不够 | 不需要 |
| 全局 Map-Reduce 搜索 | ❌ 太贵 | L2 向量检索 + L3 结构化 |
| 层级社区报告 | ❌ 不需要 | Obsidian 手动组织 |
| 实体/关系抽取 | ✅ | 小模型 structured output |
| 实体中心检索 | ✅ | PostgreSQL recursive CTE |
| 多跳遍历 | ✅ 1-2 跳足够 | SQL JOIN |

### 数据模型

```sql
-- 实体表
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'concept',  -- concept/person/project/tool/...
    description TEXT,
    project TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(name, type, project)
);

-- 关系表
CREATE TABLE relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID REFERENCES entities(id),
    target_entity_id UUID REFERENCES entities(id),
    relation_type TEXT NOT NULL,  -- depends_on/configures/relates_to/solves/...
    weight REAL DEFAULT 1.0,
    evidence TEXT,  -- 原始文本片段
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Experience-Entity 关联表
CREATE TABLE experience_entities (
    experience_id UUID,
    entity_id UUID REFERENCES entities(id),
    role TEXT DEFAULT 'mentioned',  -- mentioned/central/solved_by
    PRIMARY KEY(experience_id, entity_id)
);
```

### 抽取流程

```
草稿发布（draft → published）
  ↓
异步调用小模型（qwen2.5-7b / glm-4-flash）structured output
  ↓
提取实体列表 + 关系列表
  ↓
UPSERT 到 entities / relationships 表
  ↓
建立 experience_entities 关联
```

**成本**：~0.0001元/条（用硅基免费模型），延迟 ~1s（异步，不阻塞）

### 查询流程

```python
# 1. 从用户 query 识别实体
entities = extract_entities_from_query(query)  # 规则匹配，不调 LLM

# 2. 查 1-2 跳关系
WITH RECURSIVE graph AS (
    SELECT source_entity_id, target_entity_id, relation_type, 1 as depth
    FROM relationships WHERE source_entity_id = ANY(:entity_ids)
    UNION ALL
    SELECT r.source_entity_id, r.target_entity_id, r.relation_type, g.depth + 1
    FROM relationships r JOIN graph g ON r.source_entity_id = g.target_entity_id
    WHERE g.depth < 2
)
SELECT * FROM graph;

# 3. 合并：向量检索结果 + 实体关系上下文 → 返回 Agent
```

## 五、与 v9 设计文档的对齐

| v9 设计 | 当前实现 | L2.5 增强 |
|---------|---------|-----------|
| 痕迹（L2）→ TM Experience | ✅ 已实现 | + 实体抽取 |
| 知识（L3）→ Markdown + Obsidian | ⚠️ 框架在，未接入 Git watcher | 不变 |
| 意图路由 IntentRouter | ❌ 未实现 | 先用规则，预留接口 |
| [mem:xxx] 评估 | ✅ 已实现 | 不变 |
| Janitor promotion (L2→L3) | ❌ 未实现 | 不变 |
| Obsidian/Git watcher | ❌ 未实现 | 不变 |
| L2.5 实体关系层 | v9 无此层 | **新增** |

### 实施优先级

1. **P0**（已完成）：TM Daemon + Hooks + 收敛检测 + 日志增强
2. **P1**（下一步）：L2.5 实体关系层（entities/relationships 表 + 抽取 + 查询）
3. **P2**：Git watcher → L3 索引
4. **P3**：Janitor promotion（L2→L3 自动提升）
5. **P4**：意图路由 IntentRouter
6. **P5**：LightRAG 对特定知识库（如 ai_learning）做深度索引

## 六、关键决策记录

| 决策 | 理由 |
|------|------|
| 不用微软 GraphRAG | 成本($4/条)、延迟(2-5s)、维护重；单用户规模不匹配 |
| 自建轻量图层在 PostgreSQL | 已有基础设施，recursive CTE 够用，增量更新 trivial |
| 实体抽取用小模型 | 成本近零，延迟可接受，异步不阻塞 |
| 不做社区检测 | 单用户规模（千级节点），社区检测收益 < 复杂度 |
| LightRAG 留作 P5 | 只在知识库规模达到千+文档时才需要 |
| llm-wiki 定位为 L3 编译工具 | 不是独立系统，是 L2→L3 提升的工具链 |
