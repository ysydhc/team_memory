# TASK-11 Phase2: 借鉴 LightRAG 增强实体关系层

> 方案选择: C（借鉴 LightRAG 增强，不引入新依赖）

---

## 1. 目标

在现有 EntityExtractor + EntitySearch + WikiCompiler 基础上：
1. 补全 144 条 experience 的实体抽取
2. 增强 Wiki 交叉引用（embedding 相似度）
3. 新增 High-level 检索（实体聚类 + 主题发现）
4. 切换抽取模型到 DeepSeek-V3 提升质量

---

## 2. 现状分析

| 指标 | 当前值 |
|------|--------|
| PG entities | 9 |
| PG relationships | 8 |
| experience-entity 链接 | 6 |
| Published experiences | 146 |
| 未做实体抽取 | 144 |
| 实体抽取触发条件 | 仅 Docker 侧 enable_background=True |
| 抽取模型 | glm-4-flash（通过 LLMConfig） |
| Embedding | ollama nomic-embed-text 768d |

**根因:** EntityExtractor 只在 Docker 侧的 EventBus 回调中触发，Daemon 侧
enable_background=False 从未触发过。144 条 experience 没走过实体抽取流程。

---

## 3. 实施任务

### T11.1 补全实体抽取：CLI 批量脚本（~1.5h）

**文件:**
- 新增: `scripts/daemon/entity_backfill.py`
- 修改: `src/team_memory/config/llm.py` — 加 entity_extraction 专用配置

**实现:**
```python
# scripts/daemon/entity_backfill.py
# 用法: make entity-backfill [--model DeepSeek-V3]

1. 查询 PG: SELECT id FROM experiences WHERE exp_status='published'
   AND id NOT IN (SELECT experience_id FROM experience_entities)
2. 对每条 experience 调用 EntityExtractor.extract_and_persist()
3. 并发控制: asyncio.Semaphore(3)，避免打爆 LLM API
4. 进度条: 每10条打印进度
5. 错误处理: 单条失败不影响其他，记录 failed_ids
```

**配置:**
- 新增 `entity_extraction` 配置块到 config.yaml
- 独立于 refinement 的模型配置，支持后续升级

**Makefile:**
```makefile
entity-backfill:  ## 全量补抽实体（--model 指定模型）
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/daemon/entity_backfill.py $(OPTS)
```

**验收:** 运行后 PG entities > 100, relationships > 50

---

### T11.2 实体去重/合并（~1h）

**文件:**
- 新增: `scripts/daemon/entity_dedup.py`

**实现:**
1. 查询所有同名实体: `SELECT name, array_agg(id) FROM entities GROUP BY name HAVING count(*) > 1`
2. 合并策略:
   - 保留 source_count 最高的作为主实体
   - 将从属实体的 experience_entities 链接迁移到主实体
   - 将从属实体的 relationships 迁移到主实体
   - 删除从属实体
3. 合并后重新计算 source_count

**验收:** 无同名实体（或同名实体确实属于不同 project）

---

### T11.3 Wiki 交叉引用增强：embedding 相似度（~1.5h）

**文件:**
- 修改: `scripts/daemon/wiki_compiler.py` — `_find_related()` 增加 embedding 路径

**实现:**
```python
def _find_related(self, experience, all_experiences, max_related=5):
    # 1. 现有规则: tag overlap * 2 + same_project
    # 2. 新增: entity-graph 关联 (共享实体 = 相关)
    # 3. 新增: embedding 相似度 (title+description 的 embedding cosine > 0.7)
    
    得分 = tag_overlap * 2 + same_project * 1 + entity_overlap * 3 + embedding_sim * 2
```

**embedding 相似度:**
- 不新增 embedding 调用 — 复用 PG 中已有的 experience embedding
- 新增 `_find_related_by_embedding()`: 从 PG 查询 embedding 向量，计算 cosine similarity
- 缓存: 编译时一次性加载所有 embedding 到内存（146条 < 1MB）

**验收:** Wiki 页面"相关经验"节中出现 embedding 意义上相关的经验

---

### T11.4 High-level 检索：实体聚类 + 主题发现（~1.5h）

**文件:**
- 新增: `src/team_memory/services/topic_discovery.py`
- 修改: `src/team_memory/services/search_pipeline.py` — 集成 topic 检索
- 新增: `scripts/daemon/wiki_compiler.py` — 生成 wiki/topics/ 目录

**实现:**

**C-1: 基于 embedding 的实体聚类**
```python
class TopicDiscovery:
    """基于实体 embedding 的轻量主题发现。"""
    
    async def discover_topics(self, db_url, min_cluster_size=3):
        1. 查询所有实体 + description
        2. 对 description 做 embedding
        3. K-means 聚类（sklearn or 手写简单版）
        4. 每个聚类 = 一个 topic，生成摘要名
        5. 写入 topic 表或缓存
        
    async def find_by_topic(self, query_embedding, db_url):
        1. 计算 query 与各 topic 中心的相似度
        2. 返回最相关 topic 下的 experience IDs
```

**SearchPipeline 集成:**
- 现有搜索结果 + entity enrichment 结果 + topic 检索结果 → 合并去重
- Topic 结果排在 entity 结果之后（作为补充）

**Wiki 集成:**
- 生成 `wiki/topics/` 目录：每个 topic 一个 .md
- 内容: topic 名称 + 包含的实体列表 + 关联 experience 列表

**验收:** 搜索"网络问题"时，topic 检索能返回 Docker/Clash/LiteLLM 网络相关经验

---

### T11.5 模型升级：DeepSeek-V3（~0.5h）

**文件:**
- 修改: `scripts/hooks/config.yaml` — entity_extraction.model
- 修改: `src/team_memory/config/llm.py` — 加 entity_extraction 专用配置

**实现:**
1. config.yaml 新增 entity_extraction 配置块:
   ```yaml
   entity_extraction:
     model: "DeepSeek-V3"
     base_url: "http://localhost:4000/v1"
     api_key_env: "LITELLM_MASTER_KEY"
     timeout: 30
   ```
2. EntityExtractor 支持独立模型配置（与 refinement 分离）
3. 先用 glm-4-flash 跑 5 条做 baseline
4. 再用 DeepSeek-V3 跑同样 5 条做对比
5. 输出对比报告，确认 DeepSeek-V3 质量更好后全量跑

**验收:** DeepSeek-V3 抽取的实体更多、关系更准确

---

### T11.6 测试 + E2E 验证（~1h）

**文件:**
- 新增: `tests/test_entity_backfill.py`
- 新增: `tests/test_topic_discovery.py`

**验证清单:**
1. `make entity-backfill` — 146 条全量抽取成功
2. `make entity-dedup` — 无同名实体
3. `make wiki-compile` — Wiki 交叉引用包含 entity + embedding 链接
4. 搜索对比：同一查询，entity enrichment 前/后结果差异
5. Topic 检索：验证 topic 发现和搜索集成
6. 全量测试: pytest 通过

---

## 4. 实施顺序

```
T11.5 (模型配置) → T11.1 (补全抽取) → T11.2 (去重) → T11.3 (Wiki增强) → T11.4 (High-level) → T11.6 (验证)
```

T11.5 先行是因为抽取质量决定后续所有步骤的效果。

---

## 5. 时间预估

| 任务 | 预估 |
|------|------|
| T11.1 补全抽取 | 1.5h |
| T11.2 实体去重 | 1h |
| T11.3 Wiki 交叉引用增强 | 1.5h |
| T11.4 High-level 检索 | 1.5h |
| T11.5 模型升级 | 0.5h |
| T11.6 测试验证 | 1h |
| **总计** | **~7h** |

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| DeepSeek-V3 抽取速度慢（3s/条） | 并发3+异步，146条约7min |
| 实体去重可能误合并不同实体 | 仅合并同名+同 project 的，不同 project 保留 |
| embedding 相似度计算量大 | 146条 < 1MB，内存无压力 |
| K-means 聚类数难以确定 | 用肘部法则自动选择，或固定 10-20 个 |
| EntityExtractor 只在 Docker 侧触发 | T11.1 是一次性补全，后续新增 experience 需要 Daemon 侧也触发 |
