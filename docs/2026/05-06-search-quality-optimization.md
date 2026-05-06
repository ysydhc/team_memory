# 搜索质量优化 — 阶段性总结

**日期：** 2026-05-06
**优化范围：** Embedding 模型 + FTS 查询策略

---

## 1. 优化内容

| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| Embedding 模型 | nomic-embed-text (768d, Ollama) | qwen3-embedding:0.6b (1024d, Ollama) |
| FTS 查询策略 | plainto_tsquery (全 AND) | hybrid AND/OR (前 2 词 AND, 其余 OR) |
| 向量维度 | 768 | 1024 |

**相关提交：**
- `b7d5d45` — embedding 模型切换 + migration 017
- `deb891d` — YAML 配置更新 + backfill UUID bug fix
- `571c344` — FTS hybrid AND/OR 策略

---

## 2. 模拟验证（search_eval 10 条测试查询）

### 2.1 总体指标

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| Hit rate | 70% | 90% | **+20%** |
| Avg precision | 25.8% | 55.8% | **+30%** |
| Avg recall | 60.0% | 70.0% | **+10%** |
| Miss queries | 4 | 3 | -1 |

### 2.2 逐查询对比

| 查询 | 优化前 Hits | 优化前 Prec/Recall | 优化后 Hits | 优化后 Prec/Recall | 变化 |
|------|------------|-------------------|------------|-------------------|------|
| Docker PostgreSQL connection | 0 (miss) | - | 1 | 100% / 100% | ✅ 新增命中 |
| WKWebView cookie sync | 0 (miss) | - | 0 (miss) | - | 无变化 |
| embedding model configuration | 0 (miss) | - | 0 (miss) | - | 无变化 |
| entity extraction | 1 | 33% / 100% | 1 | 33% / 100% | 无变化 |
| task group completion | 10 | 100% / 100% | 10 | 100% / 100% | 无变化 |
| asyncpg pgvector | 3 | 100% / 100% | 3 | 100% / 100% | 无变化 |
| Makefile target | 1 | 100% / 100% | 1 | 100% / 100% | 无变化 |
| refinement worker | 3 | 75% / 100% | 3 | 75% / 100% | 无变化 |
| wiki compilation | 0 (miss) | - | 0 (miss) | - | 无变化 |
| search pipeline RRF | 2 | 50% / 100% | 2 | 50% / 100% | 无变化 |

### 2.3 Miss 查询分析

| 查询 | 原因 | 解决方向 |
|------|------|---------|
| WKWebView cookie sync | FTS `WKWebView & cookie` 无同时匹配 | 知识库缺少相关内容 |
| embedding model configuration | FTS `embedding & model` 组合未命中 | 知识库缺少相关内容 |
| wiki compilation | 知识库无 "wiki" 相关经验 | 需补充 wiki 使用经验 |

---

## 3. 实际使用验证（待补充）

### 3.1 优化前基线数据（2026-05-05）

| 指标 | 数值 | 数据来源 |
|------|------|---------|
| Hit rate (7d) | 72% | make stats |
| Use rate (7d) | 0% | make stats（marker 机制失效） |
| Total searches | 61 | make stats |
| Miss queries (top) | "Docker PostgreSQL connection", "WKWebView cookie sync", "embedding model configuration" | make stats |
| Faithfulness | N/A | 系统未上线 |

### 3.2 优化后数据（待补充）

**计划：2026-05-09（3 天后）补充**

| 指标 | 数值 | 数据来源 |
|------|------|---------|
| Hit rate (7d) | ? | make stats |
| Use rate (7d) | ? | make stats |
| Total searches | ? | make stats |
| Miss queries (top) | ? | make stats |
| Faithfulness avg | ? | make stats |

### 3.3 对比结论（待补充）

待实际数据验证后填写。

---

## 4. 后续优化方向

| 优先级 | 方向 | 预期收益 |
|--------|------|---------|
| P1 | 补充知识库缺失内容（wiki、WKWebView） | 消除 miss queries |
| P2 | Faithfulness 评估结果指导记忆质量改进 | 提升 search→use 转化率 |
| P3 | FTS AND 门槛从 2 词降到 1 词 | 进一步提升 recall |
| P4 | 换更强 embedding（qwen3-embedding:4b 或 bge-m3） | 进一步提升 precision |
