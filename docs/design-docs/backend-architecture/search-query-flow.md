# 搜索查询流程

> 技术架构 | 检索管道
> 相关： [database-schema](database-schema.md) | [rrf-hybrid-search](../tech-concepts/rrf-hybrid-search.md) | [reranker](../tech-concepts/reranker.md)

## 流程概览

```
用户查询 → 缓存检查 → Embedding → 检索(RRF) → 精确匹配加分 → 自适应过滤
         → PageIndex-Lite → Reranker → 置信度标注 → 上下文压缩 → 返回
```

## 阶段说明

| 阶段 | 说明 | 关键逻辑 |
|------|------|----------|
| 1. 缓存检查 | 命中则直接返回 | 按 query+tags+project+current_user 建 key |
| 2. Embedding | 查询向量化 | 失败时降级为 FTS-only |
| 3. 检索 + RRF | Vector 与 FTS 并行，RRF 融合 | hybrid 下双 Session（FTS 独立 session 避免并发冲突） |
| 4. 精确匹配加分 | query == title 或 query in title | 检查 root + children；exact +0.4，contains +0.15 |
| 5. 自适应过滤 | 动态阈值 + elbow 截断 | 精确匹配项不被截断 |
| 5.5 PageIndex-Lite | 长文档节点匹配 | 可选，树节点得分加权 |
| 6. Reranker | 服务端重排 | 支持 exact_title_match 元数据加分 |
| 7. 置信度标注 | high/medium/low | 按与 top-1 的得分比 |
| 8. 上下文压缩 | 控制 token 预算 | trim 后截断至 max_results |

## 检索模式

- **hybrid**：Vector + FTS 并行，RRF 融合（默认）
- **vector**：仅向量检索
- **fts**：仅全文检索（Embedding 失败时自动降级）

## FTS 方案 C（2025-03）

- 列：`fts_title_text`、`fts_desc_text`、`fts_solution_text`（jieba 分词）
- 触发：setweight(A=title, B=desc, C=solution)
- 排序：`ts_rank_cd` 考虑权重
- 存量：`make migrate-fts-jieba` 回填

## 关键配置

| 配置项 | 说明 |
|--------|------|
| search.mode | hybrid / vector / fts |
| search.adaptive_filter | 是否启用自适应过滤 |
| search.rrf_k | RRF 常数（默认 60） |
| search.vector_weight / fts_weight | RRF 源权重 |
| search.min_confidence_ratio | 动态阈值比例 |
| search.score_gap_threshold | elbow 截断 gap 阈值 |
