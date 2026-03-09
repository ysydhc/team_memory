# pgvector 与全文搜索 (FTS)

> 技术概念 | 不涉及项目实现
> 相关：[embedding-vector](embedding-vector.md) | [rrf-hybrid-search](rrf-hybrid-search.md)

## pgvector 扩展

**pgvector** 是 PostgreSQL 的一个扩展，让数据库能**原生存储和搜索向量**。

没有 pgvector 的话，你需要：
- 一个独立的向量数据库（如 Milvus、Pinecone）来存向量
- PostgreSQL 存其他数据
- 两个系统之间还要同步数据

有了 pgvector，**一个数据库搞定所有事**。

### 向量索引：IVFFlat 与 HNSW

| 索引类型 | 精确度 | 构建速度 | 内存占用 | 适用规模 |
|----------|--------|----------|----------|----------|
| IVFFlat | 中等 | 快 | 低 | 小~中型（<100 万） |
| HNSW | 高 | 慢 | 高 | 中~大型 |

**IVFFlat 原理**：用 K-means 把向量分成若干簇，查询时只搜索最接近的簇，近似最近邻。

---

## 全文搜索 (FTS)

**FTS（Full-Text Search）** 是 PostgreSQL 内置的全文检索，基于 `tsvector` 和 `tsquery`。

**LIKE 的问题**：
```sql
-- 只能匹配连续出现、无法处理词形变化、大小写
WHERE description LIKE '%docker restart%'
```

**FTS 能解决**：
- **分词**：把文本拆成独立的词
- **词干提取**：`restarting` → `restart`、`containers` → `contain`
- **忽略大小写**：`Docker` = `docker`
- **忽略停用词**：自动跳过 "the"、"a"、"is" 等
- **排名**：`ts_rank()` 按相关性排序

### tsvector 与 tsquery

- **tsvector**：存储文本分词后的结果
- **tsquery**：查询的分词表示，支持 `&`、`|`、`!` 等运算符
- **@@**：匹配运算符，检查 tsvector 是否匹配 tsquery

## 向量搜索 vs 全文搜索

| 特性 | 向量搜索 | 全文搜索（FTS） |
|------|----------|----------------|
| 原理 | 语义理解 | 关键词匹配 |
| 优势 | 能理解同义词、不同表述 | 精确匹配特定术语 |
| 劣势 | 可能返回含义相近但不相关的结果 | 不理解同义词（如"OOM"和"内存泄漏"） |
