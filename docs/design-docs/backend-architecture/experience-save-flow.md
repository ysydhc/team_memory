# 经验存入流程

> 技术架构 | 写入管道
> 相关： [database-schema](database-schema.md) | [search-query-flow](search-query-flow.md)

## 流程概览

```
MCP/API 入口 → 参数校验 → Embedding 生成 → 去重检查 → FTS 分词 → DB 写入
             → PageIndex-Lite(可选) → 事件总线 → 返回
```

## 示例图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          经验存入流程 — 主路径                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  tm_save     │  快速保存：title + problem + solution
  │  tm_save_typed│ 完整保存：+ experience_type, severity, git_refs...
  │  tm_learn    │  LLM 解析长文本 → 结构化字段 → save
  │  tm_save_group│ 父+子经验组 → save_group
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 1. 参数校验                                                                   │
  │    • JSONB 校验：structured_data, git_refs, related_links                     │
  │    • SchemaRegistry：progress_status 自动推断                                 │
  │    • tm_learn：LLM parse_content → 质量门控(quality_min) → 重试(retry_once)  │
  └──────┬───────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 2. Embedding 文本构建                                                         │
  │    embed_text = title + problem + solution + root_cause + code_snippets + tags│
  └──────┬───────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 3. Embedding 生成                                                             │
  │    sync_embedding=True  → 同步 encode_single() → embedding 就绪               │
  │    sync_embedding=False → embedding_status='pending' → 后台队列异步补全        │
  └──────┬───────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 4. 去重检查 (dedup_on_save)                                                    │
  │    check_similar(embedding, threshold) → 若命中 → 返回 duplicate_detected     │
  │    skip_dedup=True 可跳过                                                     │
  └──────┬───────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 5. AI 创建审核覆盖 (require_review_for_ai)                                     │
  │    source=auto_extract/mcp 且 publish_status≠personal → draft + pending_review │
  └──────┬───────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 6. FTS 分词 (jieba)                                                           │
  │    fts_title_text, fts_desc_text, fts_solution_text → 供加权全文检索           │
  └──────┬───────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 7. DB 写入 (repo.create / create_group)                                        │
  │    Experience 插入 → flush → 返回 experience                                  │
  └──────┬───────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 8. PageIndex-Lite (可选)                                                      │
  │    _maybe_build_tree_nodes：长文档分块 → DocumentTreeNode                      │
  └──────┬───────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ 9. 事件总线 + 返回                                                            │
  │    Events.EXPERIENCE_CREATED → 返回 experience dict                           │
  │    tm_learn 额外：_try_extract_and_save_personal_memory                       │
  └──────────────────────────────────────────────────────────────────────────────┘
```

## 阶段说明

| 阶段 | 说明 | 关键逻辑 |
|------|------|----------|
| 1. 参数校验 | JSONB 与 Schema 校验 | validate_structured_data / git_refs / related_links；tm_learn 用 LLM parse_content |
| 2. Embedding 文本 | 拼接待向量化文本 | title + problem + solution + root_cause + code_snippets + tags |
| 3. Embedding 生成 | 同步或异步 | sync：encode_single；async：embedding_status=pending + 队列 enqueue |
| 4. 去重检查 | 相似经验检测 | check_similar(embedding, threshold)；命中则返回 duplicate_detected |
| 5. AI 审核覆盖 | 自动设为草稿 | source=auto_extract/mcp 时 publish_status→draft, review_status→pending |
| 6. FTS 分词 | jieba 分词 | fts_title_text / fts_desc_text / fts_solution_text |
| 7. DB 写入 | 插入 experiences | repo.create 或 create_group（父子组） |
| 8. PageIndex-Lite | 长文档树节点 | _maybe_build_tree_nodes 建 DocumentTreeNode |
| 9. 事件与返回 | 通知与响应 | EXPERIENCE_CREATED 事件；tm_learn 额外提取 personal_memory |

## 写入入口

| 入口 | 来源 | 特点 |
|------|------|------|
| tm_save | MCP | 快速保存，source=auto_extract，默认 publish_status=personal |
| tm_save_typed | MCP | 完整类型字段，experience_type/severity/git_refs 等 |
| tm_learn | MCP | 长文本解析，LLM 提取 → save；as_group 时走 save_group |
| tm_save_group | MCP | 父+子经验组，create_group；支持 grouped_children 按类型分组 |
| Web API | parse-document / parse-url | 同 tm_learn 解析逻辑，写入后返回 |

## tm_learn 分支示意

```
                    tm_learn(conversation, as_group?)
                              │
              ┌───────────────┴───────────────┐
              │                               │
         as_group=False                  as_group=True
              │                               │
              ▼                               ▼
    parse_content(single)            parse_content(group)
              │                               │
              ▼                               ▼
    service.save(...)               service.save_group(...)
              │                               │
              │                    parent + children embeddings
              │                    create_group → PageIndex 父+子
              │                    grouped_children by experience_type
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
              _try_extract_and_save_personal_memory(conversation)
```

## 关键配置

| 配置项 | 说明 |
|--------|------|
| lifecycle.dedup_on_save | 是否启用存入时去重 |
| lifecycle.dedup_on_save_threshold | 去重相似度阈值 |
| review.require_review_for_ai | AI 创建经验是否强制草稿 |
| extraction.quality_gate | tm_learn 质量门控最低分 |
| extraction.max_retries | tm_learn 解析失败重试次数 |
