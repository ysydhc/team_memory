# FTS 存量迁移（migrate_fts）说明

> 运维文档 | 全文索引补全
> 相关：[database-operations 数据库操作](database-operations.md) | 技术概念：[tech-concepts/pgvector-fts](../tech-concepts/pgvector-fts.md)

## 一、如何使用（照着做就会）

### 1. 什么时候需要跑？

- 你发现：**在 Web 或 MCP 里用关键词搜经验，有些明明标题/内容里有的经验搜不到**。
- 或者：**你们从旧库导入了一大批经验，或很久以前就有一堆经验，想用全文检索（按关键词搜）**。

这时就需要给「还没建好全文索引」的旧数据补一遍索引，用的就是本工具。

### 2. 第一步：先「预览」，不写库（推荐必做）

```bash
python scripts/migrate_fts.py --dry-run
```

- **`--dry-run`** 表示「只看看有多少条需要补，不真的改数据库」。
- 先跑带 `--dry-run` 的这一条，确认有需要再跑真正写入的那条。

### 3. 第二步：真正执行迁移（写库）

```bash
# 方式 A：用 Make（推荐）
make migrate-fts
```

或：

```bash
# 方式 B：直接跑脚本
python scripts/migrate_fts.py
```

### 4. 可选参数

| 参数 | 含义 | 示例 |
|------|------|------|
| `--dry-run` | 只统计条数，不写库 | `python scripts/migrate_fts.py --dry-run` |
| `--batch-size 100` | 每批更新 100 条 | 数据量特别大时可调小 |
| `--limit 500` | 最多只更新 500 条 | 想先试一小部分时用 |

## 二、为什么需要？

- **新写入的经验**：写入时自动填 `fts` 字段，能全文搜到。
- **老数据 / 存量数据**：`fts` 可能为空，全文搜索会漏掉。
- **migrate_fts 做的事**：给存量经验按当前规则补算 fts，填上后全文搜索就能搜到。

## 三、相关

- 故障排查：[troubleshooting](troubleshooting.md)
- 主项目 README FAQ 有「存量数据如何支持全文检索（FTS）？」说明
