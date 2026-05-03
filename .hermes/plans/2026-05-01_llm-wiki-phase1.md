# TASK-10: Phase1 — LLM Wiki 编译层 完整实施计划

> 决策确认: D1=A(wiki/放项目内), D2=B(异步EventBus), D3=SQLite, D4=矛盾检测放Stage2

---

## 1. 目标

在 TM 现有 publish 流程后增加 Wiki 编译步骤，将 PG experiences 自动编译为结构化 markdown 文件，
形成可浏览、可交叉引用的知识 wiki，供 Obsidian 浏览和 Agent 上下文注入。

**核心公式:**
```
experience (PG) → WikiCompiler → wiki/ 目录 (markdown 文件)
```

---

## 2. 目录结构 (新增)

```
team_doc/
  wiki/                              # Wiki 编译产物 (新增, 加入 .gitignore)
    concepts/                        # 每个 experience 对应一个 .md
      跨-claude-cursor-的-agent-ssot.md
      docker-entrypoint-pg-isready.md
      ...
    queries/                         # Stage 4: 搜索回写页面 (暂空)
    index.md                         # 自动生成的目录
    log.md                           # 变更日志 (append-only)
  .wiki/                             # Wiki 元数据 (新增, 加入 .gitignore)
    cache.db                         # SQLite 编译缓存
    schema.md                        # 编译规则 (人可编辑)
```

---

## 3. 数据模型

### 3.1 SQLite cache.db

```sql
CREATE TABLE IF NOT EXISTS wiki_cache (
    experience_id TEXT PRIMARY KEY,        -- UUID of PG experience
    content_hash TEXT NOT NULL,            -- SHA-256 of title+description+solution
    wiki_path TEXT NOT NULL,               -- e.g. "concepts/跨-claude-ssot.md"
    status TEXT NOT NULL DEFAULT 'compiled',  -- compiled / stale / orphan
    project TEXT NOT NULL DEFAULT '',      -- 冗余，方便按项目过滤
    compiled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_wiki_cache_status ON wiki_cache(status);
CREATE INDEX idx_wiki_cache_project ON wiki_cache(project);
```

### 3.2 Wiki 页面模板

```markdown
---
title: 跨 Claude/Cursor 的 agent、prompt 与规则 SSOT
tags: [claude-code, cursor, agents, ssot, team_doc, mcp]
source_ids: [bac7fc35-d64e-4d70-8d7f-31214513ab08]
project: team_memory
experience_type: tech_design
created_at: 2026-04-10
updated_at: 2026-04-30
confidence: high
recall_count: 5
---

## 摘要

{自动截取 description 前 200 字，或由 LLM 生成 (Stage 2)}

## 问题描述

{description 全文}

## 解决方案

{solution 全文，无 solution 则省略此节}

## 相关经验

- [[Docker-entrypoint-pg-isready-替代方案]]  ← 同 tag 或同 project
- [[反哺排序-popularity-boost-乘法与 log 压缩]]

## 来源

- experience_id: bac7fc35-d64e-4d70-8d7f-31214513ab08
```

---

## 4. WikiCompiler 类设计

文件: `scripts/daemon/wiki_compiler.py`

```python
class WikiCompiler:
    """将 PG experiences 编译为 markdown wiki 文件。
    
    同模式参考: DraftBuffer (aiosqlite)
    """
    
    def __init__(self, wiki_root: str, db_path: str | None = None):
        self._wiki_root = wiki_root   # wiki/ 目录绝对路径
        self._db_path = db_path or os.path.join(os.path.dirname(wiki_root), ".wiki", "cache.db")
        self._db: aiosqlite.Connection | None = None
    
    # ---- 生命周期 (同 DraftBuffer) ----
    async def __aenter__(self) -> WikiCompiler: ...
    async def __aexit__(self, *exc) -> None: ...
    
    # ---- 核心编译 API ----
    async def compile_one(self, experience: dict) -> str:
        """编译单条 experience → 返回 wiki_path"""
        
    async def compile_batch(self, experiences: list[dict]) -> list[str]:
        """批量编译，返回所有 wiki_path"""
        
    async def compile_incremental(self, all_experiences: list[dict]) -> CompileResult:
        """增量编译: 只处理新增/变更的 experience"""
        # 1. 计算 content_hash
        # 2. 与 cache.db 对比
        # 3. 只编译 hash 不同的
        # 4. 删除 PG 已无对应 experience 的 wiki 页面
        # 5. 更新 index.md + log.md
    
    async def full_rebuild(self, all_experiences: list[dict]) -> CompileResult:
        """全量重编译: 清空 cache.db，重写所有 wiki 页面"""
    
    # ---- 查询 API ----
    async def get_stale(self) -> list[str]:
        """返回 status='stale' 的 experience_id 列表"""
        
    async def get_uncompiled(self, experience_ids: list[str]) -> list[str]:
        """返回 cache.db 中不存在的 experience_id"""
        
    async def get_by_project(self, project: str) -> list[dict]:
        """返回某项目下所有编译记录"""
    
    # ---- 内部方法 ----
    def _compute_hash(self, experience: dict) -> str:
        """SHA-256 of title + description + solution"""
        
    def _slugify(self, title: str) -> str:
        """标题 → 文件名 slug (中文保留, 特殊字符替换)"""
        # "跨 Claude/Cursor 的 agent SSOT" → "跨-claude-cursor-的-agent-ssot"
        
    def _render_page(self, experience: dict, related: list[dict]) -> str:
        """将 experience dict + related 列表渲染为 markdown"""
        
    def _find_related(self, experience: dict, all_experiences: list[dict]) -> list[dict]:
        """基于 tags 共享 + 同 project 找相关 experience (纯规则，无 LLM)"""
        # 规则: tags 交集 ≥ 2 OR 同 project AND tags 交集 ≥ 1
        # 最多返回 5 条相关
        
    async def _update_index(self) -> None:
        """重写 index.md"""
        
    async def _append_log(self, action: str, title: str, wiki_path: str) -> None:
        """追加 log.md 一行"""
```

### CompileResult

```python
@dataclass
class CompileResult:
    created: int = 0      # 新建的 wiki 页面
    updated: int = 0      # 更新的 wiki 页面
    skipped: int = 0      # hash 未变，跳过
    deleted: int = 0      # PG 已无对应，删除的 wiki 页面
    errors: int = 0       # 编译失败
```

---

## 5. slugify 规则

title → 文件名的转换规则:

| 输入 | 输出 | 规则 |
|------|------|------|
| 跨 Claude/Cursor 的 agent SSOT | 跨-claude-cursor-的-agent-ssot | `/` → `-`，空格 → `-`，连续 `-` 合并 |
| Docker entrypoint pg_isready 替代 | docker-entrypoint-pg_isready-替代 | 保留 `_`，中文保留 |
| 反哺排序：乘法 boost + log 压缩 | 反哺排序-乘法-boost-加-log-压缩 | `+` → `加`，`：` → `-` |
| 2026-03-30 个人记忆优化审查 | 2026-03-30-个人记忆优化审查 | 保留日期格式 |
| TM Web 使用统计与 Skills 整目录禁用/恢复 | tm-web-使用统计与-skills-整目录禁用-恢复 | 大写 → 小写，`/` → `-` |

实现: `re.sub(r'[/\\：:]+', '-', title)` → `re.sub(r'\s+', '-', ...)` → `re.sub(r'-+', '-', ...)` → `.lower()`

---

## 6. 交叉引用规则 (纯规则，无 LLM)

`_find_related()` 逻辑:

```
输入: experience A, 所有 experiences
输出: 最多 5 条相关 experience

步骤:
1. 遍历所有 published experience (排除自身)
2. 计算 tag_overlap = len(set(A.tags) & set(B.tags))
3. 同 project 加分: same_project = (A.project == B.project)
4. 得分 = tag_overlap * 2 + (1 if same_project else 0)
5. 得分 > 0 的按得分降序排列，取前 5
6. 无相关则返回空列表
```

交叉引用在 wiki 页面中表现为:
```markdown
## 相关经验

- [[Docker-entrypoint-pg-isready-替代方案]]
- [[反哺排序-popularity-boost]]
```

---

## 7. index.md 格式

```markdown
# TM Wiki Index

> 自动生成，勿手动编辑。最后更新: 2026-05-01 15:30

## team_memory (142)

### claude-code
- [[跨-claude-cursor-的-agent-ssot]] — .claude/agents 与 .cursor/agents 双轨维护...
- [[...]] — ...

### cursor
- [[...]] — ...

### docker
- [[Docker-entrypoint-pg-isready-替代方案]] — Python socket 替代 pg_isready...

## ad_learning (28)

### ...
```

按 project → tag 分组，每条: `[[slug]] — description 前 50 字`。

---

## 8. log.md 格式

```markdown
# TM Wiki Changelog

## [2026-05-01 15:30] compile | 跨 Claude/Cursor 的 agent SSOT | created
## [2026-05-01 15:30] compile | Docker entrypoint pg_isready | updated
## [2026-05-01 15:31] compile | [全量] created=142 updated=0 skipped=0
```

append-only，每条一行，`|` 分隔。

---

## 9. EventBus 集成

### 9.1 订阅事件

在 Daemon lifespan 中:

```python
# scripts/daemon/app.py — lifespan 启动部分

wiki_compiler = WikiCompiler(wiki_root=str(WIKI_ROOT))
await wiki_compiler.__aenter__()
application.state.wiki_compiler = wiki_compiler

# 订阅 experience.published 事件
from team_memory.services.event_bus import EventBus, Events

async def on_experience_published(payload: dict):
    """experience 发布后异步编译 wiki 页面"""
    exp_id = payload.get("id")
    if not exp_id:
        return
    # 通过 sink 获取 experience 详情
    experience = await sink.recall(query="", ...)  # 或直接从 PG 读
    if experience:
        await wiki_compiler.compile_one(experience)

bus = EventBus()
bus.on(Events.EXPERIENCE_PUBLISHED, on_experience_published)
application.state.event_bus = bus
```

### 9.2 发射事件

需要在 `op_draft_publish` 成功后发射事件:

```python
# src/team_memory/services/memory_operations.py — op_draft_publish 末尾

# 现有代码已返回 result，需在此处加:
from team_memory.services.event_bus import Events
# 获取全局 bus 实例，emit experience.published
```

**注意**: 当前 EventBus 是实例级（非全局单例）。需要确认:
- 方案 A: 在 Daemon 层用独立 EventBus 实例，不修改 core 层
- 方案 B: 将 EventBus 提升为全局单例，core 层也能 emit

**倾向方案 A** — Daemon 自己监听就够了，不需要 core 层感知 wiki。
触发方式: 在 `DraftRefiner.refine_and_publish()` 成功后直接调 `wiki_compiler.compile_one()`，
不需要走 EventBus。EventBus 留给 Stage 2+ 用。

### 9.3 Stage 1 简化方案

直接在 `refine_and_publish()` 后调 `wiki_compiler.compile_one()`:

```python
# scripts/daemon/draft_refiner.py — refine_and_publish 末尾

# publish 成功后，触发 wiki 编译
if hasattr(self, '_wiki_compiler') and self._wiki_compiler:
    try:
        await self._wiki_compiler.compile_one(published_experience_dict)
    except Exception:
        logger.warning("Wiki compilation failed", exc_info=True)
```

---

## 10. CLI 命令

通过 `tm_hook.py` 扩展:

```bash
tm-hook wiki-compile                    # 增量编译 (只处理新增/变更)
tm-hook wiki-compile --full             # 全量重编译
tm-hook wiki-compile --project team_memory  # 按项目编译
tm-hook wiki-lint                       # 健康检查 (Stage 3)
tm-hook wiki-status                     # 显示编译统计
```

实现: 在 `scripts/hooks/tm_hook.py` 中增加 `wiki-compile` 子命令，
通过 HTTP 调 Daemon API 或直接调 WikiCompiler (本地模式)。

---

## 11. .gitignore 更新

```
# Wiki 编译产物 (可从 PG 全量重建)
wiki/
.wiki/
```

---

## 12. 实施任务分解

### Stage 1: 核心引擎 (本次实施)

| # | 任务 | 文件 | 依赖 | 预估 |
|---|------|------|------|------|
| T10.1 | WikiCompiler 类骨架 + SQLite cache | `scripts/daemon/wiki_compiler.py` | 无 | 1h |
| T10.2 | slugify + 页面模板渲染 | `scripts/daemon/wiki_compiler.py` | T10.1 | 0.5h |
| T10.3 | 交叉引用 (_find_related) | `scripts/daemon/wiki_compiler.py` | T10.1 | 0.5h |
| T10.4 | compile_one / compile_batch | `scripts/daemon/wiki_compiler.py` | T10.2, T10.3 | 0.5h |
| T10.5 | compile_incremental + full_rebuild | `scripts/daemon/wiki_compiler.py` | T10.4 | 1h |
| T10.6 | index.md 生成 | `scripts/daemon/wiki_compiler.py` | T10.4 | 0.5h |
| T10.7 | log.md 追加 | `scripts/daemon/wiki_compiler.py` | T10.4 | 0.5h |
| T10.8 | 集成到 Daemon lifespan + DraftRefiner | `scripts/daemon/app.py`, `scripts/daemon/draft_refiner.py` | T10.5 | 0.5h |
| T10.9 | CLI 命令: wiki-compile / wiki-status | `scripts/hooks/tm_hook.py` | T10.5 | 0.5h |
| T10.10 | 测试: test_wiki_compiler.py | `tests/test_wiki_compiler.py` | T10.5 | 1h |
| T10.11 | .gitignore + 端到端验证 | `.gitignore` | T10.8 | 0.5h |

**Stage 1 总计: ~6.5h**

### Stage 2: 交叉引用增强 + 矛盾检测 (后续)

| # | 任务 | 说明 |
|---|------|------|
| T10.12 | 基于相似度的交叉引用 | 用 embedding 相似度补充 tag-based 链接 |
| T10.13 | 矛盾检测 | LLM 对比新旧 wiki 页面，标记矛盾 |
| T10.14 | SQLite → PG 迁移 (可选) | 如果 Stage 2 需要 Docker 侧查询 |

### Stage 3: Lint 与维护 (后续)

| # | 任务 | 说明 |
|---|------|------|
| T10.15 | WikiLinter | 孤儿/断链/过时检测 |
| T10.16 | wiki-lint CLI | 健康检查命令 |

### Stage 4: Query→Save 回写 (后续)

| # | 任务 | 说明 |
|---|------|------|
| T10.17 | 搜索结果回写 | 综合多条经验时自动生成 comparison page |

---

## 13. 关键实现细节

### 13.1 content_hash 计算

```python
import hashlib

def _compute_hash(self, experience: dict) -> str:
    """SHA-256 of title + description + solution."""
    parts = [
        experience.get("title", ""),
        experience.get("description", ""),
        experience.get("solution", ""),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]  # 16 chars 够了
```

### 13.2 incremental compile 核心逻辑

```python
async def compile_incremental(self, all_experiences: list[dict]) -> CompileResult:
    result = CompileResult()
    
    # 1. 计算 hash 并与缓存对比
    current_hashes = {}
    for exp in all_experiences:
        exp_id = exp["id"]
        current_hashes[exp_id] = self._compute_hash(exp)
    
    # 2. 查缓存中所有记录
    cached = await self._get_all_cached()  # {exp_id: {hash, wiki_path, ...}}
    
    # 3. 找需要编译的 (新增 + 变更)
    for exp in all_experiences:
        exp_id = exp["id"]
        new_hash = current_hashes[exp_id]
        
        if exp_id not in cached:
            # 新增
            await self.compile_one(exp)
            result.created += 1
        elif cached[exp_id]["content_hash"] != new_hash:
            # 变更
            await self.compile_one(exp)
            result.updated += 1
        else:
            result.skipped += 1
    
    # 4. 找需要删除的 (PG 已无对应)
    current_ids = set(current_hashes.keys())
    cached_ids = set(cached.keys())
    for exp_id in cached_ids - current_ids:
        wiki_path = cached[exp_id]["wiki_path"]
        os.remove(os.path.join(self._wiki_root, wiki_path))
        await self._delete_cache(exp_id)
        result.deleted += 1
    
    # 5. 更新 index.md + log.md
    if result.created + result.updated + result.deleted > 0:
        await self._update_index()
        await self._append_log("incremental", f"created={result.created} updated={result.updated} deleted={result.deleted}", "")
    
    return result
```

### 13.3 compile_one 流程

```python
async def compile_one(self, experience: dict) -> str:
    # 1. 计算 slug → wiki_path
    slug = self._slugify(experience["title"])
    wiki_path = f"concepts/{slug}.md"
    
    # 2. 找相关 experience
    related = self._find_related(experience, self._all_experiences)
    
    # 3. 渲染 markdown
    content = self._render_page(experience, related)
    
    # 4. 写文件
    full_path = os.path.join(self._wiki_root, wiki_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    async with aiofiles.open(full_path, "w") as f:
        await f.write(content)
    
    # 5. 更新缓存
    await self._upsert_cache(
        experience_id=experience["id"],
        content_hash=self._compute_hash(experience),
        wiki_path=wiki_path,
        project=experience.get("project", ""),
    )
    
    return wiki_path
```

---

## 14. 测试计划

`tests/test_wiki_compiler.py`

| 测试 | 验证点 |
|------|--------|
| test_slugify | 中文保留、特殊字符替换、去重 |
| test_compute_hash | 同内容同 hash、改一字不同 hash |
| test_render_page | frontmatter 完整、各节存在、无 solution 时省略 |
| test_find_related | tag 交集排序、同 project 加分、最多 5 条、无相关返空 |
| test_compile_one | 文件创建、缓存写入、内容正确 |
| test_compile_incremental | 新增编译、变更重编译、未变更跳过、删除清理 |
| test_full_rebuild | 清空后全量重建 |
| test_index_md | 按 project→tag 分组、格式正确 |
| test_log_md | append-only、格式正确 |
| test_cache_db | 初始化、upsert、query、delete |

---

## 15. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 183 条经验中很多 tags 为空 | 对 tags 为空的 experience，降级为仅按 project 关联；无 project 则不生成交叉引用 |
| solution 字段大多为空 | 模板中 solution 节可省略，description 为主内容 |
| slug 冲突 (两条经验 title 相同) | 冲突时加 experience_id 前 8 位后缀: `xxx-abc12345.md` |
| 编译中断 (Daemon 重启) | SQLite 事务保证：compile_one 内写文件+更新缓存在同一事务中 |
| wiki/ 目录不在 Obsidian vault 中 | 用户可在 Obsidian 中 "打开另一个仓库" 或 symlink |
