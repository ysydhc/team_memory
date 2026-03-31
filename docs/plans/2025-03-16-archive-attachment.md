# 档案馆（Archive）能力实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现档案馆能力：新增 archives（含 **overview**）/ archive_experience_links / archive_attachments 三张表，提供 tm_archive_save（写入时生成或传入 overview）、tm_search/tm_solve 的 include_archives 检索扩展（**双阶段**：先返 L0/L1 列表，再按需 **tm_get_archive(id)** 拉 L2），使历史方案可随经验一起被检索；字段选择与返回以**不丢失命中率为目标**，与 OpenViking L0/L1/L2 对齐。

**Architecture:** 档案馆为独立实体，多对多关联 Experience；默认即写即用（对创建者立即可见），团队可见由「关联经验全部已发布」推导并回写 archive status。写入走 ArchiveService + ArchiveRepository（含 overview）；检索在 SearchPipeline 中 include_archives=True 时并联查询 archives，返回 **L0/L1 混合结果**（结构见设计文档 5.3），RRF/重排后截断；详情由 **tm_get_archive(id)** 返回 L2 全文。

**Tech Stack:** SQLAlchemy async ORM、Alembic 迁移（migrations/）、FastMCP、现有 SearchPipeline/ExperienceRepository 模式。

**设计文档:** [docs/design-docs/archive-attachment-to-experience.md](../design-docs/archive-attachment-to-experience.md)

---

## 一、重要决策与技术选项

### 1.1 档案馆是否单独建向量索引

| 选项 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **A. archives 表加 embedding 列** | 与 experiences 一样，对 solution_doc/title 做向量写入与检索 | 检索路径统一，可复用现有 EmbeddingProvider、RRF 融合 | 需在 tm_archive_save 时调用 embedding；冷启动无归档时无额外成本 |
| B. 仅 FTS 检索 archives | 只对 solution_doc 建 FTS，无向量 | 实现简单，无 embedding 依赖 | 语义检索弱，与「经验用向量」体验不一致 |

**决策：选 A**。设计文档要求「同一语义检索中同时查 experiences 与 archives」，向量能保证 archives 与 experiences 在同一语义空间排序、RRF 融合合理。**Embedding 来源（以不丢命中率为目标）**：对 archives 使用 **title + overview（或 solution_doc 前段当 overview 为空）+ conversation_summary** 拼接后生成 embedding，与 Experience 的构造方式在同一语义空间；L0/L1 返回中的 solution_preview 须覆盖或来自上述参与 embedding 的文本。

### 1.2 发布状态：存储字段 vs 动态推导

| 选项 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **A. 冗余 status 字段 + 经验发布时更新** | archives.status = draft \| published；当某条 archive 的关联经验全部 published 时更新为 published | 检索时简单 WHERE status='published'，无需 JOIN 推导 | 需在经验发布/降级时维护 archive status（或定时 job） |
| B. 每次检索时动态推导 | 不存 status，检索时 JOIN archive_experience_links + experiences 判断 | 无冗余、永远一致 | 检索 SQL 复杂，且需过滤「无关联经验」的 archive |

**决策：选 A**。检索路径已较重，用冗余 status 简化查询。**即写即用**：对创建者，本人创建的 archive 始终可被本人检索（不依赖 status）；对团队，仅 status='published' 的 archive 参与团队检索。**经验状态变更时回写**：当某条 experience 从 published 改为 draft 时，须将该 experience 关联的所有 archive 的 status 更新为 draft（避免团队仍搜到「部分关联未发布」的 archive），**纳入 MVP**，在「经验发布/取消发布」逻辑中或 ArchiveService 订阅中实现。

### 1.3 迁移顺序与回滚

| 选项 | 说明 |
|------|------|
| **单次迁移** | 一张迁移文件中创建 archives、archive_experience_links、archive_attachments 三张表及索引；down_revision = 6ab06751f40e（当前 head：experience_file_locations）。 |
| 回滚 | downgrade 按依赖逆序 drop 三张表；**数据丢失**，与设计文档「Downgrade of this table drops all data」一致。**生产**：升级前必须备份（或至少备份三张新表）；回滚仅限未投产或可接受丢失的场景，回滚预案中写明「回滚即丢数据」。 |

### 1.4 检索权限与 current_user（创建者 vs 团队）

- **创建者**：本人创建的 archive（含 draft）始终可被本人检索；条件为 `created_by = current_user`（与现有 experience 检索一致）。  
- **团队**：仅 `status='published'` 且 project 匹配的 archive 参与团队检索。  
- **current_user 来源**：复用现有 tm_search/tm_solve 的 current_user 参数或 API Key 解析；**未传 current_user 时仅返回 published**（避免匿名误拿团队数据），创建者需带身份才能看到自己的 draft。  
- 在 SearchPipeline 的 include_archives 分支与 ArchiveRepository.search_archives 中按上述约定组合条件；Plan 中相关 Task 与验收均以此为准。

---

## 二、数据库设计（与迁移）

### 2.1 表结构（与设计文档 4.1–4.3 一致）

**archives**

| 列名 | 类型 | 约束 |
|------|------|------|
| id | UUID | PK, default uuid4 |
| title | VARCHAR(500) | NOT NULL |
| scope | VARCHAR(20) | NOT NULL, default 'session' |
| scope_ref | VARCHAR(200) | NULL |
| solution_doc | TEXT | NOT NULL |
| **overview** | **TEXT** | **NULL，写入时生成或传入，L1 概览，约 500–2000 字** |
| conversation_summary | TEXT | NULL |
| project | VARCHAR(100) | NOT NULL, default 'default' |
| created_by | VARCHAR(100) | NOT NULL |
| visibility | VARCHAR(20) | NOT NULL, default 'project' |
| status | VARCHAR(20) | NOT NULL, default 'draft' |
| created_at | TIMESTAMPTZ | NOT NULL, default now() |
| updated_at | TIMESTAMPTZ | NOT NULL, default now() |
| embedding | vector(768) | NULL（由 title + overview 或 solution_doc 前段 + conversation_summary 生成） |

索引：无额外唯一约束；检索按 project、status、created_at 过滤时加常规索引；**archives.embedding 建向量索引**（与 experiences 一致：ivfflat 或 hnsw，如 `ivfflat (embedding vector_cosine_ops)`），避免向量检索全表扫描，见迁移 2.2。

**archive_experience_links**

| 列名 | 类型 | 约束 |
|------|------|------|
| archive_id | UUID | NOT NULL, FK(archives.id, CASCADE) |
| experience_id | UUID | NOT NULL, FK(experiences.id, CASCADE) |
| created_at | TIMESTAMPTZ | NULL |

唯一约束：(archive_id, experience_id)。索引：archive_id, experience_id（便于双向查）。

**archive_attachments**

| 列名 | 类型 | 约束 |
|------|------|------|
| id | UUID | PK, default uuid4 |
| archive_id | UUID | NOT NULL, FK(archives.id, CASCADE) |
| kind | VARCHAR(30) | NOT NULL |
| path | VARCHAR(1000) | NULL |
| content_snapshot | TEXT | NULL |
| git_commit | VARCHAR(64) | NULL |
| git_refs | JSONB | NULL |
| snippet | TEXT | NULL |
| created_at | TIMESTAMPTZ | NULL |

索引：archive_id（列表查附件）。

### 2.2 迁移文件约定

- **路径:** `migrations/versions/<revision_id>_add_archives_tables.py`
- **revision:** 新生成（如 `a1b2c3d4e5f6`），**down_revision = "6ab06751f40e"**
- **upgrade:** 依次 create_table archives（含 vector(768)）→ archive_experience_links → archive_attachments；为 **archives.embedding** 创建与 experiences 一致的向量索引（如 ivfflat vector_cosine_ops）；project/status/created_at 常规索引。
- **downgrade:** 逆序 drop 向量索引与三张表：archive_attachments → archive_experience_links → archives。

---

## 三、代码改动预估与伪代码

### 3.1 新增/修改文件一览

| 类型 | 路径 | 预估 |
|------|------|------|
| 迁移 | migrations/versions/xxx_add_archives_tables.py | ~120 行（含 overview 列） |
| Model | src/team_memory/storage/models.py | +约 95 行（3 个 class，Archive 含 overview） |
| Repository | src/team_memory/storage/archive_repository.py（新建） | ~220 行（create_archive 含 overview；search_archives 返 L0/L1；**get_archive(id)** 返 L2） |
| Service | src/team_memory/services/archive.py（新建） | ~160 行（archive_save 含 overview；**get_archive(id)** 调 repository） |
| SearchRequest | src/team_memory/services/search_pipeline.py | +2 行（include_archives） |
| SearchPipeline | src/team_memory/services/search_pipeline.py | +约 80–120 行（archive 分支 + L0/L1 合并） |
| ExperienceService.search | src/team_memory/services/experience.py | +2 行传 include_archives |
| server.py | src/team_memory/server.py | tm_archive_save ~80 行；**tm_get_archive** ~30 行；tm_search/tm_solve 各 +2 参数 |
| 测试 | tests/test_server.py 或 tests/test_archive*.py | tm_archive_save、tm_get_archive、include_archives 检索 |
| 文档 | docs/mcp-patterns.md | 登记 tm_archive_save、tm_get_archive、include_archives |

**Repository 放置决策：** 与现有 ExperienceRepository 同文件会令 repository.py 继续膨胀；**新建 `src/team_memory/storage/archive_repository.py`** 仅负责 Archive/ArchiveExperienceLink/ArchiveAttachment 的 CRUD 与「按 project/status 查 archives」的检索接口，便于单测与后续扩展。Service 层依赖 ArchiveRepository + ExperienceRepository（查经验是否全发布）。

### 3.2 关键伪代码

**ArchiveRepository.create_archive(..., overview, linked_experience_ids, attachments)**

```python
# 1. insert archives 行（含 overview；embedding 由 title+overview/solution_doc 前段+conversation_summary 生成）
# 2. for eid in linked_experience_ids: insert archive_experience_links
# 3. for att in attachments: insert archive_attachments
# 4. 若 linked 经验全部 exp_status=='published' 则 update archives.status='published', visibility 取其一
```

**SearchPipeline 中 include_archives 分支（双阶段：列表返 L0/L1）**

```python
# 若 not request.include_archives: 保持现有逻辑，返回 pipeline_result
# 否则：
#   - repo = ArchiveRepository(session)；repo.search_archives(session, query_embedding, project, limit, min_similarity, current_user=request.current_user)（对创建者查本人全部，对团队仅 status='published' + project，未传 current_user 仅 published）
#   - 将 archive 转为 L0/L1 结构（见设计文档 5.3）：type='archive', id, title, solution_preview（solution_doc 或 overview 前 300–500 字）, score, linked_experience_ids, attachment_count；以不丢命中率为目标，solution_preview 覆盖参与 embedding 的文本
#   - 与 pipeline_result.results（experience 侧也按 L0/L1 统一形状）合并后 RRF 或按 score 重排，截断 max_results
#   - 返回新 SearchPipelineResult(results=merged, ...)
```

**tm_archive_save 入参与调用**

```python
# title, solution_doc, created_by, project=None, scope='session', scope_ref=None,
# overview=None（可选，不传则可由服务端用 solution_doc 生成或留空）, conversation_summary=None, linked_experience_ids=None, attachments=None
# 校验：title/solution_doc/created_by 必填；overview 可选；linked_experience_ids 可选 list[uuid]；attachments 可选 list[{kind, path?, ...}]
# service.archive_save(...) -> archive_id; return {"message": "...", "archive_id": str(archive_id)}
```

**tm_get_archive(archive_id)**（双阶段：详情 L2）

```python
# 按 archive_id 查询 archives + archive_attachments。
# L2 返回 JSON 结构（约定）：{ "id", "title", "scope", "scope_ref", "solution_doc", "overview", "conversation_summary", "project", "created_by", "linked_experience_ids", "attachments": [ { "id", "kind", "path"?, "content_snapshot"?, "snippet"?, "git_commit"?, "git_refs"? } ], "created_at", "updated_at" }
# 无效/不存在 archive_id：返回 404 或 MCP 错误结构（如 {"error": "archive not found", "code": 404}），与现有 tm_get_experience 对不存在 id 的约定一致。
# 无附件：attachments 为 []（不返回 null）；写入时 attachments=None 与 attachments=[] 等价，L2 中统一为 []。
```

---

## 四、任务拆分（TDD，小步提交）

### Task 1: 新增 Alembic 迁移（三张表）

**Files:**
- Create: `migrations/versions/a1b2c3d4e5f6_add_archives_tables.py`

**Step 1: 编写迁移脚本（upgrade + downgrade）**

- 使用 `alembic revision -m "add_archives_tables"` 生成模板，或手写。revision 设为唯一 ID，down_revision = `"6ab06751f40e"`。
- upgrade: 创建 `archives`（含 **overview**、vector(768) 列）、`archive_experience_links`（唯一约束 (archive_id, experience_id)）、`archive_attachments`；archives 的 **embedding 向量索引**（与 experiences 一致）及 project/status/created_at 索引见 2.1、2.2。
- downgrade: 逆序 drop 三表。

**Step 2: 运行迁移验证**

```bash
cd /Users/yeshouyou/Work/agent/team_doc && alembic upgrade head
```

Expected: 无报错，三张表存在。

**Step 3: 回滚验证**

```bash
alembic downgrade -1
```

Expected: 三表被删除。

**Step 4: 再次升级**

```bash
alembic upgrade head
```

**Step 5: Commit**

```bash
git add migrations/versions/a1b2c3d4e5f6_add_archives_tables.py
git commit -m "feat(db): add archives, archive_experience_links, archive_attachments tables"
```

---

### Task 2: 新增 ORM 模型 Archive / ArchiveExperienceLink / ArchiveAttachment

**Files:**
- Modify: `src/team_memory/storage/models.py`（在文件末尾或合适位置追加）

**Step 1: 写失败测试**

在 `tests/test_storage_models.py` 或新建 `tests/test_archive_models.py` 中：

```python
def test_archive_model_has_required_columns():
    from team_memory.storage.models import Archive, ArchiveExperienceLink, ArchiveAttachment
    assert hasattr(Archive, "title")
    assert hasattr(Archive, "solution_doc")
    assert hasattr(Archive, "status")
    assert hasattr(ArchiveExperienceLink, "archive_id")
    assert hasattr(ArchiveAttachment, "kind")
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_archive_models.py -v
```

Expected: FAIL (ImportError or missing attribute).

**Step 3: 实现三个 Model 类**

- `Archive`: 表名 `archives`，字段与 2.1 一致；relationship 到 ArchiveExperienceLink、ArchiveAttachment。
- `ArchiveExperienceLink`: 表名 `archive_experience_links`，UniqueConstraint(archive_id, experience_id)。
- `ArchiveAttachment`: 表名 `archive_attachments`。
- 使用 `mapped_column(Vector(768), nullable=True)` 需从 `pgvector.sqlalchemy` 引入 Vector（与 Experience 一致）。

**Step 4: 运行测试通过**

```bash
pytest tests/test_archive_models.py -v
```

**Step 5: Commit**

```bash
git add src/team_memory/storage/models.py tests/test_archive_models.py
git commit -m "feat(models): add Archive, ArchiveExperienceLink, ArchiveAttachment ORM"
```

---

### Task 3: 实现 ArchiveRepository（CRUD + 检索 archives）

**Files:**
- Create: `src/team_memory/storage/archive_repository.py`
- Test: `tests/test_archive_repository.py`

**Step 1: 写失败测试**

- `test_create_archive_and_links`: 创建一条 archive，插入 2 条 link，再查 archive 的 links 数量为 2。
- `test_search_archives_by_vector`: 插入一条 status=published 的 archive（带 embedding），用相同向量搜索返回该条（需 DB）。

若暂不跑集成测试，可先写 `test_archive_repository_create_returns_id` 仅测 create 返回 UUID。

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_archive_repository.py -v
```

**Step 3: 实现 ArchiveRepository**

- `create_archive(title, solution_doc, created_by, project=..., scope=..., scope_ref=..., overview=None, conversation_summary=..., visibility=..., status=..., embedding=..., linked_experience_ids=..., attachments=...)`：**overview** 必在签名中（str \| None），与 2.1 表结构及 Task 4 archive_save 对齐；插入 Archive（含 overview）；插入 links；插入 attachments；根据 linked 经验的 exp_status 决定是否把 status 设为 published。
- `search_archives(session, query_embedding, project, limit, min_similarity, current_user=None)`：**权限**：若 `current_user` 非空，则 `(created_by = current_user) OR (status='published' AND project 匹配)`；未传 current_user 时仅 `status='published'` + project。向量相似度排序，返回 list[dict]（L0/L1：id, title, solution_preview, linked_experience_ids, attachment_count 等）。
- `get_archive_by_id(session, archive_id)`：返回单条 archive + attachments（L2），供 tm_get_archive 使用。
- 需从 `team_memory.storage.models` 导入 Archive, ArchiveExperienceLink, ArchiveAttachment, Experience。

**Step 4: 运行测试通过**

**Step 5: Commit**

```bash
git add src/team_memory/storage/archive_repository.py tests/test_archive_repository.py
git commit -m "feat(storage): add ArchiveRepository create and search_archives"
```

---

### Task 4: 实现 ArchiveService 与 tm_archive_save

**Files:**
- Create: `src/team_memory/services/archive.py`
- Modify: `src/team_memory/server.py`（注册 tm_archive_save）
- Modify: `src/team_memory/bootstrap.py` 或注入点（构造 ArchiveService 并供 server 使用）

**Step 1: 写失败测试**

在 `tests/test_server.py` 中：

```python
@pytest.mark.asyncio
async def test_tm_archive_save_returns_archive_id():
    # 使用 mock 或真实 service，调用 tm_archive_save(title="T", solution_doc="D", created_by="u")
    # 断言返回中有 archive_id
```

**Step 2: 运行测试确认失败**

**Step 3: 实现 ArchiveService**

- `archive_save(..., overview=..., ...)`：overview 可选，不传则可为空或由服务端从 solution_doc 生成简短概览；embedding 由 title + overview（或 solution_doc 前段）+ conversation_summary 生成（以不丢命中率为目标）；调用 ArchiveRepository.create_archive；若 linked_experience_ids 非空且全部 published，则更新 archive.status='published'。
- `get_archive(archive_id)`：按 id 返回 L2 全文（solution_doc、overview、attachments），供 tm_get_archive 使用。
- 在 `bootstrap.py` 的 `AppContext` 中增加字段 `archive_service: ArchiveService`；在 `bootstrap()` 内构造 `ArchiveService`（需 session_factory、embedding_provider、ArchiveRepository 依赖），传入 `AppContext(..., archive_service=archive_service)`。
- 在 server 中新增 `_get_archive_service()`：`return get_context().archive_service`（与 _get_service 类似）；tm_archive_save 解析参数后调用 `await _get_archive_service().archive_save(...)`，返回 `{"message": "…", "archive_id": "…"}`；错误返回 `{"error": "…", "code": 400}`。

**Step 4: 运行测试通过**

**Step 5: Commit**

```bash
git add src/team_memory/services/archive.py src/team_memory/server.py src/team_memory/bootstrap.py tests/test_server.py
git commit -m "feat(mcp): add tm_archive_save and ArchiveService"
```

---

### Task 5: SearchRequest 与 ExperienceService.search 增加 include_archives

**Files:**
- Modify: `src/team_memory/services/search_pipeline.py`（SearchRequest 增加 include_archives: bool = False）
- Modify: `src/team_memory/services/experience.py`（search 方法增加 include_archives=False，并传入 SearchRequest）

**Step 1: 写失败测试**

- 在 test_server 或 search_pipeline 测试中：调用 search(..., include_archives=True)，断言当有 published archive 时结果中出现 type='archive' 的项（可用 mock 或集成）。
- **创建者视角**：以 current_user=创建者 调用时，本人创建的 **draft** archive 也出现在结果中。
- **团队视角**：不传 current_user 或以非创建者身份调用时，仅返回 status='published' 的 archive，draft 不出现。

**Step 2: 运行测试确认失败**

**Step 3: 实现**

- SearchRequest 增加字段 `include_archives: bool = False`。
- ExperienceService.search 增加参数 include_archives=False，构建 SearchRequest 时传入。

**Step 4: 运行测试通过**

**Step 5: Commit**

```bash
git add src/team_memory/services/search_pipeline.py src/team_memory/services/experience.py
git commit -m "feat(search): add include_archives to SearchRequest and ExperienceService.search"
```

---

### Task 6: SearchPipeline 中实现 archive 检索与结果合并

**Files:**
- Modify: `src/team_memory/services/search_pipeline.py`

**Step 1: 写失败测试**

- 当 include_archives=True 且存在 published archive 时，pipeline.search 返回的 results 中包含至少一条 type='archive'（可 mock ArchiveRepository.search_archives）。
- 传入 current_user=创建者时，该创建者的 draft archive 也出现在结果中；不传 current_user 时仅 published。

**Step 2: 运行测试确认失败**

**Step 3: 实现**

- 在 SearchPipeline.search 中，若 `request.include_archives` 为 True：在用 query_embedding 完成 experience 检索后，**在方法内用入参 `session` 实例化 `ArchiveRepository(session)`**（与现有 `ExperienceRepository(session)` 用法一致，**不在 AppContext 注入 ArchiveRepository**），调用 `search_archives(session, query_embedding, project, limit, min_similarity, current_user=request.current_user 或等价来源)`；将 archive 转为 **L0/L1** 结构（设计文档 5.3）：type='archive'、id、title、**solution_preview**（solution_doc 或 overview 前 **300–500 字**）、score、linked_experience_ids、attachment_count，以不丢命中率为目标；与 pipeline_result.results（experience 侧同构为 L0/L1）合并后 RRF 或按 score 排序，截断至 max_results；构造新 SearchPipelineResult(results=merged, ...)。

**Step 4: 运行测试通过**

**Step 5: Commit**

```bash
git add src/team_memory/services/search_pipeline.py
git commit -m "feat(search): include archives in pipeline when include_archives=True"
```

---

### Task 7: tm_search 与 tm_solve 增加 include_archives 参数并透传

**Files:**
- Modify: `src/team_memory/server.py`

**Step 1: 写失败测试**

- tm_search(query="x", include_archives=True) 调用后，若后端返回含 archive，则结果中应有 type='archive' 的条目。
- 创建者带 current_user 调用时，本人 draft archive 可出现；未传 current_user 时仅 published。

**Step 2: 运行测试确认失败**

**Step 3: 实现**

- tm_search 增加参数 `include_archives: bool = False`，传给 service.search(..., include_archives=include_archives)。
- tm_solve 同样增加 `include_archives: bool = False` 并透传。

**Step 4: 运行测试通过**

**Step 5: Commit**

```bash
git add src/team_memory/server.py
git commit -m "feat(mcp): tm_search and tm_solve accept include_archives"
```

---

### Task 7.5: 实现 tm_get_archive（双阶段详情 L2）

**Files:**
- Modify: `src/team_memory/server.py`
- Modify: `src/team_memory/services/archive.py`（get_archive）
- Modify: `src/team_memory/storage/archive_repository.py`（get_archive_by_id）

**Step 1: 写失败测试**

- tm_get_archive(archive_id) 返回 L2 JSON（含 solution_doc、overview、**attachments 为数组**，结构见 3.2）。
- **无效 archive_id**：不存在或非法 id 时返回 404 或约定错误结构（如 `{"error": "archive not found", "code": 404}`），与 tm_get_experience 对不存在 id 一致。
- **空附件**：无附件时 attachments 为 `[]`；写入时 attachments=None 与 attachments=[] 等价，L2 中统一返回 `[]`。

**Step 2: 实现** — ArchiveRepository.get_archive_by_id(session, archive_id) 返回单条 archive + attachments（无附件时 []）；ArchiveService.get_archive(archive_id)（不存在时抛或返回 None，由 server 转 404）；server 注册 tm_get_archive(archive_id)，返回 3.2 约定的 L2 JSON。

**Step 3: Commit** — `feat(mcp): add tm_get_archive for L2 detail`

---

### Task 8: 归档时根据关联经验推导 status + 经验状态变更时回写（MVP）

**Files:**
- Modify: `src/team_memory/services/archive.py`（提供 `update_archive_status_for_experience(experience_id, new_exp_status)`，根据该 experience 关联的所有 archive 重新计算「关联经验是否全部 published」并更新各 archive.status）
- Modify: 所有**经验状态变更**的调用点：如 ExperienceService 中 `change_status()`、`publish_personal()`、review 拒绝等「将 experience 设为 draft 或 published」的路径，在变更后调用 `ArchiveService.update_archive_status_for_experience(experience_id, new_exp_status)`，避免漏路径。若项目有 EventBus，可新增「经验状态变更」事件并在上述路径 emit，由 ArchiveService 订阅并执行回写（Plan 需在实现前确认调用点列表或事件契约）。

**Step 1: 写测试**

- 创建 archive 并 link 两条 experience，两条均为 exp_status='published'；断言 archive.status 为 'published'。
- 创建 archive 并 link 一条 draft 经验；断言 archive.status 为 'draft'。
- **经验→draft 回写**：将某条已 published 的 experience 改为 draft 后，关联的 archive 的 status 应被更新为 draft（检索团队结果时不再出现）。
- **经验→published 回写**：某 archive 关联两条 experience，原一条 draft 一条 published（archive 为 draft）；将那条 draft 改为 published 后，该 archive 的 status 应被更新为 'published'。

**Step 2: 实现**

- create_archive 完成后，根据 linked experience_ids 查询 Experience.exp_status；若全部为 'published' 则 update archive SET status='published', visibility=...；否则保持 draft。
- **经验状态变更时回写**：实现 `update_archive_status_for_experience(experience_id, new_exp_status)`：查该 experience 关联的所有 archive_id；对每个 archive 重新计算「其关联经验是否全部 published」；若全部为 published 则将该 archive 置为 published，否则置为 draft。在**所有**「修改 experience 的 exp_status」的调用点（如 change_status、publish_personal、review 拒绝）完成后调用该方法。

**Step 3: Commit**

```bash
git commit -m "feat(archive): derive archive status from linked experiences and backfill on experience status change"
```

---

### Task 9: 文档与 MCP 登记

**Files:**
- Modify: `docs/mcp-patterns.md`

**Step 1:** 在已注册工具列表中增加 `tm_archive_save`、`tm_get_archive`；在参数说明中增加 tm_search / tm_solve 的 `include_archives`；说明检索返回 L0/L1、solution_preview 约 300–500 字、详情用 tm_get_archive(id) 拉 L2。

**Step 2: Commit**

```bash
git add docs/mcp-patterns.md
git commit -m "docs: register tm_archive_save and include_archives in mcp-patterns"
```

---

## 五、验收标准（项目开发完毕后）

以下全部满足视为本阶段完成：

1. **迁移与模型**
   - `alembic upgrade head` 成功，存在表 `archives`（含 **overview** 列）、`archive_experience_links`、`archive_attachments`。
   - ORM 可正确读写三张表，无 import 错误。

2. **写入**
   - `tm_archive_save(..., overview=..., ...)` 返回 `archive_id`；DB 中对应 archive（含 overview）与 links、attachments 正确写入；embedding 由 title + overview（或 solution_doc 前段）+ conversation_summary 生成。
   - 当 linked 经验全部为 published 时，该 archive 的 status 为 published；否则为 draft。**经验状态变更时回写**：experience 从 published 改为 draft 后，其关联的 archive 的 status 同步为 draft；**experience 从 draft 改为 published 后，若某 archive 的关联经验已全部 published，则该 archive 的 status 同步为 published**。所有「修改 experience 的 exp_status」的调用点均触发回写（见 Task 8）。

3. **检索（双阶段、不丢命中率）**
   - `tm_search(query=..., include_archives=True)` 返回 **L0/L1** 混合结果，其中 type='archive' 的项含 id、title、**solution_preview**（约 300–500 字，覆盖参与 embedding 的文本）、score、linked_experience_ids、attachment_count。**创建者**带 current_user 时可见本人 draft；**未传 current_user** 或团队视角仅返回 published。
   - `tm_solve(..., include_archives=True)` 行为一致。`include_archives=False`（默认）时行为与当前一致，不返回 archive。
   - `tm_get_archive(archive_id)` 返回 L2 全文（JSON 结构见 3.2）：solution_doc、overview、attachments 为数组（无附件时为 `[]`）。**无效/不存在 archive_id** 返回 404 或约定错误结构。

4. **质量**
   - `make lint` 零报错。
   - `make test` 全绿；新增/修改的 MCP 与 repository 有对应测试；**创建者 vs 团队**检索、**tm_get_archive 无效 id / 空 attachments**、**经验回写（draft→draft 与 draft→published）** 有明确用例。
   - 无硬编码密钥、无裸 `print()`。
   - **可观测性**：档案馆相关 MCP（tm_archive_save、tm_get_archive、include_archives 检索）遵循 [logging-format](../design-docs/logging-format.md)，至少记录 tool 名、archive_id（如有）、duration_ms、error（如有）；便于排查与容量评估。

5. **文档**
   - `docs/mcp-patterns.md` 已登记 tm_archive_save、**tm_get_archive** 及 tm_search/tm_solve 的 include_archives；说明 L0/L1 与 solution_preview 长度约定。

**边界说明（不阻塞本次验收）：** 本阶段**不实现**从会话自动生成 solution_doc/overview 的档案馆 Agent，由调用方传入；**tm_preflight 返回 archive_summaries**、archives 表 FTS 列为后续迭代。

---

## 六、评审结论与已落实修改（Plan 虚拟评审委员会）

按 [plan-eval-multi-agent-review](../../.cursor/prompts/plan-eval-multi-agent-review.md) 对本文的评估，以下修改已直接纳入 Plan：

| 来源 | 修改项 |
|------|--------|
| **架构** | 1.4 检索权限与 current_user（创建者 vs 团队、未传仅返回 published）；Task 6 明确 SearchPipeline 内用 `session` 实例化 `ArchiveRepository(session)`、不注入 AppContext；Task 8 明确经验回写调用点（所有修改 exp_status 的路径或事件）及「经验→published 后若全部 published 则 archive→published」的测试与实现。 |
| **技术** | Task 3 `create_archive` 签名显式含 `overview`；3.2 与 Task 7.5 约定 tm_get_archive L2 JSON 结构、无效 archive_id 返回 404、空 attachments 为 `[]`。 |
| **运维** | 1.3 生产回滚预案（升级前备份、回滚即丢数据）；2.1/2.2 与 Task 1 为 archives.embedding 建向量索引（ivfflat）；验收 4 可观测性（档案馆 MCP 遵循 logging-format，记录 tool、archive_id、duration_ms、error）。 |
| **QA** | Task 5/6/7 与验收 3/4 补充：创建者检索含本人 draft、团队检索不含 draft；tm_get_archive 无效 id 与空附件测试；Task 8 与验收 2 补充经验→published 后 archive→published 的测试与验收。 |
