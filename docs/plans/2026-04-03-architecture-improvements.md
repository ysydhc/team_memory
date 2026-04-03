# TeamMemory 架构优化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 解决 7 个架构问题——跨域耦合、职责混杂、可靠性不足、配置单体、维度硬编码——使 TeamMemory 的分层更清晰、可测试性更强、可演进性更好。

**Architecture:** 严格保持 MCP/Web → Services → Storage → Models 的分层方向。通过 EventBus 解耦跨域通信，通过拆文件收窄职责，通过持久化任务表替换 fire-and-forget，通过配置分域缩小依赖面。

**Tech Stack:** Python 3.13 · FastAPI · SQLAlchemy 2.x (async) · PostgreSQL + pgvector · Pydantic v2 · pytest

**验证命令：**
```bash
source .venv/bin/activate
make lint                    # ruff 零报错
pytest tests/ -q             # 全绿
make harness-check           # import 方向检查
```

---

## Phase 0: 高优先级（跨域解耦 + Web 层拆分）

### Task 1: EventBus 解耦 Archive→PersonalMemory

**目标：** ArchiveService 不再直接调用 PatternExtractor / PersonalMemoryService，改为发事件，由 bootstrap 注册订阅者处理。

**Files:**
- Modify: `src/team_memory/services/event_bus.py:27-42` — 新增事件类型
- Modify: `src/team_memory/services/archive.py:93-176` — 移除 `_extract_patterns_bg`，改为 emit 事件
- Modify: `src/team_memory/services/archive.py:85-91` — 构造函数移除 `llm_config`
- Modify: `src/team_memory/bootstrap.py:357-360` — 新增 `_register_pattern_extraction` 订阅
- Modify: `src/team_memory/bootstrap.py:373-376` — ArchiveService 构造去掉 `llm_config`
- Test: `tests/test_archive_service_event.py` (新建)

**Step 1 — event_bus.py 新增事件常量**

```python
# src/team_memory/services/event_bus.py  Events 类内新增：
ARCHIVE_CREATED = "archive.created"
```

**Step 2 — archive.py 移除直接耦合**

`archive_save()` 中删除 `asyncio.create_task(self._extract_patterns_bg(...))` 块（当前 L151-153），替换为：

```python
if raw_conversation and created_by and created_by.lower() != "anonymous":
    await self._event_bus.emit(
        Events.ARCHIVE_CREATED,
        {
            "raw_conversation": raw_conversation,
            "user_id": created_by,
        },
    )
```

- ArchiveService.__init__ 新增 `event_bus: EventBus | None = None`，移除 `llm_config`
- 删除整个 `_extract_patterns_bg` 方法（L157-176）

**Step 3 — bootstrap.py 注册 pattern extraction 订阅**

在 `_register_cache_invalidation` 之后新增：

```python
def _register_pattern_extraction(
    event_bus: EventBus,
    embedding: EmbeddingProvider,
    llm_config,
    db_url: str,
) -> None:
    async def _on_archive_created(payload: dict) -> None:
        raw = payload.get("raw_conversation", "")
        user_id = payload.get("user_id", "")
        if not raw or not user_id:
            return
        try:
            from team_memory.services.pattern_extractor import PatternExtractor
            from team_memory.services.personal_memory import PersonalMemoryService

            pm_svc = PersonalMemoryService(embedding_provider=embedding, db_url=db_url)
            extractor = PatternExtractor()
            count = await extractor.extract_and_save(
                conversation=raw, user_id=user_id,
                llm_config=llm_config, pm_service=pm_svc,
            )
            if count:
                logger.info("Extracted %d patterns for user=%s", count, user_id)
        except Exception:
            logger.warning("Pattern extraction failed", exc_info=True)

    event_bus.on(Events.ARCHIVE_CREATED, _on_archive_created)
```

在 bootstrap() 中 `_register_cache_invalidation` 之后调用：

```python
_register_pattern_extraction(event_bus, embedding, settings.llm, db_url)
```

ArchiveService 构造传入 event_bus：

```python
archive_service = ArchiveService(
    embedding_provider=embedding,
    db_url=db_url,
    event_bus=event_bus,
)
```

**Step 4 — 更新测试**

- 修改 `tests/test_archive_overview.py` 如果它 mock 了 `_extract_patterns_bg`
- 新建 `tests/test_archive_service_event.py` 验证 emit 被调用
- 确保 `tests/test_pattern_extractor.py` 不受影响（它直接测 PatternExtractor）

**Step 5 — 验证**

```bash
pytest tests/ -q  # 全绿
make lint
```

---

### Task 2: 拆分 web/app.py — 认证逻辑独立

**目标：** 将 auth session 逻辑（~100 行）和 Pydantic schemas（~100 行）从 app.py 拆出，app.py 只保留 FastAPI 应用创建 + 中间件 + 路由挂载。

**Files:**
- Create: `src/team_memory/web/auth_session.py` — session token 编解码 + get_current_user/get_optional_user
- Create: `src/team_memory/web/schemas.py` — 所有 Pydantic request/response models
- Modify: `src/team_memory/web/app.py:302-541` — 移出代码，保留 re-export
- Modify: 7 个 route 文件的 import 路径
- Test: `tests/test_web.py` — 更新 import

**Step 1 — 创建 `web/auth_session.py`**

从 app.py 移出以下函数（当前 L302-437）：

```
_encode_api_key_cookie()       L302-303
_encode_session_token()        L306-310
_decode_session_token()        L313-330
_get_session_secret()          L333-340
_get_user_role_from_db()       L343-355
get_current_user()             L358-403
get_optional_user()            L406-437
```

新文件需要的 import：
```python
import hashlib, hmac, json, logging, os, time
from fastapi import HTTPException, Request
from team_memory.auth.provider import DbApiKeyAuth, User
from team_memory.bootstrap import get_context
from team_memory.storage.database import get_session
from team_memory.storage.models import ApiKey
from sqlalchemy import select
```

**Step 2 — 创建 `web/schemas.py`**

从 app.py 移出所有 Pydantic 类（当前 L443-541）：

```
LoginRequest, RegisterRequest, LoginResponse,
ChangePasswordRequest, ForgotPasswordResetRequest, AdminResetPasswordRequest,
ExperienceCreate, ExperienceUpdate, FeedbackCreate, SearchRequest,
ApiKeyCreateRequest, ApiKeyUpdateRequest
```

**Step 3 — app.py 保留 re-export（过渡期兼容）**

```python
# Backward-compatible re-exports (will be removed after all routes updated)
from team_memory.web.auth_session import (  # noqa: F401
    get_current_user, get_optional_user,
    _encode_api_key_cookie, _encode_session_token, _decode_session_token,
    _get_session_secret, _get_user_role_from_db,
)
from team_memory.web.schemas import (  # noqa: F401
    LoginRequest, RegisterRequest, LoginResponse, ...
)
```

**Step 4 — 逐个更新 route 文件 import**

| Route 文件 | 当前 import from | 改为 import from |
|---|---|---|
| `routes/auth.py` | `web.app` | `web.auth_session` + `web.schemas` |
| `routes/experiences.py` | `web.app` | `web.auth_session` + `web.schemas` |
| `routes/search.py` | `web.app` | `web.auth_session` + `web.schemas` |
| `routes/archives.py` | `web.app` | `web.auth_session` |
| `routes/config.py` | `web.app` | `web.auth_session` |
| `routes/dedup.py` | `web.app` | `web.auth_session` |
| `routes/personal_memory.py` | `web.app` | `web.auth_session` |
| `web/dependencies.py:40` | `web.app` | `web.auth_session` |

**Step 5 — 移除 app.py 中的 re-export**（Route 全部更新后）

**Step 6 — 更新 `tests/test_web.py`** 中涉及 `_decode_session_token` 等的 import

**Step 7 — 验证**

```bash
pytest tests/test_web.py tests/test_p0_core.py -q  # 全绿
make lint
```

---

### Task 3: 提取 `_resolve_project` 到 utils

**目标：** `_resolve_project` 在 server.py 和 web/app.py 中各有一份，逻辑相同。提取到共享位置。

**Files:**
- Create: `src/team_memory/utils/project.py` — `resolve_project()` 公共函数
- Modify: `src/team_memory/server.py` — import 替换
- Modify: `src/team_memory/web/app.py` — import 替换
- Modify: 引用 `_resolve_project` 的 route 文件

**实现：** 将 server.py 中 `_normalize_project_name` + `_resolve_project`（当前约 L113-132）移到 `utils/project.py`，两边改为 import。

---

## Phase 1: 中优先级（Service 拆分 + 任务可靠性）

### Task 4: ExperienceService 拆出 SearchOrchestrator

**目标：** 将搜索编排逻辑从 ExperienceService 独立出来，使 ExperienceService 聚焦于写操作 + 反馈。

**Files:**
- Create: `src/team_memory/services/search_orchestrator.py`
- Modify: `src/team_memory/services/experience.py:105-199` — 移出 `search()` 和 `_legacy_search()`
- Modify: `src/team_memory/bootstrap.py` — 新建 SearchOrchestrator 并注入
- Modify: `src/team_memory/server.py` — MCP 工具中搜索调用改为 SearchOrchestrator
- Test: `tests/test_search_orchestrator.py` (新建)

**新建 SearchOrchestrator：**

```python
class SearchOrchestrator:
    """Coordinate search across Experience + Archive, with implicit feedback."""

    def __init__(
        self,
        search_pipeline: SearchPipeline,
        embedding_provider: EmbeddingProvider,
        db_url: str,
    ):
        self._pipeline = search_pipeline
        self._embedding = embedding_provider
        self._db_url = db_url

    async def search(
        self, query: str, *, tags=None, max_results=5,
        min_similarity=0.6, user_name=None, source="mcp",
        grouped=True, top_k_children=3, project=None,
        include_archives=False, **kwargs,
    ) -> list[dict]:
        """Execute search pipeline + implicit feedback."""
        ...  # 从 ExperienceService.search() L105-199 移入
```

**ExperienceService 变更：**
- 移除 `search()` 方法和 `_legacy_search()`
- 移除 `search_pipeline` 依赖
- 构造函数从 8 个依赖降到 6 个

**bootstrap.py 变更：**

```python
search_orchestrator = SearchOrchestrator(
    search_pipeline=search_pipeline,
    embedding_provider=embedding,
    db_url=db_url,
)
```

AppContext 新增 `search_orchestrator` 字段。

**server.py 变更：**
- 新增 `_get_search_orchestrator()` helper
- `memory_recall` 中的 `service.search(...)` 改为 `search_orchestrator.search(...)`

---

### Task 5: 持久化后台任务替换 fire-and-forget

**目标：** 用数据库任务表替换 `asyncio.create_task`，确保 pattern 提取不会因进程退出丢失。

**Files:**
- Create: `src/team_memory/storage/models.py` — 新增 `BackgroundTask` 模型
- Create: `migrations/versions/009_background_tasks.py`
- Create: `src/team_memory/services/task_runner.py` — 轮询 + 执行
- Modify: `src/team_memory/bootstrap.py` — 注册 event handler 改为写任务表
- Modify: `src/team_memory/services/event_bus.py` — handler 写任务而非直接执行

**BackgroundTask 模型：**

```python
class BackgroundTask(Base):
    __tablename__ = "background_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g. "pattern_extraction"
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    # pending → running → completed / failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
```

**TaskRunner：**

```python
class TaskRunner:
    """Poll background_tasks table and execute pending tasks."""

    HANDLERS: dict[str, Callable] = {}  # task_type → async handler

    async def poll_and_execute(self) -> int:
        """Claim one pending task, execute, mark done. Return count processed."""
        ...

    @classmethod
    def register(cls, task_type: str, handler: Callable) -> None:
        cls.HANDLERS[task_type] = handler
```

**bootstrap 注册变更：**

Event handler 不再直接执行 pattern extraction，改为插入 BackgroundTask 行：

```python
async def _on_archive_created(payload: dict) -> None:
    async with get_session(db_url) as session:
        task = BackgroundTask(
            task_type="pattern_extraction",
            payload=payload,
        )
        session.add(task)
        await session.commit()
```

Web 应用启动时启动 TaskRunner 轮询（或在每次 archive_save 后尝试执行）。

---

## Phase 2: 低优先级（代码整洁 + 配置优化）

### Task 6: Repository 拆分 — 提取查询构造器

**目标：** 将 repository.py 中的复杂搜索查询（vector_search, fts_search, find_duplicates）提取到独立的 query builder 模块。

**Files:**
- Create: `src/team_memory/storage/query_builders.py`
- Modify: `src/team_memory/storage/repository.py:330-523` — 搜索方法调用 query builder
- Test: `tests/test_query_builders.py` (新建)

**提取内容：**

| 函数 | 当前位置 | 职责 |
|------|----------|------|
| `build_vector_search_query()` | repository.py L330-367 | 构造向量搜索 SQL |
| `build_fts_search_query()` | repository.py L369-406 | 构造全文搜索 SQL |
| `build_duplicate_detection_query()` | repository.py L449-490 | 构造去重查询 SQL |
| `_active_filter()` | repository.py L76-96 | 可见性 WHERE 条件 |
| `_apply_scope_filter()` | repository.py L98-118 | 项目范围过滤 |

Repository 方法变为：

```python
async def search_by_vector(self, embedding, ...) -> list[dict]:
    stmt = build_vector_search_query(embedding, filters=..., limit=...)
    result = await self._session.execute(stmt)
    return [self._row_to_dict(r) for r in result]
```

---

### Task 7: 配置按域拆分

**目标：** 将 Settings 的 20 个字段按使用域分组到子模块，每个 Service 只接收自己需要的配置。

**Files:**
- Create: `src/team_memory/config/` 目录，拆出：
  - `config/database.py` — DatabaseConfig
  - `config/embedding.py` — EmbeddingConfig + 4 个子配置
  - `config/search.py` — SearchConfig + CacheConfig + VectorConfig + RetrievalConfig + PageIndexLiteConfig
  - `config/llm.py` — LLMConfig + ExtractionConfig
  - `config/auth.py` — AuthConfig
  - `config/web.py` — WebConfig + UploadsConfig
  - `config/mcp.py` — MCPConfig
  - `config/settings.py` — Settings 聚合类 + load_settings()
  - `config/__init__.py` — re-export 所有
- Modify: `src/team_memory/config.py` → 改为 `config/__init__.py` 的 re-export（兼容期）
- Test: `tests/test_config.py` — 确保不破坏

**原则：** 先拆文件，不改 Settings 接口。下一步再让 Service 只接收子配置而非整个 Settings。

---

## Phase 3: 低优先级（数据模型演进）

### Task 8: Embedding 维度配置化

**目标：** 让模型层的 Vector 维度从配置读取，而非硬编码 768。

**Files:**
- Modify: `src/team_memory/storage/models.py:64,213,384` — Vector(768) → 动态维度
- Create: `migrations/versions/010_flexible_embedding_dimension.py` — 迁移脚本
- Modify: `src/team_memory/bootstrap.py` — 初始化时传入维度

**方案：**

由于 SQLAlchemy 的 `Vector(dim)` 在建表时就确定维度，运行时无法更改。实际方案是：

1. **models.py 中使用变量**：

```python
from team_memory.config import get_settings

_EMBEDDING_DIM = 768  # 默认值，bootstrap 时更新

class Experience(Base):
    embedding = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
```

2. **迁移脚本**：当用户切换 provider（如从 Ollama 768 到 OpenAI 1536）时，需要运行迁移修改列类型 + 清空旧向量：

```sql
ALTER TABLE experiences ALTER COLUMN embedding TYPE vector(1536);
UPDATE experiences SET embedding = NULL;  -- 需要重新 embed
```

3. **bootstrap.py** 中校验：启动时检查配置维度与 DB 列维度是否一致，不一致则告警。

**注意：** 此 Task 影响面大（需要重新 embed 所有数据），建议作为独立版本发布，不与其他 Task 混在一次 PR 中。

---

## 任务依赖关系

```
Task 1 (EventBus 解耦)
  └── Task 5 (持久化任务) — 依赖 Task 1 的事件模式
Task 2 (web/app 拆分)
  └── Task 3 (resolve_project 提取) — 可同步进行
Task 4 (SearchOrchestrator) — 独立
Task 6 (Query Builder) — 独立
Task 7 (Config 拆分) — 独立
Task 8 (Embedding 维度) — 独立，但建议最后做
```

**建议执行顺序：** 1 → 2+3（并行）→ 4 → 5 → 6 → 7 → 8

---

## 验证清单

每个 Task 完成后：
- [ ] `make lint` 零报错
- [ ] `pytest tests/ -q` 全绿（排除已知的 2 个 personal_memory 环境问题）
- [ ] `make harness-check` import 方向合规
- [ ] 新建文件已有对应测试
- [ ] 无循环 import（`python -c "from team_memory.server import mcp"` 无报错）
