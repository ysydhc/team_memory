# Layer Boundary Enforcement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all 24 daemon→storage direct imports and 4 hooks→daemon internal imports, enforcing the layered architecture (storage ← services ← daemon ← hooks).

**Architecture:**
- **C (Facade):** Services layer exposes all needed methods; daemon only calls services, never storage directly.
- **A (Anti-Corruption Layer):** TMSink remains the single gateway from daemon to services; auto_dream/wiki_compiler route through it or through new service methods.
- **New `services/maintenance/batch_ops.py`:** Handles bulk read/write operations needed by backfill scripts (no User auth required — internal system operations).

**Tech Stack:** Python async/await, SQLAlchemy async, existing ExperienceService, TMSink pattern.

---

## Overview of Changes

```
BEFORE (violations):
  daemon/integration/auto_dream.py  → storage.database (4x)
  daemon/integration/tm_sink.py     → storage.database (2x)
  daemon/wiki/wiki_compiler.py      → storage.database (1x)
  daemon/wiki/wiki_cli.py           → storage.database (3x)
  daemon/backfill/embedding_backfill.py → storage.database (1x)
  daemon/backfill/graph_backfill.py     → storage.database (1x)
  daemon/backfill/entity_backfill.py    → storage.database (3x)
  daemon/backfill/entity_dedup.py       → storage.database (4x)
  daemon/backfill/solution_backfill.py  → storage.database (1x)
  hooks/session_timeout.py          → daemon.buffers.draft_buffer (1x)
  hooks/session_timeout.py          → daemon.refinement.draft_refiner (1x)
  hooks/integrations/git_watcher.py → daemon.wiki.markdown_indexer (1x)
  hooks/integrations/initial_index.py → daemon.wiki.markdown_indexer (1x)

AFTER (clean):
  daemon/* → services/* → storage/*
  hooks/*  → HTTP (daemon port 3901) only
```

---

## Task 1: Add missing methods to ExperienceService

**Why:** auto_dream and tm_sink bypass services because these methods don't exist yet.

**Files:**
- Modify: `src/team_memory/services/memory/experience.py`
- Test: `tests/services/test_experience_service.py`

**New methods to add:**

```python
async def list_published(
    self,
    project: str | None = None,
    quality_tier: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[dict]:
    """List published/promoted experiences. Used by auto_dream."""

async def count_published(
    self,
    project: str | None = None,
    since: datetime | None = None,
) -> int:
    """Count published experiences. Used by auto_dream._count_new_experiences."""

async def soft_delete_by_id(self, experience_id: str) -> bool:
    """Soft-delete by string ID. Wraps existing soft_delete(UUID)."""

async def increment_used_count(self, experience_id: str) -> None:
    """Increment recall usage counter. Used by TMSink."""
```

**Step 1: Write failing tests**

```python
# tests/services/test_experience_service.py
async def test_list_published_returns_published_only():
    svc = ExperienceService(...)
    result = await svc.list_published(project="test")
    assert all(r["status"] in ("published", "promoted") for r in result)

async def test_count_published():
    svc = ExperienceService(...)
    count = await svc.count_published(project="test")
    assert isinstance(count, int)

async def test_increment_used_count():
    svc = ExperienceService(...)
    await svc.increment_used_count(str(some_id))
    exp = await svc.get_by_id(str(some_id))
    assert exp["used_count"] == 1
```

**Step 2:** Run `pytest tests/services/test_experience_service.py -k "list_published or count_published or increment_used" -v` → expect FAIL

**Step 3: Implement**

In `experience.py`, add after `list_experiences()`:

```python
async def list_published(
    self,
    project: str | None = None,
    quality_tier: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[dict]:
    async with self._session() as session:
        conditions = [
            Experience.is_deleted == False,
            Experience.exp_status.in_(["published", "promoted"]),
        ]
        if project:
            conditions.append(Experience.project == project)
        if quality_tier:
            conditions.append(Experience.quality_tier == quality_tier)
        if since:
            conditions.append(Experience.created_at >= since)
        stmt = select(Experience).where(and_(*conditions)).limit(limit)
        result = await session.execute(stmt)
        return [self._to_dict(e) for e in result.scalars()]

async def count_published(
    self,
    project: str | None = None,
    since: datetime | None = None,
) -> int:
    async with self._session() as session:
        conditions = [Experience.is_deleted == False]
        if project:
            conditions.append(Experience.project == project)
        if since:
            conditions.append(Experience.created_at >= since)
        stmt = select(func.count(Experience.id)).where(and_(*conditions))
        result = await session.execute(stmt)
        return result.scalar() or 0

async def soft_delete_by_id(self, experience_id: str) -> bool:
    import uuid as _uuid
    return await self.soft_delete(_uuid.UUID(experience_id))

async def increment_used_count(self, experience_id: str) -> None:
    import uuid as _uuid
    async with self._session() as session:
        stmt = select(Experience).where(
            Experience.id == _uuid.UUID(experience_id)
        )
        result = await session.execute(stmt)
        exp = result.scalar_one_or_none()
        if exp is not None:
            exp.used_count = (exp.used_count or 0) + 1
```

**Step 4:** Run tests → expect PASS

**Step 5: Commit**
```bash
git add src/team_memory/services/memory/experience.py tests/services/test_experience_service.py
git commit -m "feat(services): add list_published, count_published, soft_delete_by_id, increment_used_count"
```

---

## Task 2: Fix TMSink.increment_used_count (daemon→storage violation)

**Why:** `tm_sink.py:316-317` directly imports `get_session` + `ExperienceRepository`. Now that ExperienceService has `increment_used_count`, route through it.

**Files:**
- Modify: `src/team_memory/daemon/integration/tm_sink.py`

**Step 1: Write failing test**

```python
# tests/daemon/test_tm_sink.py
async def test_increment_used_count_calls_service_not_storage():
    """TMSink.increment_used_count must not import storage directly."""
    import ast, inspect
    import team_memory.daemon.integration.tm_sink as mod
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, 'module', '') or ''
            assert 'storage.database' not in name, \
                f"tm_sink imports storage directly: {name}"
```

**Step 2:** Run → expect FAIL (storage import found)

**Step 3: Replace implementation**

In `LocalTMSink.increment_used_count` (around line 143):

```python
async def increment_used_count(self, experience_id: str) -> None:
    """Increment used_count via ExperienceService (no direct DB access)."""
    _ensure_ops()
    ctx = _daemon_context()
    await ctx.service.increment_used_count(experience_id)
```

Remove the `get_session` + `ExperienceRepository` imports from this method.

**Step 4:** Run test → PASS. Also run `make lint` → 0 errors.

**Step 5: Commit**
```bash
git add src/team_memory/daemon/integration/tm_sink.py
git commit -m "fix(daemon): route TMSink.increment_used_count through ExperienceService"
```

---

## Task 3: Fix auto_dream.py (4 daemon→storage violations)

**Why:** auto_dream queries/updates Experience ORM directly. Now routes through ExperienceService.

**Files:**
- Modify: `src/team_memory/daemon/integration/auto_dream.py`

**The 4 violations and their replacements:**

| Method | Current | Replacement |
|--------|---------|-------------|
| `_get_dream_candidates()` | `get_session` + ORM query | `ctx.service.list_published(quality_tier="Bronze", since=last_dream)` |
| `_prune_stale()` | `get_session` + ORM delete | `ctx.service.list_published(quality_tier="Bronze")` + `ctx.service.soft_delete_by_id()` loop |
| `_soft_delete_experience()` | `get_session` + ORM update | `ctx.service.soft_delete_by_id(experience_id)` |
| `_count_new_experiences()` | `get_session` + ORM count | `ctx.service.count_published(since=last_dream)` |

**Step 1: Write failing test**

```python
# tests/daemon/test_auto_dream.py
def test_auto_dream_no_storage_imports():
    import ast, pathlib
    src = pathlib.Path("src/team_memory/daemon/integration/auto_dream.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("team_memory.storage"), \
                f"auto_dream imports storage: {node.module} (line {node.lineno})"
```

**Step 2:** Run → FAIL

**Step 3: Replace each method**

`_get_dream_candidates()` — replace body:
```python
async def _get_dream_candidates(self, last_dream=None) -> list[dict]:
    ctx = _daemon_context()
    return await ctx.service.list_published(
        quality_tier="Bronze",
        since=last_dream,
        limit=200,
    )
```

`_count_new_experiences()` — replace body:
```python
async def _count_new_experiences(self, last_dream=None) -> int:
    ctx = _daemon_context()
    return await ctx.service.count_published(since=last_dream)
```

`_soft_delete_experience()` — replace body:
```python
async def _soft_delete_experience(self, experience_id: str) -> None:
    ctx = _daemon_context()
    await ctx.service.soft_delete_by_id(experience_id)
```

`_prune_stale()` — replace body:
```python
async def _prune_stale(self) -> int:
    from datetime import timedelta
    cutoff = _utcnow() - timedelta(days=self._cfg.prune_max_age_days)
    ctx = _daemon_context()
    candidates = await ctx.service.list_published(quality_tier="Bronze")
    pruned = 0
    for exp in candidates:
        if (exp.get("recall_count", 1) == 0
                and not exp.get("is_pinned")
                and exp.get("created_at") < cutoff):
            await ctx.service.soft_delete_by_id(exp["id"])
            pruned += 1
    return pruned
```

Remove all `from team_memory.storage.*` imports from auto_dream.py.

**Step 4:** Run test → PASS. Run `make test -k auto_dream` → PASS.

**Step 5: Commit**
```bash
git add src/team_memory/daemon/integration/auto_dream.py
git commit -m "fix(daemon): remove auto_dream direct storage access, route through ExperienceService"
```

---

## Task 4: Create services/maintenance/batch_ops.py

**Why:** backfill scripts need bulk DB access without User auth. New `BatchOpsService` in the existing `services/maintenance/` package provides this cleanly.

**Files:**
- Create: `src/team_memory/services/maintenance/batch_ops.py`
- Modify: `src/team_memory/services/maintenance/__init__.py`
- Test: `tests/services/maintenance/test_batch_ops.py`

**Interface:**

```python
class BatchOpsService:
    """Internal-only bulk operations for backfill scripts.
    No user auth required — system-level access.
    """

    def __init__(self, db_url: str) -> None: ...

    async def list_experiences_missing_embedding(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]: ...

    async def list_archives_missing_embedding(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]: ...

    async def list_personal_memories_missing_embedding(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]: ...

    async def update_experience_embedding(
        self, experience_id: str, embedding: list[float]
    ) -> None: ...

    async def update_archive_embedding(
        self, archive_id: str, embedding: list[float]
    ) -> None: ...

    async def list_published_experiences(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]: ...

    async def list_published_archives(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]: ...

    async def list_entities_missing_embedding(
        self, limit: int = 50
    ) -> list[dict]: ...

    async def update_entity_embedding(
        self, entity_id: str, embedding: list[float]
    ) -> None: ...

    async def list_experiences_missing_solution(
        self, limit: int = 50
    ) -> list[dict]: ...

    async def update_experience_solution(
        self, experience_id: str, solution: str
    ) -> None: ...
```

**Step 1: Write failing tests**

```python
# tests/services/maintenance/test_batch_ops.py
async def test_list_experiences_missing_embedding_returns_list():
    svc = BatchOpsService(db_url=TEST_DB_URL)
    result = await svc.list_experiences_missing_embedding(limit=5)
    assert isinstance(result, list)
    for r in result:
        assert "id" in r
        assert r.get("embedding") is None
```

**Step 2:** Run → FAIL

**Step 3: Implement** using `get_session` + raw SQLAlchemy text queries (this layer IS allowed to use storage directly — it's in services/).

**Step 4:** Run tests → PASS

**Step 5: Commit**
```bash
git add src/team_memory/services/maintenance/batch_ops.py tests/services/maintenance/test_batch_ops.py
git commit -m "feat(services): add BatchOpsService for bulk backfill operations"
```

---

## Task 5: Migrate backfill scripts to BatchOpsService

**Why:** 5 backfill scripts directly import `storage.database`. Replace with `BatchOpsService`.

**Files:**
- Modify: `src/team_memory/daemon/backfill/embedding_backfill.py`
- Modify: `src/team_memory/daemon/backfill/graph_backfill.py`
- Modify: `src/team_memory/daemon/backfill/entity_backfill.py`
- Modify: `src/team_memory/daemon/backfill/entity_dedup.py`
- Modify: `src/team_memory/daemon/backfill/solution_backfill.py`

**Pattern for each script:**

```python
# BEFORE
from team_memory.storage.database import get_session
from team_memory.storage.models import Experience
async with get_session(db_url) as session:
    result = await session.execute(text("SELECT ..."))

# AFTER
from team_memory.services.maintenance.batch_ops import BatchOpsService
svc = BatchOpsService(db_url=db_url)
rows = await svc.list_experiences_missing_embedding(limit=chunk_size, cursor_id=cursor)
```

**Step 1: Write failing test (static import check)**

```python
# tests/daemon/backfill/test_no_storage_imports.py
import ast, pathlib

BACKFILL_DIR = pathlib.Path("src/team_memory/daemon/backfill")

def test_backfill_scripts_no_storage_imports():
    for py_file in BACKFILL_DIR.glob("*.py"):
        src = py_file.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("team_memory.storage"), \
                    f"{py_file.name} imports storage: {node.module} (line {node.lineno})"
```

**Step 2:** Run → FAIL (5 files violate)

**Step 3: Migrate each file**

For `embedding_backfill.py`: replace `get_session` queries with `BatchOpsService.list_experiences_missing_embedding()` and `update_experience_embedding()`.

For `graph_backfill.py`: replace `get_session` query with `BatchOpsService.list_published_experiences()` and `list_published_archives()`.

For `entity_backfill.py`: replace with `BatchOpsService.list_entities_missing_embedding()` + `update_entity_embedding()`.

For `entity_dedup.py`: replace with `BatchOpsService` queries for entity dedup operations.

For `solution_backfill.py`: replace with `BatchOpsService.list_experiences_missing_solution()` + `update_experience_solution()`.

**Step 4:** Run test → PASS. Run `python -m team_memory.daemon.backfill.embedding_backfill --dry-run` → works.

**Step 5: Commit**
```bash
git add src/team_memory/daemon/backfill/
git commit -m "fix(daemon): migrate all backfill scripts from storage to BatchOpsService"
```

---

## Task 6: Fix wiki_compiler.py and wiki_cli.py (daemon→storage violations)

**Why:** wiki_compiler loads experience→entity mapping directly from DB. wiki_cli queries Experience ORM.

**Files:**
- Modify: `src/team_memory/daemon/wiki/wiki_compiler.py`
- Modify: `src/team_memory/daemon/wiki/wiki_cli.py`

**wiki_compiler fix:**

The `_load_pg_data(pg_url)` method loads experience→entity mapping. Add to `BatchOpsService`:

```python
async def load_experience_entity_map(self) -> dict[str, list[str]]:
    """Returns {experience_id: [entity_names]} mapping for wiki compilation."""
```

Then in `wiki_compiler.py`:
```python
# BEFORE
from team_memory.storage.database import get_session
async with get_session(pg_url) as session:
    result = await session.execute(text("SELECT ..."))

# AFTER
from team_memory.services.maintenance.batch_ops import BatchOpsService
svc = BatchOpsService(db_url=self._pg_url)
self._exp_entity_map = await svc.load_experience_entity_map()
```

**wiki_cli fix:**

wiki_cli queries Experience for wiki generation. Add to `BatchOpsService`:
```python
async def list_experiences_for_wiki(self, project: str | None = None) -> list[dict]: ...
```

**Step 1: Write failing test**
```python
def test_wiki_compiler_no_storage_imports():
    import ast, pathlib
    for f in ["wiki_compiler.py", "wiki_cli.py"]:
        src = (pathlib.Path("src/team_memory/daemon/wiki") / f).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("team_memory.storage"), \
                    f"{f} imports storage: {node.module}"
```

**Step 2:** Run → FAIL

**Step 3:** Add methods to `BatchOpsService`, update wiki_compiler and wiki_cli.

**Step 4:** Run test → PASS

**Step 5: Commit**
```bash
git add src/team_memory/daemon/wiki/ src/team_memory/services/maintenance/batch_ops.py
git commit -m "fix(daemon): remove wiki_compiler/wiki_cli direct storage access"
```

---

## Task 7: Fix hooks/session_timeout.py (hooks→daemon violation)

**Why:** `SessionTimeoutManager` imports `DraftBuffer` and `DraftRefiner` from daemon internals. Hooks must only communicate with daemon via HTTP.

**Files:**
- Modify: `src/team_memory/hooks/session_timeout.py`

**New approach:** session_timeout calls `POST /hooks/session_end` on the daemon instead of directly calling DraftRefiner.

```python
# BEFORE
from team_memory.daemon.buffers.draft_buffer import DraftBuffer
from team_memory.daemon.refinement.draft_refiner import DraftRefiner

class SessionTimeoutManager:
    def __init__(self, draft_refiner: DraftRefiner, draft_buffer: DraftBuffer, ...):
        self._refiner = draft_refiner
        self._buffer = draft_buffer

    async def _check_timeouts(self):
        pending = await self._buffer.get_all_pending()
        ...
        await self._refiner.refine_and_publish(session_id)

# AFTER
from team_memory.hooks.core.daemon_http import post_hook

class SessionTimeoutManager:
    def __init__(self, daemon_url: str = "http://localhost:3901", timeout_minutes: int = 30, ...):
        self._daemon_url = daemon_url
        self._timeout = timedelta(minutes=timeout_minutes)

    async def _check_timeouts(self):
        # Query pending sessions via daemon HTTP
        resp = await post_hook("session_timeout_check", {})
        timed_out = resp.get("timed_out_sessions", [])
        for session_id in timed_out:
            await post_hook("session_end", {
                "conversation_id": session_id,
                "reason": "timeout",
            })
```

**Daemon side:** add `/hooks/session_timeout_check` endpoint that returns sessions pending > 30 min (reads from ConversationBuffer — already in daemon state).

**Step 1: Write failing test**
```python
def test_session_timeout_no_daemon_imports():
    import ast, pathlib
    src = pathlib.Path("src/team_memory/hooks/session_timeout.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("team_memory.daemon"), \
                f"session_timeout imports daemon: {node.module}"
```

**Step 2:** Run → FAIL

**Step 3:** Add `/hooks/session_timeout_check` to `daemon/runtime/app.py`, rewrite `SessionTimeoutManager` to use HTTP.

**Step 4:** Run test → PASS. Run existing session_timeout tests → PASS.

**Step 5: Commit**
```bash
git add src/team_memory/hooks/session_timeout.py src/team_memory/daemon/runtime/app.py
git commit -m "fix(hooks): remove session_timeout direct daemon imports, use HTTP"
```

---

## Task 8: Fix hooks/integrations git_watcher and initial_index (hooks→daemon violation)

**Why:** Both import `MarkdownIndexer` from `daemon.wiki` directly.

**Files:**
- Modify: `src/team_memory/hooks/integrations/git_watcher.py`
- Modify: `src/team_memory/hooks/integrations/initial_index.py`

**Fix:** Add a `/wiki/index` HTTP endpoint to the daemon. Hooks POST to it instead of importing MarkdownIndexer.

```python
# BEFORE
from team_memory.daemon.wiki.markdown_indexer import MarkdownIndexer
indexer = MarkdownIndexer(...)
await indexer.index_file(path)

# AFTER
from team_memory.hooks.core.daemon_http import post_hook
await post_hook("wiki_index", {"path": str(path), "project": project})
```

**Daemon side:** add `POST /wiki/index` endpoint that delegates to `app.state.wiki_compiler`.

**Step 1: Write failing test**
```python
def test_hooks_integrations_no_daemon_imports():
    import ast, pathlib
    for f in ["git_watcher.py", "initial_index.py"]:
        src = (pathlib.Path("src/team_memory/hooks/integrations") / f).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("team_memory.daemon"), \
                    f"{f} imports daemon: {node.module}"
```

**Step 2:** Run → FAIL

**Step 3:** Add `/wiki/index` endpoint to daemon, update both hook files.

**Step 4:** Run test → PASS

**Step 5: Commit**
```bash
git add src/team_memory/hooks/integrations/ src/team_memory/daemon/runtime/app.py
git commit -m "fix(hooks): remove git_watcher/initial_index direct daemon imports, use HTTP"
```

---

## Task 9: Add harness import check rules for new boundaries

**Why:** Without CI enforcement, violations will creep back in.

**Files:**
- Modify: `scripts/harness_import_check.py`

**Add rules:**
```python
# In LAYER_MAP or equivalent:
"daemon": {
    "forbidden_imports": ["team_memory.storage"],
    "message": "daemon/ must not import storage/ directly — use services/ layer",
},
"hooks": {
    "forbidden_imports": ["team_memory.daemon"],
    "message": "hooks/ must not import daemon/ directly — use HTTP (daemon_http.post_hook)",
},
```

**Step 1:** Run `make harness-check` → should now FAIL on any remaining violations.

**Step 2:** Fix any remaining violations found.

**Step 3:** Run `make harness-check` → PASS

**Step 4: Commit**
```bash
git add scripts/harness_import_check.py
git commit -m "chore(harness): enforce daemon→storage and hooks→daemon import boundaries"
```

---

## Task 10: Full verification

**Step 1:** Run full test suite
```bash
make verify
```
Expected: lint 0 errors, all tests pass.

**Step 2:** Run harness check
```bash
make harness-check
```
Expected: 0 violations.

**Step 3:** Verify no storage imports in daemon
```bash
grep -rn "from team_memory.storage" src/team_memory/daemon/ --include="*.py" | grep -v __pycache__
```
Expected: empty output.

**Step 4:** Verify no daemon imports in hooks
```bash
grep -rn "from team_memory.daemon" src/team_memory/hooks/ --include="*.py" | grep -v __pycache__
```
Expected: empty output.

**Step 5: Final commit**
```bash
git add -A
git commit -m "chore: layer boundary enforcement complete — 24 daemon→storage + 4 hooks→daemon violations resolved"
```

---

## Summary

| Task | What | Files Changed |
|------|------|---------------|
| 1 | Add 4 methods to ExperienceService | `services/memory/experience.py` |
| 2 | Fix TMSink.increment_used_count | `daemon/integration/tm_sink.py` |
| 3 | Fix auto_dream 4 DB calls | `daemon/integration/auto_dream.py` |
| 4 | Create BatchOpsService | `services/maintenance/batch_ops.py` |
| 5 | Migrate 5 backfill scripts | `daemon/backfill/*.py` |
| 6 | Fix wiki_compiler + wiki_cli | `daemon/wiki/*.py` |
| 7 | Fix session_timeout HTTP | `hooks/session_timeout.py` + daemon endpoint |
| 8 | Fix git_watcher + initial_index | `hooks/integrations/*.py` + daemon endpoint |
| 9 | Add harness rules | `scripts/harness_import_check.py` |
| 10 | Full verification | — |

**Total violations resolved:** 28 (24 daemon→storage + 4 hooks→daemon)
