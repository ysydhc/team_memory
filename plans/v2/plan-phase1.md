# Phase 1: Daemon HTTP API + 内部模块迁移

**Goal:** 创建 daemon 主进程，暴露 HTTP API，将现有 hook 逻辑迁入 daemon 内部模块。

## Task 1-1: Daemon 主进程骨架

**Files:**
- Create: `scripts/daemon/app.py` — FastAPI app + 生命周期管理
- Create: `scripts/daemon/__main__.py` — 入口
- Test: `tests/test_daemon_app.py`

**Step 1: Write failing test**

```python
# tests/test_daemon_app.py
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.fixture
async def client():
    from daemon.app import create_app
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_status_endpoint(client):
    resp = await client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"

@pytest.mark.asyncio
async def test_hooks_after_response(client):
    resp = await client.post("/hooks/after_response", json={
        "conversation_id": "test-123",
        "prompt": "some agent response text",
        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] in ("draft_saved", "published", "ok")

@pytest.mark.asyncio
async def test_hooks_session_start(client):
    resp = await client.post("/hooks/session_start", json={
        "conversation_id": "test-456",
        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
    })
    assert resp.status_code == 200

@pytest.mark.asyncio
async def test_hooks_before_prompt(client):
    resp = await client.post("/hooks/before_prompt", json={
        "conversation_id": "test-789",
        "prompt": "之前遇到的问题怎么解决的",
        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
    })
    assert resp.status_code == 200

@pytest.mark.asyncio
async def test_recall_endpoint(client):
    resp = await client.get("/recall", params={"query": "test", "project": "default"})
    assert resp.status_code == 200
```

**Step 2: Implement daemon app**

```python
# scripts/daemon/app.py
"""TM Daemon — local HTTP API server for hook events + Obsidian watching."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from daemon.config import DaemonConfig, load_config
from daemon.tm_sink import TMSink, create_sink

logger = logging.getLogger("tm_daemon")

# ---------------------------------------------------------------------------
# Global state (initialized in lifespan)
# ---------------------------------------------------------------------------
_config: DaemonConfig | None = None
_sink: TMSink | None = None
_draft_buffer = None
_convergence_detector = None
_draft_refiner = None
_session_timeout = None
_watcher_task: asyncio.Task | None = None


class HookPayload(BaseModel):
    conversation_id: str = ""
    prompt: str = ""
    workspace_roots: list[str] = []
    model: str = ""


class DraftSavePayload(BaseModel):
    title: str
    content: str
    project: str | None = None
    group_key: str | None = None
    conversation_id: str | None = None


class DraftPublishPayload(BaseModel):
    draft_id: str
    refined_content: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init TMSink, DraftBuffer, FileWatcher. Shutdown: cleanup."""
    global _config, _sink, _draft_buffer, _convergence_detector, _draft_refiner, _session_timeout, _watcher_task

    _config = load_config()
    _sink = create_sink({"tm": _config.tm})

    # Import and init internal modules
    from daemon.draft_buffer import DraftBuffer
    from daemon.convergence_detector import ConvergenceDetector
    from daemon.draft_refiner import DraftRefiner

    db_path = _config.draft.db_path
    _draft_buffer = DraftBuffer(db_path)
    await _draft_buffer.__aenter__()

    _convergence_detector = ConvergenceDetector(
        keywords=_config.retrieval.keyword_triggers,
    )
    _draft_refiner = DraftRefiner(_sink, _draft_buffer)

    # Start Obsidian file watcher
    if _config.obsidian.vaults:
        from daemon.watcher import start_watcher
        _watcher_task = asyncio.create_task(start_watcher(_config, _sink))

    logger.info("TM Daemon started on %s:%d", _config.daemon.host, _config.daemon.port)
    yield

    # Shutdown
    if _watcher_task:
        _watcher_task.cancel()
    await _draft_buffer.__aexit__(None, None, None)
    logger.info("TM Daemon stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="TM Daemon", lifespan=lifespan)

    @app.get("/status")
    async def status():
        return {"status": "running", "tm_mode": _config.tm.mode if _config else "unknown"}

    @app.post("/hooks/after_response")
    async def hooks_after_response(payload: HookPayload):
        """Process agent response: accumulate draft, detect convergence, publish."""
        from daemon.pipeline import process_after_response
        result = await process_after_response(
            payload.dict(), _config, _sink, _draft_buffer,
            _convergence_detector, _draft_refiner,
        )
        return result

    @app.post("/hooks/session_start")
    async def hooks_session_start(payload: HookPayload):
        """Inject project context for new session."""
        from daemon.pipeline import process_session_start
        result = await process_session_start(payload.dict(), _config, _sink)
        return result

    @app.post("/hooks/before_prompt")
    async def hooks_before_prompt(payload: HookPayload):
        """Retrieve relevant memories for user prompt."""
        from daemon.pipeline import process_before_prompt
        result = await process_before_prompt(payload.dict(), _config, _sink)
        return result

    @app.post("/hooks/session_end")
    async def hooks_session_end(payload: HookPayload):
        """Flush remaining drafts on session end."""
        from daemon.pipeline import process_session_end
        result = await process_session_end(
            payload.dict(), _config, _sink, _draft_buffer, _draft_refiner,
        )
        return result

    @app.post("/draft/save")
    async def draft_save(payload: DraftSavePayload):
        result = await _sink.draft_save(
            title=payload.title, content=payload.content,
            project=payload.project, group_key=payload.group_key,
            conversation_id=payload.conversation_id,
        )
        return result

    @app.post("/draft/publish")
    async def draft_publish(payload: DraftPublishPayload):
        result = await _sink.draft_publish(
            draft_id=payload.draft_id, refined_content=payload.refined_content,
        )
        return result

    @app.get("/recall")
    async def recall(query: str = "", project: str | None = None, max_results: int = 5):
        results = await _sink.recall(query=query, project=project, max_results=max_results)
        return {"results": results}

    return app
```

```python
# scripts/daemon/__main__.py
"""Entry point: python -m daemon"""
from daemon.app import create_app
from daemon.config import load_config
import uvicorn

def main():
    config = load_config()
    app = create_app()
    uvicorn.run(app, host=config.daemon.host, port=config.daemon.port)

if __name__ == "__main__":
    main()
```

**Step 3: Run tests + commit**

---

## Task 1-2: Pipeline 逻辑迁移

**Files:**
- Create: `scripts/daemon/pipeline.py` — 从 hooks 脚本提取的管线逻辑
- Migrate: draft_buffer.py → `scripts/daemon/draft_buffer.py`
- Migrate: convergence_detector.py → `scripts/daemon/convergence_detector.py`
- Migrate: draft_refiner.py → `scripts/daemon/draft_refiner.py` (TMClient → TMSink)
- Test: `tests/test_daemon_pipeline.py`

核心变更：DraftRefiner 不再持有 TMClient，改为持有 TMSink。

```python
# scripts/daemon/pipeline.py
"""Pipeline logic — extracted from hook scripts, runs inside daemon."""
from __future__ import annotations
import logging
from typing import Any

from daemon.config import DaemonConfig
from daemon.tm_sink import TMSink

logger = logging.getLogger("tm_daemon.pipeline")


def _resolve_project(workspace_roots: list[str], config: DaemonConfig) -> str | None:
    for root in workspace_roots:
        for pm in config.projects:
            for pattern in pm.path_patterns:
                if pattern in root:
                    return pm.name
    return None


async def process_after_response(
    input_data: dict, config: DaemonConfig, sink: TMSink,
    buf, detector, refiner,
) -> dict[str, Any]:
    session_id = input_data.get("conversation_id", "") or "unknown"
    response_text = input_data.get("prompt", "") or ""
    workspace_roots = input_data.get("workspace_roots", [])
    project = _resolve_project(workspace_roots, config)

    if project is None:
        return {"action": "ok", "convergence": False, "draft_id": ""}

    # Check for existing pending draft
    existing = await buf.get_pending(session_id)

    # Accumulate text
    accumulated = (existing[0]["content"] + "\n" + response_text) if existing else response_text

    # Detect convergence
    converged = detector.detect(response_text, current_path=project)

    if converged and existing:
        draft_id = existing[0]["id"]
        await buf.update_draft(draft_id, accumulated)
        result = await refiner.refine_and_publish(session_id)
        if result:
            return {"action": "published", "convergence": True, "draft_id": result.get("draft_id", draft_id)}
        return {"action": "ok", "convergence": True, "draft_id": draft_id}
    elif converged and not existing:
        title = f"Session {session_id[:8]} draft"
        tm_resp = await refiner.save_draft(session_id, title, accumulated, project=project)
        result = await refiner.refine_and_publish(session_id)
        if result:
            return {"action": "published", "convergence": True, "draft_id": result.get("draft_id", tm_resp.get("id", ""))}
        return {"action": "draft_saved", "convergence": True, "draft_id": tm_resp.get("id", "")}
    else:
        title = f"Session {session_id[:8]} draft"
        tm_resp = await refiner.save_draft(session_id, title, accumulated, project=project)
        return {"action": "draft_saved", "convergence": False, "draft_id": tm_resp.get("id", "")}


async def process_session_start(
    input_data: dict, config: DaemonConfig, sink: TMSink,
) -> dict[str, Any]:
    workspace_roots = input_data.get("workspace_roots", [])
    project = _resolve_project(workspace_roots, config)
    if project is None:
        return {"additional_context": "", "project": None}

    result = await sink.context(project=project)
    return {"additional_context": result, "project": project}


async def process_before_prompt(
    input_data: dict, config: DaemonConfig, sink: TMSink,
) -> dict[str, Any]:
    query = input_data.get("prompt", "")
    workspace_roots = input_data.get("workspace_roots", [])
    project = _resolve_project(workspace_roots, config)

    results = await sink.recall(query=query, project=project, max_results=config.retrieval.session_start_top_k)
    return {"results": results, "project": project}


async def process_session_end(
    input_data: dict, config: DaemonConfig, sink: TMSink,
    buf, refiner,
) -> dict[str, Any]:
    session_id = input_data.get("conversation_id", "") or "unknown"
    existing = await buf.get_pending(session_id)
    if not existing:
        return {"action": "ok", "flushed": False}

    result = await refiner.refine_and_publish(session_id)
    if result:
        return {"action": "published", "flushed": True, "draft_id": result.get("draft_id", "")}
    return {"action": "ok", "flushed": False}
```

**Step 3: Migrate internal modules**

Copy `scripts/hooks/draft_buffer.py` → `scripts/daemon/draft_buffer.py` (no changes needed — pure SQLite, no TMClient dependency)

Copy `scripts/hooks/convergence_detector.py` → `scripts/daemon/convergence_detector.py` (no changes needed)

Copy `scripts/hooks/draft_refiner.py` → `scripts/daemon/draft_refiner.py`, change TMClient → TMSink:
- Constructor: `def __init__(self, sink: TMSink, buf: DraftBuffer)`
- `save_draft`: call `self._sink.draft_save(...)` instead of `self._tm.draft_save(...)`
- `refine_and_publish`: call `self._sink.draft_publish(...)` instead of `self._tm.draft_publish(...)`

**Step 4: Run tests + commit**

---

## Task 1-3: File Watcher (Obsidian)

**Files:**
- Create: `scripts/daemon/watcher.py`
- Test: `tests/test_daemon_watcher.py`

```python
# scripts/daemon/watcher.py
"""Obsidian vault file watcher using watchfiles."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from watchfiles import awatch, Change

from daemon.config import DaemonConfig
from daemon.markdown_indexer import MarkdownIndexer
from daemon.tm_sink import TMSink

logger = logging.getLogger("tm_daemon.watcher")


async def start_watcher(config: DaemonConfig, sink: TMSink) -> None:
    """Watch all configured Obsidian vaults and index changes."""
    indexer = MarkdownIndexer()
    paths = []
    for vault in config.obsidian.vaults:
        p = Path(vault.path).expanduser()
        if p.exists():
            paths.append(p)
            logger.info("Watching vault: %s (project=%s)", p, vault.project)

    if not paths:
        logger.warning("No Obsidian vaults configured or paths don't exist")
        return

    try:
        async for changes in awatch(*paths, debounce=2000):
            await _process_changes(changes, config, indexer, sink)
    except asyncio.CancelledError:
        logger.info("File watcher stopped")
    except Exception:
        logger.exception("File watcher error")


async def _process_changes(
    changes: set[tuple[Change, str]],
    config: DaemonConfig,
    indexer: MarkdownIndexer,
    sink: TMSink,
) -> None:
    """Process a batch of file changes."""
    for change_type, path_str in changes:
        path = Path(path_str)

        # Only process .md files
        if path.suffix != ".md":
            continue

        # Find matching vault config
        vault = _find_vault(path, config)
        if vault is None:
            continue

        # Check exclude patterns
        if any(exc in path_str for exc in vault.exclude):
            continue

        if change_type in (Change.added, Change.modified):
            try:
                entries = indexer.index_file(path, project=vault.project)
                for entry in entries:
                    await sink.save(
                        title=entry["title"],
                        content=entry["content"],
                        tags=entry.get("tags"),
                        project=vault.project,
                        source="obsidian",
                        group_key=entry.get("group_key"),
                    )
                logger.info("Indexed: %s (%d entries)", path.name, len(entries))
            except Exception:
                logger.exception("Failed to index: %s", path)

        elif change_type == Change.deleted:
            logger.info("Deleted: %s (no-op, TM has no delete API yet)", path.name)


def _find_vault(path: Path, config: DaemonConfig):
    """Find matching VaultConfig for a given file path."""
    for vault in config.obsidian.vaults:
        vault_path = Path(vault.path).expanduser()
        try:
            path.relative_to(vault_path)
            return vault
        except ValueError:
            continue
    return None
```

**Step 4: Run tests + commit**
