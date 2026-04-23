# Phase 0: TMSink 抽象层

**Goal:** 创建 TMSink ABC + LocalTMSink + RemoteTMSink，统一本地/远端 TM 存储访问。

## Task 0-1: TMSink 抽象接口

**Files:**
- Create: `scripts/daemon/tm_sink.py`
- Test: `tests/test_tm_sink.py`

**Step 1: Write failing test for TMSink ABC**

```python
# tests/test_tm_sink.py
import pytest
from daemon.tm_sink import TMSink, LocalTMSink, RemoteTMSink

def test_tmsink_is_abstract():
    """TMSink cannot be instantiated directly."""
    with pytest.raises(TypeError):
        TMSink()

@pytest.mark.asyncio
async def test_local_sink_draft_save():
    sink = LocalTMSink()
    result = await sink.draft_save(
        title="test draft",
        content="test content",
        project="default",
    )
    assert "id" in result
    assert result["status"] == "draft"

@pytest.mark.asyncio
async def test_local_sink_draft_publish():
    sink = LocalTMSink()
    saved = await sink.draft_save(title="t", content="c", project="default")
    result = await sink.draft_publish(draft_id=saved["id"])
    assert result["status"] == "published"

@pytest.mark.asyncio
async def test_local_sink_recall():
    sink = LocalTMSink()
    result = await sink.recall(query="test", project="default")
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_local_sink_context():
    sink = LocalTMSink()
    result = await sink.context(project="default")
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_remote_sink_draft_save(httpx_mock):
    """RemoteTMSink calls HTTP endpoint."""
    httpx_mock.add_response(
        url="http://remote:3900/memory_draft_save",
        json={"id": "abc-123", "status": "draft"},
    )
    sink = RemoteTMSink(base_url="http://remote:3900")
    result = await sink.draft_save(title="t", content="c", project="default")
    assert result["id"] == "abc-123"

@pytest.mark.asyncio
async def test_remote_sink_recall(httpx_mock):
    httpx_mock.add_response(
        url="http://remote:3900/memory_recall",
        json={"results": []},
    )
    sink = RemoteTMSink(base_url="http://remote:3900")
    result = await sink.recall(query="test")
    assert "results" in result
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tm_sink.py -v`
Expected: FAIL — module not found

**Step 3: Implement TMSink**

```python
# scripts/daemon/tm_sink.py
"""TMSink — abstract interface for TM storage access.

LocalTMSink: direct Python import of team_memory.services.memory_operations
RemoteTMSink: HTTP client to remote TM server
"""
from __future__ import annotations

import abc
from typing import Any

import httpx


class TMSink(abc.ABC):
    """Abstract interface for TeamMemory storage operations."""

    @abc.abstractmethod
    async def draft_save(
        self, *, title: str, content: str, project: str | None = None,
        group_key: str | None = None, conversation_id: str | None = None,
    ) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def draft_publish(
        self, *, draft_id: str, refined_content: str | None = None,
    ) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def save(
        self, *, title: str, content: str, tags: list[str] | None = None,
        project: str | None = None, source: str | None = None,
        group_key: str | None = None,
    ) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def recall(
        self, *, query: str, project: str | None = None, max_results: int = 5,
    ) -> list[dict[str, Any]]:
        ...

    @abc.abstractmethod
    async def context(
        self, *, project: str | None = None,
    ) -> dict[str, Any]:
        ...


class LocalTMSink(TMSink):
    """Direct Python import — calls memory_operations functions."""

    def __init__(self, user: str = "daemon") -> None:
        self._user = user
        from team_memory.services.memory_operations import (
            op_context, op_draft_publish, op_draft_save, op_recall, op_save,
        )
        self._op_draft_save = op_draft_save
        self._op_draft_publish = op_draft_publish
        self._op_save = op_save
        self._op_recall = op_recall
        self._op_context = op_context

    async def draft_save(self, *, title, content, project=None, group_key=None, conversation_id=None):
        return await self._op_draft_save(
            self._user, title=title, content=content, project=project,
            group_key=group_key, conversation_id=conversation_id,
        )

    async def draft_publish(self, *, draft_id, refined_content=None):
        return await self._op_draft_publish(
            self._user, draft_id=draft_id, refined_content=refined_content,
        )

    async def save(self, *, title, content, tags=None, project=None, source=None, group_key=None):
        return await self._op_save(
            self._user, title=title, content=content, tags=tags,
            project=project, source=source, group_key=group_key,
        )

    async def recall(self, *, query, project=None, max_results=5):
        raw = await self._op_recall(
            self._user, query=query, project=project, max_results=max_results,
        )
        # op_recall returns JSON string or dict — normalize
        if isinstance(raw, str):
            import json
            raw = json.loads(raw)
        return raw.get("results", []) if isinstance(raw, dict) else raw

    async def context(self, *, project=None):
        return await self._op_context(self._user, project=project)


class RemoteTMSink(TMSink):
    """HTTP client — calls remote TM server."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def _call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{tool_name}"
        payload = {"arguments": arguments}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def draft_save(self, *, title, content, project=None, group_key=None, conversation_id=None):
        args = {"title": title, "content": content}
        if project: args["project"] = project
        if group_key: args["group_key"] = group_key
        if conversation_id: args["conversation_id"] = conversation_id
        return await self._call("memory_draft_save", args)

    async def draft_publish(self, *, draft_id, refined_content=None):
        args = {"draft_id": draft_id}
        if refined_content: args["refined_content"] = refined_content
        return await self._call("memory_draft_publish", args)

    async def save(self, *, title, content, tags=None, project=None, source=None, group_key=None):
        args = {"title": title, "content": content}
        if tags: args["tags"] = tags
        if project: args["project"] = project
        if source: args["source"] = source
        if group_key: args["group_key"] = group_key
        return await self._call("memory_save", args)

    async def recall(self, *, query, project=None, max_results=5):
        args = {"query": query, "max_results": max_results}
        if project: args["project"] = project
        raw = await self._call("memory_recall", args)
        return raw.get("results", []) if isinstance(raw, dict) else raw

    async def context(self, *, project=None):
        args = {}
        if project: args["project"] = project
        return await self._call("memory_context", args)


def create_sink(config: dict) -> TMSink:
    """Factory: create LocalTMSink or RemoteTMSink from config dict."""
    tm_config = config.get("tm", {})
    mode = tm_config.get("mode", "local")
    if mode == "remote":
        return RemoteTMSink(base_url=tm_config.get("base_url", "http://localhost:3900"))
    return LocalTMSink()
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_tm_sink.py -v`
Expected: LocalTMSink tests pass (need TM DB bootstrap); RemoteTMSink tests need httpx mock

**Step 5: Commit**

```bash
git add scripts/daemon/tm_sink.py tests/test_tm_sink.py
git commit -m "feat(v2): TMSink abstraction — LocalTMSink + RemoteTMSink"
```

---

## Task 0-2: Daemon 目录结构 + 配置加载

**Files:**
- Create: `scripts/daemon/__init__.py`
- Create: `scripts/daemon/config.py`
- Modify: `scripts/hooks/config.yaml` — 加 daemon/tm/obsidian 配置
- Test: `tests/test_daemon_config.py`

**Step 1: Write failing test**

```python
# tests/test_daemon_config.py
from daemon.config import DaemonConfig, load_config

def test_load_config_defaults():
    cfg = load_config()
    assert cfg.daemon.host == "127.0.0.1"
    assert cfg.daemon.port == 3901

def test_load_config_tm_local():
    cfg = load_config()
    assert cfg.tm.mode == "local"

def test_load_config_obsidian_vaults():
    cfg = load_config()
    assert len(cfg.obsidian.vaults) >= 0
```

**Step 2: Implement config**

```python
# scripts/daemon/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DaemonSettings:
    host: str = "127.0.0.1"
    port: int = 3901


@dataclass
class TMSettings:
    mode: str = "local"  # local | remote
    base_url: str = "http://localhost:3900"
    user: str = "daemon"


@dataclass
class VaultConfig:
    path: str = ""
    project: str = ""
    exclude: list[str] = field(default_factory=lambda: [".obsidian", ".trash"])


@dataclass
class ObsidianSettings:
    vaults: list[VaultConfig] = field(default_factory=list)


@dataclass
class DraftSettings:
    max_age_minutes: int = 30
    db_path: str = "~/.cache/tm-pipeline/drafts.db"


@dataclass
class RetrievalSettings:
    session_start_top_k: int = 3
    keyword_triggers: list[str] = field(default_factory=lambda: ["之前", "上次", "经验", "踩坑", "遇到过"])


@dataclass
class ProjectMapping:
    name: str = ""
    path_patterns: list[str] = field(default_factory=list)


@dataclass
class DaemonConfig:
    daemon: DaemonSettings = field(default_factory=DaemonSettings)
    tm: TMSettings = field(default_factory=TMSettings)
    obsidian: ObsidianSettings = field(default_factory=ObsidianSettings)
    draft: DraftSettings = field(default_factory=DraftSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    projects: list[ProjectMapping] = field(default_factory=list)


def load_config(config_path: str | None = None) -> DaemonConfig:
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "hooks" / "config.yaml")
    p = Path(config_path).expanduser()
    if not p.exists():
        return DaemonConfig()
    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    daemon_raw = raw.get("daemon", {})
    tm_raw = raw.get("tm", {})
    obs_raw = raw.get("obsidian", {})
    draft_raw = raw.get("draft", {})
    ret_raw = raw.get("retrieval", {})
    proj_raw = raw.get("projects", {})

    vaults = []
    for v in obs_raw.get("vaults", []):
        vaults.append(VaultConfig(
            path=v.get("path", ""),
            project=v.get("project", ""),
            exclude=v.get("exclude", [".obsidian", ".trash"]),
        ))

    projects = []
    for name, patterns in proj_raw.items():
        if isinstance(patterns, dict):
            patterns = patterns.get("path_patterns", [])
        projects.append(ProjectMapping(name=name, path_patterns=patterns if isinstance(patterns, list) else [patterns]))

    return DaemonConfig(
        daemon=DaemonSettings(
            host=daemon_raw.get("host", "127.0.0.1"),
            port=daemon_raw.get("port", 3901),
        ),
        tm=TMSettings(
            mode=tm_raw.get("mode", "local"),
            base_url=tm_raw.get("base_url", "http://localhost:3900"),
            user=tm_raw.get("user", "daemon"),
        ),
        obsidian=ObsidianSettings(vaults=vaults),
        draft=DraftSettings(
            max_age_minutes=draft_raw.get("max_age_minutes", 30),
            db_path=draft_raw.get("db_path", "~/.cache/tm-pipeline/drafts.db"),
        ),
        retrieval=RetrievalSettings(
            session_start_top_k=ret_raw.get("session_start_top_k", 3),
            keyword_triggers=ret_raw.get("keyword_triggers", ["之前", "上次", "经验", "踩坑", "遇到过"]),
        ),
        projects=projects,
    )
```

**Step 3: Update config.yaml**

Add to `scripts/hooks/config.yaml`:
```yaml
daemon:
  host: "127.0.0.1"
  port: 3901

tm:
  mode: "local"          # local | remote
  base_url: "http://localhost:3900"
  user: "daemon"

obsidian:
  vaults:
    - path: ""           # TODO: fill in Obsidian vault path
      project: "knowledge"
      exclude: [".obsidian", ".trash"]
```

**Step 4: Run tests + commit**

Run: `.venv/bin/python -m pytest tests/test_daemon_config.py -v`
Commit: `feat(v2): daemon config module`
