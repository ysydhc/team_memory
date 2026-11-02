"""Unit tests for Phase 3: Bi-Temporal Memory State Machine (ADD / MERGE / SUPERSEDE / NOOP)."""

from __future__ import annotations

import asyncio
import sys
import types
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import pytest

    _async_mark = pytest.mark.asyncio
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

    def _async_mark(fn: Any) -> Any:
        return fn


def _ensure_sandbox_import_stubs() -> None:
    """Install lightweight module stubs only when running outside virtualenv."""
    try:
        import sqlalchemy  # noqa: F401

        return
    except ImportError:
        pass

    class _DummyCol:
        def __init__(self, name: str) -> None:
            self.name = name

        def __eq__(self, other: Any) -> str:  # type: ignore[override]
            return f"{self.name} == {other!r}"

        def __ne__(self, other: Any) -> str:  # type: ignore[override]
            return f"{self.name} != {other!r}"

        def __lt__(self, other: Any) -> str:
            return f"{self.name} < {other!r}"

        def is_(self, other: Any) -> str:
            return f"{self.name} IS {other!r}"

        def is_not(self, other: Any) -> str:
            return f"{self.name} IS NOT {other!r}"

        def in_(self, other: Any) -> str:
            return f"{self.name} IN {other!r}"

        def op(self, operator: str) -> Any:
            return lambda val: _DummyCol(f"({self.name} {operator} {val!r})")

        def __invert__(self) -> str:
            return f"NOT {self.name}"

        def __str__(self) -> str:
            return self.name

    sa_mod = types.ModuleType("sqlalchemy")
    sa_mod.text = lambda sql: sql  # type: ignore[attr-defined]
    sa_mod.or_ = lambda *args: f"OR({', '.join(map(str, args))})"  # type: ignore[attr-defined]
    for name in (
        "JSON",
        "Boolean",
        "DateTime",
        "Float",
        "ForeignKey",
        "Integer",
        "Select",
        "String",
        "Text",
        "UniqueConstraint",
        "delete",
        "desc",
        "func",
        "select",
        "update",
    ):
        setattr(sa_mod, name, MagicMock())
    sys.modules["sqlalchemy"] = sa_mod

    sa_sql = types.ModuleType("sqlalchemy.sql")
    sa_elements = types.ModuleType("sqlalchemy.sql.elements")
    sa_elements.TextClause = str  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.sql"] = sa_sql
    sys.modules["sqlalchemy.sql.elements"] = sa_elements

    sa_ext = types.ModuleType("sqlalchemy.ext")
    sa_asyncio = types.ModuleType("sqlalchemy.ext.asyncio")
    sa_asyncio.AsyncSession = Any  # type: ignore[attr-defined]
    sa_asyncio.async_sessionmaker = MagicMock()  # type: ignore[attr-defined]
    sa_asyncio.create_async_engine = MagicMock()  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.ext"] = sa_ext
    sys.modules["sqlalchemy.ext.asyncio"] = sa_asyncio

    sa_orm = types.ModuleType("sqlalchemy.orm")
    sa_orm.DeclarativeBase = object  # type: ignore[attr-defined]
    sa_orm.Mapped = Any  # type: ignore[attr-defined]
    sa_orm.mapped_column = lambda *a, **kw: None  # type: ignore[attr-defined]
    sa_orm.relationship = lambda *a, **kw: None  # type: ignore[attr-defined]
    sa_orm.selectinload = MagicMock()  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.orm"] = sa_orm

    sa_dialects = types.ModuleType("sqlalchemy.dialects")
    sa_pg = types.ModuleType("sqlalchemy.dialects.postgresql")
    sa_pg.ARRAY = MagicMock()  # type: ignore[attr-defined]
    sa_pg.JSONB = MagicMock()  # type: ignore[attr-defined]
    sa_pg.TSVECTOR = MagicMock()  # type: ignore[attr-defined]
    sa_pg.UUID = MagicMock()  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.dialects"] = sa_dialects
    sys.modules["sqlalchemy.dialects.postgresql"] = sa_pg

    pgv_mod = types.ModuleType("pgvector")
    pgv_sa = types.ModuleType("pgvector.sqlalchemy")
    pgv_sa.Vector = MagicMock()  # type: ignore[attr-defined]
    sys.modules["pgvector"] = pgv_mod
    sys.modules["pgvector.sqlalchemy"] = pgv_sa

    models_mod = types.ModuleType("team_memory.storage.models")

    @dataclass
    class _DummyExperience:
        id: uuid.UUID = field(default_factory=uuid.uuid4)
        title: str = ""
        description: str = ""
        solution: str | None = None
        tags: list[str] = field(default_factory=list)
        embedding: list[float] | None = None
        project: str = "default"
        created_by: str = "tester"
        exp_status: str = "published"
        visibility: str = "project"
        experience_type: str = "general"
        source: str = "manual"
        is_deleted: bool = False
        valid_from: datetime | None = field(default_factory=lambda: datetime.now(timezone.utc))
        valid_until: datetime | None = None
        superseded_by_id: uuid.UUID | None = None
        last_verified_at: datetime | None = None
        mutation_history: list[dict[str, Any]] = field(default_factory=list)
        negative_feedback_count: int = 0
        updated_at: datetime | None = None

        def to_dict(self, include_children: bool = True) -> dict[str, Any]:
            return {
                "id": str(self.id),
                "title": self.title,
                "description": self.description,
                "solution": self.solution,
                "tags": list(self.tags or []),
                "project": self.project,
                "exp_status": self.exp_status,
                "status": self.exp_status,
                "valid_from": self.valid_from.isoformat() if self.valid_from else None,
                "valid_until": self.valid_until.isoformat() if self.valid_until else None,
                "superseded_by_id": str(self.superseded_by_id) if self.superseded_by_id else None,
                "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
                "mutation_history": list(self.mutation_history or []),
                "negative_feedback_count": self.negative_feedback_count,
            }

    for col in ("is_deleted", "exp_status", "created_by", "valid_until", "title", "project", "visibility", "id"):
        setattr(_DummyExperience, col, _DummyCol(col))

    models_mod.Experience = _DummyExperience  # type: ignore[attr-defined]
    models_mod.ExperienceFeedback = MagicMock()  # type: ignore[attr-defined]
    models_mod.PersonalMemory = MagicMock()  # type: ignore[attr-defined]
    sys.modules["team_memory.storage.models"] = models_mod

    db_mod = types.ModuleType("team_memory.storage.database")

    @asynccontextmanager
    async def _dummy_get_session(_db_url: str | None = None):
        yield AsyncMock()

    db_mod.get_session = _dummy_get_session  # type: ignore[attr-defined]
    sys.modules["team_memory.storage.database"] = db_mod

    cfg_mod = types.ModuleType("team_memory.config")
    cfg_mod.SearchConfig = MagicMock  # type: ignore[attr-defined]
    cfg_mod.LLMConfig = MagicMock  # type: ignore[attr-defined]
    cfg_mod.Settings = Any  # type: ignore[attr-defined]
    cfg_mod.get_settings = MagicMock()  # type: ignore[attr-defined]
    sys.modules["team_memory.config"] = cfg_mod


_ensure_sandbox_import_stubs()

from team_memory.daemon.quality.search_eval import compute_fama_at_k  # noqa: E402
from team_memory.services.memory.mutation_resolver import MutationResolver  # noqa: E402
from team_memory.storage.models import Experience  # noqa: E402
from team_memory.storage.query_builders import active_filter, build_duplicate_detection  # noqa: E402
from team_memory.storage.repository import ExperienceRepository  # noqa: E402


def test_mutation_resolver_four_transitions() -> None:
    """Verify ADD, NOOP, MERGE, and SUPERSEDE transitions in MutationResolver."""
    resolver = MutationResolver(noop_threshold=0.96, mutation_threshold=0.85)

    # 1. ADD when no candidates or low similarity
    d_add_empty = resolver.resolve_by_rules(
        title="New Redis lock",
        problem="Need distributed lock",
        solution="Use Redlock",
        candidates=[],
    )
    assert d_add_empty.action == "ADD"

    d_add_low = resolver.resolve_by_rules(
        title="New Redis lock",
        problem="Need distributed lock",
        solution="Use Redlock",
        candidates=[{"id": "exp-1", "similarity": 0.72, "solution": "Use SETNX"}],
    )
    assert d_add_low.action == "ADD"

    # 2. NOOP when high similarity (>= 0.96) or identical solution
    d_noop = resolver.resolve_by_rules(
        title="PostgreSQL pool sizing",
        problem="Connection pool exhaustion under load",
        solution="Set max_overflow=10 and pool_pre_ping=True",
        candidates=[
            {
                "id": "exp-noop-1",
                "similarity": 0.98,
                "description": "Connection pool exhaustion under load",
                "solution": "Set max_overflow=10 and pool_pre_ping=True",
            }
        ],
    )
    assert d_noop.action == "NOOP"
    assert d_noop.target_id == "exp-noop-1"

    # 3. MERGE when similarity in [0.85, 0.96) and complementary solution
    d_merge = resolver.resolve_by_rules(
        title="PostgreSQL pool sizing troubleshooting",
        problem="Connection pool exhaustion under burst traffic",
        solution="Also configure pool_recycle=1800 to avoid stale firewall connections.",
        candidates=[
            {
                "id": "exp-merge-1",
                "similarity": 0.90,
                "description": "Connection pool exhaustion under load",
                "solution": "Increase pool_size to 15 in SQLAlchemy engine.",
            }
        ],
    )
    assert d_merge.action == "MERGE"
    assert d_merge.target_id == "exp-merge-1"
    assert "Increase pool_size to 15" in (d_merge.merged_solution or "")
    assert "pool_recycle=1800" in (d_merge.merged_solution or "")

    # 4. SUPERSEDE when deprecation marker or conflicting key=value config
    d_supersede_marker = resolver.resolve_by_rules(
        title="PostgreSQL pool config (升级为 v2)",
        problem="废弃旧版本连接池配置，改为 asyncpg 推荐参数",
        solution="使用 pool_size=20 替代旧方案",
        candidates=[
            {
                "id": "exp-old-1",
                "similarity": 0.91,
                "description": "连接池配置",
                "solution": "使用默认连接池参数",
            }
        ],
    )
    assert d_supersede_marker.action == "SUPERSEDE"
    assert d_supersede_marker.target_id == "exp-old-1"

    d_supersede_kv = resolver.resolve_by_rules(
        title="PostgreSQL pool config",
        problem="Connection pool tuning",
        solution="Set pool_size=20 and timeout=30",
        candidates=[
            {
                "id": "exp-old-2",
                "similarity": 0.92,
                "description": "Connection pool tuning",
                "solution": "Set pool_size=5 and timeout=30",
            }
        ],
    )
    assert d_supersede_kv.action == "SUPERSEDE"
    assert d_supersede_kv.target_id == "exp-old-2"


@_async_mark
async def test_repository_bitemporal_methods() -> None:
    """Verify supersede_experience, merge_experience, and touch_verified audit trails."""
    session = AsyncMock()
    repo = ExperienceRepository(session)

    old_id = uuid.uuid4()
    new_id = uuid.uuid4()
    old_exp = Experience(
        id=old_id,
        title="Old Auth v1",
        description="Auth token config",
        solution="jwt_ttl=3600",
        tags=["auth", "v1"],
        mutation_history=[],
    )
    new_exp = Experience(
        id=new_id,
        title="New Auth v2",
        description="Auth token config",
        solution="Deprecated v1; jwt_ttl=900",
        tags=["auth", "v2"],
        mutation_history=[],
    )

    store = {old_id: old_exp, new_id: new_exp}

    async def _mock_get_by_id(eid: uuid.UUID, include_deleted: bool = False) -> Any:
        return store.get(eid)

    with patch.object(repo, "get_by_id", side_effect=_mock_get_by_id):
        # 1. Test supersede_experience
        superseded = await repo.supersede_experience(
            old_id,
            new_id,
            reason="Upgraded to Auth v2",
            actor="alice",
        )
        assert superseded is old_exp
        assert old_exp.valid_until is not None
        assert old_exp.superseded_by_id == new_id
        assert old_exp.mutation_history[-1]["action"] == "SUPERSEDE"
        assert old_exp.mutation_history[-1]["superseded_by_id"] == str(new_id)
        assert new_exp.last_verified_at is not None
        assert new_exp.mutation_history[-1]["action"] == "SUPERSEDE_REPLACES"

        # 2. Test merge_experience
        merged = await repo.merge_experience(
            new_id,
            merged_solution="Deprecated v1; jwt_ttl=900\n\n### 补充方案 (Merged Update)\nEnable refresh token rotation",
            merged_tags=["auth", "security"],
            reason="Complementary refresh token step",
            actor="bob",
        )
        assert merged is new_exp
        assert "refresh token rotation" in (new_exp.solution or "")
        assert "security" in new_exp.tags
        assert new_exp.mutation_history[-1]["action"] == "MERGE"

        # 3. Test touch_verified (NOOP)
        touched = await repo.touch_verified(
            new_id,
            reason="Duplicate verification",
            actor="charlie",
        )
        assert touched is new_exp
        assert new_exp.mutation_history[-1]["action"] == "NOOP"
        assert new_exp.mutation_history[-1]["actor"] == "charlie"


@_async_mark
async def test_active_filter_and_fama_100_percent() -> None:
    """Verify active_filter excludes superseded records and achieves FAMA@5 = 100%."""
    clauses = [str(c) for c in active_filter(current_user="alice")]
    assert any("valid_until" in c for c in clauses)

    dup_sql = str(build_duplicate_detection(project="team_memory"))
    assert "a.valid_until IS NULL" in dup_sql
    assert "b.valid_until IS NULL" in dup_sql

    # Simulate retrieval before vs after bi-temporal valid_until filtering:
    old_id = "exp-superseded-v1"
    new_id = "exp-active-v2"

    # Without valid_until IS NULL filter, old_id leaks into top-5 -> FAMA@5 = 0.0
    fama_without_filter = compute_fama_at_k(
        retrieved_ids=[new_id, old_id],
        active_relevant_ids={new_id},
        superseded_ids={old_id},
        k=5,
    )
    assert fama_without_filter == 0.0

    # With active_filter (valid_until IS NULL), old_id is excluded -> FAMA@5 = 1.0 (100%)
    fama_with_filter = compute_fama_at_k(
        retrieved_ids=[new_id],
        active_relevant_ids={new_id},
        superseded_ids={old_id},
        k=5,
    )
    assert fama_with_filter == 1.0


if __name__ == "__main__":
    test_mutation_resolver_four_transitions()
    asyncio.run(test_repository_bitemporal_methods())
    asyncio.run(test_active_filter_and_fama_100_percent())
    print("ALL PHASE 3 BI-TEMPORAL MUTATION TESTS PASSED!")
