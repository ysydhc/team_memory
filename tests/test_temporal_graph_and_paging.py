"""Tests for Phase 1–3 upgrades:
- Phase 1: Bi-temporal validity & memory consolidation state machine (ADD / MERGE / SUPERSEDE / NOOP)
- Phase 2: Lightweight PPR (Personalized PageRank) & cross-project entity graph traversal
- Phase 3: SessionWorkingSet memory paging & progressive disclosure folding
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from team_memory.daemon.refinement.pipeline import _format_l1_item
from team_memory.services.entity.entity_search import (
    find_weighted_experiences_by_entities,
    run_lightweight_ppr,
)
from team_memory.services.memory.mutation_resolver import MutationResolver
from team_memory.services.search.progressive_disclosure import (
    SessionWorkingSet,
    apply_progressive_disclosure,
    get_session_working_set,
)
from team_memory.services.search.search_pipeline import _apply_bi_temporal_annotations
from team_memory.storage.models import Experience

# ============================================================
# Phase 1: Temporal Knowledge Graph & Consolidation State Machine
# ============================================================


class TestPhase1TemporalGraph:
    """Tests for bi-temporal columns, consolidation classification, and supersession."""

    def test_experience_to_dict_includes_temporal_when_superseded(self):
        old_id = uuid.uuid4()
        new_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        exp = Experience(
            id=old_id,
            title="Old Approach",
            description="Auth config",
            solution="Use JWT v1",
            created_by="tester",
            valid_from=now,
            valid_until=now,
            superseded_by_id=new_id,
            mutation_history=[{"action": "SUPERSEDE", "superseded_by_id": str(new_id)}],
        )
        data = exp.to_dict()
        assert data["superseded_by_id"] == str(new_id)
        assert data["valid_until"] == now.isoformat()
        assert data["mutation_history"][0]["action"] == "SUPERSEDE"

    def test_mutation_resolver_noop(self):
        resolver = MutationResolver()
        decision = resolver.resolve_by_rules(
            title="Fix Redis timeout",
            problem="Redis connection times out under load",
            solution="Set socket_timeout=5 and retry_on_timeout=True",
            candidates=[
                {
                    "id": "exp-1",
                    "similarity": 0.97,
                    "description": "Redis connection times out under load",
                    "solution": "Set socket_timeout=5 and retry_on_timeout=True",
                }
            ],
        )
        assert decision.action == "NOOP"
        assert decision.target_id == "exp-1"

    def test_mutation_resolver_supersede_on_conflicting_kv_or_marker(self):
        resolver = MutationResolver()
        decision = resolver.resolve_by_rules(
            title="Configure token auth in API gateway",
            problem="How to configure token authentication in API gateway",
            solution="Deprecated auth_mode=static_secret; use auth_mode=oidc_jwks",
            candidates=[
                {
                    "id": "exp-old",
                    "similarity": 0.91,
                    "description": "How to configure token authentication in API gateway",
                    "solution": "Use auth_mode=static_secret",
                }
            ],
        )
        assert decision.action == "SUPERSEDE"
        assert decision.target_id == "exp-old"

    def test_mutation_resolver_merge_on_complementary_solution(self):
        resolver = MutationResolver()
        decision = resolver.resolve_by_rules(
            title="Configure token auth in API gateway",
            problem="How to configure token authentication in API gateway",
            solution="Also enable clock skew tolerance of 30s on token verifier.",
            candidates=[
                {
                    "id": "exp-base",
                    "similarity": 0.90,
                    "description": "How to configure token authentication in API gateway",
                    "solution": "Configure JWKS endpoint URL in gateway.yaml.",
                }
            ],
        )
        assert decision.action == "MERGE"
        assert decision.target_id == "exp-base"
        assert "clock skew tolerance" in (decision.merged_solution or "")

    def test_apply_bi_temporal_annotations_downweights_and_marks_superseded(self):
        new_id = str(uuid.uuid4())
        old_id = str(uuid.uuid4())
        results = [
            {
                "id": old_id,
                "title": "Legacy Auth Flow",
                "description": "Auth problem",
                "solution": "Old solution",
                "score": 0.90,
                "similarity": 0.90,
                "valid_until": "2026-03-01T00:00:00+00:00",
                "superseded_by_id": new_id,
                "mutation_history": [],
            },
            {
                "id": new_id,
                "title": "Modern OAuth2 Flow",
                "description": "Auth problem",
                "solution": "New solution",
                "score": 0.60,
                "similarity": 0.60,
                "valid_until": None,
                "superseded_by_id": None,
                "mutation_history": [
                    {"action": "SUPERSEDE_REPLACES", "supersedes_id": old_id}
                ],
            },
        ]

        annotated = _apply_bi_temporal_annotations(results)
        # New active item should rank above superseded item after 0.3x penalty (0.90 * 0.3 = 0.27 < 0.60)
        assert annotated[0]["id"] == new_id
        assert annotated[1]["id"] == old_id
        assert annotated[1]["is_superseded"] is True
        assert annotated[1]["score"] == pytest.approx(0.27, rel=1e-3)
        assert f"tm://exp/{new_id[:8]}" in annotated[1]["superseded_marker"]
        assert f"[已演进/Superseded by tm://exp/{new_id[:8]}]" in annotated[1]["title"]
        assert f"tm://exp/{old_id[:8]}" in annotated[0]["evolution_marker"]


# ============================================================
# Phase 2: Lightweight PPR & Cross-Project Entity Sharing
# ============================================================


class TestPhase2LightweightPPR:
    """Tests for damped Personalized PageRank and cross-project entity graph walk."""

    def test_run_lightweight_ppr_multi_hop_and_cross_project_discount(self):
        entities = [
            {"id": "e1", "name": "ServiceA", "project": "proj1", "source_count": 1},
            {"id": "e2", "name": "RedisPool", "project": "proj1", "source_count": 1},
            {"id": "e3", "name": "SharedAuth", "project": "proj2", "source_count": 1},
        ]
        relationships = [
            {"source_entity_id": "e1", "target_entity_id": "e2", "weight": 0.9, "valid_until": None},
            {"source_entity_id": "e2", "target_entity_id": "e3", "weight": 0.8, "valid_until": None},
        ]
        experience_entities = [
            {"experience_id": "exp-1", "entity_id": "e1"},
            {"experience_id": "exp-2", "entity_id": "e2"},
            {"experience_id": "exp-3", "entity_id": "e3"},
        ]

        ranked = dict(
            run_lightweight_ppr(
                seed_entities=["ServiceA"],
                entities=entities,
                relationships=relationships,
                experience_entities=experience_entities,
                primary_projects=["proj1"],
                max_hops=3,
                damping=0.55,
                cross_project_discount=0.85,
            )
        )
        assert "exp-1" in ranked
        assert "exp-2" in ranked
        assert "exp-3" in ranked
        # Seed > 1-hop same project > 2-hop cross-project
        assert ranked["exp-1"] > ranked["exp-2"] > ranked["exp-3"] > 0.0

    @pytest.mark.asyncio
    async def test_find_weighted_experiences_by_entities_executes_cte(self):
        exp_id = uuid.uuid4()
        mock_row = (exp_id, 0.8075)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        results = await find_weighted_experiences_by_entities(
            "postgresql+asyncpg://localhost/test",
            ["RedisTimeout", "AuthGateway"],
            projects=["proj_a"],
            max_depth=3,
            limit=10,
            session=mock_session,
        )
        assert len(results) == 1
        r_id, weight = results[0]
        assert r_id == str(exp_id)
        assert weight == pytest.approx(0.8075)

        call_args = mock_session.execute.call_args
        params = call_args[0][1]
        assert params["max_depth"] == 3
        assert params["limit"] == 10


# ============================================================
# Phase 3: SessionWorkingSet Memory Paging
# ============================================================


class TestPhase3MemoryPaging:
    """Tests for SessionWorkingSet and L2 -> L1 folding across turns."""

    def test_session_working_set_record_and_evict(self):
        ws = SessionWorkingSet(ttl_seconds=3600)
        s_id = "sess-test-001"
        assert ws.has_l2(s_id, "exp-1") is False

        ws.record_l2(s_id, ["exp-1"])
        assert ws.has_l2(s_id, "exp-1") is True
        assert ws.get_loaded_ids(s_id) == {"exp-1"}

        ws.evict(s_id, "exp-1")
        assert ws.has_l2(s_id, "exp-1") is False

    def test_progressive_disclosure_folds_already_loaded_l2_and_promotes_next(self):
        ws = SessionWorkingSet(ttl_seconds=3600)
        s_id = "sess-paging-demo"

        def _make_items():
            return [
                {
                    "id": f"exp-{i}",
                    "title": f"Experience {i}",
                    "description": f"Detailed problem description for item {i} " * 5,
                    "solution": f"Comprehensive step by step solution for item {i} " * 8,
                    "tags": ["tag1", "tag2"],
                    "score": 0.95 - i * 0.05,
                }
                for i in range(1, 5)
            ]

        # Turn 1: Top-2 (exp-1, exp-2) should be full (L2); exp-3, exp-4 should be summary (L1)
        turn1_results, turn1_meta = apply_progressive_disclosure(
            _make_items(),
            detail_level="progressive",
            full_top_k=2,
            session_id=s_id,
            working_set=ws,
        )
        assert turn1_results[0]["disclosure_level"] == "full"
        assert turn1_results[1]["disclosure_level"] == "full"
        assert turn1_results[2]["disclosure_level"] == "summary"
        assert turn1_results[3]["disclosure_level"] == "summary"
        assert turn1_meta["working_set_paged_out"] == 0
        assert turn1_meta["working_set_size"] == 2

        # Turn 2 (same session, same candidates):
        # exp-1 and exp-2 are already in Working Set -> folded to L1 summary with marker!
        # exp-3 and exp-4 get promoted to fill the 2 L2 slots!
        turn2_results, turn2_meta = apply_progressive_disclosure(
            _make_items(),
            detail_level="progressive",
            full_top_k=2,
            session_id=s_id,
            working_set=ws,
        )
        assert turn2_results[0]["disclosure_level"] == "summary"
        assert turn2_results[0]["working_set_status"] == "paged_out"
        assert "Already in Working Set" in turn2_results[0]["summary"]

        assert turn2_results[1]["disclosure_level"] == "summary"
        assert turn2_results[1]["working_set_status"] == "paged_out"

        assert turn2_results[2]["disclosure_level"] == "full"
        assert turn2_results[2]["working_set_status"] == "paged_in"
        assert turn2_results[3]["disclosure_level"] == "full"
        assert turn2_results[3]["working_set_status"] == "paged_in"
        assert turn2_meta["working_set_paged_out"] == 2
        assert turn2_meta["working_set_size"] == 4

        # Turn 3 with force_l2=True (Page-in): forces L2 even if already in working set
        turn3_results, _ = apply_progressive_disclosure(
            _make_items(),
            detail_level="progressive",
            full_top_k=2,
            session_id=s_id,
            working_set=ws,
            force_l2=True,
        )
        assert turn3_results[0]["disclosure_level"] == "full"
        assert turn3_results[1]["disclosure_level"] == "full"

    def test_pipeline_format_l1_item_marks_working_set_hit(self):
        ws = get_session_working_set()
        s_id = f"sess-{uuid.uuid4().hex[:8]}"
        exp_id = str(uuid.uuid4())
        ws.record_l2(s_id, [exp_id])

        item = {
            "id": exp_id,
            "title": "Fix Alembic Migration Conflict",
            "description": "Multiple heads in alembic migration",
            "solution": "Run alembic merge heads",
            "tags": ["alembic", "db"],
        }
        formatted = _format_l1_item(item, session_id=s_id)
        assert "已在当前会话加载 L2 / Already in Working Set" in formatted
        ws.clear(s_id)
