"""Unit tests for CrossEncoderRerankerProvider and 4-way Golden Eval ablation."""

from __future__ import annotations

import builtins
import json
import sys
import time
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from team_memory.config.reranker import RerankerConfig, RerankerCrossEncoderConfig
from team_memory.reranker.base import RerankerProvider, RerankResult
from team_memory.reranker.cross_encoder_provider import (
    CrossEncoderRerankerProvider,
    clear_model_cache,
    resolve_device,
)
from team_memory.reranker.factory import create_reranker
from team_memory.services.analysis.golden_eval import (
    build_ablation_specs,
    compute_ablation_lifts,
    format_eval_report,
    run_ablation_eval,
)


@pytest.fixture(autouse=True)
def _clean_ce_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def _fake_torch(*, mps: bool = False, cuda: bool = False) -> ModuleType:
    mod = ModuleType("torch")
    mod.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    mod.cuda = SimpleNamespace(is_available=lambda: cuda)
    return mod


class TestDeviceResolution:
    def test_auto_prefers_mps_over_cuda_and_cpu(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(mps=True, cuda=True)}):
            assert resolve_device("auto") == "mps"

    def test_auto_falls_back_to_cuda_then_cpu(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(mps=False, cuda=True)}):
            assert resolve_device("auto") == "cuda"
        with patch.dict(sys.modules, {"torch": _fake_torch(mps=False, cuda=False)}):
            assert resolve_device("auto") == "cpu"

    def test_explicit_mps_falls_back_to_cpu_when_unavailable(self, caplog):
        with patch.dict(sys.modules, {"torch": _fake_torch(mps=False, cuda=False)}):
            with caplog.at_level("WARNING"):
                assert resolve_device("mps") == "cpu"
        assert "falling back to cpu" in caplog.text

    def test_explicit_cpu_never_switches(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(mps=True, cuda=True)}):
            assert resolve_device("cpu") == "cpu"

    def test_torch_import_error_falls_back_to_cpu(self):
        real_import = builtins.__import__

        def _deny_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"torch": None}):
            with patch("builtins.__import__", side_effect=_deny_torch):
                assert resolve_device("auto") == "cpu"


class TestCrossEncoderSingletonAndFallback:
    @pytest.mark.asyncio
    async def test_singleton_lazy_loading_shared_across_instances(self):
        init_calls: list[tuple[str, str]] = []

        class _FakeCrossEncoder:
            def __init__(self, model_name: str, device: str = "cpu"):
                init_calls.append((model_name, device))

            def predict(self, pairs):
                # Rank doc with "target" highest
                return [0.95 if "target" in doc else 0.1 for _, doc in pairs]

        fake_st = ModuleType("sentence_transformers")
        fake_st.CrossEncoder = _FakeCrossEncoder

        with patch.dict(
            sys.modules,
            {"sentence_transformers": fake_st, "torch": _fake_torch(mps=True)},
        ):
            p1 = CrossEncoderRerankerProvider(model_name="tiny-ce", device="auto", top_k=5)
            p2 = CrossEncoderRerankerProvider(model_name="tiny-ce", device="auto", top_k=5)

            assert p1.device == "mps"
            assert p2.device == "mps"
            # Lazy: not loaded on __init__
            assert init_calls == []

            r1 = await p1.rank("query", ["noise", "target doc"])
            r2 = await p2.rank("query", ["other", "target doc"])

            # Singleton: constructed exactly once across both instances
            assert len(init_calls) == 1
            assert init_calls[0] == ("tiny-ce", "mps")
            assert [r.index for r in r1] == [1, 0]
            assert [r.index for r in r2] == [1, 0]

    @pytest.mark.asyncio
    async def test_missing_dependency_gracefully_passes_through_original_order(self, caplog):
        real_import = builtins.__import__

        def _deny_st(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return real_import(name, *args, **kwargs)

        provider = CrossEncoderRerankerProvider(model_name="tiny-ce", device="cpu", top_k=3)
        docs = ["doc-0", "doc-1", "doc-2"]

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with patch("builtins.__import__", side_effect=_deny_st):
                with caplog.at_level("WARNING"):
                    results = await provider.rank("test query", docs)

        assert [r.index for r in results] == [0, 1, 2]
        assert [r.text for r in results] == docs
        assert results[0].score > results[1].score > results[2].score > 0
        assert "dependency missing" in caplog.text

    @pytest.mark.asyncio
    async def test_timeout_protection_gracefully_passes_through_original_order(self, caplog):
        class _SlowCrossEncoder:
            def __init__(self, model_name: str, device: str = "cpu"):
                pass

            def predict(self, pairs):
                time.sleep(0.5)
                return [1.0 for _ in pairs]

        fake_st = ModuleType("sentence_transformers")
        fake_st.CrossEncoder = _SlowCrossEncoder

        with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            provider = CrossEncoderRerankerProvider(
                model_name="slow-ce",
                device="cpu",
                top_k=3,
                timeout_seconds=0.05,
            )
            docs = ["first", "second", "third"]
            t0 = time.monotonic()
            with caplog.at_level("WARNING"):
                results = await provider.rank("query", docs)
            elapsed = time.monotonic() - t0

        assert elapsed < 0.35
        assert [r.index for r in results] == [0, 1, 2]
        assert [r.text for r in results] == docs
        assert "timed out" in caplog.text

    def test_config_and_factory_pass_timeout_and_device(self):
        cfg = RerankerConfig(
            provider="cross_encoder",
            cross_encoder=RerankerCrossEncoderConfig(
                model_name="custom/ce",
                device="cpu",
                top_k=7,
                timeout_seconds=1.5,
            ),
        )
        provider = create_reranker(cfg)
        assert isinstance(provider, CrossEncoderRerankerProvider)
        assert provider.device == "cpu"
        assert provider.timeout_seconds == 1.5
        assert "custom/ce" in provider.provider_name


class TestGoldenEvalFourWayAblation:
    def test_build_ablation_specs_four_way_isolates_graph_and_rerank(self):
        specs = build_ablation_specs(prod_rerank_enabled=True, include_rerank_ablation=True)
        assert [s.name for s in specs] == ["vector_only", "hybrid", "full", "full_rerank"]
        by_name = {s.name: s for s in specs}

        assert by_name["vector_only"].rerank_enabled is False
        assert by_name["hybrid"].rerank_enabled is False
        assert by_name["full"].rerank_enabled is False
        assert by_name["full"].enable_graph is True
        assert by_name["full_rerank"].enable_graph is True
        assert by_name["full_rerank"].rerank_enabled is True
        assert "Reranker" in by_name["full_rerank"].label

    def test_compute_ablation_lifts_and_format_report(self):
        modes = {
            "vector_only": {
                "label": "A: Vector-only",
                "recall_at_5": 0.40,
                "mrr_at_5": 0.35,
                "ndcg_at_5": 0.36,
                "p50_latency_ms": 100.0,
                "p95_latency_ms": 200.0,
                "avg_tokens": 1000.0,
            },
            "hybrid": {
                "label": "B: Hybrid (Vec+FTS)",
                "recall_at_5": 0.50,
                "mrr_at_5": 0.42,
                "ndcg_at_5": 0.44,
                "p50_latency_ms": 120.0,
                "p95_latency_ms": 240.0,
                "avg_tokens": 1100.0,
            },
            "full": {
                "label": "C: Full (Hybrid+Graph)",
                "recall_at_5": 0.55,
                "mrr_at_5": 0.48,
                "ndcg_at_5": 0.50,
                "p50_latency_ms": 150.0,
                "p95_latency_ms": 300.0,
                "avg_tokens": 1150.0,
            },
            "full_rerank": {
                "label": "D: Full + Reranker",
                "recall_at_5": 0.60,
                "mrr_at_5": 0.56,
                "ndcg_at_5": 0.58,
                "p50_latency_ms": 210.0,
                "p95_latency_ms": 420.0,
                "avg_tokens": 1150.0,
            },
        }
        lifts = compute_ablation_lifts(modes)
        assert lifts["graph_lift"] == pytest.approx(0.06)
        assert lifts["graph_recall_lift"] == pytest.approx(0.05)
        assert lifts["rerank_lift"] == pytest.approx(0.08)
        assert lifts["rerank_recall_lift"] == pytest.approx(0.05)
        assert lifts["rerank_ndcg_lift"] == pytest.approx(0.08)
        assert lifts["rerank_p50_delta_ms"] == pytest.approx(60.0)
        assert lifts["rerank_p95_delta_ms"] == pytest.approx(120.0)

        record = {
            "run_id": "eval-test4way",
            "golden_count": 10,
            "prod_rerank_enabled": False,
            "duration_s": 1.2,
            "git_sha": "abc1234",
            "modes": modes,
            **lifts,
        }
        report = format_eval_report(record)
        assert "D: Full + Reranker" in report
        assert "Rerank Lift (MRR@5 D-C): +0.0800" in report
        assert "ΔP50: +60.0ms, ΔP95: +120.0ms" in report

    @pytest.mark.asyncio
    async def test_run_ablation_eval_four_way_end_to_end_mocked(self, tmp_path):
        golden_file = tmp_path / "golden.json"
        golden_file.write_text(
            json.dumps(
                {
                    "queries": [
                        {
                            "id": "q1",
                            "project": "team_memory",
                            "bucket": "semantic_qa",
                            "query": "how does reranker timeout work",
                            "expected_ids": ["exp_gold"],
                        },
                        {
                            "id": "q2",
                            "project": "team_memory",
                            "bucket": "exact_identifier",
                            "query": "CrossEncoderRerankerProvider mps",
                            "expected_ids": ["exp_gold_2"],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        class _DummyReranker(RerankerProvider):
            async def rank(self, query, documents, top_k=10, document_metadata=None):
                return [RerankResult(index=i, score=1.0 - i * 0.1, text=d) for i, d in enumerate(documents[:top_k])]

            @property
            def provider_name(self) -> str:
                return "dummy_reranker"

        class _FakePipeline:
            def __init__(self, *, search_config, rerank_enabled=False, **kwargs):
                self._mode = search_config.mode
                self._graph = search_config.enable_graph
                self._rerank = rerank_enabled

            async def search(self, session, request):
                gold = "exp_gold" if "timeout" in request.query else "exp_gold_2"
                if self._rerank:
                    # D: gold ranked #1 (MRR = 1.0)
                    ids = [gold, "other_1", "other_2"]
                elif self._graph:
                    # C: gold ranked #2 (MRR = 0.5)
                    ids = ["other_1", gold, "other_2"]
                elif self._mode == "hybrid":
                    # B: gold ranked #3 (MRR = 0.3333)
                    ids = ["other_1", "other_2", gold]
                else:
                    # A: gold not in top-3 (MRR = 0.0)
                    ids = ["other_1", "other_2", "other_3"]
                return SimpleNamespace(
                    results=[{"id": eid, "title": eid, "description": "d", "solution": "s"} for eid in ids]
                )

        @asynccontextmanager
        async def _fake_session(_db_url: str):
            yield object()

        fake_settings = SimpleNamespace(
            search={"mode": "hybrid", "enable_graph": True, "enable_topic": True},
            retrieval=SimpleNamespace(),
            llm=SimpleNamespace(),
            reranker=RerankerConfig(provider="none"),
            tag_synonyms={},
        )

        with (
            patch("team_memory.services.analysis.golden_eval.get_session", _fake_session),
            patch("team_memory.services.search.search_pipeline.SearchPipeline", _FakePipeline),
        ):
            record = await run_ablation_eval(
                db_url="postgresql+asyncpg://localhost/test",
                embedding_provider=object(),
                settings=fake_settings,
                reranker_provider=_DummyReranker(),
                include_rerank=True,
                golden_set_path=golden_file,
                history_path=None,
                k=5,
            )

        assert record["rerank_ablation_enabled"] is True
        assert record["reranker_provider"] == "dummy_reranker"
        assert set(record["modes"].keys()) == {"vector_only", "hybrid", "full", "full_rerank"}
        assert record["modes"]["full"]["mrr_at_5"] == pytest.approx(0.5)
        assert record["modes"]["full_rerank"]["mrr_at_5"] == pytest.approx(1.0)
        assert record["rerank_lift"] == pytest.approx(0.5)
        assert record["graph_lift"] == pytest.approx(0.1667, abs=1e-4)
        assert "rerank_p50_delta_ms" in record
        assert "rerank_p95_delta_ms" in record
