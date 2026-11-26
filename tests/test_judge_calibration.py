"""Tests for Judge Calibration Set schema, metrics calculation, and CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_judge as cj  # noqa: E402, I001


# ===========================================================================
# 1. Calibration Dataset Schema Completeness
# ===========================================================================


class TestCalibrationSetSchema:
    def test_dataset_exists_and_has_at_least_20_balanced_cases(self) -> None:
        assert cj.DEFAULT_DATASET_PATH.exists(), "evals/judge_calibration_set.json must exist"
        data = cj.load_calibration_set(cj.DEFAULT_DATASET_PATH)

        cases = data["cases"]
        assert len(cases) >= 20

        counts: dict[str, int] = {lbl: 0 for lbl in cj.REQUIRED_LABELS}
        seen_ids: set[str] = set()

        for case in cases:
            cid = case["id"]
            assert cid not in seen_ids, f"Duplicate case id: {cid}"
            seen_ids.add(cid)

            label = case["label"]
            assert label in counts, f"Unexpected label: {label}"
            counts[label] += 1

            assert isinstance(case["query"], str) and len(case["query"].strip()) >= 8
            assert isinstance(case["response"], str) and len(case["response"].strip()) >= 20
            assert isinstance(case["contexts"], list) and len(case["contexts"]) >= 1

            ctx_ids = {str(c["id"]) for c in case["contexts"]}
            target_id = str(case["target_experience_id"])
            assert target_id in ctx_ids, f"{cid}: target_experience_id {target_id} not in contexts"

            for ctx in case["contexts"]:
                assert ctx.get("title")
                assert ctx.get("problem")
                assert ctx.get("solution")

            exp = case["expected"]
            assert isinstance(exp["adopted"], bool)
            assert isinstance(exp["contradicted"], bool)
            rel = float(exp["relevance"])
            assert 0.0 <= rel <= 1.0

            # Semantic label invariants
            if label == "adopted":
                assert exp["adopted"] is True
                assert exp["contradicted"] is False
                assert rel >= 0.7
                kws = exp.get("evidence_keywords") or []
                assert len(kws) >= 1
                assert any(kw.lower() in case["response"].lower() for kw in kws), (
                    f"{cid}: at least one evidence keyword must appear in response"
                )
            elif label == "relevant_only":
                assert exp["adopted"] is False
                assert exp["contradicted"] is False
                assert rel >= 0.5
            elif label == "irrelevant":
                assert exp["adopted"] is False
                assert exp["contradicted"] is False
                assert rel <= 0.3
            elif label == "contradicted":
                assert exp["adopted"] is False
                assert exp["contradicted"] is True
                assert rel >= 0.6

        for lbl, cnt in counts.items():
            assert cnt >= 5, f"Expected >= 5 cases for label '{lbl}', got {cnt}"


# ===========================================================================
# 2. Metric Computation & Precision Gate Logic
# ===========================================================================


class TestCalibrationMetrics:
    def _make_pred(
        self,
        cid: str,
        label: str,
        *,
        exp_adopted: bool,
        pred_adopted: bool,
        exp_contra: bool = False,
        pred_contra: bool = False,
        exp_rel: float = 0.9,
        pred_rel: float = 0.9,
        error: str | None = None,
    ) -> cj.CasePrediction:
        return cj.CasePrediction(
            case_id=cid,
            label=label,
            target_experience_id=f"exp-{cid}",
            expected_adopted=exp_adopted,
            expected_contradicted=exp_contra,
            expected_relevance=exp_rel,
            pred_adopted=pred_adopted,
            pred_contradicted=pred_contra,
            pred_relevance=pred_rel,
            error=error,
        )

    def test_perfect_predictions_pass_gate(self) -> None:
        preds = [
            self._make_pred("c1", "adopted", exp_adopted=True, pred_adopted=True, exp_rel=0.95, pred_rel=0.95),
            self._make_pred("c2", "relevant_only", exp_adopted=False, pred_adopted=False, exp_rel=0.75, pred_rel=0.75),
            self._make_pred("c3", "irrelevant", exp_adopted=False, pred_adopted=False, exp_rel=0.05, pred_rel=0.05),
            self._make_pred(
                "c4",
                "contradicted",
                exp_adopted=False,
                pred_adopted=False,
                exp_contra=True,
                pred_contra=True,
                exp_rel=0.85,
                pred_rel=0.85,
            ),
        ]
        m = cj.compute_calibration_metrics(preds, min_precision=0.85)
        assert m.adopted_precision == pytest.approx(1.0)
        assert m.adopted_recall == pytest.approx(1.0)
        assert m.adopted_f1 == pytest.approx(1.0)
        assert m.relevance_mae == pytest.approx(0.0)
        assert m.contradicted_detection_rate == pytest.approx(1.0)
        assert m.contradicted_precision == pytest.approx(1.0)
        assert m.error_count == 0
        assert m.passed is True

    def test_precision_below_threshold_fails_gate(self) -> None:
        # 4 TP + 1 FP => Precision = 4/5 = 0.80 < 0.85 => FAIL
        preds = [
            self._make_pred(f"tp{i}", "adopted", exp_adopted=True, pred_adopted=True, exp_rel=0.9, pred_rel=0.8)
            for i in range(4)
        ] + [
            self._make_pred("fp1", "relevant_only", exp_adopted=False, pred_adopted=True, exp_rel=0.7, pred_rel=0.9),
            self._make_pred("fn1", "adopted", exp_adopted=True, pred_adopted=False, exp_rel=0.9, pred_rel=0.6),
            self._make_pred(
                "ct1",
                "contradicted",
                exp_adopted=False,
                pred_adopted=False,
                exp_contra=True,
                pred_contra=True,
                exp_rel=0.9,
                pred_rel=0.9,
            ),
            self._make_pred(
                "ct2",
                "contradicted",
                exp_adopted=False,
                pred_adopted=False,
                exp_contra=True,
                pred_contra=False,
                exp_rel=0.9,
                pred_rel=0.8,
            ),
        ]
        m = cj.compute_calibration_metrics(preds, min_precision=0.85)
        assert m.adopted_tp == 4
        assert m.adopted_fp == 1
        assert m.adopted_fn == 1
        assert m.adopted_precision == pytest.approx(0.8)
        assert m.adopted_recall == pytest.approx(0.8)
        assert m.adopted_f1 == pytest.approx(0.8)
        assert m.contradicted_detection_rate == pytest.approx(0.5)
        # MAE = (0.1*4 + 0.2 + 0.3 + 0.0 + 0.1) / 8 = 1.0 / 8 = 0.125
        assert m.relevance_mae == pytest.approx(0.125)
        assert m.passed is False

    def test_all_errors_fail_gate(self) -> None:
        preds = [
            self._make_pred("e1", "adopted", exp_adopted=True, pred_adopted=False, error="timeout"),
        ]
        m = cj.compute_calibration_metrics(preds, min_precision=0.85)
        assert m.evaluated_cases == 0
        assert m.error_count == 1
        assert m.passed is False


# ===========================================================================
# 3. End-to-End Offline Mock Mode & CLI
# ===========================================================================


class TestCalibrationOfflineMock:
    @pytest.mark.asyncio
    async def test_run_calibration_offline_mock_passes_gate(self) -> None:
        data = cj.load_calibration_set(cj.DEFAULT_DATASET_PATH)
        metrics, preds = await cj.run_calibration(data, offline_mock=True, min_precision=0.85)

        assert metrics.total_cases >= 20
        assert metrics.evaluated_cases == metrics.total_cases
        assert metrics.error_count == 0
        assert metrics.adopted_precision >= 0.85
        assert metrics.adopted_recall >= 0.85
        assert metrics.contradicted_detection_rate >= 0.85
        assert metrics.relevance_mae < 0.15
        assert metrics.passed is True
        assert len(preds) == metrics.total_cases

    def test_cli_offline_mock_writes_json_report(self, tmp_path: Path) -> None:
        out_path = tmp_path / "cal_report.json"
        rc = cj.main(["--offline-mock", "--json-out", str(out_path)])
        assert rc == 0
        assert out_path.exists()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["mode"] == "offline-mock"
        assert payload["metrics"]["passed"] is True
        assert payload["metrics"]["adopted_precision"] >= 0.85
        assert len(payload["predictions"]) >= 20
