#!/usr/bin/env python3
"""Calibrate AttributionJudge against evals/judge_calibration_set.json.

Computes:
- Adopted Precision / Recall / F1 (built-in gate: Precision >= 0.85)
- Relevance MAE (Mean Absolute Error against annotated relevance)
- Contradicted Detection Rate (Recall on contradicted cases) & Precision
- Per-label breakdown across adopted / relevant_only / irrelevant / contradicted

Supports ``--offline-mock`` for zero-cost deterministic CI verification without
calling an external LLM gateway.

Usage:
    uv run python scripts/calibrate_judge.py
    uv run python scripts/calibrate_judge.py --offline-mock
    uv run python scripts/calibrate_judge.py --dataset evals/judge_calibration_set.json --min-precision 0.85
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from team_memory.daemon.quality.faithfulness_judge import (  # noqa: E402
    AttributionItem,
    AttributionJudge,
    AttributionResult,
)
from team_memory.project_resolver import load_dotenv_if_exists  # noqa: E402

DEFAULT_DATASET_PATH = ROOT / "evals" / "judge_calibration_set.json"
DEFAULT_MIN_PRECISION = 0.85
REQUIRED_LABELS = ("adopted", "relevant_only", "irrelevant", "contradicted")

# Lightweight lexical cues used exclusively in --offline-mock mode
_CONTRADICTION_CUES = (
    "过时",
    "失效",
    "废弃",
    "不再适用",
    "已下线",
    "已移除",
    "已被移除",
    "完全错误",
    "不能再采用",
    "Connection Refused",
    "refuses",
    "deprecated",
    "obsolete",
)


@dataclass
class CasePrediction:
    """Normalized prediction record for a single calibration case."""

    case_id: str
    label: str
    target_experience_id: str
    expected_adopted: bool
    expected_contradicted: bool
    expected_relevance: float
    pred_adopted: bool
    pred_contradicted: bool
    pred_relevance: float
    evidence_quote: str = ""
    downgraded: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationMetrics:
    """Aggregated calibration metrics and acceptance gate verdict."""

    total_cases: int
    evaluated_cases: int
    error_count: int
    adopted_tp: int
    adopted_fp: int
    adopted_tn: int
    adopted_fn: int
    adopted_precision: float
    adopted_recall: float
    adopted_f1: float
    relevance_mae: float
    contradicted_tp: int
    contradicted_fp: int
    contradicted_fn: int
    contradicted_detection_rate: float
    contradicted_precision: float
    min_precision_threshold: float
    passed: bool
    per_label: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_calibration_set(path: Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    """Load and return the calibration dataset JSON payload."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"Invalid calibration dataset format in {path}")
    return data


def compute_calibration_metrics(
    predictions: list[CasePrediction],
    *,
    min_precision: float = DEFAULT_MIN_PRECISION,
) -> CalibrationMetrics:
    """Compute Precision/Recall/F1, Relevance MAE, Contradicted detection rate, and gate status."""
    total_cases = len(predictions)
    valid = [p for p in predictions if p.error is None]
    error_count = total_cases - len(valid)

    adopted_tp = sum(1 for p in valid if p.expected_adopted and p.pred_adopted)
    adopted_fp = sum(1 for p in valid if (not p.expected_adopted) and p.pred_adopted)
    adopted_tn = sum(1 for p in valid if (not p.expected_adopted) and (not p.pred_adopted))
    adopted_fn = sum(1 for p in valid if p.expected_adopted and (not p.pred_adopted))

    if adopted_tp + adopted_fp > 0:
        adopted_precision = adopted_tp / (adopted_tp + adopted_fp)
    else:
        adopted_precision = 1.0 if (adopted_fn == 0 and valid) else 0.0

    if adopted_tp + adopted_fn > 0:
        adopted_recall = adopted_tp / (adopted_tp + adopted_fn)
    else:
        adopted_recall = 1.0 if valid else 0.0

    if adopted_precision + adopted_recall > 0:
        adopted_f1 = 2 * adopted_precision * adopted_recall / (adopted_precision + adopted_recall)
    else:
        adopted_f1 = 0.0

    if valid:
        relevance_mae = sum(abs(p.pred_relevance - p.expected_relevance) for p in valid) / len(valid)
    else:
        relevance_mae = 1.0

    contradicted_tp = sum(1 for p in valid if p.expected_contradicted and p.pred_contradicted)
    contradicted_fp = sum(1 for p in valid if (not p.expected_contradicted) and p.pred_contradicted)
    contradicted_fn = sum(1 for p in valid if p.expected_contradicted and (not p.pred_contradicted))

    if contradicted_tp + contradicted_fn > 0:
        contradicted_detection_rate = contradicted_tp / (contradicted_tp + contradicted_fn)
    else:
        contradicted_detection_rate = 1.0 if valid else 0.0

    if contradicted_tp + contradicted_fp > 0:
        contradicted_precision = contradicted_tp / (contradicted_tp + contradicted_fp)
    else:
        contradicted_precision = 1.0 if valid else 0.0

    per_label: dict[str, dict[str, Any]] = {}
    labels_present = list(REQUIRED_LABELS)
    for p in predictions:
        if p.label not in labels_present:
            labels_present.append(p.label)

    for lbl in labels_present:
        group = [p for p in valid if p.label == lbl]
        if not group:
            continue
        exact_matches = sum(
            1
            for p in group
            if p.pred_adopted == p.expected_adopted and p.pred_contradicted == p.expected_contradicted
        )
        per_label[lbl] = {
            "count": len(group),
            "exact_match_rate": round(exact_matches / len(group), 4),
            "avg_pred_relevance": round(sum(p.pred_relevance for p in group) / len(group), 4),
            "avg_expected_relevance": round(sum(p.expected_relevance for p in group) / len(group), 4),
            "relevance_mae": round(
                sum(abs(p.pred_relevance - p.expected_relevance) for p in group) / len(group),
                4,
            ),
            "pred_adopted_count": sum(1 for p in group if p.pred_adopted),
            "pred_contradicted_count": sum(1 for p in group if p.pred_contradicted),
        }

    passed = (len(valid) > 0) and (adopted_precision >= min_precision)

    return CalibrationMetrics(
        total_cases=total_cases,
        evaluated_cases=len(valid),
        error_count=error_count,
        adopted_tp=adopted_tp,
        adopted_fp=adopted_fp,
        adopted_tn=adopted_tn,
        adopted_fn=adopted_fn,
        adopted_precision=round(adopted_precision, 4),
        adopted_recall=round(adopted_recall, 4),
        adopted_f1=round(adopted_f1, 4),
        relevance_mae=round(relevance_mae, 4),
        contradicted_tp=contradicted_tp,
        contradicted_fp=contradicted_fp,
        contradicted_fn=contradicted_fn,
        contradicted_detection_rate=round(contradicted_detection_rate, 4),
        contradicted_precision=round(contradicted_precision, 4),
        min_precision_threshold=min_precision,
        passed=passed,
        per_label=per_label,
    )


def _build_offline_mock_payload(case: dict[str, Any]) -> str:
    """Synthesize a deterministic JSON array payload for --offline-mock mode.

    Uses lightweight lexical matching against the case's response and contexts
    so AttributionJudge._parse_attribution_result() and its Evidence-Required
    hard gate are exercised end-to-end without network calls.
    """
    response = str(case.get("response") or "")
    response_lower = response.lower()
    expected = case.get("expected") or {}
    evidence_keywords = [str(k) for k in expected.get("evidence_keywords") or [] if str(k).strip()]
    target_id = str(case.get("target_experience_id") or "")

    items: list[dict[str, Any]] = []
    for ctx in case.get("contexts") or []:
        cid = str(ctx.get("id") or "")
        is_target = cid == target_id

        # Check explicit evidence substring in response
        matched_quote = ""
        if is_target:
            for kw in evidence_keywords:
                if kw.lower() in response_lower:
                    matched_quote = kw
                    break

        expected_rel = float(expected.get("relevance", 0.0))
        has_contradiction = (
            is_target
            and expected_rel >= 0.5
            and any(cue.lower() in response_lower for cue in _CONTRADICTION_CUES)
        )

        if matched_quote and not has_contradiction:
            relevance = 0.92
            adopted = True
            contradicted = False
        elif has_contradiction:
            relevance = 0.88
            adopted = False
            contradicted = True
        elif is_target and expected_rel >= 0.5:
            relevance = 0.74
            adopted = False
            contradicted = False
        else:
            relevance = 0.08
            adopted = False
            contradicted = False

        items.append(
            {
                "experience_id": cid,
                "relevance": relevance,
                "adopted": adopted,
                "evidence_quote": matched_quote,
                "contradicted": contradicted,
            }
        )

    return json.dumps(items, ensure_ascii=False)


def _extract_target_item(result: AttributionResult, target_id: str) -> AttributionItem | None:
    for item in result.items:
        if item.experience_id == target_id:
            return item
    return None


async def run_calibration(
    dataset: dict[str, Any],
    *,
    offline_mock: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    min_precision: float = DEFAULT_MIN_PRECISION,
    concurrency: int = 4,
) -> tuple[CalibrationMetrics, list[CasePrediction]]:
    """Run AttributionJudge across all cases in ``dataset`` and compute metrics."""
    cases: list[dict[str, Any]] = list(dataset.get("cases") or [])
    judge = AttributionJudge(
        base_url=base_url,
        model=model,
        api_key=api_key or ("sk-offline-mock" if offline_mock else None),
    )

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _eval_one(case: dict[str, Any]) -> CasePrediction:
        case_id = str(case["id"])
        label = str(case["label"])
        target_id = str(case["target_experience_id"])
        expected = case["expected"]
        exp_adopted = bool(expected["adopted"])
        exp_contradicted = bool(expected["contradicted"])
        exp_relevance = float(expected["relevance"])

        async with sem:
            if offline_mock:
                raw_json = _build_offline_mock_payload(case)
                id_to_index = {str(c["id"]): idx + 1 for idx, c in enumerate(case["contexts"])}
                result = judge._parse_attribution_result(raw_json, id_to_index=id_to_index)
            else:
                result = await judge.evaluate(
                    query=str(case["query"]),
                    contexts=list(case["contexts"]),
                    response=str(case["response"]),
                )

        if result.score < 0:
            return CasePrediction(
                case_id=case_id,
                label=label,
                target_experience_id=target_id,
                expected_adopted=exp_adopted,
                expected_contradicted=exp_contradicted,
                expected_relevance=exp_relevance,
                pred_adopted=False,
                pred_contradicted=False,
                pred_relevance=0.0,
                error=result.raw_response[:200] or "judge_error",
            )

        target_item = _extract_target_item(result, target_id)
        if target_item is None:
            return CasePrediction(
                case_id=case_id,
                label=label,
                target_experience_id=target_id,
                expected_adopted=exp_adopted,
                expected_contradicted=exp_contradicted,
                expected_relevance=exp_relevance,
                pred_adopted=False,
                pred_contradicted=False,
                pred_relevance=0.0,
                evidence_quote="",
                downgraded=False,
            )

        return CasePrediction(
            case_id=case_id,
            label=label,
            target_experience_id=target_id,
            expected_adopted=exp_adopted,
            expected_contradicted=exp_contradicted,
            expected_relevance=exp_relevance,
            pred_adopted=target_item.adopted,
            pred_contradicted=target_item.contradicted,
            pred_relevance=target_item.relevance,
            evidence_quote=target_item.evidence_quote,
            downgraded=target_item.downgraded,
        )

    predictions = list(await asyncio.gather(*(_eval_one(c) for c in cases)))
    metrics = compute_calibration_metrics(predictions, min_precision=min_precision)
    return metrics, predictions


def _print_report(
    metrics: CalibrationMetrics,
    predictions: list[CasePrediction],
    *,
    mode: str,
    model_name: str,
    verbose: bool = False,
) -> None:
    gate_str = "PASS" if metrics.passed else "FAIL"
    print("=" * 72)
    print(f"AttributionJudge Calibration Report  [mode={mode}, model={model_name}]")
    print("=" * 72)
    print(
        f"Cases: total={metrics.total_cases}  evaluated={metrics.evaluated_cases}  "
        f"errors={metrics.error_count}"
    )
    print(
        f"Adopted      : Precision={metrics.adopted_precision:.4f}  "
        f"Recall={metrics.adopted_recall:.4f}  F1={metrics.adopted_f1:.4f}  "
        f"(TP={metrics.adopted_tp}, FP={metrics.adopted_fp}, "
        f"TN={metrics.adopted_tn}, FN={metrics.adopted_fn})"
    )
    print(f"Relevance MAE: {metrics.relevance_mae:.4f}")
    print(
        f"Contradicted : DetectionRate={metrics.contradicted_detection_rate:.4f}  "
        f"Precision={metrics.contradicted_precision:.4f}  "
        f"(TP={metrics.contradicted_tp}, FP={metrics.contradicted_fp}, FN={metrics.contradicted_fn})"
    )
    print("-" * 72)
    print("Per-Label Breakdown:")
    for lbl, stats in metrics.per_label.items():
        print(
            f"  {lbl:<14s} n={stats['count']:<2d} "
            f"exact={stats['exact_match_rate']:.2f}  "
            f"rel_pred={stats['avg_pred_relevance']:.2f} "
            f"(exp={stats['avg_expected_relevance']:.2f}, mae={stats['relevance_mae']:.2f})  "
            f"adopted={stats['pred_adopted_count']}  contradicted={stats['pred_contradicted_count']}"
        )

    mismatches = [
        p
        for p in predictions
        if p.error is not None
        or p.pred_adopted != p.expected_adopted
        or p.pred_contradicted != p.expected_contradicted
    ]
    if mismatches:
        print("-" * 72)
        print(f"Mismatches / Errors ({len(mismatches)}):")
        for p in mismatches:
            if p.error:
                print(f"  [ERROR] {p.case_id} ({p.label}): {p.error}")
            else:
                print(
                    f"  [DIFF]  {p.case_id} ({p.label}): "
                    f"adopted={p.pred_adopted} (exp={p.expected_adopted}), "
                    f"contradicted={p.pred_contradicted} (exp={p.expected_contradicted}), "
                    f"rel={p.pred_relevance:.2f} (exp={p.expected_relevance:.2f})"
                )

    if verbose:
        print("-" * 72)
        print("All Case Predictions:")
        for p in predictions:
            print(
                f"  {p.case_id} [{p.label:<13s}] "
                f"rel={p.pred_relevance:.2f}/{p.expected_relevance:.2f} "
                f"adopted={str(p.pred_adopted):<5s}/{str(p.expected_adopted):<5s} "
                f"contra={str(p.pred_contradicted):<5s}/{str(p.expected_contradicted):<5s} "
                f"quote={p.evidence_quote[:40]!r}"
            )

    print("=" * 72)
    print(
        f"GATE [{gate_str}]: Adopted Precision {metrics.adopted_precision:.4f} "
        f"{'>=' if metrics.passed else '<'} {metrics.min_precision_threshold:.2f}"
    )
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_exists(str(ROOT))
    parser = argparse.ArgumentParser(
        description="Calibrate AttributionJudge against evals/judge_calibration_set.json"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to calibration JSON dataset (default: evals/judge_calibration_set.json)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_MIN_PRECISION,
        help=f"Minimum adopted Precision gate (default: {DEFAULT_MIN_PRECISION})",
    )
    parser.add_argument(
        "--offline-mock",
        action="store_true",
        help="Run deterministic offline mock mode (zero LLM cost, suitable for CI)",
    )
    parser.add_argument("--model", type=str, default=None, help="Override Judge LLM model name")
    parser.add_argument("--base-url", type=str, default=None, help="Override Judge LLM base URL")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent LLM calls (default: 4)")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON report")
    parser.add_argument("--verbose", action="store_true", help="Print per-case prediction table")
    args = parser.parse_args(argv)

    dataset = load_calibration_set(args.dataset)
    metrics, predictions = asyncio.run(
        run_calibration(
            dataset,
            offline_mock=args.offline_mock,
            model=args.model,
            base_url=args.base_url,
            min_precision=args.min_precision,
            concurrency=args.concurrency,
        )
    )

    mode = "offline-mock" if args.offline_mock else "live"
    judge_preview = AttributionJudge(base_url=args.base_url, model=args.model)
    model_name = "offline-mock" if args.offline_mock else judge_preview._model
    _print_report(metrics, predictions, mode=mode, model_name=model_name, verbose=args.verbose)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "mode": mode,
            "model": model_name,
            "dataset": str(args.dataset),
            "metrics": metrics.to_dict(),
            "predictions": [p.to_dict() for p in predictions],
        }
        args.json_out.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Report written to: {args.json_out}")

    return 0 if metrics.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
