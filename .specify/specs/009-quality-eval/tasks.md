# Tasks: Refinement Quality Evaluation

**Plan**: plan.md
**Total tasks**: 5

## Task 1: TDD — quality gate metrics (2 fail)
- File(s): tests/test_refinement_worker_unit.py (追加)
- Action:
  1. test_quality_score_recorded_on_pass: mock quality_gate=passed, assert metrics.record("quality_score", 85, {result="passed"})
  2. test_quality_rejected_recorded_below_threshold: mock score=30 < min=60, assert metrics.record("quality_rejected", 1) + quality_score record
- Test: 2 FAIL (metrics.record 未被调用)
- Depends on: none
- Status: [x] done

## Task 2: 实现 quality gate → metrics (2 pass)
- File(s): src/team_memory/daemon/refinement_worker.py
- Action:
  1. _structural_check 失败 → metrics.record("quality_rejected", 1, {reason="structure_failed"})
  2. _quality_score < threshold → metrics.record("quality_score", score, {result="rejected"}) + quality_rejected
  3. 通过 → metrics.record("quality_score", score, {result="passed"})
- Test: 2 tests PASS
- Depends on: Task 1
- Status: [x] done

## Task 3: TDD — title quality (1 fail)
- File(s): tests/test_refinement_worker_unit.py (追加)
- Action: test_title_quality_good_title_recorded: mock title="Good Title", quality_gate disabled → assert metrics.record("title_quality", 1.0)
- Test: 1 FAIL
- Depends on: none (just needs TestTitleQuality class)
- Status: [x] done

## Task 4: 实现标题质量 (1 pass)
- File(s): src/team_memory/daemon/refinement_worker.py
- Action: mark_published 前检查 title (not empty + not starts_with "Draft " + len>3) → record 1 or 0
- Test: 1 test PASS
- Depends on: Task 3
- Status: [x] done

## Task 5: 全量验证 + commit
- Action: ruff → pytest -q → make e2e-verify → commit
- Depends on: Task 2, Task 4
- Status: [x] done