# Implementation Plan: Refinement Quality Evaluation

**Spec**: spec.md (009)
**Status**: Ready

---

## Step 1: quality gate → metrics (TDD)

### 现状
- `_refine_one` 中 quality gate 段 (refinement_worker.py:535-558)
- 两条拒绝路径:
  - `_structural_check` 失败 → logger.info + draft_publish + mark_published + return
  - `_quality_score < quality_min_score` → logger.info + draft_publish + mark_published + return
- 一条通过路径: 继续往下走到 draft_publish + update_experience + mark_published
- `self._metrics` 在方法作用域内可用 (006a 已接线)
- quality gate 只在 `_cfg.quality_gate_enabled=True` 时执行

### 目标
- 每次 quality gate 执行后（无论通过或拒绝），对 metrics.record 记录

### 具体改动

**refinement_worker.py** — _refine_one 方法中 quality gate 段:

```python
# ---- Quality Gate ----
if self._cfg.quality_gate_enabled:
    passed, reason = self._structural_check(structured)
    if not passed:
        logger.info(...)
        if self._metrics:
            self._metrics.record(
                "quality_rejected", 1,
                {"project": project, "reason": "structure_failed"},
            )
        await self._sink.draft_publish(draft_id=experience_id)
        await self._buf.mark_published(draft_id)
        return

    score = await self._quality_score(structured)
    if score < self._cfg.quality_min_score:
        logger.info(...)
        if self._metrics:
            self._metrics.record(
                "quality_score", float(score),
                {"project": project, "result": "rejected"},
            )
            self._metrics.record(
                "quality_rejected", 1,
                {"project": project, "reason": "below_threshold"},
            )
        await self._sink.draft_publish(draft_id=experience_id)
        await self._buf.mark_published(draft_id)
        return

    # 通过 gate
    if self._metrics:
        self._metrics.record(
            "quality_score", float(score),
            {"project": project, "result": "passed"},
        )
```

**tests/test_refinement_worker_unit.py** — 追加 2 tests:

```python
class TestQualityMetrics:
    @pytest.mark.asyncio
    async def test_quality_score_recorded_on_pass(self):
        """quality gate 通过 → record quality_score."""
        cfg = _make_config(quality_gate_enabled=True, quality_min_score=30)
        buf, sink, metrics = AsyncMock(), AsyncMock(), MagicMock()
        buf.mark_publishing_in_progress = AsyncMock()
        buf.mark_published = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-q"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})
        worker = RefinementWorker(cfg, buf, sink, metrics=metrics)
        with patch.object(worker, "_call_llm", AsyncMock(return_value={"title":"T","problem":"P","solution":"S"})):
            with patch.object(worker, "_quality_score", AsyncMock(return_value=85)):
                with patch.object(worker, "_structural_check", return_value=(True, "")):
                    await worker._refine_one(_make_draft())
        metrics.record.assert_any_call("quality_score", 85.0, {"project": "default", "result": "passed"})

    @pytest.mark.asyncio
    async def test_quality_rejected_recorded_below_threshold(self):
        """quality score < threshold → record quality_rejected + quality_score."""
        cfg = _make_config(quality_gate_enabled=True, quality_min_score=60)
        buf, sink, metrics = AsyncMock(), AsyncMock(), MagicMock()
        buf.mark_publishing_in_progress = AsyncMock()
        buf.mark_published = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-r"})
        sink.draft_publish = AsyncMock(return_value={})
        worker = RefinementWorker(cfg, buf, sink, metrics=metrics)
        with patch.object(worker, "_call_llm", AsyncMock(return_value={"title":"T","problem":"P","solution":"S"})):
            with patch.object(worker, "_quality_score", AsyncMock(return_value=30)):
                with patch.object(worker, "_structural_check", return_value=(True, "")):
                    await worker._refine_one(_make_draft())
        metrics.record.assert_any_call("quality_score", 30.0, {"project": "default", "result": "rejected"})
        metrics.record.assert_any_call("quality_rejected", 1, {"project": "default", "reason": "below_threshold"})
```

### 验收: 2 tests PASS

---

## Step 2: 标题质量 → metrics (TDD)

### 现状
- mark_published 前不判断标题
- 标题可能为 `"Draft e2e-sim-..."` 或空字符串
- 已在 006a 确认过这个现象

### 目标
- 好标题: not empty + not startswith "Draft " + len > 3
- 坏标题: 其他
- → record binary: `title_quality=1 or 0`
- 时机: mark_published 之前（确保 session 仍可访问）

### 具体改动

**refinement_worker.py** — _refine_one 末尾，mark_published 之前:

```python
# Step 5: mark local draft published
if self._metrics:
    title = structured.get("title", "")
    is_good = bool(title) and not title.startswith("Draft ") and len(title) > 3
    self._metrics.record(
        "title_quality",
        1.0 if is_good else 0.0,
        {"project": project},
    )
await self._buf.mark_published(draft_id)
```

**tests/test_refinement_worker_unit.py** — 追加 1 test:

```python
class TestTitleQuality:
    @pytest.mark.asyncio
    async def test_title_quality_good_title_recorded(self):
        """Good title → title_quality=1 recorded."""
        cfg = _make_config(quality_gate_enabled=False)
        buf, sink, metrics = AsyncMock(), AsyncMock(), MagicMock()
        buf.mark_publishing_in_progress = AsyncMock()
        buf.mark_published = AsyncMock()
        sink.draft_save = AsyncMock(return_value={"id": "exp-t"})
        sink.draft_publish = AsyncMock(return_value={})
        sink.update_experience = AsyncMock(return_value={})
        worker = RefinementWorker(cfg, buf, sink, metrics=metrics)
        with patch.object(worker, "_call_llm", AsyncMock(return_value={"title":"Good Title","problem":"P","solution":"S"})):
            await worker._refine_one(_make_draft())
        metrics.record.assert_any_call("title_quality", 1.0, {"project": "default"})
```

### 验收: 1 test PASS

---

## Step 3: 全量验证 + commit
- ruff check → clean
- pytest -q -m "not integration" → 1865+ tests pass
- git commit