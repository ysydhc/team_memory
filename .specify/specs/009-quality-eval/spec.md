# Feature Specification: Refinement Quality Evaluation

**Created**: 2026-05-14
**Status**: Draft
**Input**: 精炼后自动记录质量指标 (高/低质量通过, 标题质量), CLI 可查看趋势

## Existing
- quality_gate: _structural_check + _quality_score (score 0-100)
- quality_min_score = 60
- 通过: 继续 publish + update_experience
- 拒绝: draft_publish without update (保留原始内容)

## User Scenarios

### P1: 质量门禁指标可见
- As 管理员
- I want to 知道多少草稿被质量门禁拒绝
- Acceptance:
  1. 每次 quality gate 执行 → record("quality_score", score, {project})
  2. 被拒 → record("quality_rejected", 1, {project, reason})
  3. tm-cli metrics show --name quality_* 能看到趋势

### P2: 标题质量评估
- As 管理员
- I want to 知道精炼后的标题质量
- Acceptance:
  1. 标题为空/为默认值 "Draft xxx" → record title_quality=0
  2. 标题有实质内容 → record title_quality=1
  3. CLI 可查

## Technical Requirements

### refinement_worker.py 改动
- quality gate 通过: metrics.record("quality_score", score, {project})
- quality gate 拒绝: metrics.record("quality_rejected", 1, {project, reason})
- mark_published 前: metrics.record("title_quality", 1 if good_title else 0, {project})

## Out of Scope
- faithfulness_judge 接入 (太重)
- prompt 自动优化
