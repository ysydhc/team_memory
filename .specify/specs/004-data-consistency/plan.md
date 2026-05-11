# Implementation Plan: Session State Machine Consistency

**Spec**: spec.md
**Status**: Ready for implementation
**Updated**: 2026-05-13 (incorporated 5-role review — 17 findings)

## 5-Role Review Summary

| Role | Found | Key insight |
|------|-------|-------------|
| Backend Dev | experience_id gap, _init_db naming, draft/session confusion | recovery path technically broken |
| QA Engineer | missing test file, quality-gate ordering, stale publishing | test coverage gaps |
| Architect | fragile migration, skip_llm signal, PendingQueue interaction | interface contract underspecified |
| DevOps/SRE | WAL mode missing, JSON.dumps unguarded, burst risk | operational safety gaps |
| End User | naming collision, publish idempotency, no integration test | reliability concerns |

## P0 Fixes (incorporated into plan)

1. **experience_id 持久化** → refined_payload = {"experience_id": "...", "structured": {...}}
2. **get_needs_refinement 返回新字段** → refined_payload (parsed dict or None), skip_llm (bool)
3. **ConversationBuffer 启用 WAL** → `PRAGMA journal_mode=WAL` 在 __aenter__
4. **迁移改用 PRAGMA table_info** → 不用 try/except 吞错误

## P1 Fixes (incorporated)
5. **mark_publishing 在 quality gate 之前** → LLM 成功后立即持久化
6. **JSON.dumps guard** → try/except in mark_publishing
7. **命名** → mark_publishing → mark_publishing_in_progress
8. **_init_db → __aenter__** → 文档修正为正确方法名

## P2 Fixes (documented, deferred)
9-12: PendingQueue交互、同run内恢复、集成测试、重启限流 → 标注为已知限制/后续优化

## Architecture

### Modified Modules
| Module | File | Key change |
|--------|------|------------|
| ConversationBuffer | conversation_buffer.py | WAL, refined_payload列, mark_publishing_in_progress(), reset_publishing_sessions(), get_needs_refinement扩展 |
| RefinementWorker | refinement_worker.py | LLM后立即mark_publishing, skip_llm检查在mark_refining之前, corrupt fallback |
| Daemon app | app.py | 启动时reset_publishing_sessions |
| CLI diagnostics | diagnose-pipeline.py | 状态分布+图标 |

### Session State Machine
```
active → needs_refinement → refining → publishing → published
                                                    ↑
                                         fallback ──┘

refining:   LLM 调用中，断了重来 OK (幂等)
publishing: LLM 已成功 + 结果已持久化 (含experience_id)
            断了只重试 PG write，不重跑 LLM
```

### Data Flow (正常路径)
```
_refine_one:
  1. mark_refining(session_id)                   → status=refining
  2. draft_save(content)                         → experience_id (PG draft)
  3. structured = _call_llm(content)             → refined fields
  4. mark_publishing_in_progress(draft_id, payload={
       "experience_id": experience_id,
       "structured": structured
     })                                           → status=publishing, refined_payload写入
  5. Quality Gate (uses structured)              → 通过继续, 不通过fallback
  6. draft_publish(experience_id)                → PG 发布
  7. update_experience(experience_id, structured) → PG 更新
  8. mark_published(session_id)                  → status=published
```

### Data Flow (重启恢复)
```
daemon 重启:
  reset_refining_sessions()    → refining → needs_refinement
  reset_publishing_sessions()  → publishing → needs_refinement (refined_payload 保留)

RefinementWorker._tick:
  处理 needs_refinement session:
    → 检查 refined_payload 是否存在
    → 不存在/None → 正常 refine 流程 (mark_refining → _refine_one)
    → 存在 → JSON parse:
      - 成功 + 有 experience_id + experience_type is dict:
        skip_llm=true → 从 payload 提取 experience_id + structured
        → 直接 draft_publish + update_experience + mark_published
        (不调用 mark_refining)
      - 成功但缺 experience_id 或 structured 不是 dict:
        logger.error → 走正常 refine (安全回退)
      - 失败 (corrupt): logger.error → 走正常 refine (安全回退)
```

### sessions 表变更
```sql
ALTER TABLE sessions ADD COLUMN refined_payload TEXT;
```
- 迁移: PRAGMA table_info 检查列存在，不存在才 ALTER
- NULL 默认值 → 旧数据安全回退

### get_needs_refinement 返回格式 (新)
```python
{
    "id": str,              # session_id
    "content": str,         # 拼接的消息内容
    "project": str,
    "conversation_id": str,
    "status": "needs_refinement",
    "refined_payload": dict | None,  # NEW: parsed JSON or None
    "skip_llm": bool,                 # NEW: True if refined_payload is valid
}
```

## Implementation Steps

### Step 1: 启用 WAL + refined_payload 列 + 新方法 (TDD)
- File: src/team_memory/daemon/conversation_buffer.py
- New test file: tests/test_conversation_buffer.py (新建)
- Action:
  1. __aenter__: 加 `PRAGMA journal_mode=WAL` (在已有 CREATE TABLE 之后)
  2. 列迁移: `PRAGMA table_info(sessions)` → 检查 refined_payload 不存在 → ALTER TABLE
  3. mark_publishing_in_progress(session_id, payload):
     - json.dumps 用 try/except 保护 (JSON serialization guard)
     - UPDATE status='publishing', refined_payload=json_str
  4. reset_publishing_sessions(): UPDATE status='needs_refinement' WHERE status='publishing', RETURN count
  5. get_needs_refinement(): SELECT 加 refined_payload 列
     - 如果 refined_payload 不为空 → parse JSON → 验证 keys → skip_llm=true + 返回 parsed dict
     - 否则 skip_llm=false, refined_payload=None
- Tests: 6 个新测试
- Checklist 覆盖: [M3] migrate, [M1] experience_id, [M2] return shape, WAL, json guard

### Step 2: RefinementWorker 集成 publishing (TDD)
- File: src/team_memory/daemon/refinement_worker.py
- Tests: tests/test_refinement_worker_unit.py (追加)
- Action:
  1. _refine_one: _call_llm 成功后 → 立即 mark_publishing_in_progress (before quality gate)
     存储 {"experience_id": experience_id, "structured": structured}
  2. _tick: skip_llm 检查必须在 mark_refining 之前:
     ```python
     if draft.get("skip_llm"):
         structured = draft["refined_payload"]["structured"]
         experience_id = draft["refined_payload"]["experience_id"]
         # skip mark_refining + _call_llm, directly:
         await self._sink.draft_publish(draft_id=experience_id)
         await self._sink.update_experience(experience_id, structured)
         await self._buf.mark_published(draft_id)
         continue
     # Normal path: mark_refining → _refine_one
     ```
  3. corrupt/partial payload fallback: 不满足 skip_llm 条件 → 走正常 refine
  4. _fallback_publish: 成功后 `UPDATE sessions SET refined_payload=NULL`
- Tests: 4 个新测试
- Checklist 覆盖: P1-5 (quality gate ordering), P1-6 (JSON guard), P0 skip_llm before mark_refining

### Step 3: 启动时重置 + 诊断工具
- File: src/team_memory/daemon/app.py, scripts/diagnose-pipeline.py
- Action:
  1. app.py: reset_refining_sessions() + reset_publishing_sessions()
     日志: "Reset N refining, M publishing sessions (crash recovery)"
  2. diagnose-pipeline.py:
     - 加 Status Distribution: active/needs_refinement/refining/publishing/published 计数
     - 更新 status_icon: {"refining": "🔵", "publishing": "🟣"}
     - 更新 show_convergence_status 包含新状态
- Tests: 手动验证

### Step 4: 全量验证 + commit
- Action: ruff check → pytest 全量 → make e2e-verify → commit → update roadmap

## Known Limitations (documented)
- [L1] 同 run 内 stale publishing 只在 daemon 重启时恢复
- [L2] PG 孤儿 draft (draft_save后crash) — 预存问题，非本次引入
- [L3] 重启恢复爆发 — 顺序处理，未限流
- [L4] draft_publish 幂等性 — 依赖 TM sink API，未在本 spec 验证
- [L5] PendingQueue + publishing 交互 — 后续 003+004 整合时处理
