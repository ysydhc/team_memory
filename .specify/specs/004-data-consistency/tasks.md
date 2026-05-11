# Tasks: Session State Machine Consistency

**Plan**: plan.md (updated with 5-role review)
**Total tasks**: 8

## Task 1: TDD — 创建 test_conversation_buffer.py (6个失败测试)
- File(s): tests/test_conversation_buffer.py (新建)
- Action: 写 6 个测试 (全部应先失败):
  1. test_mark_publishing_in_progress_stores_payload
  2. test_reset_publishing_sessions_returns_count
  3. test_existing_sessions_have_null_refined_payload
  4. test_get_needs_refinement_includes_refined_payload
  5. test_get_needs_refinement_skip_llm_when_valid_payload
  6. test_wal_mode_enabled
- Test: pytest tests/test_conversation_buffer.py → 6 FAIL
- Depends on: none
- Status: [x] done

## Task 2: 实现 ConversationBuffer publishing (6个通过)
- File(s): src/team_memory/daemon/conversation_buffer.py
- Action:
  1. __aenter__: 加 PRAGMA journal_mode=WAL (含 await)
  2. __aenter__: 加 PRAGMA table_info 检查 → ALTER TABLE ADD COLUMN refined_payload TEXT
  3. mark_publishing_in_progress(session_id, payload):
     try: data = json.dumps(payload); except: logger.error; data = "{}"
     UPDATE sessions SET status='publishing', refined_payload=data
  4. reset_publishing_sessions(): 同上，返回 count
  5. get_needs_refinement(): SELECT 加 refined_payload, 解析 JSON + 验证 keys
     返回新字段: refined_payload (dict|None), skip_llm (bool)
  6. 保留旧的 mark_published/mark_session/mark_refining 不变
- Test: 6 tests PASS, existing tests no regression
- Depends on: Task 1
- Status: [x] done

## Task 3: TDD — RefinementWorker publishing test (4个失败)
- File(s): tests/test_refinement_worker_unit.py (追加)
- Action: 写 4 个测试 (应先失败):
  1. test_refine_one_marks_publishing_after_llm
     验证 _call_llm 成功后 → mark_publishing_in_progress 被调用
     参数: (draft_id, {"experience_id": exp_id, "structured": {...}})
  2. test_tick_skip_llm_when_payload_valid
     构造 skip_llm=true 的 needs_refinement session
     验证: mark_refining NOT called, _call_llm NOT called
     验证: draft_publish + update_experience + mark_published called
  3. test_tick_corrupt_payload_falls_back_to_full_refine
     refined_payload 存在但 JSON parse 失败 → 走正常 refine
     验证: mark_refining IS called, _refine_one IS called
  4. test_fallback_publish_clears_refined_payload
     _fallback_publish 成功后 → sessions.refined_payload = NULL
- Test: 4 FAIL
- Depends on: Task 2
- Status: [x] done

## Task 4: 实现 RefinementWorker publishing (4个通过)
- File(s): src/team_memory/daemon/refinement_worker.py
- Action:
  1. _refine_one: _call_llm 成功后 → 立即:
     await self._buf.mark_publishing_in_progress(draft_id, {
         "experience_id": experience_id,
         "structured": structured,
     })
     (before quality gate!)
  2. _tick: needs_refinement 循环改为:
     if draft.get("skip_llm"):
         # 跳过 mark_refining + _call_llm
         payload = draft["refined_payload"]
         await self._sink.draft_publish(draft_id=payload["experience_id"])
         await self._sink.update_experience(payload["experience_id"], payload["structured"])
         await self._buf.mark_published(draft["id"])
         continue
     # Normal path unchanged
  3. _fallback_publish: 成功后 await self._buf mark_published + 清理 refined_payload
  4. _tick 中 queue retry section: 不动 (P2 defer)
- Test: 4 tests PASS, existing tests no regression
- Depends on: Task 3
- Status: [x] done

## Task 5: 启动恢复 + 诊断工具
- File(s): src/team_memory/daemon/app.py, scripts/diagnose-pipeline.py
- Action:
  1. app.py: 在 reset_refining_sessions 后加:
     pub_count = await buf.reset_publishing_sessions()
     logger.info("Reset %d refining, %d publishing (crash recovery)", ref_count, pub_count)
  2. diagnose-pipeline.py:
     - 加 Status Distribution 章节 (per-status count, bar chart)
     - 更新 status_icon map: refining/publishing
     - show_convergence_status 加新状态
- Test: 手动 — daemon 启动日志 + make diagnose-pipeline
- Depends on: Task 2
- Status: [x] done

## Task 6: 全量测试 + lint
- Action:
  1. ruff check
  2. pytest tests/ -q -m "not integration"
  3. make e2e-verify
- Test: all tests pass, lint clean
- Depends on: Task 4, Task 5
- Status: [x] done

## Task 7: Commit + update roadmap
- Action: git commit, mark P0-3 [x] in roadmap.md
- Depends on: Task 6
- Status: [x] done
