# Tasks: Graceful Degradation with Pending Queue

**Plan**: plan.md
**Total tasks**: 9

## Task 1: Create PendingQueue class (TDD)
- File(s): src/team_memory/daemon/pending_queue.py (new), tests/test_pending_queue.py (new)
- Action:
  1. Write failing tests: test_enqueue_dequeue, test_enqueue_duplicate_rejected,
     test_backoff_sequence, test_queue_full_rejected, test_get_due_ops_only_expired,
     test_stats_counts, test_list_ops
  2. Implement PendingQueue class:
     - __init__(db_path): CREATE TABLE IF NOT EXISTS pending_ops
     - enqueue(op_type, session_id, payload): dedup + size check + INSERT
     - get_due_ops(): SELECT where next_retry_at <= now AND status != 'failed'
     - dequeue(op_id): DELETE
     - bump_retry(op_id, error, backoff_seconds)
     - mark_failed(op_id)
     - stats(), list_ops()
     - BACKOFF = [30, 60, 120], MAX_RETRIES = 3, MAX_QUEUE_SIZE = 200
  3. Tests pass
- Test: 7 tests, all passing
- Depends on: none
- Status: [ ] pending

## Task 2: Add "refining" status to ConversationBuffer (TDD)
- File(s): src/team_memory/daemon/conversation_buffer.py, tests/test_conversation_buffer.py
- Action:
  1. Write failing tests: test_mark_refining, test_refining_excluded_from_needs_refinement,
     test_reset_refining_sessions
  2. Add methods:
     - mark_refining(session_id): UPDATE status='refining'
     - reset_refining_sessions(): UPDATE status='needs_refinement' WHERE status='refining'
  3. Modify get_needs_refinement(): add WHERE status='needs_refinement' (already there,
     but verify refining is excluded — it should be since we don't query for it)
  4. Tests pass
- Test: 3 tests, all passing; existing tests unaffected
- Depends on: none
- Status: [ ] pending

## Task 3: Integrate PendingQueue into RefinementWorker (TDD)
- File(s): src/team_memory/daemon/refinement_worker.py, tests/test_refinement_worker.py
- Action:
  1. Write failing tests:
     - test_failure_enqueues_to_pending_queue
     - test_retry_from_queue_succeeds
     - test_max_retries_triggers_fallback
     - test_queue_retried_before_new_tasks
  2. Constructor: accept queue parameter (optional, for backward compat)
  3. _tick() changes:
     a. Process queue.get_due_ops() FIRST (before new needs_refinement)
        - For each: load payload -> retry _refine_one -> success: dequeue + mark_published
        - Failure: bump_retry or mark_failed (if retries >= 3)
     b. Then needs_refinement loop:
        - mark_refining(draft_id) before processing
        - On _refine_one exception: enqueue(payload=draft) instead of _fallback_publish
  4. When queue marks as failed (retries >= 3): trigger _fallback_publish
  5. Remove in-memory _retry_counts dict
  6. Tests pass
- Test: 4 tests, all passing; existing RefinementWorker tests unaffected
- Depends on: Task 1, Task 2
- Status: [ ] pending

## Task 4: Return errors from LocalTMSink on PG failure
- File(s): src/team_memory/daemon/tm_sink.py, tests/test_tm_sink.py
- Action:
  1. Write failing test: test_draft_publish_returns_error_dict_on_connection_refused
  2. In draft_publish() and draft_save(): catch httpx errors, return {"error": ...} dict
     instead of raising exception
  3. Tests pass
- Test: 1 test, passing
- Depends on: none
- Status: [ ] pending

## Task 5: Wire PendingQueue in daemon/app.py
- File(s): src/team_memory/daemon/app.py
- Action:
  1. Instantiate PendingQueue with same db_path as ConversationBuffer
  2. Call buf.reset_refining_sessions() on startup (recovery from crash)
  3. Pass queue to RefinementWorker constructor
  4. Store in application.state for potential access
- Test: Manual — start daemon, check logs show "reset N refining sessions"
- Depends on: Task 1, Task 2, Task 3
- Status: [ ] pending

## Task 6: Add CLI queue commands
- File(s): src/team_memory/cli.py, tests/test_cli_config_setup.py (or new test_cli_queue.py)
- Action:
  1. db_path resolution: use same logic as existing stats (from config or default)
  2. cmd_queue_list: open PendingQueue, call list_ops(), print table
     --type X filter support
  3. cmd_queue_stats: call stats(), print summary
  4. cmd_queue_retry --id X: open queue, update next_retry_at='now'
  5. cmd_queue_clear --id X | --all: delete matching rows
  6. Register subcommands in argparse
  7. Tests: test_queue_list_output, test_queue_stats_output, test_queue_retry_resets,
     test_queue_clear_removes
- Test: 4 tests, passing; existing CLI tests unaffected
- Depends on: Task 1
- Status: [ ] pending

## Task 7: Update Makefile — queue in stats
- File(s): Makefile
- Action:
  1. Add to stats target: after existing stats, call tm-cli queue stats
  2. Format: "Queue: N pending, N retrying, N failed"
  3. Or: add as separate make queue-stats target
- Test: make stats shows queue line
- Depends on: Task 6
- Status: [ ] pending

## Task 8: Write E2E verification script
- File(s): scripts/e2e-verify-graceful.sh (new)
- Action:
  1. Check PG running -> make e2e-verify passes
  2. docker compose stop postgres
  3. make e2e-verify (data enters queue)
  4. tm-cli queue stats (shows 1+ pending)
  5. docker compose start postgres
  6. Wait for tick + retry (sleep 45)
  7. tm-cli queue stats (shows 0 pending)
  8. docker exec psql verify experience exists
  9. Cleanup
- Test: Run script, all steps pass
- Depends on: Task 5, Task 6
- Status: [ ] pending

## Task 9: Final verification + commit
- Action:
  1. ruff check all modified files
  2. pytest tests/ -q -m "not integration" (full suite)
  3. make e2e-verify
  4. git commit
  5. Update roadmap.md: mark P0-2 [x]
- Depends on: Task 8
- Status: [ ] pending
