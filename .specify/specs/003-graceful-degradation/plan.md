# Implementation Plan: Graceful Degradation with Pending Queue

**Spec**: spec.md
**Status**: Ready for review
**Updated**: 2026-05-12 (incorporated review findings)

## Architecture

### New Module
| Module | File | Role |
|--------|------|------|
| PendingQueue | daemon/pending_queue.py | SQLite queue, retry/backoff/dedup/limits |

### Modified Modules
| Module | File | Key change |
|--------|------|------------|
| ConversationBuffer | daemon/conversation_buffer.py | add "refining" status, startup recovery |
| RefinementWorker | daemon/refinement_worker.py | enqueue on failure, retry from queue, drop _retry_counts |
| LocalTMSink | daemon/tm_sink.py | return error status on PG failure (caller enqueues) |
| Daemon app | daemon/app.py | instantiate + wire PendingQueue |
| CLI | cli.py | cmd_queue_* commands, db_path resolution |

### Data Flow
```
_refine_one fails (LLM 429 or PG down)
  |
  +--> PendingQueue.enqueue(op_type="refine", session_id, payload)
  |       |-- dedup check: if session already in queue -> skip
  |       |-- queue_size check: if >= 200 -> reject + Feishu alert
  |       +-- INSERT pending_ops (retries=0, next_retry_at=now+30s)

RefinementWorker._tick():
  |
  +--> 1. Process queue retries first (积压优先)
  |      get_due_ops() -> for each:
  |        try exec_retry(op) -> success: dequeue
  |        except -> bump_retry (backoff) or mark_failed (if retries>=3)
  |
  +--> 2. Process new needs_refinement sessions
         for each: mark_refining -> _refine_one -> on failure: enqueue
```

### Session Status Flow
```
active -> needs_refinement -> refining -> published
                                  |            ^
                          (fail)  +-- enqueue --+ (retry later)
                          (3x)    +-- mark_failed + _fallback_publish -> published
```

### Startup Recovery
Daemon startup -> reset all "refining" -> "needs_refinement"
(Prevents stuck sessions after daemon kill mid-refinement)

## Implementation Steps

### Step 1: Create PendingQueue class
- File: src/team_memory/daemon/pending_queue.py (new)
- API:
  - enqueue(op_type, session_id, payload) -> op_id | None
    - before INSERT: check if session_id already queued -> return None (dedup)
    - before INSERT: count pending ops -> if >= 200 -> return None + log alert
  - get_due_ops() -> list[dict]   (WHERE next_retry_at <= now AND status != 'failed')
  - dequeue(op_id): DELETE
  - bump_retry(op_id, error, backoff_seconds): retries++, next_retry_at += backoff
  - mark_failed(op_id): status='failed'
  - stats() -> {"pending": N, "retrying": N, "failed": N}
  - list_ops(op_type=None) -> list of all non-completed ops
- Constants:
  - BACKOFF_SECONDS = [30, 60, 120]
  - MAX_RETRIES = 3
  - MAX_QUEUE_SIZE = 200
- Tests: test_enqueue_dequeue, test_enqueue_duplicate_rejected, test_backoff_sequence,
  test_queue_full_rejected, test_get_due_ops_only_expired

### Step 2: Add "refining" status to ConversationBuffer
- File: src/team_memory/daemon/conversation_buffer.py
- New methods:
  - mark_refining(session_id): UPDATE status='refining'
  - reset_refining_sessions(): UPDATE status='needs_refinement' WHERE status='refining'
  - get_needs_refinement(): modified to exclude refining (WHERE status='needs_refinement' only)
- Tests: test_mark_refining, test_refining_excluded_from_needs_refinement,
  test_reset_refining_on_startup

### Step 3: Integrate PendingQueue into RefinementWorker
- File: src/team_memory/daemon/refinement_worker.py
- Constructor: add queue parameter
- _tick() changes:
  1. Process queue retries FIRST (get_due_ops)
     - For each: load payload, retry _refine_one
     - Success -> dequeue, mark_published
     - Failure -> bump_retry or mark_failed
  2. Then process needs_refinement as before, but:
     - Before _refine_one: mark_refining()
     - On failure: enqueue(payload=draft) instead of _fallback_publish
     - On max retries (already tracked by queue): _fallback_publish + mark_failed
- Remove: _retry_counts dict (replaced by queue)
- Tests: test_failure_enqueues, test_retry_succeeds, test_max_retries_fallback,
  test_queue_retried_before_new_tasks

### Step 4: Handle PG failures in LocalTMSink
- File: src/team_memory/daemon/tm_sink.py
- draft_publish(): on httpx error -> return {"error": "publish_failed", "detail": str(e)}
- draft_save(): same pattern
- Note: sink does NOT enqueue itself. RefinementWorker catches the error and enqueues.
- Tests: test_draft_publish_returns_error_on_connection_refused

### Step 5: Add CLI queue commands
- File: src/team_memory/cli.py
- db_path resolution:
  - Default: ~/.cache/tm-pipeline/drafts.db
  - Config override: daemon_config.cache_dir / "drafts.db"
  - Env override: TM_PIPELINE_DB_PATH
  - Use same path as ConversationBuffer (load from config or default)
- Commands:
  - cmd_queue_list: print table (id, type, session, retries, next_retry, status)
    --type X to filter
  - cmd_queue_stats: print summary + breakdown
  - cmd_queue_retry --id X: SET next_retry_at='now', status='pending'
  - cmd_queue_clear --id X | --all: DELETE matching rows
- Tests: test_queue_list_format, test_queue_stats_counts, test_queue_retry_resets_timer,
  test_queue_clear_removes

### Step 6: Integrate into Makefile
- make stats: append queue summary line
  "Queue: 3 pending, 0 retrying, 1 failed"
  (calls tm-cli queue stats internally)

### Step 7: E2E verification
- Script: scripts/e2e-verify-with-failures.sh (new, or extend existing e2e-verify)
- Steps:
  1. docker compose stop postgres
  2. make e2e-verify (should pass — data goes to queue)
  3. tm-cli queue stats (should show 1+ pending)
  4. docker compose start postgres
  5. sleep 45 (wait for tick + retry)
  6. tm-cli queue stats (should show 0 pending)
  7. docker exec psql check experience exists
  8. docker compose stop -> start for normal state restoration

## Test Strategy
- Unit: PendingQueue CRUD, backoff calc, limit enforcement, dedup
- Integration: RefinementWorker + PendingQueue, ConversationBuffer status flow
- E2E: PG stop/start cycle with data verification

## Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| kill daemon mid-refining -> session stuck | startup: reset_all_refining -> needs_refinement |
| same session queued twice | enqueue() dedup check by session_id |
| CLI can't find db_path | documented resolution chain, same as ConversationBuffer |
| queue full while PG down | alert at 150 (before 200 limit), Feishu notification |
| concurrent CLI + worker on pending_ops | SQLite WAL mode, worker is single writer |
