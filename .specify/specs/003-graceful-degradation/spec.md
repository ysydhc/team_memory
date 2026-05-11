# Feature Specification: Graceful Degradation with Pending Queue

**Created**: 2026-05-12
**Status**: Draft
**Input**: When PostgreSQL or LiteLLM is unreachable, queue failed operations in SQLite and retry with exponential backoff. CLI command to view queue status.

## Current State

| Failure | Current behavior | Problem |
|---------|-----------------|---------|
| LLM call fails | _fallback_publish (save without refinement) | retry would be better than immediate fallback |
| PG publish fails | exception, mark_published anyway | data lost, wrong status |
| _retry_counts | in-memory dict | lost on daemon restart |
| No queue visibility | nothing exposed | user doesn't know what failed |

## Design Decisions

- A: Independent PendingQueue class (not inlined)
- B: Refinement failure -> pending_ops queue, retry with backoff
- C: Same SQLite file as drafts.db (no extra lock complexity)

## User Scenarios

### P1: PG temporarily down, data queued and retried
- As a TeamMemory user
- I want to continue using AI normally when PG is briefly unavailable
- So that no experience data is lost
- Acceptance Criteria:
  1. Stop PG -> draft_publish fails -> operation queued in pending_ops
  2. Start PG -> next tick retries succeed -> experience appears in PG
  3. tm-cli queue stats shows 0 pending after recovery

### P2: LiteLLM rate-limited (429), refinement retried with backoff
- As a TeamMemory user
- I want LLM refinement to retry with backoff instead of immediate fallback
- Acceptance Criteria:
  1. LLM returns 429 -> operation queued, not fallback-published
  2. After backoff (30s -> 60s -> 120s) -> retry
  3. After 3 failures -> permanent_failure + fallback_publish
  4. Session stays "refining" during retries, not "published"

### P3: View queue status via CLI
- As a system administrator
- I want to see what operations are in the retry queue
- Acceptance Criteria:
  1. tm-cli queue list -> table of pending ops
  2. tm-cli queue stats -> summary counts
  3. tm-cli queue retry <id> -> force immediate retry
  4. tm-cli queue clear --id <id> -> remove entry
  5. make stats includes queue summary

### P4: Queue survives daemon restart
- As a user who restarts daemon mid-retry
- I want queued operations to survive restart
- Acceptance Criteria:
  1. Queue in SQLite, survives process restart
  2. On startup, PendingQueue loads existing entries
  3. Next tick picks up where it left off

## Technical Requirements

### Data Flow
Operation fails (LLM 429 / PG down)
  -> PendingQueue.enqueue(op_type, payload, session_id)
  -> SQLite pending_ops table

RefinementWorker._tick():
  -> PendingQueue.get_due_ops()         (expired tasks)
  -> for each op: retry or escalate
  -> on success: PendingQueue.dequeue(id)
  -> on failure: backoff / mark permanent

### Interface Contracts

#### PendingQueue class
| Method | Input | Output |
|--------|-------|--------|
| enqueue | op_type, session_id, payload | op_id |
| get_due_ops | — | list of ops past next_retry_at |
| dequeue | op_id | — |
| mark_failed | op_id | — |
| stats | — | {pending, retrying, failed} |
| list_ops | op_type (optional) | list of all queued ops |

#### CLI commands
| Command | Args | Output |
|---------|------|--------|
| tm-cli queue list | --type X | table of pending ops |
| tm-cli queue stats | — | N pending, N retrying, N failed |
| tm-cli queue retry | --id X | "Retrying..." or error |
| tm-cli queue clear | --id X, --all | "Cleared N entries" |

#### Changes to existing modules
| Module | Change |
|--------|--------|
| RefinementWorker | LLM failure -> enqueue, not _fallback_publish |
| RefinementWorker | _fallback_publish only after max_retries exceeded |
| tm_sink | PG failure -> enqueue for retry |
| daemon/app.py | instantiate PendingQueue, pass to RefinementWorker |
| cli.py | add cmd_queue_* commands |
| conversation_buffer.py | add "refining" status state |

### pending_ops table
CREATE TABLE pending_ops (
  id TEXT PRIMARY KEY,
  op_type TEXT NOT NULL,        -- 'publish' | 'refine' | 'memory' | 'embed'
  session_id TEXT,
  payload_json TEXT NOT NULL,
  retries INTEGER DEFAULT 0,
  next_retry_at TEXT NOT NULL,
  last_error TEXT,
  status TEXT DEFAULT 'pending',
  created_at TEXT DEFAULT (datetime('now'))
);

### Backoff strategy
- 1st retry: 30s after failure
- 2nd retry: 60s
- 3rd retry: 120s
- After 3rd failure: permanent_failure, fallback, Feishu alert

## Edge Cases
1. Queue > 200 entries -> Feishu alert, refuse new entries
2. All 3 deps down -> queue handles gracefully
3. Corrupted payload -> skip, mark failed
4. Kill daemon mid-retry -> entry stays pending after restart
5. Duplicate session -> dedup in enqueue()

## Out of Scope
- Auto-scaling workers
- Priority queue
- Cross-machine persistence
- Web UI for queue
