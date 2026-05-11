# Feature Specification: Session State Machine Consistency

**Created**: 2026-05-13
**Status**: Draft
**Input**: Add "publishing" intermediate state, harden the session state machine so no partial writes or stuck sessions survive daemon restart.

## Current State Machine

```
active ──(convergence/idle)──→ needs_refinement
                                    │
                              (tick picks up)
                                    │
                                    ↓
                               refining ──(success)──→ published
                                    │
                              (fail) │
                                    ↓
                              pending_ops queue ──(retry)──→ refining
                                    │
                              (3x fail)
                                    ↓
                              mark_failed + fallback → published
```

## Gaps Found

| Gap | Symptom | Cause |
|-----|---------|-------|
| G1 | Kill daemon after LLM succeeds but before PG write | refined content lost, session stuck in `refining` |
| G2 | PG write succeeds, mark_published fails | duplicate experience on next tick |
| G3 | Startup resets all `refining` to `needs_refinement` | loses visibility into whether a session was being refined or about to publish |
| G4 | No distinction between "waiting for LLM" and "waiting for PG" | same `refining` status for both phases |

## Solution: Add "publishing" State

New state machine:
```
active → needs_refinement → refining → publishing → published
                                │           │
                         (fail) │    (fail) │
                                ↓           ↓
                         pending_ops    pending_ops
                         (retry LLM)    (retry publish)
```

- **refining**: LLM is being called. If daemon dies here → retry from queue is OK (idempotent).
- **publishing**: LLM succeeded, PG write in progress. If daemon dies here → on restart, retry only the PG publish step (NOT re-run LLM). The refined content is already in the payload.

## User Scenarios

### P1: Kill daemon during LLM call — safe retry
- As the system
- I want to retry the LLM call safely after daemon restart
- Acceptance Criteria:
  1. Session status is `refining` when daemon is killed mid-LLM
  2. On restart, `reset_refining_sessions` resets it to `needs_refinement`
  3. RefinementWorker picks it up, re-runs LLM (idempotent)
  4. No data corruption

### P2: Kill daemon during PG publish — no LLM re-run
- As the system
- I want to only retry the PG write after restart, not re-run the expensive LLM call
- Acceptance Criteria:
  1. Session enters `publishing` after LLM succeeds and before PG write
  2. On daemon restart, `reset_publishing_sessions` resets `publishing` to `needs_refinement` with preserved payload (containing refined content)
  3. RefinementWorker picks it up, skips LLM, goes straight to publish

### P3: Startup recovery handles all stuck states
- As the system
- I want all stuck sessions to be recovered correctly on daemon restart
- Acceptance Criteria:
  1. `refining` sessions → reset to `needs_refinement` (safe to re-run LLM)
  2. `publishing` sessions → reset to `needs_refinement`, payload includes flag `skip_llm: true` so worker skips LLM
  3. Log counts of each recovery

### P4: CLI shows healthy state machine
- As an admin
- I want to see how many sessions are in each state
- Acceptance Criteria:
  1. `tm-cli queue stats` or `make diagnose-pipeline` shows per-status counts

## Technical Requirements

### Data Flow

_refine_one:
  1. mark_refining(session_id)                   → status=refining
  2. draft_save(content)                         → experience_id (draft)
  3. _call_llm(content)                          → refined fields
  4. mark_publishing(session_id)                  → status=publishing, save refined payload
  5. draft_publish(experience_id)                → PG published
  6. update_experience(experience_id, fields)     → PG updated
  7. mark_published(session_id)                  → status=published

### ConversationBuffer changes
- `mark_publishing(session_id, payload)`: SET status='publishing', store payload in a `session_payload` column or a new `refined_content` column
- `reset_publishing_sessions()`: UPDATE status='needs_refinement' WHERE status='publishing' (payload preserved)
- `get_needs_refinement()`: also include sessions with status='publishing' that carry skip_llm flag

### RefinementWorker changes
- _tick: when processing a publishing→needs_refinement session, check for skip_llm flag in payload → go straight to draft_publish
- _refine_one: mark as publishing after LLM success, before PG write

### sessions table schema change
```sql
ALTER TABLE sessions ADD COLUMN refined_payload TEXT;
```
(Stores the LLM-refined content so it survives restart)

## Edge Cases
1. draft_save fails before publishing → session stays refining, retry from queue
2. draft_publish fails after publishing → restart retries only publish step
3. update_experience fails after publish → publish already done, just retry update

## Out of Scope
- Transactions across PG + SQLite (impossible — separate DBs)
- Atomic multi-operation support
