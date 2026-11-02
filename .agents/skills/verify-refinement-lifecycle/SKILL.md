---
name: verify-refinement-lifecycle
description: "Formally verify the TeamMemory Daemon session refinement, crash-recovery, importer idempotency, tool-observation trajectory distillation, and PendingQueue retry state machine. Use whenever modifying src/team_memory/daemon/refinement/refinement_worker.py, src/team_memory/daemon/refinement/pipeline.py, src/team_memory/daemon/refinement/chatgpt_importer.py, src/team_memory/daemon/observe/agy_transcript_watcher.py, src/team_memory/daemon/buffers/conversation_buffer.py, src/team_memory/daemon/buffers/pending_queue.py, src/team_memory/daemon/runtime/routes/hooks.py, or src/team_memory/daemon/runtime/app.py."
---

# Verify Daemon Session Refinement Lifecycle

This skill formally verifies the **TeamMemory Daemon session refinement, crash-recovery, importer idempotency, tool-observation trajectory distillation, and retry queue state machine** using TLA+ (`RefinementLifecycle.tla` checked with TLC 1.7.4) paired with a live Python trace-replay harness against the real `RefinementWorker`, `ConversationBuffer`, `PendingQueue`, `import_chatgpt_sessions`, `import_agy_sessions`, `process_tool_observation`, and `ConvergenceDetector` classes.

## Coverage Record & Sync Commit

- **Last confirmed sync commit**: `6362c9d3c951a44fe1f716e371864071c8701c05` *(plus working-tree importer, tool-observation, and lifecycle fixes)*
- **Covered files & functions**:
  - `src/team_memory/daemon/refinement/refinement_worker.py`
    - `RefinementWorker._tick` (lines 377–519)
    - `RefinementWorker._refine_one` (lines 520–759)
    - `RefinementWorker._call_llm` (lines 874–959)
    - `RefinementWorker._fallback_publish_from_payload` & `RefinementWorker._fallback_publish` (lines 960–1029)
  - `src/team_memory/daemon/buffers/conversation_buffer.py`
    - `ConversationBuffer.mark_refining` (lines 212–215)
    - `ConversationBuffer.reset_refining_sessions` (lines 216–229)
    - `ConversationBuffer.mark_publishing_in_progress` (lines 230–252)
    - `ConversationBuffer.reset_publishing_sessions` (lines 253–265)
    - `ConversationBuffer.add_tool_observation` & `ConversationBuffer.get_tool_observations` (lines 363–432)
    - `ConversationBuffer.mark_published` & `ConversationBuffer.mark_needs_refinement` (lines 494–501)
    - `ConversationBuffer.get_needs_refinement` (lines 536–581)
  - `src/team_memory/daemon/buffers/pending_queue.py`
    - `PendingQueue.enqueue` (lines 51–83)
    - `PendingQueue.get_due_ops` (lines 84–111)
    - `PendingQueue.dequeue` & `PendingQueue.dequeue_by_session` (lines 112–123)
    - `PendingQueue.bump_retry` (lines 125–136)
    - `PendingQueue.mark_failed` (lines 137–144)
  - `src/team_memory/daemon/refinement/chatgpt_importer.py`
    - `import_chatgpt_sessions` (lines 296–397)
  - `src/team_memory/daemon/observe/agy_transcript_watcher.py`
    - `import_agy_sessions` (lines 154–238)
  - `src/team_memory/daemon/refinement/pipeline.py`
    - `process_stop` tool-observation loading (lines 458–509)
    - `process_tool_observation` (lines 596–650)
  - `src/team_memory/daemon/quality/convergence_detector.py`
    - `ConvergenceDetector.check_tool_patterns` & `ConvergenceDetector.check_error_to_pass_resolution` (lines 80–146)
  - `src/team_memory/daemon/runtime/routes/hooks.py`
    - `tool_observation`, `ingest_session`, and `ingest_batch` endpoints (lines 443–529)
  - `src/team_memory/daemon/runtime/app.py`
    - Startup crash recovery (`reset_refining_sessions` and `reset_publishing_sessions`, lines 458–467)

When any of the covered files or functions change after `6362c9d3c951a44fe1f716e371864071c8701c05`, run this skill. Move the sync commit forward **only after** you confirm that `RefinementLifecycle.tla` and `scripts/verify_refinement_lifecycle.py` accurately describe the code at the new commit.

## Properties Verified & Harm

1. **`AtMostOnePendingQueueEntry`** *(Holds on current code)*
   - **Property**: A session never has more than one active (`status != 'failed'`) retry entry in `PendingQueue`.
   - **Harm if broken**: Repeated failures enqueue duplicate retry rows for the same session, causing redundant LLM calls and duplicate Experience records in PostgreSQL.
2. **`CrashRecoveryAndSkipLLMReplay` / `TypeOK`** *(Holds on current code)*
   - **Property**: Startup crash recovery resets `refining` and `publishing` sessions back to `needs_refinement`, and a session that already persisted `refined_payload` via `mark_publishing_in_progress` replays publication (`skip_llm=True`) without calling the LLM a second time.
   - **Harm if broken**: Daemon restarts either lose in-flight sessions or re-run expensive LLM extraction after structured output was already saved.
3. **`NoStrandedRefiningSession`** *(Holds on current code — fixed in `refinement_worker.py:493–498`)*
   - **Property**: Between worker ticks, an unpublished session is never left in `status = 'refining'` or `'publishing'` without an active retry in `PendingQueue`.
   - **Harm if broken**: If `mark_refining(draft_id)` sets `sessions.status = 'refining'` and `_call_llm` raises `LLMUnavailableError` without resetting `status` to `'needs_refinement'`, the session is stranded in `'refining'` until the daemon restarts.
4. **`RejectedDraftNeverPublished`** *(Holds on current code — fixed in `refinement_worker.py:658–699`)*
   - **Property**: A draft rejected by the quality gate (`_structural_check` or `_quality_score`) is never published to TeamMemory.
   - **Harm if broken**: Calling `self._sink.draft_publish(draft_id=experience_id)` inside the quality-gate rejection branches publishes rejected low-quality drafts into TeamMemory.
5. **`AtMostOnePublishedExperience`** *(Holds on current code — fixed in `refinement_worker.py:404–409, 483–485, 747–749` & `pending_queue.py:117–123`)*
   - **Property**: A single session is never published more than once to TeamMemory across retries, importer re-runs, and daemon restarts.
   - **Harm if broken**: If Step 2 publishes a session via `skip_llm` replay after a restart without clearing `PendingQueue`, or if Step 1 retries an op whose session is already `published`, the same session is published twice to TeamMemory.
6. **`ImporterPreservesPublishingSession`** *(Holds on current code — fixed in `chatgpt_importer.py:354` & `agy_transcript_watcher.py:184`)*
   - **Property**: Re-running `import_chatgpt_sessions` or `import_agy_sessions` never overwrites an in-flight `status = 'publishing'` session back to `'needs_refinement'`.
   - **Harm if broken**: Omitting `"publishing"` from the importer skip-status tuple causes re-imports to append duplicate messages and reset an in-flight session back to `'needs_refinement'`.
7. **`ToolObservationTrajectoryContract`** *(Holds on current code — fixed in `pipeline.py:624–645`, `conversation_buffer.py:363–432`, `refinement_worker.py:625–630`, `convergence_detector.py:92, 128`)*
   - **Property**: Tool observations recorded via `process_tool_observation` and stored in `ConversationBuffer.tool_observations` preserve command and target-file details through `ConvergenceDetector` and `RefinementWorker._refine_one`.
   - **Harm if broken**: Mismatched parameter/column names (`command` vs `command_summary`, `target_file` vs `target_files`) either raise `TypeError` on `/hooks/tool_observation` or silently drop command/file trajectory details during convergence detection and LLM prompt enrichment.
8. **`IngestRoutesSignatureContract`** *(Holds on current code — fixed in `routes/hooks.py:465–529`)*
   - **Property**: The `/ingest/session` and `/ingest/batch` routes in `routes/hooks.py` pass only valid keyword arguments accepted by `import_chatgpt_sessions` and `import_agy_sessions`.
   - **Harm if broken**: Passing unsupported keyword arguments raises `TypeError` on every `/ingest/session` or `/ingest/batch` HTTP request.

## Scope, Bounds, and What Is Left Out

- **Bounds**: 1 conversation session (`session_id`), up to 3 PostgreSQL draft saves (`pg_drafts \in 0..3`), up to 2 PostgreSQL publications (`pg_published \in 0..2`), and up to 2 active queue entries (`queue_count \in 0..2`). TLC explores 23 distinct states and 34 transitions on the fixed state machine in ~5 seconds.
- **What is left out**:
  - Cross-session vector deduplication (`_dedup_search` cosine thresholds and embedding provider behavior).
  - SQLite WAL locking across multiple OS processes (the daemon runs a single `RefinementWorker` loop per process).
  - LLM prompt quality and JSON parsing internals (`_refine_via_openai`).

## How to Run the Verification

Run the single verification command from the repository root:

```bash
/opt/homebrew/bin/python3.13 .agents/skills/verify-refinement-lifecycle/scripts/verify_refinement_lifecycle.py --strict
```

- Verifies that all invariants (`TypeOK`, `AtMostOnePendingQueueEntry`, `NoStrandedRefiningSession`, `RejectedDraftNeverPublished`, `AtMostOnePublishedExperience`, `ImporterPreservesPublishingSession`, `CrashRecoveryAndSkipLLMReplay`, `ToolObservationTrajectoryContract`, and `IngestRoutesSignatureContract`) pass in both TLC and the live Python code harness. Exits `0` when all checks pass, or `1` on any regression.

## Reading TLA+ Results with `/statechart`

Read every TLA+ result with the pstack `/statechart` skill on `.agents/skills/verify-refinement-lifecycle/RefinementLifecycle.tla` (working in a scratch folder outside the repository). `/statechart` shows each invariant violation TLC finds as its shortest path in plain words, compares `RefinementLifecycle.tla` against `RefinementLifecycleFixed.tla`, and draws the statechart when checks pass.

## Telling a Real Bug from a Modeling Error & Updating the Verification

1. **Compare TLC with Part 3 (Live Python Code Trace Replay)**:
   - If TLC reports an invariant violation on `RefinementLifecycle.tla` **and** Part 3 reproduces the violation against the real `RefinementWorker` / `ConversationBuffer` / `PendingQueue` classes, it is a **real bug in the code**.
   - If TLC reports a violation that does not reproduce against the real Python classes (because a guard in the Python code prevents that transition), the TLA+ action in `RefinementLifecycle.tla` is missing that guard (**modeling error**). Add the guard with its `\* file.py:line` comment.
2. **When making an intentional state-machine change**:
   - Update `RefinementLifecycle.tla`, `RefinementLifecycleFixed.tla`, `RefinementLifecycle.map.mjs`, and `scripts/verify_refinement_lifecycle.py`, re-run the verification command and `/statechart`, and only then update the **Last confirmed sync commit** at the top of this file.
