# Feature Specification: Pipeline Acceptance Test System

**Created**: 2026-05-10
**Status**: Draft
**Input**: "对系统可靠性进行评估 — 能抓到用户数据，评估 LLM 能跑通，数据能推送给 TM 服务，检索能正常检索到，检索后的评估系统也能正常工作"

## User Scenarios & Testing

### P1: End-to-End Data Flow Verification (端到端数据流验证)
- **As a** developer
- **I want to** verify that data flows from hook input to PostgreSQL storage without breaking
- **So that** I can trust the pipeline works in production
- **Acceptance Criteria**:
  1. Send a simulated hook payload → data appears in ConversationBuffer
  3+ assistant messages → session marked as needs_refinement
  2. RefinementWorker processes the session → experience saved to PostgreSQL
  3. The saved experience has correct project, title, content

### P2: Search Retrieval Verification (检索验证)
- **As a** developer
- **I want to** verify that saved experiences can be retrieved via search
- **So that** the AI can actually use the stored knowledge
- **Acceptance Criteria**:
  1. Save an experience with known content
  2. Search for keywords from that content
  3. The experience appears in search results with score > 0
  4. Results are filtered by project

### P3: Evaluation System Verification (评估系统验证)
- **As a** developer
- **I want to** verify that the evaluation pipeline works end-to-end
- **So that** I can measure search quality and response faithfulness
- **Acceptance Criteria**:
  1. Search log is created when recall is called
  2. [mem:xxx] markers are injected into results
  3. was_used detection works (marker found in response)
  4. Fuzzy match detection works (keyword overlap)
  5. Faithfulness judge LLM can be called and returns a score

### P4: Hook → Daemon → Buffer Verification (Hook 链路验证)
- **As a** developer
- **I want to** verify that all hook endpoints accept real payload formats
- **So that** Cursor/Claude/Hermes hooks actually work
- **Acceptance Criteria**:
  1. /hooks/before_prompt accepts Cursor format and stores user message
  2. /hooks/after_response accepts Cursor format (agent_response field) and stores assistant message
  3. /hooks/session_start creates session in ConversationBuffer
  4. /hooks/session_end triggers personal memory extraction
  5. /hooks/mcp_callback stores search correlation

### P5: Convergence Detection Verification (收敛检测验证)
- **As a** developer
- **I want to** verify that convergence detection actually triggers
- **So that** sessions don't stay active forever
- **Acceptance Criteria**:
  1. After 3+ assistant messages, convergence triggers
  2. Session status changes to needs_refinement
  3. Personal memory extraction is triggered

## Technical Requirements

### Data Flow (完整链路)

```
Hook Payload (Cursor/Claude/Hermes)
    ↓ POST /hooks/before_prompt
pipeline.process_before_prompt()
    ↓ buf.add_message(user) + sink.recall()
ConversationBuffer (SQLite) + PostgreSQL search
    ↓ POST /hooks/after_response
pipeline.process_after_response()
    ↓ buf.add_message(assistant)
ConversationBuffer (SQLite)
    ↓ detect_convergence() — 3+ messages threshold
mark_needs_refinement()
    ↓ RefinementWorker._tick() — every 30s
_refine_one(draft)
    ↓ _dedup_search() + _call_llm() + sink.draft_publish()
PostgreSQL (experiences table)
    ↓ WikiCompiler.compile_incremental()
wiki/*.md files
```

### Interface Contracts

| Module | Input | Output | Storage |
|--------|-------|--------|---------|
| HookPayload | conversation_id, response_text/agent_response | validated Pydantic model | — |
| pipeline.process_before_prompt | HookPayload + config + sink + buf + wiki | {results, project, persona} | ConversationBuffer |
| pipeline.process_after_response | HookPayload + config + sink + buf + detector + refiner | {action, convergence, draft_id} | ConversationBuffer |
| ConversationBuffer | session_id, role, content, project | message_id | SQLite |
| RefinementWorker._refine_one | draft dict | None | PostgreSQL |
| SearchLogWriter | query, project, result_ids | log_id | PostgreSQL search_logs |
| ResponseBuffer | query, response, result_ids | None | PostgreSQL response_buffer |
| FaithfulnessJudge | query, response, context | score (0-1) | PostgreSQL faithfulness_scores |
| EvaluationService | results, agent_response | was_used dict | PostgreSQL search_logs.was_used |

### Dependencies
- PostgreSQL (port 5433) — experiences, search_logs, response_buffer, personal_memories
- Ollama (port 11434) — embedding (qwen3-embedding:0.6b)
- LiteLLM (port 4000) — LLM for refinement + faithfulness judgment
- SQLite — ConversationBuffer, DraftBuffer, PayloadCapture

## Test Scenarios

### Scenario 1: Full Pipeline (Happy Path)
1. Send 3 before_prompt + after_response pairs
2. Wait for RefinementWorker tick (30s)
3. Verify: PostgreSQL has new experience, project=team_memory
4. Verify: search recall returns the new experience
5. Verify: search_log records the query
6. Verify: [mem:xxx] marker injection works

### Scenario 2: Edge Cases
1. Empty response_text → should be skipped, not stored
2. Unknown fields in payload → should be accepted (lenient)
3. Same content sent twice → should dedup
4. Session idle > 30min → should auto-mark for refinement

### Scenario 3: Evaluation Pipeline
1. Call recall → search_log created
2. Send response with [mem:xxx] marker → was_used=True
3. Send response without marker → fuzzy match check
4. Buffer 5 responses → faithfulness judge triggered

## Out of Scope
- Obsidian watcher integration (separate feature)
- Wiki compilation quality (separate feature)
- Performance/load testing
- UI dashboard
- Multi-user authentication

## Verification Command

```bash
# Run the acceptance test suite
PYTHONPATH=src python -m pytest tests/test_pipeline_acceptance.py -v

# Or manually:
make test-acceptance
```
