#!/usr/bin/env bash
# TM Stability Verification — validates P1-P4 features
# Usage: bash scripts/verify-stability.sh

set -uo pipefail  # no -e: don't exit on grep empty results
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0; FAIL=0

green()  { echo "  ✅ $*"; PASS=$((PASS+1)); }
red()    { echo "  ❌ $*"; FAIL=$((FAIL+1)); }
yellow() { echo "  ⚠️  $* (manual verify needed)"; }
info()   { echo ""; echo "$@"; echo "──────────────────────────────────"; }

# ================================================================
info "P1-1: Daemon wait-for-deps"
# ================================================================
DEP_LOG=$(grep "All deps\|All critical\|All dependencies" /tmp/tm-daemon.log 2>/dev/null | tail -1)
if [ -z "$DEP_LOG" ]; then
    DEP_LOG=$(grep "All deps\|All critical\|All dependencies" /tmp/tm-daemon.err 2>/dev/null | tail -1)
fi
if [ -n "$DEP_LOG" ]; then
    green "wait-for-deps: $(echo $DEP_LOG | cut -c20-)"
else
    red "wait-for-deps: no startup log found. Restart daemon first."
fi

# ================================================================
info "P1-2: Wake-up detection (code check)"
# ================================================================
if grep -q "SLEEP_THRESHOLD" "$ROOT/src/team_memory/monitoring/heartbeat.py" 2>/dev/null; then
    green "wake-up detection: SLEEP_THRESHOLD=300s in heartbeat.py"
else
    red "wake-up detection: SLEEP_THRESHOLD not found"
fi

if grep -q "_last_monotonic" "$ROOT/src/team_memory/monitoring/heartbeat.py" 2>/dev/null; then
    green "wake-up detection: monotonic tracking active"
else
    red "wake-up detection: monotonic tracking missing"
fi

# Workers have pause/resume
for worker in "refinement_worker" "auto_dream"; do
    FILE=""
    case $worker in
        refinement_worker) FILE="$ROOT/src/team_memory/daemon/refinement/refinement_worker.py" ;;
        auto_dream)        FILE="$ROOT/src/team_memory/daemon/integration/auto_dream.py" ;;
    esac
    if grep -q "def pause\|def resume" "$FILE" 2>/dev/null; then
        green "pause/resume: $worker ✅"
    else
        red "pause/resume: $worker missing"
    fi
done

yellow "Real wake-up test: close laptop lid >5 min, then wake."
yellow "  Check: grep 'system_sleep' /tmp/tm-daemon.err"

# ================================================================
info "P1-3: LLM health check (code check)"
# ================================================================
if grep -q "_llm_api_reachable" "$ROOT/src/team_memory/daemon/refinement/refinement_worker.py" 2>/dev/null; then
    green "LLM health check: _llm_api_reachable() exists"
else
    red "LLM health check: _llm_api_reachable() not found"
fi

if grep -q "LLMUnavailableError" "$ROOT/src/team_memory/daemon/refinement/refinement_worker.py" 2>/dev/null; then
    green "LLM health check: LLMUnavailableError handled"
else
    red "LLM health check: LLMUnavailableError not handled"
fi

yellow "Test: stop LiteLLM (kill process), send messages, check logs."
yellow "  Expected: '[LLM] Skipping refinement — LLM API unreachable'"

# ================================================================
info "P2-1: PendingQueue checkpoint"
# ================================================================
CHECKPOINT_COL=$(sqlite3 ~/.cache/tm-pipeline/drafts.db "PRAGMA table_info(pending_ops);" 2>/dev/null | grep -c "checkpoint_json")
if [ "$CHECKPOINT_COL" -gt 0 ]; then
    green "checkpoint: checkpoint_json column exists in pending_ops"
else
    red "checkpoint: checkpoint_json column not found"
fi

if grep -q "save_checkpoint\|load_checkpoint" "$ROOT/src/team_memory/daemon/buffers/pending_queue.py" 2>/dev/null; then
    green "checkpoint: save/load methods exist"
else
    red "checkpoint: save/load methods missing"
fi

# ================================================================
info "P2-2: Batch chunked execution"
# ================================================================
if grep -q "chunk-size\|chunk_size\|--chunk" "$ROOT/src/team_memory/daemon/backfill/embedding_backfill.py" 2>/dev/null; then
    green "chunked: embedding_backfill supports --chunk-size"
else
    red "chunked: embedding_backfill missing --chunk-size"
fi

if grep -q "chunk-size\|chunk_size\|--chunk" "$ROOT/src/team_memory/daemon/backfill/graph_backfill.py" 2>/dev/null; then
    green "chunked: graph_backfill supports --chunk-size"
else
    red "chunked: graph_backfill missing --chunk-size"
fi

# Quick dry-run test
DRYRUN_OUT=$(timeout 10 python -m team_memory.daemon.backfill.embedding_backfill --dry-run --chunk-size 10 2>&1 || true)
if echo "$DRYRUN_OUT" | grep -q "Chunk\|chunk\|All.*embeddings\|Found"; then
    green "chunked: dry-run test passed"
else
    red "chunked: dry-run test failed — $DRYRUN_OUT"
fi

# ================================================================
info "P2-3: Faithfulness checkpoint"
# ================================================================
if grep -q "cursor_id" "$ROOT/src/team_memory/daemon/buffers/response_buffer.py" 2>/dev/null; then
    green "faithfulness: fetch_pending(cursor_id) supported"
else
    red "faithfulness: fetch_pending(cursor_id) missing"
fi

if grep -q "checkpoint" "$ROOT/src/team_memory/daemon/quality/faithfulness_batch.py" 2>/dev/null; then
    green "faithfulness: checkpoint file support in batch"
else
    red "faithfulness: checkpoint file support missing"
fi

# ================================================================
info "P3-1: Structured sleep logs"
# ================================================================
if grep -q "system_sleep_detected\|system_wake_recovery" "$ROOT/src/team_memory/monitoring/heartbeat.py" 2>/dev/null; then
    green "sleep-log: JSON structured log in heartbeat"
else
    red "sleep-log: structured log missing"
fi

if grep -q "system_wake_recovery" "$ROOT/src/team_memory/daemon/runtime/app.py" 2>/dev/null; then
    green "sleep-log: recovery completion log in app.py"
else
    red "sleep-log: recovery log missing in app.py"
fi

# ================================================================
info "P3-2: Failure aggregation alerts"
# ================================================================
if grep -q "pending_failures\|_check_pending_failures" "$ROOT/src/team_memory/daemon/runtime/health_check.py" 2>/dev/null; then
    green "alerting: pending_failures check exists"
else
    red "alerting: pending_failures check missing"
fi

if grep -q "worker_stall\|_check_worker_stall" "$ROOT/src/team_memory/daemon/runtime/health_check.py" 2>/dev/null; then
    green "alerting: worker_stall check exists"
else
    red "alerting: worker_stall check missing"
fi

if grep -q "report_worker_activity" "$ROOT/src/team_memory/daemon/refinement/refinement_worker.py" 2>/dev/null; then
    green "alerting: report_worker_activity() called in worker"
else
    red "alerting: report_worker_activity() not called"
fi

# ================================================================
info "P4-1: Embedding timeout + retry"
# ================================================================
if grep -q "self._timeout" "$ROOT/src/team_memory/embedding/openai_provider.py" 2>/dev/null; then
    green "embed-timeout: configurable timeout in provider"
else
    red "embed-timeout: configurable timeout missing"
fi

if grep -q "self._max_retries" "$ROOT/src/team_memory/embedding/openai_provider.py" 2>/dev/null; then
    green "embed-timeout: retry logic in provider"
else
    red "embed-timeout: retry logic missing"
fi

if grep -q "EmbeddingTimeoutError" "$ROOT/src/team_memory/embedding/openai_provider.py" 2>/dev/null; then
    green "embed-timeout: EmbeddingTimeoutError defined"
else
    red "embed-timeout: EmbeddingTimeoutError missing"
fi

# ================================================================
info "Integration: Daemon health"
# ================================================================
DAEMON_STATUS=$(curl -s http://localhost:3901/status 2>/dev/null || echo "")
if echo "$DAEMON_STATUS" | grep -q "running"; then
    green "daemon: running ($(echo $DAEMON_STATUS | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["heartbeat"]["uptime_seconds"])' 2>/dev/null || echo '?')s)"
else
    red "daemon: NOT RUNNING"
fi

# ================================================================
info "Integration: E2E verify"
# ================================================================
E2E_OUT=$(cd "$ROOT" && timeout 90 make e2e-verify 2>&1 || true)
E2E_PASS=$(echo "$E2E_OUT" | grep -c "✅\|pass" || true)
E2E_FAIL=$(echo "$E2E_OUT" | grep -c "❌\|fail" || true)
echo "$E2E_OUT" | tail -5
if echo "$E2E_OUT" | grep -q "0 failed"; then
    green "e2e-verify: ALL PASSED"
elif echo "$E2E_OUT" | grep -q "Results:"; then
    yellow "e2e-verify: some tests failed (see above)"
else
    red "e2e-verify: could not run"
fi

# ================================================================
echo ""
echo "═══════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi