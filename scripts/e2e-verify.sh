#!/bin/bash
# TM E2E Verify — 端到端链路验证
# Usage: ./scripts/e2e-verify.sh
#
# 模拟 3 轮 AI 对话，验证全链路:
#   hook → ConversationBuffer → 收敛 → RefinementWorker → PostgreSQL → recall

set -e

PYTHON="/Users/yeshouyou/miniconda3/envs/team_memory/bin/python3"
DB="$HOME/.cache/tm-pipeline/drafts.db"
DAEMON="http://localhost:3901"
SESSION="e2e-verify-$(date +%s)"
PASS=0
FAIL=0

check() {
    if [ "$2" = "pass" ]; then
        echo "  ✅ $1"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $1"
        FAIL=$((FAIL + 1))
    fi
}

echo "═══════════════════════════════════════════"
echo "  TM E2E Verify — Session: $SESSION"
echo "═══════════════════════════════════════════"
echo ""

# --- Pre-check ---
echo "0. Pre-check"
curl -sf "$DAEMON/status" >/dev/null 2>&1 && check "Daemon running" "pass" || { check "Daemon not running" "fail"; exit 1; }
echo ""

# --- Simulate 3-turn conversation ---
echo "1. Simulating 3-turn conversation..."

curl -s -X POST "$DAEMON/hooks/session_start" -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"$SESSION\",\"workspace_roots\":[\"/Users/yeshouyou/Work/agent/team_doc\"]}" >/dev/null

for i in 1 2 3; do
    curl -s -X POST "$DAEMON/hooks/before_prompt" -H "Content-Type: application/json" \
      -d "{\"conversation_id\":\"$SESSION\",\"prompt\":\"Test question $i about tm-cli config\",\"workspace_roots\":[\"/Users/yeshouyou/Work/agent/team_doc\"]}" >/dev/null

    RESP=$(curl -s -X POST "$DAEMON/hooks/after_response" -H "Content-Type: application/json" \
      -d "{\"conversation_id\":\"$SESSION\",\"response_text\":\"Test answer $i: tm-cli config manages user configuration with priority env > .env > toml > default.\",\"workspace_roots\":[\"/Users/yeshouyou/Work/agent/team_doc\"]}")

    CONVERGENCE=$(echo "$RESP" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('convergence',False))" 2>/dev/null)
    if [ "$i" = "3" ] && [ "$CONVERGENCE" = "True" ]; then
        check "Turn $i: convergence triggered" "pass"
    elif [ "$i" -lt "3" ] && [ "$CONVERGENCE" = "False" ]; then
        check "Turn $i: no convergence (expected)" "pass"
    else
        check "Turn $i: convergence=$CONVERGENCE" "fail"
    fi
done

curl -s -X POST "$DAEMON/hooks/session_end" -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"$SESSION\",\"workspace_roots\":[\"/Users/yeshouyou/Work/agent/team_doc\"]}" >/dev/null
echo ""

# --- Verify SQLite ---
echo "2. Verify SQLite"

STATUS=$($PYTHON -c "
import sqlite3
db = sqlite3.connect('$DB')
s = db.execute(\"SELECT status FROM sessions WHERE id = '$SESSION'\").fetchone()
print(s[0] if s else 'not_found')
" 2>/dev/null)

if [ "$STATUS" = "needs_refinement" ] || [ "$STATUS" = "published" ]; then
    check "Session status: $STATUS" "pass"
else
    check "Session status: $STATUS (expected needs_refinement or published)" "fail"
fi

MSG_CHECK=$($PYTHON -c "
import sqlite3
db = sqlite3.connect('$DB')
u = db.execute(\"SELECT COUNT(*) FROM messages WHERE session_id = '$SESSION' AND role = 'user'\").fetchone()[0]
a = db.execute(\"SELECT COUNT(*) FROM messages WHERE session_id = '$SESSION' AND role = 'assistant'\").fetchone()[0]
print(f'{u},{a}')
" 2>/dev/null)

USER_N=$(echo "$MSG_CHECK" | cut -d, -f1)
ASST_N=$(echo "$MSG_CHECK" | cut -d, -f2)

if [ "$USER_N" = "3" ] && [ "$ASST_N" = "3" ]; then
    check "Messages: $USER_N user + $ASST_N assistant" "pass"
else
    check "Messages: $USER_N user + $ASST_N assistant (expected 3+3)" "fail"
fi
echo ""

# --- Wait for RefinementWorker ---
echo "3. Wait for RefinementWorker (max 60s)..."
PUBLISHED=false
for i in $(seq 1 6); do
    sleep 10
    STATUS=$($PYTHON -c "
import sqlite3
db = sqlite3.connect('$DB')
print(db.execute(\"SELECT status FROM sessions WHERE id = '$SESSION'\").fetchone()[0])
" 2>/dev/null)
    if [ "$STATUS" = "published" ]; then
        check "RefinementWorker processed (${i}0s)" "pass"
        PUBLISHED=true
        break
    fi
done
if [ "$PUBLISHED" = "false" ]; then
    check "RefinementWorker timeout (status=$STATUS)" "fail"
fi
echo ""

# --- Verify PostgreSQL ---
echo "4. Verify PostgreSQL"
PG_COUNT=$(docker exec team_doc-postgres-1 psql -U developer -d team_memory -t -c \
  "SELECT COUNT(*) FROM experiences WHERE created_at > NOW() - INTERVAL '3 minutes' AND project = 'team_memory';" 2>/dev/null | tr -d ' ')

if [ "$PG_COUNT" -gt 0 ]; then
    check "PostgreSQL: $PG_COUNT new experience(s)" "pass"
else
    check "PostgreSQL: no new experiences" "fail"
fi
echo ""

# --- Verify Search ---
echo "5. Verify Search"
SEARCH_COUNT=$(curl -s "$DAEMON/recall?query=tm-cli+config&project=team_memory&max_results=3" | \
  $PYTHON -c "import sys,json; d=json.load(sys.stdin); r=d if isinstance(d,list) else d.get('results',[]); print(len(r))" 2>/dev/null)

if [ "$SEARCH_COUNT" -gt 0 ]; then
    check "Search: $SEARCH_COUNT result(s)" "pass"
else
    check "Search: no results" "fail"
fi
echo ""

# --- Cleanup ---
echo "6. Cleanup"
$PYTHON -c "
import sqlite3
db = sqlite3.connect('$DB')
db.execute(\"DELETE FROM messages WHERE session_id = '$SESSION'\")
db.execute(\"DELETE FROM sessions WHERE id = '$SESSION'\")
db.commit()
" 2>/dev/null
docker exec team_doc-postgres-1 psql -U developer -d team_memory -c \
  "DELETE FROM experiences WHERE created_at > NOW() - INTERVAL '3 minutes' AND project = 'team_memory' AND title LIKE 'Draft e2e-%';" >/dev/null 2>&1
check "Test data cleaned" "pass"
echo ""

# --- Summary ---
echo "═══════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
