#!/bin/bash
# TM Pipeline Health Check — 人工验证脚本
# Usage: ./scripts/verify-pipeline.sh
# 
# 验证整个数据链路是否正常工作

set -e

PYTHON="/Users/yeshouyou/miniconda3/envs/team_memory/bin/python"
DB_PATH="$HOME/.cache/tm-pipeline/drafts.db"
DAEMON_URL="http://localhost:3901"
WEB_URL="http://localhost:9111"
PASS=0
FAIL=0
WARN=0

check() {
    local desc="$1"
    local result="$2"
    if [ "$result" = "pass" ]; then
        echo "  ✅ $desc"
        PASS=$((PASS + 1))
    elif [ "$result" = "warn" ]; then
        echo "  ⚠️  $desc"
        WARN=$((WARN + 1))
    else
        echo "  ❌ $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================="
echo "  TM Pipeline Health Check"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

# --- 1. Infrastructure ---
echo "1. Infrastructure"

# PostgreSQL
if docker exec team_doc-postgres-1 pg_isready -U developer -q 2>/dev/null; then
    check "PostgreSQL" "pass"
else
    check "PostgreSQL not running" "fail"
fi

# Ollama
if curl -sf --max-time 3 "http://localhost:11434/api/tags" >/dev/null 2>&1; then
    check "Ollama" "pass"
else
    check "Ollama not running" "fail"
fi

# Web
HEALTH=$(curl -sf --max-time 3 "$WEB_URL/health" 2>/dev/null)
if [ $? -eq 0 ]; then
    STATUS=$(echo "$HEALTH" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    if [ "$STATUS" = "healthy" ]; then
        check "Web ($STATUS)" "pass"
    else
        check "Web ($STATUS)" "warn"
    fi
else
    check "Web not running" "fail"
fi

# Daemon
DAEMON_STATUS=$(curl -sf --max-time 3 "$DAEMON_URL/status" 2>/dev/null)
if [ $? -eq 0 ]; then
    check "Daemon" "pass"
else
    check "Daemon not running" "fail"
fi

echo ""

# --- 2. SQLite State ---
echo "2. SQLite State"

if [ -f "$DB_PATH" ]; then
    SESSIONS=$($PYTHON -c "
import sqlite3
db = sqlite3.connect('$DB_PATH')
active = db.execute(\"SELECT COUNT(*) FROM sessions WHERE status='active'\").fetchone()[0]
needs = db.execute(\"SELECT COUNT(*) FROM sessions WHERE status='needs_refinement'\").fetchone()[0]
pub = db.execute(\"SELECT COUNT(*) FROM sessions WHERE status='published'\").fetchone()[0]
print(f'active={active} needs_refinement={needs} published={pub}')
" 2>/dev/null)
    check "Sessions: $SESSIONS" "pass"

    MSGS=$($PYTHON -c "
import sqlite3
db = sqlite3.connect('$DB_PATH')
total = db.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
users = db.execute(\"SELECT COUNT(*) FROM messages WHERE role='user'\").fetchone()[0]
assts = db.execute(\"SELECT COUNT(*) FROM messages WHERE role='assistant'\").fetchone()[0]
print(f'total={total} user={users} assistant={assts}')
" 2>/dev/null)
    check "Messages: $MSGS" "pass"

    # Check if there are assistant messages (key indicator)
    ASST_COUNT=$($PYTHON -c "
import sqlite3
db = sqlite3.connect('$DB_PATH')
print(db.execute(\"SELECT COUNT(*) FROM messages WHERE role='assistant'\").fetchone()[0])
" 2>/dev/null)
    if [ "$ASST_COUNT" -gt 0 ]; then
        check "Assistant messages exist ($ASST_COUNT)" "pass"
    else
        check "No assistant messages — hooks may not be working" "fail"
    fi
else
    check "SQLite DB not found at $DB_PATH" "fail"
fi

echo ""

# --- 3. PostgreSQL State ---
echo "3. PostgreSQL State"

PG_EXP=$(docker exec team_doc-postgres-1 psql -U developer -d team_memory -t -c "SELECT COUNT(*) FROM experiences WHERE is_deleted=false;" 2>/dev/null | tr -d ' ')
if [ -n "$PG_EXP" ]; then
    check "Experiences: $PG_EXP" "pass"
else
    check "Cannot query PostgreSQL" "fail"
fi

PG_LOGS=$(docker exec team_doc-postgres-1 psql -U developer -d team_memory -t -c "SELECT COUNT(*) FROM search_logs;" 2>/dev/null | tr -d ' ')
if [ -n "$PG_LOGS" ]; then
    check "Search logs: $PG_LOGS" "pass"
else
    check "Cannot query search_logs" "fail"
fi

# Check recent activity (last 24h)
RECENT=$(docker exec team_doc-postgres-1 psql -U developer -d team_memory -t -c "SELECT COUNT(*) FROM search_logs WHERE created_at > NOW() - INTERVAL '24 hours';" 2>/dev/null | tr -d ' ')
if [ "$RECENT" -gt 0 ]; then
    check "Recent search activity (24h): $RECENT" "pass"
else
    check "No search activity in last 24h" "warn"
fi

echo ""

# --- 4. PayloadCapture ---
echo "4. PayloadCapture"

CAPTURE_DB="$HOME/.cache/tm-pipeline/drafts_capture.db"
if [ -f "$CAPTURE_DB" ]; then
    CAPTURED=$($PYTHON -c "
import sqlite3
db = sqlite3.connect('$CAPTURE_DB')
total = db.execute('SELECT COUNT(*) FROM raw_payloads').fetchone()[0]
print(total)
" 2>/dev/null)
    if [ "$CAPTURED" -gt 0 ]; then
        check "Captured payloads: $CAPTURED" "pass"
    else
        check "No payloads captured yet" "warn"
    fi
else
    check "Capture DB not found (will be created on first hook)" "warn"
fi

echo ""

# --- 5. Hook Configuration ---
echo "5. Hook Configuration"

# Claude Code
if [ -f "$HOME/.claude/settings.json" ]; then
    if grep -q "team_memory" "$HOME/.claude/settings.json" 2>/dev/null; then
        check "Claude Code hooks configured" "pass"
    else
        check "Claude Code hooks not configured" "warn"
    fi
else
    check "Claude Code settings not found" "warn"
fi

# Cursor
if [ -f ".cursor/hooks.json" ]; then
    if grep -q "team_memory" ".cursor/hooks.json" 2>/dev/null; then
        check "Cursor hooks configured" "pass"
    else
        check "Cursor hooks not configured" "warn"
    fi
else
    check "Cursor hooks.json not found" "warn"
fi

# Hermes
if grep -q "team_memory" "$HOME/.hermes/config.yaml" 2>/dev/null; then
    check "Hermes hooks configured" "pass"
else
    check "Hermes hooks not configured" "warn"
fi

echo ""

# --- 6. Daemon Debug Endpoints ---
echo "6. Daemon Debug"

CONTRACTS=$(curl -sf --max-time 3 "$DAEMON_URL/debug/contracts" 2>/dev/null)
if [ $? -eq 0 ]; then
    check "/debug/contracts accessible" "pass"
else
    check "/debug/contracts not accessible" "warn"
fi

PAYLOADS=$(curl -sf --max-time 3 "$DAEMON_URL/debug/payloads?limit=1" 2>/dev/null)
if [ $? -eq 0 ]; then
    check "/debug/payloads accessible" "pass"
else
    check "/debug/payloads not accessible" "warn"
fi

echo ""

# --- Summary ---
echo "========================================="
echo "  Results: $PASS passed, $FAIL failed, $WARN warnings"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "  ❌ Pipeline has issues. Check the failures above."
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo ""
    echo "  ⚠️  Pipeline mostly OK but has warnings."
    exit 0
else
    echo ""
    echo "  ✅ Pipeline is healthy!"
    exit 0
fi
