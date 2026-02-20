#!/bin/bash
# team_memory 一键健康检查
# Usage: ./scripts/healthcheck.sh [web_port]
#
# Checks all team_memory components and reports their status.
# Exit code 0 = all critical services OK, non-zero = at least one FAIL.

PORT="${1:-9111}"
WEB_URL="http://localhost:${PORT}"
FAILURES=0

echo "===== team_memory Health Check ====="
echo ""

# 1. PostgreSQL
printf "  %-22s" "PostgreSQL:"
if command -v pg_isready >/dev/null 2>&1; then
    if pg_isready -h localhost -p 5432 -U developer -q 2>/dev/null; then
        echo "OK"
    else
        echo "FAIL  (run: docker compose up -d postgres)"
        FAILURES=$((FAILURES + 1))
    fi
else
    # pg_isready not installed, try psql or docker
    if docker compose exec -T postgres pg_isready -U developer -q 2>/dev/null; then
        echo "OK (via docker)"
    else
        echo "FAIL  (run: docker compose up -d postgres)"
        FAILURES=$((FAILURES + 1))
    fi
fi

# 2. Ollama
printf "  %-22s" "Ollama:"
if curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    # Check if embedding model is available
    MODEL="${TEAM_MEMORY_OLLAMA_MODEL:-nomic-embed-text}"
    if curl -sf --max-time 3 http://localhost:11434/api/tags | grep -q "\"${MODEL}\""; then
        echo "OK (model: ${MODEL})"
    else
        echo "WARN  (service up, but model '${MODEL}' not pulled)"
        echo "                        Fix: ollama pull ${MODEL}"
    fi
else
    echo "FAIL  (run: docker compose up -d ollama)"
    FAILURES=$((FAILURES + 1))
fi

# 3. Redis (optional)
printf "  %-22s" "Redis (optional):"
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli ping >/dev/null 2>&1; then
        echo "OK"
    else
        echo "SKIP  (not running; cache falls back to memory)"
    fi
else
    if docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        echo "OK (via docker)"
    else
        echo "SKIP  (not running; cache falls back to memory)"
    fi
fi

# 4. Web Service
printf "  %-22s" "Web (port ${PORT}):"
HEALTH_RESP=$(curl -sf --max-time 3 "${WEB_URL}/health" 2>/dev/null)
if [ $? -eq 0 ]; then
    STATUS=$(echo "$HEALTH_RESP" | python3 -c \
        "import sys,json; print(json.loads(sys.stdin.read()).get('status','ok'))" \
        2>/dev/null)
    echo "${STATUS:-OK}"
else
    echo "FAIL  (run: python -m team_memory.web.app)"
    FAILURES=$((FAILURES + 1))
fi

# 5. Ollama LLM model (optional, for parsing/reranking)
printf "  %-22s" "LLM model (optional):"
LLM_MODEL="${TEAM_MEMORY_LLM_MODEL:-}"
if [ -n "$LLM_MODEL" ]; then
    if curl -sf --max-time 3 http://localhost:11434/api/tags | grep -q "\"${LLM_MODEL}\""; then
        echo "OK (model: ${LLM_MODEL})"
    else
        echo "WARN  (model '${LLM_MODEL}' not found)"
    fi
else
    echo "SKIP  (no LLM model configured)"
fi

echo ""
echo "================================="
if [ "$FAILURES" -gt 0 ]; then
    echo "Result: ${FAILURES} critical service(s) FAILED"
    exit 1
else
    echo "Result: All critical services OK"
    exit 0
fi
