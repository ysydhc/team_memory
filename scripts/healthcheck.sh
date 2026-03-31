#!/bin/bash
# team_memory 一键健康检查
# Usage: ./scripts/healthcheck.sh [web_port]
#
# Checks all team_memory components and reports their status.
# Ollama URL and embedding model are read from config.development.yaml / load_settings (same as app);
# override with TEAM_MEMORY_EMBEDDING__OLLAMA__BASE_URL / TEAM_MEMORY_OLLAMA_MODEL.
# Exit code 0 = all critical services OK, non-zero = at least one FAIL.

PORT="${1:-9111}"
WEB_URL="http://localhost:${PORT}"
FAILURES=0

# Resolve project root (directory containing config.*.yaml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load Ollama base_url and embedding model from config (same as app); fallback to defaults
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_EMBED_MODEL="nomic-embed-text:latest"
OLLAMA_LLM_MODEL_CONFIG=""
if [ -f "${ROOT_DIR}/config.development.yaml" ] || [ -f "${ROOT_DIR}/config.production.yaml" ]; then
  CONFIG_OUT=$(cd "${ROOT_DIR}" && python3 -c "
try:
    from team_memory.config import load_settings
    s = load_settings()
    if getattr(s.embedding, 'ollama', None) and getattr(s.embedding, 'provider', None) == 'ollama':
        print(s.embedding.ollama.base_url.rstrip('/'))
        print(s.embedding.ollama.model or 'nomic-embed-text:latest')
    else:
        print('http://localhost:11434')
        print('nomic-embed-text:latest')
    if getattr(s, 'llm', None) and getattr(s.llm, 'model', None):
        print(s.llm.model)
except Exception:
    pass
" 2>/dev/null)
  if [ -n "$CONFIG_OUT" ]; then
    OLLAMA_BASE_URL=$(echo "$CONFIG_OUT" | sed -n '1p')
    OLLAMA_EMBED_MODEL=$(echo "$CONFIG_OUT" | sed -n '2p')
    OLLAMA_LLM_MODEL_CONFIG=$(echo "$CONFIG_OUT" | sed -n '3p')
    [ -z "$OLLAMA_BASE_URL" ] && OLLAMA_BASE_URL="http://localhost:11434"
    [ -z "$OLLAMA_EMBED_MODEL" ] && OLLAMA_EMBED_MODEL="nomic-embed-text:latest"
  fi
fi
# Env overrides (same prefix as app)
[ -n "${TEAM_MEMORY_EMBEDDING__OLLAMA__BASE_URL}" ] && OLLAMA_BASE_URL="${TEAM_MEMORY_EMBEDDING__OLLAMA__BASE_URL}"
[ -n "${TEAM_MEMORY_OLLAMA_MODEL}" ] && OLLAMA_EMBED_MODEL="${TEAM_MEMORY_OLLAMA_MODEL}"

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

# 2. Ollama (base_url from config / TEAM_MEMORY_EMBEDDING__OLLAMA__BASE_URL)
printf "  %-22s" "Ollama:"
if curl -sf --max-time 3 "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
    # Tags use "name":"model:tag" — match base name so both nomic-embed-text and nomic-embed-text:latest work
    OLLAMA_EMBED_MODEL_BASE="${OLLAMA_EMBED_MODEL%%:*}"
    if curl -sf --max-time 3 "${OLLAMA_BASE_URL}/api/tags" | grep -q "\"name\":\"${OLLAMA_EMBED_MODEL_BASE}"; then
        echo "OK (model: ${OLLAMA_EMBED_MODEL})"
    else
        echo "WARN  (service up, but model '${OLLAMA_EMBED_MODEL}' not pulled)"
        echo "                        Fix: ollama pull ${OLLAMA_EMBED_MODEL}"
    fi
else
    echo "FAIL  (run: docker compose up -d ollama or check config embedding.ollama.base_url)"
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

# 5. Ollama LLM model (optional, for parsing/reranking; same base_url as embedding)
printf "  %-22s" "LLM model (optional):"
LLM_MODEL="${TEAM_MEMORY_LLM_MODEL:-${OLLAMA_LLM_MODEL_CONFIG:-}}"
if [ -n "$LLM_MODEL" ]; then
    LLM_MODEL_BASE="${LLM_MODEL%%:*}"
    if curl -sf --max-time 3 "${OLLAMA_BASE_URL}/api/tags" | grep -q "\"name\":\"${LLM_MODEL_BASE}"; then
        echo "OK (model: ${LLM_MODEL})"
    else
        echo "WARN  (model '${LLM_MODEL}' not found)"
    fi
else
    echo "SKIP  (no LLM model configured)"
fi

# 6. Dashboard /stats (if Web returned health JSON)
if [ -n "$HEALTH_RESP" ]; then
    DASH=$(echo "$HEALTH_RESP" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d.get('checks', {}).get('dashboard_stats', {})
    if c.get('status') == 'down':
        print('FAIL')
        print(c.get('error', 'unknown'))
        print(c.get('ops_hint', ''))
    else:
        print('OK')
except Exception:
    print('SKIP')
" 2>/dev/null)
    if [ -n "$DASH" ]; then
        printf "  %-22s" "Dashboard /stats:"
        echo "$DASH" | head -1
        if echo "$DASH" | tail -n +2 | grep -q .; then
            echo "$DASH" | tail -n +2 | sed 's/^/                        /'
            echo ""
            echo "  → 若 Web 仪表盘报「加载仪表盘失败」请按上面 ops_hint 排查（如：执行 alembic upgrade head、检查 config 中 database.url）。"
        fi
    fi
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
