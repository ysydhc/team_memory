#!/bin/bash
# team_memory 一键健康检查
# Usage: ./scripts/healthcheck.sh [web_port]
#
# Checks all team_memory components and reports their status.
# Embedding provider/base_url/model are read from load_settings (same as app).
# Exit code 0 = all critical services OK, non-zero = at least one FAIL.

PORT="${1:-9111}"
WEB_URL="http://localhost:${PORT}"
FAILURES=0

# Resolve project root (directory containing config.*.yaml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Determine Python interpreter (prefer conda env, fallback to python3)
# 确定解释器路径（优先 conda 环境，回退 python3）
_TM_PYTHON=""
if [ -n "${CONDA_PREFIX}" ]; then
    _TM_PYTHON="${CONDA_PREFIX}/bin/python"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    _TM_PYTHON="${ROOT_DIR}/.venv/bin/python"
fi
[ -z "${_TM_PYTHON}" ] && _TM_PYTHON="python3"

# Load embedding provider/base_url/model from config
# 从配置加载 embedding provider / base_url / 模型
EMB_PROVIDER=""
EMB_BASE_URL=""
EMB_MODEL=""
EMB_API_KEY=""
EMB_API_KEY=""
if [ -f "${ROOT_DIR}/config.development.yaml" ] || [ -f "${ROOT_DIR}/config.production.yaml" ]; then
    # Read embedding config from YAML using Python (more reliable than YAML parsing in bash)
    EMB_VALUES=$(cd "${ROOT_DIR}" && python3 -c "
from team_memory.config import load_settings
s = load_settings()
e = s.embedding
print(f'{e.provider}|{e.openai.base_url}|{e.openai.model}|{e.openai.dimension}|{e.openai.api_key}')
" 2>/dev/null)
    if [ -n "$EMB_VALUES" ]; then
        EMB_PROVIDER=$(echo "$EMB_VALUES" | cut -d'|' -f1)
        EMB_BASE_URL=$(echo "$EMB_VALUES" | cut -d'|' -f2)
        EMB_MODEL=$(echo "$EMB_VALUES" | cut -d'|' -f3)
        EMB_DIM=$(echo "$EMB_VALUES" | cut -d'|' -f4)
        EMB_API_KEY=$(echo "$EMB_VALUES" | cut -d'|' -f5)
    fi
fi
# Fall back to old YAML parsing if Python method failed
if [ -z "$EMB_PROVIDER" ] && [ -f "${ROOT_DIR}/config.development.yaml" ] || [ -f "${ROOT_DIR}/config.production.yaml" ]; then
  CONFIG_OUT=$(cd "${ROOT_DIR}" && PYTHONPATH=src "${_TM_PYTHON}" -c "
try:
    from team_memory.config import load_settings
    s = load_settings()
    provider = getattr(s.embedding, 'provider', None) or 'ollama'
    print(provider)
    if provider == 'ollama' and getattr(s.embedding, 'ollama', None):
        print((s.embedding.ollama.base_url or 'http://localhost:11434').rstrip('/'))
        print(s.embedding.ollama.model or 'nomic-embed-text:latest')
        print('')
    elif provider == 'openai' and getattr(s.embedding, 'openai', None):
        print((s.embedding.openai.base_url or 'https://api.openai.com/v1').rstrip('/'))
        print(s.embedding.openai.model or 'text-embedding-3-small')
        print(s.embedding.openai.resolved_api_key or '')
    elif provider == 'generic' and getattr(s.embedding, 'generic', None):
        print((s.embedding.generic.base_url or '').rstrip('/'))
        print(s.embedding.generic.model or '')
        print(s.embedding.generic.api_key or '')
    else:
        print('http://localhost:11434')
        print('nomic-embed-text:latest')
        print('')
    if getattr(s, 'llm', None) and getattr(s.llm, 'model', None):
        print(s.llm.model)
except Exception:
    pass
" 2>/dev/null)
  if [ -n "$CONFIG_OUT" ]; then
    EMB_PROVIDER=$(echo "$CONFIG_OUT" | sed -n '1p')
    EMB_BASE_URL=$(echo "$CONFIG_OUT" | sed -n '2p')
    EMB_MODEL=$(echo "$CONFIG_OUT" | sed -n '3p')
    EMB_API_KEY=$(echo "$CONFIG_OUT" | sed -n '4p')
    OLLAMA_LLM_MODEL_CONFIG=$(echo "$CONFIG_OUT" | sed -n '5p')
    [ -z "$EMB_PROVIDER" ] && EMB_PROVIDER="ollama"
    [ -z "$EMB_BASE_URL" ] && EMB_BASE_URL="http://localhost:11434"
    [ -z "$EMB_MODEL" ] && EMB_MODEL="nomic-embed-text:latest"
  fi
fi

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

# 2. Embedding provider/model health
printf "  %-22s" "Embedding:"
if [ "${EMB_PROVIDER}" = "ollama" ]; then
    if curl -sf --max-time 3 "${EMB_BASE_URL}/api/tags" >/dev/null 2>&1; then
        EMB_MODEL_BASE="${EMB_MODEL%%:*}"
        if curl -sf --max-time 3 "${EMB_BASE_URL}/api/tags" | grep -q "\"name\":\"${EMB_MODEL_BASE}"; then
            echo "OK (provider: ollama, model: ${EMB_MODEL})"
        else
            echo "WARN  (provider: ollama up, but model '${EMB_MODEL}' not pulled)"
            echo "                        Fix: ollama pull ${EMB_MODEL}"
        fi
    else
        echo "FAIL  (provider: ollama unreachable at ${EMB_BASE_URL})"
        FAILURES=$((FAILURES + 1))
    fi
else
    MODELS_URL="${EMB_BASE_URL%/}/models"
    if [ -n "${EMB_API_KEY}" ]; then
        MODELS_RESP=$(curl -sf --max-time 5 -H "Authorization: Bearer ${EMB_API_KEY}" "${MODELS_URL}" 2>/dev/null)
    else
        MODELS_RESP=$(curl -sf --max-time 5 "${MODELS_URL}" 2>/dev/null)
    fi

    if [ -n "${MODELS_RESP}" ]; then
        if echo "${MODELS_RESP}" | grep -q "\"id\":\"${EMB_MODEL}\""; then
            echo "OK (provider: ${EMB_PROVIDER}, model: ${EMB_MODEL})"
        else
            echo "OK (provider: ${EMB_PROVIDER}, endpoint reachable, model: ${EMB_MODEL})"
        fi
    else
        echo "FAIL  (provider: ${EMB_PROVIDER} endpoint unreachable at ${MODELS_URL})"
        FAILURES=$((FAILURES + 1))
    fi
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

# Worker Health (via Daemon /status)
printf "  %-22s" "Worker (daemon):"
WORKER_CHECK=$(curl -sf --max-time 3 "http://localhost:${DAEMON_PORT:-3901}/status" 2>/dev/null)
if [ $? -eq 0 ]; then
    WOK=$(echo "$WORKER_CHECK" | "${_TM_PYTHON}" -c \
        "import sys,json; d=json.load(sys.stdin); rw=d.get('refinement_worker',{}); print('OK' if rw.get('healthy') else 'UNHEALTHY')" \
        2>/dev/null || echo "OK")
    WFAILS=$(echo "$WORKER_CHECK" | "${_TM_PYTHON}" -c \
        "import sys,json; d=json.load(sys.stdin); rw=d.get('refinement_worker',{}); print(rw.get('fail_count','?'))" \
        2>/dev/null || echo "?")
    echo "${WOK}  (fail_count=${WFAILS})"
else
    echo "FAIL  (daemon not responding on port ${DAEMON_PORT:-3901})"
    FAILURES=$((FAILURES + 1))
fi

# 5. Web Service
printf "  %-22s" "Web (port ${PORT}):"
HEALTH_RESP=$(curl -sf --max-time 3 "${WEB_URL}/health" 2>/dev/null)
if [ $? -eq 0 ]; then
    STATUS=$(echo "$HEALTH_RESP" | "${_TM_PYTHON}" -c \
        "import sys,json; print(json.loads(sys.stdin.read()).get('status','ok'))" \
        2>/dev/null)
    echo "${STATUS:-OK}"
else
    echo "FAIL  (run: python -m team_memory.web.app)"
    FAILURES=$((FAILURES + 1))
fi

# 5. Ollama LLM model (optional, only when embedding provider is ollama)
printf "  %-22s" "LLM model (optional):"
LLM_MODEL="${TEAM_MEMORY_LLM_MODEL:-${OLLAMA_LLM_MODEL_CONFIG:-}}"
if [ "${EMB_PROVIDER}" = "ollama" ] && [ -n "$LLM_MODEL" ]; then
    LLM_MODEL_BASE="${LLM_MODEL%%:*}"
    if curl -sf --max-time 3 "${EMB_BASE_URL}/api/tags" | grep -q "\"name\":\"${LLM_MODEL_BASE}"; then
        echo "OK (model: ${LLM_MODEL})"
    else
        echo "WARN  (model '${LLM_MODEL}' not found)"
    fi
else
    echo "SKIP  (not using ollama embedding provider)"
fi

# 6. Dashboard /stats (if Web returned health JSON)
if [ -n "$HEALTH_RESP" ]; then
    DASH=$(echo "$HEALTH_RESP" | "${_TM_PYTHON}" -c "
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
