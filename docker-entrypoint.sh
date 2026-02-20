#!/bin/bash
set -e

echo "========================================="
echo "  team_memory — Starting up"
echo "========================================="

# --- 1. Wait for PostgreSQL ---
PG_HOST="${TEAM_MEMORY_PG_HOST:-localhost}"
PG_PORT="${TEAM_MEMORY_PG_PORT:-5432}"
PG_USER="${TEAM_MEMORY_PG_USER:-developer}"
PG_DB="${TEAM_MEMORY_PG_DB:-team_memory}"

echo "[1/5] Waiting for PostgreSQL at ${PG_HOST}:${PG_PORT}..."
MAX_RETRIES=30
RETRY=0
while ! pg_isready -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -q 2>/dev/null; do
    RETRY=$((RETRY + 1))
    if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
        echo "ERROR: PostgreSQL not ready after ${MAX_RETRIES} attempts. Exiting."
        exit 1
    fi
    echo "  Attempt ${RETRY}/${MAX_RETRIES}..."
    sleep 2
done
echo "  PostgreSQL is ready!"

# --- 2. Run database migrations ---
echo "[2/5] Running database migrations..."
if command -v alembic &>/dev/null; then
    alembic upgrade head
    echo "  Migrations complete."
else
    echo "  WARNING: alembic not found, skipping migrations."
fi

# --- 3. Check Ollama availability & auto-pull model ---
OLLAMA_URL="${TEAM_MEMORY_OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${TEAM_MEMORY_OLLAMA_MODEL:-nomic-embed-text}"
echo "[3/5] Checking Ollama at ${OLLAMA_URL}..."
if curl -s --max-time 5 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    echo "  Ollama is available."
    # Auto-pull embedding model if not already present
    if curl -s --max-time 5 "${OLLAMA_URL}/api/tags" | grep -q "\"${OLLAMA_MODEL}\""; then
        echo "  Model '${OLLAMA_MODEL}' already available."
    else
        echo "  Pulling model '${OLLAMA_MODEL}'... (first-time only, may take a few minutes)"
        curl -s -X POST "${OLLAMA_URL}/api/pull" \
            -d "{\"name\":\"${OLLAMA_MODEL}\"}" \
            | while IFS= read -r line; do
                STATUS=$(echo "$line" | python3 -c \
                    "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('status',''))" \
                    2>/dev/null)
                [ -n "$STATUS" ] && echo "    $STATUS"
            done
        echo "  Model '${OLLAMA_MODEL}' is ready."
    fi
else
    echo "  WARNING: Ollama is not reachable at ${OLLAMA_URL}."
    echo "  Embedding generation may fail. Configure a different embedding provider if needed."
fi

# --- 4. Load environment-specific config ---
TEAM_MEMORY_ENV="${TEAM_MEMORY_ENV:-development}"
echo "[4/5] Environment: ${TEAM_MEMORY_ENV}"
if [ -f "config.${TEAM_MEMORY_ENV}.yaml" ]; then
    echo "  Loading config.${TEAM_MEMORY_ENV}.yaml"
    export TEAM_MEMORY_CONFIG_PATH="config.${TEAM_MEMORY_ENV}.yaml"
fi

# --- 5. Auto-create admin API Key on first start ---
echo "[5/5] Checking first-run setup..."
if [ -n "$TEAM_MEMORY_AUTO_ADMIN_KEY" ]; then
    echo "  Admin key configured via TEAM_MEMORY_AUTO_ADMIN_KEY."
elif [ -z "$TEAM_MEMORY_API_KEY" ]; then
    # Generate a random API key if none is configured
    GENERATED_KEY=$(python -c "import secrets; print(secrets.token_hex(16).upper())")
    export TEAM_MEMORY_API_KEY="$GENERATED_KEY"
    echo ""
    echo "  ╔════════════════════════════════════════════════╗"
    echo "  ║  AUTO-GENERATED ADMIN API KEY (save this!)     ║"
    echo "  ║  ${GENERATED_KEY}  ║"
    echo "  ╚════════════════════════════════════════════════╝"
    echo ""
fi

echo "========================================="
echo "  Starting team_memory server..."
echo "========================================="

# Execute the main command (uvicorn or whatever CMD is)
exec "$@"
