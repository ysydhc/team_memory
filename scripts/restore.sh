#!/bin/bash
# team_memory native PostgreSQL restore & verification drill script
# Usage:
#   ./scripts/restore.sh --verify-only [backup_file]
#   ./scripts/restore.sh [--yes] <backup_file>
#
# Supports native PostgreSQL custom dumps (.dump via pg_restore) and legacy
# compressed SQL dumps (.sql.gz via psql). Automatically verifies SHA256SUMS
# when present. Never uses docker exec.

set -euo pipefail

export PATH="/opt/homebrew/opt/postgresql@16/bin:/opt/homebrew/bin:/usr/local/opt/postgresql@16/bin:/usr/local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERIFY_ONLY=0
AUTO_YES=0
BACKUP_FILE=""

usage() {
    cat <<EOF
Usage:
  $0 --verify-only [backup_file]   Restore into temporary DB, verify core tables > 0 rows, and drop temp DB
  $0 [--yes] <backup_file>         Restore backup into target database (interactive confirm unless --yes)

Available backups:
EOF
    ls -lh "${ROOT_DIR}/backups/daily"/team_memory_*.dump "${ROOT_DIR}/backups/weekly"/team_memory_*.dump "${ROOT_DIR}/backups"/team_memory_*.sql.gz 2>/dev/null || echo "  No backups found in ${ROOT_DIR}/backups/"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --verify-only)
            VERIFY_ONLY=1
            shift
            ;;
        -y|--yes)
            AUTO_YES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [ -n "$BACKUP_FILE" ]; then
                echo "ERROR: Unexpected extra argument: $1" >&2
                usage >&2
                exit 1
            fi
            BACKUP_FILE="$1"
            shift
            ;;
    esac
done

# Preserve caller-supplied PG* overrides before sourcing .env
_ORIG_PGHOST="${PGHOST:-}"
_ORIG_PGPORT="${PGPORT:-}"
_ORIG_PGUSER="${PGUSER:-}"
_ORIG_PGPASSWORD="${PGPASSWORD:-}"
_ORIG_PGDATABASE="${PGDATABASE:-}"

if [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${ROOT_DIR}/.env"
    set +a
fi

_TM_PYTHON="${PYTHON:-}"
if [ -z "${_TM_PYTHON}" ] || [ ! -x "${_TM_PYTHON}" ]; then
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        _TM_PYTHON="${CONDA_PREFIX}/bin/python"
    elif [ -x "/Users/yeshouyou/miniconda3/envs/team_memory/bin/python" ]; then
        _TM_PYTHON="/Users/yeshouyou/miniconda3/envs/team_memory/bin/python"
    elif [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
        _TM_PYTHON="${ROOT_DIR}/.venv/bin/python"
    else
        _TM_PYTHON="$(command -v python3 || echo python3)"
    fi
fi

# Auto-pick latest backup when --verify-only is called without an explicit file
BACKUP_ROOT="${TEAM_MEMORY_BACKUP_DIR:-${ROOT_DIR}/backups}"
if [ -z "$BACKUP_FILE" ] && [ "$VERIFY_ONLY" -eq 1 ]; then
    if [ -f "${BACKUP_ROOT}/LATEST" ]; then
        _REL_DUMP=$("${_TM_PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('pg_dump',''))" "${BACKUP_ROOT}/LATEST" 2>/dev/null || true)
        if [ -n "$_REL_DUMP" ] && [ -f "${BACKUP_ROOT}/${_REL_DUMP}" ]; then
            BACKUP_FILE="${BACKUP_ROOT}/${_REL_DUMP}"
        fi
    fi
    if [ -z "$BACKUP_FILE" ]; then
        BACKUP_FILE=$(ls -t "${BACKUP_ROOT}/daily"/team_memory_*.dump "${BACKUP_ROOT}"/team_memory_*.dump "${BACKUP_ROOT}"/team_memory_*.sql.gz 2>/dev/null | head -n 1 || true)
    fi
fi

if [ -z "$BACKUP_FILE" ]; then
    usage
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE" >&2
    exit 1
fi

# Parse native PostgreSQL connection from team_memory config
_CFG_HOST=""
_CFG_PORT=""
_CFG_USER=""
_CFG_PASS=""
_CFG_DB=""
if [ -f "${ROOT_DIR}/config.development.yaml" ] || [ -f "${ROOT_DIR}/config.production.yaml" ]; then
    _PG_URL_INFO=$(cd "${ROOT_DIR}" && PYTHONPATH=src "${_TM_PYTHON}" -c "
from urllib.parse import urlparse, unquote
from team_memory.config import load_settings
s = load_settings()
u = urlparse(s.database.url.replace('+asyncpg', ''))
print(u.hostname or '')
print(u.port or '')
print(unquote(u.username or ''))
print(unquote(u.password or ''))
print(u.path.lstrip('/') or '')
" 2>/dev/null || true)
    if [ -n "$_PG_URL_INFO" ]; then
        _CFG_HOST=$(printf '%s\n' "$_PG_URL_INFO" | sed -n '1p')
        _CFG_PORT=$(printf '%s\n' "$_PG_URL_INFO" | sed -n '2p')
        _CFG_USER=$(printf '%s\n' "$_PG_URL_INFO" | sed -n '3p')
        _CFG_PASS=$(printf '%s\n' "$_PG_URL_INFO" | sed -n '4p')
        _CFG_DB=$(printf '%s\n' "$_PG_URL_INFO" | sed -n '5p')
    fi
fi

export PGHOST="${_ORIG_PGHOST:-${_CFG_HOST:-${PGHOST:-localhost}}}"
export PGPORT="${_ORIG_PGPORT:-${_CFG_PORT:-${PGPORT:-5432}}}"
export PGUSER="${_ORIG_PGUSER:-${_CFG_USER:-${PGUSER:-developer}}}"
export PGDATABASE="${_ORIG_PGDATABASE:-${_CFG_DB:-${PGDATABASE:-team_memory}}}"
if [ -n "${_ORIG_PGPASSWORD}" ]; then
    export PGPASSWORD="${_ORIG_PGPASSWORD}"
elif [ -n "${_CFG_PASS}" ]; then
    export PGPASSWORD="${_CFG_PASS}"
fi

PG_RESTORE_BIN="$(command -v pg_restore || true)"
PSQL_BIN="$(command -v psql || true)"
if [ -z "$PSQL_BIN" ]; then
    echo "ERROR: psql not found in PATH." >&2
    exit 1
fi

# Verify SHA256 checksum if matching SHA256SUMS_<TS>.txt exists
BACKUP_DIR="$(cd "$(dirname "$BACKUP_FILE")" && pwd)"
BACKUP_BASE="$(basename "$BACKUP_FILE")"
BACKUP_TS=$(printf '%s' "$BACKUP_BASE" | sed -E 's/^team_memory_([0-9]{8}_[0-9]{6})\..*$/\1/')
SHA_FILE="${BACKUP_DIR}/SHA256SUMS_${BACKUP_TS}.txt"

if [ -f "$SHA_FILE" ]; then
    echo "[restore] Verifying SHA256 checksums (${SHA_FILE})..."
    (cd "$BACKUP_DIR" && shasum -a 256 -c "$(basename "$SHA_FILE")")
    echo "[restore] SHA256 verification PASSED."
else
    echo "[restore] Note: No matching SHA256SUMS file found at ${SHA_FILE}; skipping checksum verification."
fi

restore_into_db() {
    local target_db="$1"
    local clean_flag="$2"  # "clean" or "fresh"

    case "$BACKUP_FILE" in
        *.dump|*.pgdump|*.backup)
            if [ -z "$PG_RESTORE_BIN" ]; then
                echo "ERROR: pg_restore not found in PATH." >&2
                exit 1
            fi
            local extra_args=(--no-owner --no-privileges --exit-on-error)
            if [ "$clean_flag" = "clean" ]; then
                extra_args+=(--clean --if-exists)
            fi
            "$PG_RESTORE_BIN" \
                "${extra_args[@]}" \
                -h "$PGHOST" \
                -p "$PGPORT" \
                -U "$PGUSER" \
                -d "$target_db" \
                "$BACKUP_FILE"
            ;;
        *.sql.gz)
            gunzip -c "$BACKUP_FILE" | "$PSQL_BIN" -X \
                -h "$PGHOST" \
                -p "$PGPORT" \
                -U "$PGUSER" \
                -d "$target_db" \
                -v ON_ERROR_STOP=1 \
                --quiet
            ;;
        *.sql)
            "$PSQL_BIN" -X \
                -h "$PGHOST" \
                -p "$PGPORT" \
                -U "$PGUSER" \
                -d "$target_db" \
                -v ON_ERROR_STOP=1 \
                --quiet \
                -f "$BACKUP_FILE"
            ;;
        *)
            echo "ERROR: Unsupported backup extension: $BACKUP_FILE (expected .dump, .sql.gz, or .sql)" >&2
            exit 1
            ;;
    esac
}

if [ "$VERIFY_ONLY" -eq 1 ]; then
    VERIFY_DB="team_memory_verify_$(date +%s)_$$"
    cleanup_verify_db() {
        echo "[verify] Cleaning up temporary database '${VERIFY_DB}'..."
        "$PSQL_BIN" -X -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -v ON_ERROR_STOP=0 -q \
            -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${VERIFY_DB}' AND pid <> pg_backend_pid();" >/dev/null 2>&1 || true
        "$PSQL_BIN" -X -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -v ON_ERROR_STOP=0 -q \
            -c "DROP DATABASE IF EXISTS \"${VERIFY_DB}\";" >/dev/null 2>&1 || true
        echo "[verify] Temporary database '${VERIFY_DB}' dropped."
    }
    trap cleanup_verify_db EXIT

    echo "[verify] Starting restore verification drill"
    echo "[verify] Backup file: ${BACKUP_FILE}"
    echo "[verify] Target host: ${PGUSER}@${PGHOST}:${PGPORT}"
    echo "[verify] Creating temporary database: ${VERIFY_DB}"

    "$PSQL_BIN" -X -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -v ON_ERROR_STOP=1 -q \
        -c "CREATE DATABASE \"${VERIFY_DB}\";"

    echo "[verify] Importing backup into ${VERIFY_DB}..."
    restore_into_db "$VERIFY_DB" "fresh"

    echo "[verify] Checking core table row counts..."
    VERIFY_FAILED=0
    for tbl in experiences archives entities tenants; do
        CNT=$("$PSQL_BIN" -X -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$VERIFY_DB" -tAc \
            "SELECT count(*) FROM public.\"${tbl}\";" 2>/dev/null | tr -d '[:space:]' || echo "ERR")
        if [ "$CNT" = "ERR" ] || ! printf '%s' "$CNT" | grep -Eq '^[0-9]+$'; then
            printf "  %-14s FAIL (table missing or unreadable)\n" "${tbl}:"
            VERIFY_FAILED=1
        elif [ "$CNT" -le 0 ]; then
            printf "  %-14s FAIL (rows=%s, expected > 0)\n" "${tbl}:" "$CNT"
            VERIFY_FAILED=1
        else
            printf "  %-14s OK   (rows=%s)\n" "${tbl}:" "$CNT"
        fi
    done

    # Also verify RLS policies & pgvector extension were restored
    RLS_COUNT=$("$PSQL_BIN" -X -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$VERIFY_DB" -tAc \
        "SELECT count(*) FROM pg_policies WHERE schemaname = 'public';" 2>/dev/null | tr -d '[:space:]' || echo "0")
    VEC_EXT=$("$PSQL_BIN" -X -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$VERIFY_DB" -tAc \
        "SELECT extversion FROM pg_extension WHERE extname = 'vector';" 2>/dev/null | tr -d '[:space:]' || echo "")
    echo "  pgvector:      ${VEC_EXT:-not-installed}  |  RLS policies: ${RLS_COUNT}"

    if [ "$VERIFY_FAILED" -ne 0 ]; then
        echo "[verify] FAILED: One or more core tables failed verification." >&2
        exit 1
    fi

    echo "[verify] PASSED: Restore verification drill succeeded!"
    exit 0
fi

# Actual restore into target database
echo "WARNING: This will overwrite database '${PGDATABASE}' on ${PGHOST}:${PGPORT}!"
echo "Backup file: ${BACKUP_FILE}"
if [ "$AUTO_YES" -ne 1 ]; then
    read -r -p "Continue? (y/N) " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "[restore] Restoring database '${PGDATABASE}'..."
restore_into_db "$PGDATABASE" "clean"
echo "[restore] Restore complete."

echo "[restore] Running migrations to ensure schema is up-to-date..."
if command -v uv >/dev/null 2>&1; then
    (cd "$ROOT_DIR" && uv run python -m alembic upgrade head) || echo "  WARNING: alembic migration skipped/failed."
elif command -v alembic >/dev/null 2>&1; then
    (cd "$ROOT_DIR" && alembic upgrade head) || echo "  WARNING: alembic migration skipped/failed."
else
    echo "  WARNING: alembic not available, skip migration."
fi
echo "[restore] Done."
