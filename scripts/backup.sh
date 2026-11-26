#!/bin/bash
# team_memory native PostgreSQL + SQLite backup script
# Usage: ./scripts/backup.sh [output_dir]
#
# Creates a native PostgreSQL custom-format dump (--format=custom, including
# schema, RLS policies, extensions, and pgvector data) and a WAL-safe SQLite
# hot backup of ~/.cache/tm-pipeline/drafts.db, plus manifest & SHA256SUMS.
#
# Retention policy (configurable via env):
#   DAILY_RETENTION_DAYS  (default: 14)
#   WEEKLY_RETENTION_DAYS (default: 28, i.e. 4 weeks)
#
# Connection resolution priority:
#   1. Explicit PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE env vars
#   2. .env + team_memory.config.load_settings() database.url
#   3. Defaults: localhost:5432 / developer / team_memory

set -euo pipefail

# Ensure native Homebrew PostgreSQL binaries are discoverable under launchd
export PATH="/opt/homebrew/opt/postgresql@16/bin:/opt/homebrew/bin:/usr/local/opt/postgresql@16/bin:/usr/local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

# Resolve Python interpreter for config parsing & manifest generation
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

# Verify native tools
PG_DUMP_BIN="$(command -v pg_dump || true)"
PSQL_BIN="$(command -v psql || true)"
SQLITE_BIN="$(command -v sqlite3 || true)"
if [ -z "$PG_DUMP_BIN" ]; then
    echo "ERROR: pg_dump not found in PATH. Install native PostgreSQL (e.g. brew install postgresql@16)." >&2
    exit 1
fi

BACKUP_ROOT="${1:-${TEAM_MEMORY_BACKUP_DIR:-${ROOT_DIR}/backups}}"
DAILY_DIR="${BACKUP_ROOT}/daily"
WEEKLY_DIR="${BACKUP_ROOT}/weekly"
DAILY_RETENTION_DAYS="${DAILY_RETENTION_DAYS:-14}"
WEEKLY_RETENTION_DAYS="${WEEKLY_RETENTION_DAYS:-28}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CREATED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ISO_WEEK="$(date +%G-W%V)"
DAY_OF_WEEK="$(date +%u)"  # 1=Mon .. 7=Sun

mkdir -p "$DAILY_DIR" "$WEEKLY_DIR"

PG_BASENAME="team_memory_${TIMESTAMP}.dump"
PG_FILE="${DAILY_DIR}/${PG_BASENAME}"
PG_TMP="${PG_FILE}.tmp"

echo "[backup] Starting TeamMemory native backup (${TIMESTAMP})"
echo "[backup] PostgreSQL: ${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE}"
echo "[backup] pg_dump:    $("$PG_DUMP_BIN" --version)"

"$PG_DUMP_BIN" \
    --format=custom \
    --compress=6 \
    --no-owner \
    --no-privileges \
    -h "$PGHOST" \
    -p "$PGPORT" \
    -U "$PGUSER" \
    -d "$PGDATABASE" \
    -f "$PG_TMP"

mv "$PG_TMP" "$PG_FILE"
PG_BYTES=$(stat -f%z "$PG_FILE" 2>/dev/null || stat -c%s "$PG_FILE")
PG_HUMAN=$(du -h "$PG_FILE" | cut -f1 | tr -d '[:space:]')
echo "[backup] PostgreSQL dump complete: ${PG_FILE} (${PG_HUMAN})"

# WAL-safe SQLite hot backup (drafts.db)
SQLITE_DB="${TM_SQLITE_DB:-${HOME}/.cache/tm-pipeline/drafts.db}"
SQLITE_BASENAME=""
SQLITE_FILE=""
SQLITE_BYTES=0
SQLITE_INTEGRITY="skipped"
if [ -f "$SQLITE_DB" ]; then
    if [ -z "$SQLITE_BIN" ]; then
        echo "ERROR: sqlite3 not found in PATH; cannot safely hot-backup ${SQLITE_DB}." >&2
        exit 1
    fi
    SQLITE_BASENAME="drafts_${TIMESTAMP}.db"
    SQLITE_FILE="${DAILY_DIR}/${SQLITE_BASENAME}"
    SQLITE_TMP="${SQLITE_FILE}.tmp"
    "$SQLITE_BIN" -init /dev/null -batch "$SQLITE_DB" ".timeout 10000" ".backup '${SQLITE_TMP}'"
    SQLITE_INTEGRITY=$("$SQLITE_BIN" -init /dev/null -batch -noheader -list "$SQLITE_TMP" "PRAGMA integrity_check;" | head -n 1 | tr -d '\r')
    if [ "$SQLITE_INTEGRITY" != "ok" ]; then
        rm -f "$SQLITE_TMP"
        echo "ERROR: SQLite backup integrity check failed: ${SQLITE_INTEGRITY}" >&2
        exit 1
    fi
    mv "$SQLITE_TMP" "$SQLITE_FILE"
    SQLITE_BYTES=$(stat -f%z "$SQLITE_FILE" 2>/dev/null || stat -c%s "$SQLITE_FILE")
    SQLITE_HUMAN=$(du -h "$SQLITE_FILE" | cut -f1 | tr -d '[:space:]')
    echo "[backup] SQLite hot backup complete: ${SQLITE_FILE} (${SQLITE_HUMAN}, integrity=${SQLITE_INTEGRITY})"
else
    echo "[backup] SQLite database not found at ${SQLITE_DB}, skipping."
fi

# Optional core table row-count snapshot for manifest
TABLE_COUNTS_JSON="{}"
if [ -n "$PSQL_BIN" ]; then
    TABLE_COUNTS_JSON=$("$PSQL_BIN" -X -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -tAc "
SELECT json_build_object(
  'experiences', COALESCE((SELECT count(*) FROM experiences), 0),
  'archives',    COALESCE((SELECT count(*) FROM archives), 0),
  'entities',    COALESCE((SELECT count(*) FROM entities), 0),
  'tenants',     COALESCE((SELECT count(*) FROM tenants), 0)
);
" 2>/dev/null || echo "{}")
    [ -z "$TABLE_COUNTS_JSON" ] && TABLE_COUNTS_JSON="{}"
fi

# Generate SHA256 checksums
SHA_BASENAME="SHA256SUMS_${TIMESTAMP}.txt"
SHA_FILE="${DAILY_DIR}/${SHA_BASENAME}"
(
    cd "$DAILY_DIR"
    if [ -n "$SQLITE_BASENAME" ]; then
        shasum -a 256 "$PG_BASENAME" "$SQLITE_BASENAME" > "$SHA_BASENAME"
    else
        shasum -a 256 "$PG_BASENAME" > "$SHA_BASENAME"
    fi
)
PG_SHA256=$(awk -v f="$PG_BASENAME" '$2 == f {print $1}' "$SHA_FILE")
SQLITE_SHA256=""
if [ -n "$SQLITE_BASENAME" ]; then
    SQLITE_SHA256=$(awk -v f="$SQLITE_BASENAME" '$2 == f {print $1}' "$SHA_FILE")
fi
echo "[backup] Checksums written: ${SHA_FILE}"

# Generate manifest JSON & update LATEST pointer atomically
MANIFEST_BASENAME="manifest_${TIMESTAMP}.json"
MANIFEST_FILE="${DAILY_DIR}/${MANIFEST_BASENAME}"
LATEST_FILE="${BACKUP_ROOT}/LATEST"

"${_TM_PYTHON}" - \
    "$MANIFEST_FILE" "$LATEST_FILE" "$TIMESTAMP" "$CREATED_AT_UTC" "$ISO_WEEK" \
    "$PGHOST" "$PGPORT" "$PGDATABASE" \
    "$PG_BASENAME" "$PG_BYTES" "$PG_SHA256" \
    "$SQLITE_BASENAME" "$SQLITE_BYTES" "$SQLITE_SHA256" "$SQLITE_INTEGRITY" \
    "$SHA_BASENAME" "$TABLE_COUNTS_JSON" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_path,
    latest_path,
    timestamp,
    created_at_utc,
    iso_week,
    pg_host,
    pg_port,
    pg_db,
    pg_file,
    pg_bytes,
    pg_sha256,
    sqlite_file,
    sqlite_bytes,
    sqlite_sha256,
    sqlite_integrity,
    sha_file,
    table_counts_raw,
) = sys.argv[1:18]

try:
    table_counts = json.loads(table_counts_raw) if table_counts_raw else {}
except Exception:
    table_counts = {}

manifest = {
    "timestamp": timestamp,
    "created_at_utc": created_at_utc,
    "iso_week": iso_week,
    "database": {
        "host": pg_host,
        "port": int(pg_port) if pg_port.isdigit() else pg_port,
        "name": pg_db,
        "format": "custom",
        "file": pg_file,
        "size_bytes": int(pg_bytes),
        "sha256": pg_sha256,
        "core_table_counts": table_counts,
    },
    "sqlite": {
        "file": sqlite_file or None,
        "size_bytes": int(sqlite_bytes),
        "sha256": sqlite_sha256 or None,
        "integrity_check": sqlite_integrity,
    },
    "sha256sums": sha_file,
}

Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

latest = {
    "timestamp": timestamp,
    "created_at_utc": created_at_utc,
    "iso_week": iso_week,
    "pg_dump": f"daily/{pg_file}",
    "sqlite_backup": f"daily/{sqlite_file}" if sqlite_file else None,
    "manifest": f"daily/{Path(manifest_path).name}",
    "sha256sums": f"daily/{sha_file}",
    "pg_size_bytes": int(pg_bytes),
    "sqlite_size_bytes": int(sqlite_bytes),
}
tmp_latest = Path(latest_path + ".tmp")
tmp_latest.write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp_latest.replace(Path(latest_path))
PY
echo "[backup] Manifest written:  ${MANIFEST_FILE}"

# Weekly retention: promote on Sunday OR if current ISO week has no weekly backup yet
NEED_WEEKLY=0
if [ "$DAY_OF_WEEK" = "7" ]; then
    NEED_WEEKLY=1
elif ! grep -q "\"iso_week\": \"${ISO_WEEK}\"" "${WEEKLY_DIR}"/manifest_*.json 2>/dev/null; then
    NEED_WEEKLY=1
fi

if [ "$NEED_WEEKLY" -eq 1 ]; then
    cp -p "$PG_FILE" "${WEEKLY_DIR}/${PG_BASENAME}"
    [ -n "$SQLITE_FILE" ] && cp -p "$SQLITE_FILE" "${WEEKLY_DIR}/${SQLITE_BASENAME}"
    cp -p "$MANIFEST_FILE" "${WEEKLY_DIR}/${MANIFEST_BASENAME}"
    cp -p "$SHA_FILE" "${WEEKLY_DIR}/${SHA_BASENAME}"
    echo "[backup] Weekly snapshot promoted to ${WEEKLY_DIR} (${ISO_WEEK})"
fi

# Retention cleanup: daily > 14d, weekly > 28d (4 weeks)
DAILY_REMOVED=$(find "$DAILY_DIR" -maxdepth 1 -type f -mtime +"${DAILY_RETENTION_DAYS}" -print -delete | wc -l | tr -d '[:space:]')
WEEKLY_REMOVED=$(find "$WEEKLY_DIR" -maxdepth 1 -type f -mtime +"${WEEKLY_RETENTION_DAYS}" -print -delete | wc -l | tr -d '[:space:]')
echo "[backup] Retention cleanup done (daily>${DAILY_RETENTION_DAYS}d removed=${DAILY_REMOVED}, weekly>${WEEKLY_RETENTION_DAYS}d removed=${WEEKLY_REMOVED})"
echo "[backup] SUCCESS: ${PG_FILE}"
