#!/bin/bash
# team_memory database backup script
# Usage: ./scripts/backup.sh [output_dir]
#
# Creates a compressed PostgreSQL backup with timestamp.
# Environment variables:
#   PGHOST (default: localhost)
#   PGPORT (default: 5432)
#   PGUSER (default: developer)
#   PGDATABASE (default: team_memory)

set -e

OUTPUT_DIR="${1:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-developer}"
PGDATABASE="${PGDATABASE:-team_memory}"

mkdir -p "$OUTPUT_DIR"

BACKUP_FILE="${OUTPUT_DIR}/team_memory_${TIMESTAMP}.sql.gz"

echo "Backing up database: ${PGDATABASE}@${PGHOST}:${PGPORT}"
echo "Output: ${BACKUP_FILE}"

pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    --no-owner --no-privileges --clean --if-exists \
    | gzip > "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup complete: ${BACKUP_FILE} (${SIZE})"

# Keep only last 10 backups (POSIX-compatible, works on macOS and Linux)
cd "$OUTPUT_DIR"
ls -t team_memory_*.sql.gz 2>/dev/null | tail -n +11 | while read -r f; do rm -f "$f"; done
echo "Cleanup done. Keeping last 10 backups."
