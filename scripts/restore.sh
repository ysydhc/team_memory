#!/bin/bash
# team_memory database restore script
# Usage: ./scripts/restore.sh <backup_file>
#
# Restores a compressed PostgreSQL backup.
# Environment variables:
#   PGHOST (default: localhost)
#   PGPORT (default: 5432)
#   PGUSER (default: developer)
#   PGDATABASE (default: team_memory)

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    echo ""
    echo "Available backups:"
    ls -lh backups/team_memory_*.sql.gz 2>/dev/null || echo "  No backups found in ./backups/"
    exit 1
fi

BACKUP_FILE="$1"
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-developer}"
PGDATABASE="${PGDATABASE:-team_memory}"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: File not found: $BACKUP_FILE"
    exit 1
fi

echo "WARNING: This will overwrite the database '${PGDATABASE}'!"
echo "Backup file: ${BACKUP_FILE}"
read -p "Continue? (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo "Restoring database..."
gunzip -c "$BACKUP_FILE" | psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" --quiet

echo "Restore complete."
echo "Running migrations to ensure schema is up-to-date..."
alembic upgrade head 2>/dev/null || echo "  WARNING: alembic not available, skip migration."
echo "Done."
