#!/bin/bash
# Cleanup test data from SQLite + PG / 清理测试数据
set -e
DB="$HOME/.cache/tm-pipeline/drafts.db"
if [ -f "$DB" ]; then
    sqlite3 "$DB" "DELETE FROM sessions WHERE id LIKE 'fail-%'; DELETE FROM messages WHERE session_id LIKE 'fail-%';" 2>/dev/null || true
fi
docker exec team_doc-postgres-1 psql -U developer -d team_memory -c "DELETE FROM experiences WHERE title LIKE 'Draft fail-%';" 2>/dev/null || true
echo "Test data cleaned"
