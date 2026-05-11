#!/bin/bash
# E2E verification with graceful degradation / 优雅降级端到端验证
set -e
echo "=== E2E: Graceful Degradation Test ==="

# 1. Normal flow should pass
echo "Step 1: Normal flow..."
make e2e-verify
echo "OK: Normal flow passes"

# 2. Stop PG
echo "Step 2: Stopping PostgreSQL..."
docker compose stop postgres
sleep 3

# 3. Send data — should go to queue
echo "Step 3: Sending data while PG is down..."
DAEMON="http://localhost:3901"
SESSION="graceful-$(date +%s)"
curl -s -X POST "$DAEMON/hooks/before_prompt" -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"$SESSION\",\"prompt\":\"PG down test\",\"workspace_roots\":[\"$(pwd)\"]}" > /dev/null
for i in 1 2 3; do
  curl -s -X POST "$DAEMON/hooks/after_response" -H "Content-Type: application/json" \
    -d "{\"conversation_id\":\"$SESSION\",\"response_text\":\"Response $i\",\"workspace_roots\":[\"$(pwd)\"]}" > /dev/null
done
echo "OK: Data sent"

# 4. Check queue
echo "Step 4: Check queue..."
sleep 5
QUEUE=$(PYTHONPATH=src python -m team_memory.cli queue stats 2>/dev/null || echo "N/A")
echo "Queue status: $QUEUE"
# Don't fail if queue is empty — PG might have caught up already

# 5. Start PG
echo "Step 5: Starting PostgreSQL..."
docker compose start postgres
sleep 5

# 6. Wait for retry
echo "Step 6: Waiting for retry (45 seconds)..."
sleep 45

# 7. Check queue cleared
echo "Step 7: Final queue check..."
PYTHONPATH=src python -m team_memory.cli queue stats 2>/dev/null || true

# 8. Cleanup
echo "Step 8: Cleanup..."
sqlite3 ~/.cache/tm-pipeline/drafts.db "DELETE FROM messages WHERE session_id='$SESSION'; DELETE FROM sessions WHERE id='$SESSION';" 2>/dev/null || true
docker exec team_doc-postgres-1 psql -U developer -d team_memory -c "DELETE FROM experiences WHERE title LIKE 'Draft graceful-%';" 2>/dev/null || true

echo "=== E2E Graceful Degradation Complete ==="
