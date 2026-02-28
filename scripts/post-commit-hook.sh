#!/usr/bin/env bash
# Git post-commit hook for TeamMemory task auto-update.
#
# Parses commit messages for [TM-xxx] markers and updates
# the referenced tasks via the TeamMemory API.
#
# Install: make hooks-install
# Or manually: cp scripts/post-commit-hook.sh .git/hooks/post-commit && chmod +x .git/hooks/post-commit

set -euo pipefail

TM_BASE_URL="${TEAM_MEMORY_URL:-http://localhost:9111}"
TM_API_KEY="${TEAM_MEMORY_API_KEY:-}"

COMMIT_MSG=$(git log -1 --format='%s%n%n%b')
COMMIT_HASH=$(git log -1 --format='%H')
COMMIT_AUTHOR=$(git log -1 --format='%an')

# Extract all [TM-xxx] markers
TASK_IDS=$(echo "$COMMIT_MSG" | grep -oE '\[TM-[a-f0-9-]+\]' | sed 's/\[TM-//;s/\]//' || true)

if [ -z "$TASK_IDS" ]; then
    exit 0
fi

AUTH_HEADER=""
if [ -n "$TM_API_KEY" ]; then
    AUTH_HEADER="-H \"Cookie: api_key=$TM_API_KEY\""
fi

for TASK_ID in $TASK_IDS; do
    echo "[TeamMemory] Updating task $TASK_ID from commit $COMMIT_HASH"

    # Post a message on the task
    curl -sf -X POST "${TM_BASE_URL}/api/v1/tasks/${TASK_ID}/messages" \
        -H "Content-Type: application/json" \
        -b "api_key=${TM_API_KEY}" \
        -d "{\"content\":\"Commit ${COMMIT_HASH:0:8} by ${COMMIT_AUTHOR}: $(echo "$COMMIT_MSG" | head -1 | sed 's/"/\\"/g')\"}" \
        > /dev/null 2>&1 || echo "[TeamMemory] Warning: Failed to post message to task $TASK_ID"
done
