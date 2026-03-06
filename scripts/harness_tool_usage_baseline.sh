#!/usr/bin/env bash
# Tool usage baseline for Harness Phase 0-1
# Fetches aggregate tool stats (tool_name, count, avg_duration) from /api/v1/analytics/tool-usage.
# Requires: TEAM_MEMORY_API_KEY in env. Base URL: TEAM_MEMORY_WEB_URL or http://localhost:9111
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_URL="${TEAM_MEMORY_WEB_URL:-http://localhost:9111}"
API_KEY="${TEAM_MEMORY_API_KEY:-}"

if [ -z "$API_KEY" ]; then
  echo "需登录 Web 后手动导出使用统计"
  exit 1
fi

URL="${BASE_URL}/api/v1/analytics/tool-usage?group_by=tool&days=7"
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT
HTTP_CODE=$(curl -sf -w "%{http_code}" -o "$TMPFILE" \
  -H "Authorization: Bearer ${API_KEY}" \
  "$URL" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "403" ] || [ "$HTTP_CODE" = "000" ]; then
  echo "需登录 Web 后手动导出使用统计"
  exit 1
fi

RESP_FILE="$TMPFILE" python3 -c "
import json, os, sys
try:
    with open(os.environ.get('RESP_FILE', '')) as f:
        d = json.load(f)
    data = d.get('data', [])
    if not data:
        print('tool_name\tcount\tavg_duration')
        sys.exit(0)
    print('tool_name\tcount\tavg_duration')
    for r in data:
        name = r.get('tool_name', '')
        count = r.get('count', 0)
        avg = r.get('avg_duration_ms', 0)
        print(f'{name}\t{count}\t{avg}')
except Exception as e:
    print('需登录 Web 后手动导出使用统计', file=sys.stderr)
    sys.exit(1)
"
