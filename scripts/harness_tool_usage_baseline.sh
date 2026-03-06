#!/usr/bin/env bash
# Tool usage baseline for Harness Phase 0-1
# Fetches aggregate tool stats from /api/v1/analytics/tool-usage.
# Output: .debug/docs/harness_tool_usage_YYYYMMDD.json
# If service unavailable: writes placeholder JSON to same path.
# Requires: TEAM_MEMORY_API_KEY in env (optional for placeholder). Base URL: TEAM_MEMORY_WEB_URL or http://localhost:9111
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_URL="${TEAM_MEMORY_WEB_URL:-http://localhost:9111}"
API_KEY="${TEAM_MEMORY_API_KEY:-}"
OUT_DIR="${ROOT_DIR}/.debug/docs"
DATE_SUFFIX=$(date +%Y%m%d)
OUT_FILE="${OUT_DIR}/harness_tool_usage_${DATE_SUFFIX}.json"

mkdir -p "$OUT_DIR"

write_placeholder() {
  local reason="${1:-service_unavailable}"
  python3 -c "
import json, sys
out = {
  'data': [],
  'group_by': 'tool',
  'days': 7,
  'placeholder': True,
  'reason': sys.argv[1]
}
print(json.dumps(out, ensure_ascii=False, indent=2))
" "$reason" > "$OUT_FILE"
  echo "Wrote placeholder to $OUT_FILE (reason: $reason)" >&2
}

if [ -z "$API_KEY" ]; then
  write_placeholder "no_api_key"
  exit 0
fi

URL="${BASE_URL}/api/v1/analytics/tool-usage?group_by=tool&days=7"
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT
HTTP_CODE=$(curl -sf -w "%{http_code}" -o "$TMPFILE" \
  -H "Authorization: Bearer ${API_KEY}" \
  "$URL" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" != "200" ]; then
  case "$HTTP_CODE" in
    401) write_placeholder "unauthorized" ;;
    403) write_placeholder "forbidden" ;;
    000|000000) write_placeholder "connection_failed" ;;
    *)   write_placeholder "http_${HTTP_CODE}" ;;
  esac
  exit 0
fi

RESP_FILE="$TMPFILE" OUT_FILE="$OUT_FILE" python3 -c "
import json, os, sys
try:
    with open(os.environ.get('RESP_FILE', '')) as f:
        d = json.load(f)
    out_path = os.environ.get('OUT_FILE', '')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(d.get(\"data\", []))} tools to {out_path}', file=sys.stderr)
except Exception as e:
    out_path = os.environ.get('OUT_FILE', '')
    placeholder = {'data': [], 'group_by': 'tool', 'days': 7, 'placeholder': True, 'reason': 'parse_error'}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(placeholder, f, ensure_ascii=False, indent=2)
    print(f'Parse error, wrote placeholder to {out_path}: {e}', file=sys.stderr)
    sys.exit(0)
"
