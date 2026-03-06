#!/usr/bin/env bash
# Send system notification for Harness Plan execution status.
# Usage: ./scripts/notify_plan_status.sh "Title" "Body"
# Used by Agent at key points: plan start, human decision, interrupt, completion.
set -e
TITLE="${1:-Harness Plan}"
BODY="${2:-Status update}"

case "$(uname -s)" in
  Darwin)
    osascript -e "display notification \"$BODY\" with title \"$TITLE\""
    ;;
  Linux)
    if command -v notify-send >/dev/null 2>&1; then
      notify-send "$TITLE" "$BODY"
    else
      echo "[$TITLE] $BODY" >&2
    fi
    ;;
  *)
    echo "[$TITLE] $BODY" >&2
    ;;
esac
