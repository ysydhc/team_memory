#!/usr/bin/env bash
set -uo pipefail

# Graphiti task worker controller.
# Usage: ./scripts/taskctl.sh <command> [args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${ROOT_DIR}/.task_worker.log"
STATE_FILE="${ROOT_DIR}/.task_worker.state"
KUZU_DB="${ROOT_DIR}/.kuzu_db"
DEFAULT_TASK_TYPE="graphiti_ingest"

if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON="python3"
fi

# ── helpers ──────────────────────────────────────────────────────────

db_exec() {
  docker compose exec -T postgres psql -U developer -d team_memory -v ON_ERROR_STOP=1 "$@"
}

worker_pid() {
  pgrep -f "process_graph_tasks.py" 2>/dev/null | head -1 || true
}

kuzu_lock_holder_pid() {
  lsof "${KUZU_DB}" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | head -1 || true
}

# Read a field from the JSON state file (requires python for parsing)
state_field() {
  local key="$1"
  if [ -f "${STATE_FILE}" ]; then
    "${PYTHON}" -c "
import json, sys
try:
    d = json.load(open('${STATE_FILE}'))
    v = d.get('${key}')
    if v is not None:
        if isinstance(v, dict):
            print(json.dumps(v))
        else:
            print(v)
except Exception:
    pass
" 2>/dev/null
  fi
}

human_duration() {
  local secs="$1"
  if [ "${secs}" -ge 3600 ]; then
    printf "%dh%02dm" $((secs/3600)) $(( (secs%3600)/60 ))
  elif [ "${secs}" -ge 60 ]; then
    printf "%dm%02ds" $((secs/60)) $((secs%60))
  else
    printf "%ds" "${secs}"
  fi
}

# ── commands ─────────────────────────────────────────────────────────

cmd_status() {
  local pid
  pid="$(worker_pid)"

  echo "== Worker =="
  if [ -n "${pid}" ]; then
    local etime cpu mem
    etime="$(ps -p "${pid}" -o etime= 2>/dev/null | xargs)" || etime="?"
    cpu="$(ps -p "${pid}" -o %cpu= 2>/dev/null | xargs)" || cpu="?"
    mem="$(ps -p "${pid}" -o rss= 2>/dev/null | xargs)" || mem="?"
    if [ -n "${mem}" ] && [ "${mem}" != "?" ]; then
      mem="$((mem / 1024))MB"
    fi

    local state
    state="$(state_field status)"
    [ -z "${state}" ] && state="running"

    echo "  Status:    RUNNING (${state})"
    echo "  PID:       ${pid}"
    echo "  Uptime:    ${etime}"
    echo "  CPU:       ${cpu}%"
    echo "  Memory:    ${mem}"

    # Current task
    local cur
    cur="$(state_field current_task)"
    if [ -n "${cur}" ] && [ "${cur}" != "null" ]; then
      local tid rid st
      tid="$("${PYTHON}" -c "import json; d=json.loads('${cur}'); print(d.get('task_id','?')[:8])" 2>/dev/null)"
      rid="$("${PYTHON}" -c "import json; d=json.loads('${cur}'); print(d.get('record_id','?')[:12])" 2>/dev/null)"
      st="$("${PYTHON}" -c "import json; d=json.loads('${cur}'); print(d.get('started_at','?')[:19])" 2>/dev/null)"
      echo "  Current:   task=${tid} record=${rid} since=${st}"
    else
      echo "  Current:   (idle)"
    fi

    # Last completed
    local last
    last="$(state_field last_completed)"
    if [ -n "${last}" ] && [ "${last}" != "null" ]; then
      local ltid lrid lsec lat
      ltid="$("${PYTHON}" -c "import json; d=json.loads('${last}'); print(d.get('task_id','?')[:8])" 2>/dev/null)"
      lrid="$("${PYTHON}" -c "import json; d=json.loads('${last}'); print(d.get('record_id','?')[:12])" 2>/dev/null)"
      lsec="$("${PYTHON}" -c "import json; d=json.loads('${last}'); print(d.get('elapsed_s','?'))" 2>/dev/null)"
      lat="$("${PYTHON}" -c "import json; d=json.loads('${last}'); print(d.get('at','?')[:19])" 2>/dev/null)"
      echo "  Last done: task=${ltid} record=${lrid} ${lsec}s at ${lat}"
    fi

    # Session stats
    local sdone sfail
    sdone="$(state_field tasks_completed)"
    sfail="$(state_field tasks_failed)"
    [ -n "${sdone}" ] && echo "  Session:   ${sdone} completed, ${sfail:-0} failed"

    # Heartbeat age
    local hb
    hb="$(state_field heartbeat)"
    if [ -n "${hb}" ]; then
      local age_s
      age_s="$("${PYTHON}" -c "
from datetime import datetime, timezone
try:
    hb = datetime.fromisoformat('${hb}')
    now = datetime.now(timezone.utc)
    print(int((now - hb).total_seconds()))
except Exception:
    print(-1)
" 2>/dev/null)"
      if [ -n "${age_s}" ] && [ "${age_s}" -ge 0 ]; then
        if [ "${age_s}" -gt 120 ]; then
          echo "  Heartbeat: $(human_duration "${age_s}") ago ⚠️  STALE (>2min)"
        else
          echo "  Heartbeat: $(human_duration "${age_s}") ago"
        fi
      fi
    fi
  else
    echo "  Status:    STOPPED"
    # Check if it exited recently
    local last_state
    last_state="$(state_field status)"
    if [ -n "${last_state}" ]; then
      local reason
      reason="$(state_field exit_reason)"
      echo "  Last state: ${last_state}${reason:+ (${reason})}"
    fi
  fi

  echo ""

  # Kùzu lock
  echo "== Kùzu Lock =="
  local lock_pid
  lock_pid="$(kuzu_lock_holder_pid)"
  if [ -n "${lock_pid}" ]; then
    local lock_cmd
    lock_cmd="$(ps -p "${lock_pid}" -o args= 2>/dev/null | head -c 80)"
    echo "  HELD by PID ${lock_pid} (${lock_cmd})"
  else
    echo "  FREE"
  fi

  echo ""

  # Queue
  echo "== Queue (${DEFAULT_TASK_TYPE}) =="
  db_exec -c "
    SELECT status, count(*) AS cnt
    FROM background_tasks
    WHERE task_type='${DEFAULT_TASK_TYPE}'
    GROUP BY status
    ORDER BY status;
  "

  # Throughput (last hour)
  echo "== Throughput (last 1h) =="
  db_exec -c "
    SELECT count(*) AS completed_last_hour,
           CASE WHEN count(*) > 0
                THEN round(avg(EXTRACT(EPOCH FROM (completed_at - started_at)))::numeric, 1)
                ELSE 0
           END AS avg_seconds
    FROM background_tasks
    WHERE task_type='${DEFAULT_TASK_TYPE}'
      AND status='completed'
      AND completed_at > now() - interval '1 hour';
  "
}

cmd_start() {
  local pid
  pid="$(worker_pid)"
  if [ -n "${pid}" ]; then
    echo "Worker already running (PID ${pid})."
    return 0
  fi

  local lock_pid
  lock_pid="$(kuzu_lock_holder_pid)"
  if [ -n "${lock_pid}" ]; then
    local lock_cmd
    lock_cmd="$(ps -p "${lock_pid}" -o args= 2>/dev/null | head -c 80)"
    echo "ERROR: .kuzu_db locked by PID ${lock_pid}"
    echo "  ${lock_cmd}"
    echo ""
    echo "Stop the lock holder first, then retry."
    return 1
  fi

  echo "Starting worker..."
  cd "${ROOT_DIR}"

  # Rotate log if > 10MB
  if [ -f "${LOG_FILE}" ]; then
    local size
    size="$(stat -f%z "${LOG_FILE}" 2>/dev/null || stat -c%s "${LOG_FILE}" 2>/dev/null || echo 0)"
    if [ "${size}" -gt 10485760 ]; then
      mv "${LOG_FILE}" "${LOG_FILE}.old"
      echo "Log rotated (was ${size} bytes)"
    fi
  fi

  nohup "${PYTHON}" scripts/process_graph_tasks.py >> "${LOG_FILE}" 2>&1 &
  local nohup_pid=$!
  sleep 3

  pid="$(worker_pid)"
  if [ -n "${pid}" ]; then
    echo "Started (PID ${pid}). Log: ${LOG_FILE}"
    echo ""
    echo "Monitor:"
    echo "  ./scripts/taskctl.sh status    # overview"
    echo "  ./scripts/taskctl.sh logs -f   # live log"
  else
    echo "Failed to start. Last 20 lines:"
    echo "---"
    tail -20 "${LOG_FILE}" 2>/dev/null || echo "(no log)"
    return 1
  fi
}

cmd_stop() {
  local pid
  pid="$(worker_pid)"
  if [ -z "${pid}" ]; then
    echo "Worker not running."
    return 0
  fi

  echo "Stopping PID ${pid}..."
  kill "${pid}" 2>/dev/null || true
  sleep 2

  if [ -n "$(worker_pid)" ]; then
    kill -9 "$(worker_pid)" 2>/dev/null || true
    echo "Force-killed."
  else
    echo "Stopped."
  fi
}

cmd_restart() {
  cmd_stop
  sleep 1
  cmd_start
}

cmd_retry_failed() {
  local task_type="${1:-${DEFAULT_TASK_TYPE}}"
  local count
  count="$(db_exec -t -c "
    SELECT count(*) FROM background_tasks
    WHERE task_type='${task_type}' AND status='failed';
  " | xargs)"

  if [ "${count}" = "0" ]; then
    echo "No failed tasks to retry."
    return 0
  fi

  echo "Retrying ${count} failed tasks (${task_type})..."
  db_exec -c "
    UPDATE background_tasks
    SET status = 'pending',
        retry_count = 0,
        error_message = NULL,
        started_at = NULL,
        completed_at = NULL
    WHERE task_type='${task_type}' AND status='failed';
  "
  echo "Done. Run './scripts/taskctl.sh start' if worker is stopped."
}

cmd_logs() {
  local follow=false
  local lines=50

  while [ $# -gt 0 ]; do
    case "$1" in
      -f|--follow) follow=true; shift ;;
      *) lines="$1"; shift ;;
    esac
  done

  if [ ! -f "${LOG_FILE}" ]; then
    echo "No log file yet: ${LOG_FILE}"
    return 0
  fi

  if [ "${follow}" = true ]; then
    echo "=== Live log (Ctrl-C to stop) ==="
    tail -"${lines}" -f "${LOG_FILE}"
  else
    tail -"${lines}" "${LOG_FILE}"
  fi
}

cmd_watch() {
  echo "=== Live status (Ctrl-C to stop, refresh every ${1:-5}s) ==="
  while true; do
    clear
    echo "$(date '+%H:%M:%S') — taskctl watch"
    echo "──────────────────────────────────"
    cmd_status
    sleep "${1:-5}"
  done
}

# ── main ─────────────────────────────────────────────────────────────

usage() {
  cat <<'EOF'
Graphiti task worker controller.

Usage: ./scripts/taskctl.sh <command> [args]

Commands:
  status                Rich status: PID, uptime, CPU, current task, throughput
  start                 Start the worker (refuses if Kùzu locked)
  stop                  Stop the worker
  restart               Stop + start
  retry-failed [type]   Reset failed tasks back to pending
  logs [N] [-f]         Show log (default 50 lines; -f for live tail)
  watch [interval]      Continuous status refresh (default 5s)
  summary [task_type]   Raw task counts by status

EOF
}

main() {
  local cmd="${1:-status}"
  shift || true

  case "${cmd}" in
    status)       cmd_status ;;
    summary)
      local task_type="${1:-}"
      if [ -n "${task_type}" ]; then
        db_exec -c "
          SELECT status, count(*) AS cnt
          FROM background_tasks
          WHERE task_type='${task_type}'
          GROUP BY status ORDER BY status;
        "
      else
        db_exec -c "
          SELECT task_type, status, count(*) AS cnt
          FROM background_tasks
          GROUP BY task_type, status
          ORDER BY task_type, status;
        "
      fi
      ;;
    start)        cmd_start ;;
    stop)         cmd_stop ;;
    restart)      cmd_restart ;;
    retry-failed) cmd_retry_failed "${1:-}" ;;
    logs)         cmd_logs "$@" ;;
    watch)        cmd_watch "${1:-5}" ;;
    -h|--help|help) usage ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      return 1
      ;;
  esac
}

main "$@"
