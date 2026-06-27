#!/usr/bin/env bash
# Manage two evolution daemon instances so they survive independently of any
# controlling terminal or agent session (resilient to usage-limit interruptions).
#
#   evo_daemon.sh start|ensure   start any instance that isn't running
#   evo_daemon.sh stop           stop both instances
#   evo_daemon.sh restart        stop then start both (picks up a rebuilt binary)
#   evo_daemon.sh status         print running/stopped + pids
#
# Instance 1: --config config.toml  → populations/
# Instance 2: --config config2.toml → populations2/
# Both write saves to fractals/ (dedup runs every 2h to clean near-dups).
set -u
cd "$(dirname "$0")/.." || exit 1

BIN_REL="target/release/nnfractals"
BIN="./$BIN_REL"
LOG="evolution.log"

# Per-instance configuration
CONFIGS=("config.toml" "config2.toml")
PIDFILES=("evolution.pid" "evolution2.pid")
PATS=("${BIN_REL} --config config.toml" "${BIN_REL} --config config2.toml")

running_pid() { pgrep -f "${PATS[$1]}" | head -1; }

start_instance() {
  local idx=$1
  local pid; pid="$(running_pid $idx)"
  if [ -n "$pid" ]; then echo "[$((idx+1))] already running (pid $pid)"; return 0; fi
  if [ ! -x "$BIN" ]; then echo "binary missing: $BIN (cargo build --release)"; return 1; fi
  echo "=== $(date '+%F %T') daemon start instance $((idx+1)) ===" >> "$LOG"
  setsid "$BIN" --config "${CONFIGS[$idx]}" >> "$LOG" 2>&1 < /dev/null &
  sleep 1
  pid="$(running_pid $idx)"
  if [ -n "$pid" ]; then
    echo "$pid" > "${PIDFILES[$idx]}"
    echo "[$((idx+1))] started pid $pid (${CONFIGS[$idx]})"
  else
    echo "[$((idx+1))] FAILED to start"
    return 1
  fi
}

stop_instance() {
  local idx=$1
  local pid; pid="$(running_pid $idx)"
  if [ -z "$pid" ]; then echo "[$((idx+1))] not running"; rm -f "${PIDFILES[$idx]}"; return 0; fi
  pkill -f "${PATS[$idx]}"
  for _ in 1 2 3 4 5; do [ -z "$(running_pid $idx)" ] && break; sleep 1; done
  pkill -9 -f "${PATS[$idx]}" 2>/dev/null
  rm -f "${PIDFILES[$idx]}"
  echo "[$((idx+1))] stopped (was pid $pid)"
}

start_all()  { start_instance 0; start_instance 1; }
stop_all()   { stop_instance  0; stop_instance  1; }
status_all() {
  for idx in 0 1; do
    local pid; pid="$(running_pid $idx)"
    if [ -n "$pid" ]; then echo "[$((idx+1))] running pid $pid (${CONFIGS[$idx]})"
    else echo "[$((idx+1))] stopped (${CONFIGS[$idx]})"; fi
  done
}

case "${1:-status}" in
  start|ensure) start_all ;;
  stop)         stop_all ;;
  restart)      stop_all; start_all ;;
  status)       status_all ;;
  *) echo "usage: $0 start|ensure|stop|restart|status"; exit 1 ;;
esac
