#!/usr/bin/env bash
# Manage the evolution daemon so it survives independently of any controlling
# terminal or agent session (resilient to usage-limit interruptions).
#
#   evo_daemon.sh start|ensure   start if not already running (setsid-detached)
#   evo_daemon.sh stop           stop the running daemon
#   evo_daemon.sh restart        stop then start (picks up a rebuilt binary)
#   evo_daemon.sh status         print running/stopped + pid
#
# The daemon is identified by its full binary path so we never touch unrelated
# processes. Output goes to evolution.log (append).
set -u
cd "$(dirname "$0")/.." || exit 1

BIN_REL="target/release/nnfractals"
BIN="./$BIN_REL"
PIDF="evolution.pid"
LOG="evolution.log"
PAT="${BIN_REL}\$"          # match "…/target/release/nnfractals" at end of cmdline

running_pid() { pgrep -f "$PAT" | head -1; }

start() {
  local pid; pid="$(running_pid)"
  if [ -n "$pid" ]; then echo "already running (pid $pid)"; return 0; fi
  if [ ! -x "$BIN" ]; then echo "binary missing: $BIN (cargo build --release)"; return 1; fi
  echo "=== $(date '+%F %T') daemon start ===" >> "$LOG"
  setsid "$BIN" >> "$LOG" 2>&1 < /dev/null &
  sleep 1
  pid="$(running_pid)"
  if [ -n "$pid" ]; then echo "$pid" > "$PIDF"; echo "started pid $pid"; else echo "FAILED to start"; return 1; fi
}

stop() {
  local pid; pid="$(running_pid)"
  if [ -z "$pid" ]; then echo "not running"; rm -f "$PIDF"; return 0; fi
  pkill -f "$PAT"
  for _ in 1 2 3 4 5; do [ -z "$(running_pid)" ] && break; sleep 1; done
  pkill -9 -f "$PAT" 2>/dev/null
  rm -f "$PIDF"
  echo "stopped (was pid $pid)"
}

case "${1:-status}" in
  start|ensure) start ;;
  stop)         stop ;;
  restart)      stop; start ;;
  status)
    pid="$(running_pid)"
    if [ -n "$pid" ]; then echo "running pid $pid"; else echo "stopped"; fi ;;
  *) echo "usage: $0 start|ensure|stop|restart|status"; exit 1 ;;
esac
