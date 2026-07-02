#!/usr/bin/env bash
# Manage up to four evolution daemon instances (one per fractals_N/ pool) so
# they survive independently of any controlling terminal or agent session
# (resilient to usage-limit interruptions).
#
#   evo_daemon.sh start|ensure [N]   start instances that aren't running (default: first 2 — VRAM)
#   evo_daemon.sh stop                stop ALL configured instances
#   evo_daemon.sh restart [N]         stop all, then start [N] (picks up a rebuilt binary)
#   evo_daemon.sh status              print running/stopped + pids for all 4 slots
#
# Instance 1: --config config.toml  → fractals_1/ populations_1/
# Instance 2: --config config2.toml → fractals_2/ populations_2/
# Instance 3: --config config3.toml → fractals_3/ populations_3/
# Instance 4: --config config4.toml → fractals_4/ populations_4/
# Each instance has its own save_dir (no shared pool) — dedup still runs every
# 2h per-dir to clean near-dups within each.
#
# Only ~2 instances fit in 12GB VRAM at once (each aesthetic sidecar ~3GB), so
# start/ensure/restart default to the first N=2 slots; pass a count to run more.
set -u
cd "$(dirname "$0")/.." || exit 1

BIN_REL="target/release/nnfractals"
BIN="./$BIN_REL"
DEFAULT_N=2

# Per-instance configuration (4 slots; only the first N are started by default)
CONFIGS=("config.toml" "config2.toml" "config3.toml" "config4.toml")
PIDFILES=("evolution.pid" "evolution2.pid" "evolution3.pid" "evolution4.pid")
LOGS=("evo1.log" "evo2.log" "evo3.log" "evo4.log")
PATS=(
  "${BIN_REL} --config config.toml"
  "${BIN_REL} --config config2.toml"
  "${BIN_REL} --config config3.toml"
  "${BIN_REL} --config config4.toml"
)

running_pid() { pgrep -f "${PATS[$1]}" | head -1; }

start_instance() {
  local idx=$1
  local pid; pid="$(running_pid $idx)"
  if [ -n "$pid" ]; then echo "[$((idx+1))] already running (pid $pid)"; return 0; fi
  if [ ! -x "$BIN" ]; then echo "binary missing: $BIN (cargo build --release)"; return 1; fi
  local log="${LOGS[$idx]}"
  echo "=== $(date '+%F %T') daemon start instance $((idx+1)) ===" >> "$log"
  setsid "$BIN" --config "${CONFIGS[$idx]}" >> "$log" 2>&1 < /dev/null &
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

start_all() {
  local n="${1:-$DEFAULT_N}"
  for ((idx=0; idx<n && idx<${#CONFIGS[@]}; idx++)); do start_instance "$idx"; done
}
stop_all() {
  for idx in "${!CONFIGS[@]}"; do stop_instance "$idx"; done
}
status_all() {
  for idx in "${!CONFIGS[@]}"; do
    local pid; pid="$(running_pid $idx)"
    if [ -n "$pid" ]; then echo "[$((idx+1))] running pid $pid (${CONFIGS[$idx]})"
    else echo "[$((idx+1))] stopped (${CONFIGS[$idx]})"; fi
  done
}

case "${1:-status}" in
  start|ensure) start_all "${2:-$DEFAULT_N}" ;;
  stop)         stop_all ;;
  restart)      stop_all; start_all "${2:-$DEFAULT_N}" ;;
  status)       status_all ;;
  *) echo "usage: $0 start|ensure|stop|restart|status [N]"; exit 1 ;;
esac
