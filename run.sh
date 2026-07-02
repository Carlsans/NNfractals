#!/usr/bin/env bash
# run.sh — run N evolution instances that share one fractal gallery.
#
#   ./run.sh [N] [--config FILE] [--build] [--fg]
#
#   N           number of instances to launch (default: 1)
#   --config F  base config file (default: config.toml)
#   --build     cargo build --release before launching
#   --fg        run a single instance in the foreground (ignores N; Ctrl-C to stop)
#
# All instances use the same config, so they write discoveries to the same
# save_dir (hash-named files → no collisions, more explorers filling one gallery).
# Temp probe/score files are PID-scoped in the binary, so concurrent instances
# don't corrupt each other's aesthetic scoring.
#
#   ./run.sh                 one instance, background
#   ./run.sh 4               four instances, background
#   ./run.sh --fg            one instance in the foreground
#   ./run.sh 3 --build       rebuild, then launch three instances
#   ./run.sh stop            stop all instances started by this script
#   ./run.sh status          show running instances
set -u
cd "$(dirname "$0")" || exit 1

BIN="./target/release/nnfractals"
LOG="evolution.log"
PIDFILE=".run_pids"
CONFIG="config.toml"
N=1
BUILD=0
FG=0

# ── Sub-commands ────────────────────────────────────────────────────────────
case "${1:-}" in
  stop)
    if [ -f "$PIDFILE" ]; then
      while read -r pid; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null && echo "stopped pid $pid"
      done < "$PIDFILE"
      rm -f "$PIDFILE"
    else
      echo "no $PIDFILE — nothing to stop"
    fi
    # Also stop the aesthetic scorer sidecar(s). They are child processes of the
    # evolution instances, but can outlive an abrupt kill (e.g. blocked in a GPU
    # score) and keep holding VRAM. SIGTERM first, then SIGKILL any stragglers.
    # (pkill never matches its own PID, and "bash run.sh stop" doesn't contain the
    # pattern, so this can't self-terminate the script.)
    if pkill -f "aesthetic_scorer.py" 2>/dev/null; then
      echo "stopped aesthetic scorer sidecar(s)"
      sleep 1
      pkill -9 -f "aesthetic_scorer.py" 2>/dev/null
    fi
    exit 0 ;;
  status)
    if [ -f "$PIDFILE" ]; then
      while read -r pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "running  pid $pid"
        else echo "dead     pid $pid"; fi
      done < "$PIDFILE"
    else
      echo "no instances tracked ($PIDFILE absent)"
    fi
    exit 0 ;;
esac

# ── Parse args ──────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --build)  BUILD=1; shift ;;
    --fg)     FG=1; shift ;;
    ''|*[!0-9]*) echo "unknown arg: $1"; echo "usage: $0 [N] [--config FILE] [--build] [--fg] | stop | status"; exit 1 ;;
    *)        N="$1"; shift ;;
  esac
done

# ── Build if requested (or if the binary is missing) ────────────────────────
if [ "$BUILD" = 1 ] || [ ! -x "$BIN" ]; then
  echo "building release binary…"
  cargo build --release || { echo "build failed"; exit 1; }
fi

# ── Foreground: single instance, no PID tracking ────────────────────────────
if [ "$FG" = 1 ]; then
  echo "running 1 instance (foreground) --config $CONFIG — Ctrl-C to stop"
  exec "$BIN" --config "$CONFIG"
fi

# ── Background: launch N instances, track PIDs ──────────────────────────────
: > "$PIDFILE"
echo "=== $(date '+%F %T') run.sh launching $N instance(s) --config $CONFIG ===" >> "$LOG"
for i in $(seq 1 "$N"); do
  setsid "$BIN" --config "$CONFIG" >> "$LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" >> "$PIDFILE"
  echo "[$i/$N] started pid $pid"
done
echo "logs → $LOG   |   ./run.sh status   |   ./run.sh stop"
