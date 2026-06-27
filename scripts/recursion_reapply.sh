#!/usr/bin/env bash
# One iteration boundary of the recursion-evolution loop:
#   1. retrain the formula-only recursion model on freshly accumulated archive data
#   2. rebuild (no-op if code unchanged) and restart the daemon so it loads the
#      updated recursion_model.json and keeps evolving with the refreshed predictor.
# The metric stays a major selection criterion alongside file-size (PNG) entropy.
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== $(date '+%F %T') recursion reapply ==="
echo "-- retrain formula-only recursion model --"
python3 scripts/fit_recursion_model.py || { echo "model fit FAILED"; exit 1; }
echo "-- rebuild --"
cargo build --release 2>&1 | tail -1
echo "-- restart daemon --"
bash scripts/evo_daemon.sh restart
bash scripts/evo_daemon.sh status
