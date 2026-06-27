#!/usr/bin/env bash
# Run at the end of the colormap experiment:
#   1. Generate the CLIP/LAION comparison chart
#   2. Switch config to the winning colormap
#   3. Restart daemon with winner
set -eu
cd "$(dirname "$0")/.."
echo "=== $(date '+%F %T') colormap experiment finalize ==="

# Generate chart
echo "-- generating chart --"
python3 scripts/colormap_chart.py colormap_loop_state.json colormap_scores.png

# Determine winner from state (highest laion_mean)
WINNER=$(python3 -c "
import json, sys
h = json.load(open('colormap_loop_state.json'))['history']
if not h: sys.exit(1)
best = max(h, key=lambda x: x.get('laion_mean', 0))
print(best['colormap'])
")
echo "-- winner: $WINNER --"
python3 scripts/colormap_switch.py "$WINNER"
echo "-- done. Chart saved to colormap_scores.png --"
