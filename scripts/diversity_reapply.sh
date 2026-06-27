#!/usr/bin/env bash
# 2h diversity loop iteration boundary:
#   1. Analyze formula diversity of recent saves
#   2. Adjust formula_diversity_weight in config.toml based on results
#   3. Rebuild (fast — no code change expected) and restart daemon
set -eu
cd "$(dirname "$0")/.."

ITER=${1:-"?"}
SINCE=${2:-0}

echo "=== $(date '+%F %T') diversity reapply (iter $ITER) ==="

echo "-- analyzing formula diversity --"
python3 scripts/analyze_diversity.py --since "$SINCE"

# Read current weight
CURRENT_WEIGHT=$(python3 -c "
import re
txt = open('config.toml').read()
m = re.search(r'formula_diversity_weight\s*=\s*([\d.]+)', txt)
print(float(m.group(1)) if m else 0.30)
")
echo "Current formula_diversity_weight: $CURRENT_WEIGHT"

# Let the analysis script emit a recommendation via exit code:
#   0 = maintain, 1 = increase, 2 = decrease
RECOMMEND=$(python3 -c "
import re, glob, json, os, math, statistics, sys
from collections import Counter

SINCE = float(sys.argv[1]) if len(sys.argv) > 1 else 0
N_BASIS = 58
files = glob.glob('fractals/*.nn')
recs = [json.load(open(f)) for f in files if os.path.getmtime(f) >= SINCE]

if not recs:
    print('maintain')
    sys.exit(0)

basis_counts = Counter()
for g in recs:
    used = set(min(t.get('basis',0), N_BASIS-1) for t in g.get('terms',[]))
    for b in used: basis_counts[b] += 1
total = len(recs)
probs = [basis_counts[b] / total for b in range(N_BASIS)]
H = -sum(p * math.log2(p) for p in probs if p > 0)

laions = [g['laion_score'] for g in recs if g.get('laion_score')]
laion_mean = statistics.mean(laions) if laions else 5.30

print(f'H={H:.3f} laion={laion_mean:.4f}', file=__import__('sys').stderr)

if H < 3.5 and laion_mean >= 5.25:
    print('increase')   # low diversity, beauty still fine → push harder
elif laion_mean < 5.25:
    print('decrease')   # beauty dropping → ease off
else:
    print('maintain')
" "$SINCE" 2>&1)

# Extract recommendation (last word of output)
ACTION=$(echo "$RECOMMEND" | tail -1)
echo "Analysis: $RECOMMEND"
echo "Action: $ACTION"

# Adjust weight
NEW_WEIGHT=$(python3 -c "
w = float('$CURRENT_WEIGHT')
action = '$ACTION'
if action == 'increase':
    w = min(w + 0.10, 0.60)
elif action == 'decrease':
    w = max(w - 0.08, 0.05)
print(f'{w:.2f}')
")

if [ "$NEW_WEIGHT" != "$CURRENT_WEIGHT" ]; then
    echo "Adjusting formula_diversity_weight: $CURRENT_WEIGHT → $NEW_WEIGHT"
    python3 -c "
import re
txt = open('config.toml').read()
txt = re.sub(r'(formula_diversity_weight\s*=\s*)[\d.]+', r'\g<1>$NEW_WEIGHT', txt)
open('config.toml', 'w').write(txt)
print('config.toml updated')
"
else
    echo "formula_diversity_weight unchanged at $CURRENT_WEIGHT"
fi

echo "-- rebuild --"
cargo build --release 2>&1 | tail -2

echo "-- restart daemon --"
bash scripts/evo_daemon.sh restart
bash scripts/evo_daemon.sh status
