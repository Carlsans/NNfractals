#!/usr/bin/env bash
# Overnight parameter sweep + quality evaluation for video-zoom-explore.
#
# Evaluates the `--min-file-size-ratio` floor fix (2026-08-13) against the
# genome whose run Carl reported as failing: "after about 1/4 of the zoom
# the algo gets stuck in a low entropy zone and keeps zooming in on it."
#
# ratio=0.0 is the deliberate CONTROL — it disables the floor entirely,
# reproducing the exact pre-fix behavior, so any improvement at higher
# ratios is measured against the real baseline rather than assumed.
#
# Quality metric (no new Rust code needed): re-render every waypoint of the
# winning chain via `explorer shot` at a fixed resolution and record each
# PNG's byte size. At fixed resolution PNG size is a direct proxy for
# `png_compression_entropy` — so the SHAPE of that trajectory across the
# chain is exactly the reported symptom, measured: a run that degrades into
# a low-entropy zone shows sizes collapsing partway through, while a good
# run holds them up all the way to the leaf.
set -u
cd /home/carl/rust_projects/NNfractals

OUT=${VZ_OUT:-/tmp/claude-1000/-home-carl-rust-projects-NNfractals/f717be76-d887-47cc-a431-78a4d7202c8c/scratchpad/sweep}
mkdir -p "$OUT"
SEED=viewer_output/vae_explore/5854a6e40a7ad417/_seed_genome.nn
CX=0.0575101301074028; CY=0.0873841941356659; Z=1.0199525655935573
SHOT_RES=256
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

for RATIO in ${VZ_RATIOS:-0.0 0.6 0.7 0.8}; do
  TAG="r${RATIO}"
  DIR="explorer_out/vz_sweep_${TAG}"
  rm -rf "$DIR"
  echo "=== ratio=$RATIO start $(date +%H:%M:%S) ===" >> "$SUMMARY"
  START=$(date +%s)
  ./target/release/explorer video-zoom-explore "$SEED" "$CX" "$CY" "$Z" "$DIR" \
      --depth 5 --finalists 3 --lookahead-plies 2 --method mixed \
      --final-width 1080 --top-winners 5 \
      --min-file-size-ratio ${VZ_SEED_RATIO:-0.45} --min-file-size-step-ratio ${VZ_STEP_RATIO:-0.80} \
      --min-step-zoom "$RATIO" \
      > "$OUT/run_${TAG}.log" 2>&1
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))

  if [ ! -f "$DIR/video_zoom_winners.jsonl" ]; then
    echo "ratio=$RATIO FAILED (rc=$RC, ${ELAPSED}s) - no winners file" >> "$SUMMARY"
    continue
  fi

  # Per-waypoint richness trajectory of the best winner.
  python3 - "$DIR" "$SEED" "$SHOT_RES" "$OUT" "$TAG" "$ELAPSED" "$RATIO" >> "$SUMMARY" <<'PY'
import json, subprocess, sys, os, statistics
d, seed, res, out, tag, elapsed, ratio = sys.argv[1:8]
res = int(res)
wins = [json.loads(l) for l in open(f'{d}/video_zoom_winners.jsonl')]
if not wins:
    print(f'ratio={ratio} NO WINNERS ({elapsed}s)'); sys.exit()
w = wins[0]
sizes = []
tmp = f'{out}/_shot_{tag}.png'
for i, wp in enumerate(w['chain']):
    subprocess.run(['./target/release/explorer', 'shot', seed,
                    str(wp['cx']), str(wp['cy']), str(wp['zoom']), str(res), tmp],
                   capture_output=True)
    sizes.append(os.path.getsize(tmp) if os.path.exists(tmp) else 0)
    if os.path.exists(tmp): os.remove(tmp)
n = len(sizes)
head = statistics.mean(sizes[:max(1, n//4)])          # first quarter
tail = statistics.mean(sizes[max(1, n//4):]) if n > 1 else head  # rest
floor_rejects = sum(1 for l in open(f'{d}/vae_explore_log.jsonl')
                    if '"n_above_floor":0' in l.replace(' ', ''))
print(f'ratio={ratio} legs={w["n_legs"]} vid_ratio={w["final_probe_ratio"]:.4f} '
      f'ended={w["ended_reason"]} {elapsed}s floor_rejects={floor_rejects}')
print(f'   waypoint_kb: ' + ' '.join(f'{s//1024}' for s in sizes))
print(f'   head_avg_kb={head/1024:.0f} tail_avg_kb={tail/1024:.0f} '
      f'retention={tail/head if head else 0:.2f}  (1.0=holds up, <<1=degrades)')
print(f'   all_winner_vid_ratios: ' + ' '.join(f'{x["final_probe_ratio"]:.4f}' for x in wins))
PY
done
echo "=== SWEEP DONE $(date +%H:%M:%S) ===" >> "$SUMMARY"
