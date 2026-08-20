#!/usr/bin/env bash
# Batch pipeline: for each shortlisted genome, run a deep video-zoom search,
# verify the best winner's frames offline, and render it as a full video.
#
# Verification happens BEFORE the multi-hour full-resolution render, using
# `verify-chain`'s flood statistic (fraction of a frame that is one colour).
# A chain that dies partway is skipped and the next-ranked winner tried
# instead — the whole point is never to spend hours rendering a dead zoom.
#
# Usage: scripts/zoom_batch.sh <genome.nn> <tag>
set -u

GENOME="$1"
TAG="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# 30, not 15: at 15 every deep chain in the first batch ended `DepthReached`
# — the cap, not the fractal, was what stopped them, and the deepest (f08)
# halted at zoom 1.3e10 with the f64 wall still 179x further out. Steps
# average 3.1-4.4x zoom each, so ~25 plies are needed before precision
# becomes the binding constraint. Costs roughly 2x the search time, but only
# for chains that keep finding structure; ones that dead-end still exit early.
DEPTH="${DEPTH:-30}"
STEPS="${STEPS:-2400}"
FPS="${FPS:-30}"
W="${W:-1080}"
H="${H:-1920}"
TOPW="${TOPW:-6}"
# Render only every Nth frame and warp the rest (see
# export_video_chain_interpolated). Measured on a real 2400-frame chain:
# ~8x faster AND slightly cleaner, because resampling softens the aliasing
# speckle that point-sampling produces. 1 disables it.
KFSTRIDE="${KFSTRIDE:-16}"
# ANGLE=1 searches AND renders in exit-angle colouring. It must be set for
# both halves or neither: the search ranks chains by the compressed size of a
# real encode, so a chain chosen under one colouring was never evaluated under
# the other. verify-chain picks it up from the winners manifest on its own;
# this only needs to force it when re-rendering an older manifest.
ANGLE_FLAG=""
[ "${ANGLE:-0}" = "1" ] && ANGLE_FLAG="--angle-coloring"
# MUST match `explorer video-zoom-explore`'s OWN default output directory,
# rather than passing one: its positionals are `<genome> [cx] [cy] [zoom]
# [out_dir]` and parsing stops at the first flag (`pos = &args[..flag_boundary]`),
# so an out_dir written after `--depth ...` is silently DROPPED and the run
# lands here anyway. Specifying it positionally would mean also supplying
# cx/cy/zoom, which for a .nn input are meant to come from the genome itself.
# Deriving the default is the honest fix. (Same trap as the documented
# explorer positional/flag bug — hit again on the first real batch run.)
OUT="explorer_out/$(basename "$GENOME" .nn)_video_zoom"
LOG="$OUT/batch.log"

# Headless batch: use every core. The in-process default deliberately keeps
# half of them free for the interactive viewer, which isn't running here.
export NNFRACTALS_SAVE_THREADS="${NNFRACTALS_SAVE_THREADS:-$(nproc)}"

mkdir -p "$OUT" viewer_output

# SKIP_EXPLORE=1 reuses an existing winners manifest and goes straight to
# verify+render. The search is by far the expensive half and depends only on
# --final-width (never on height), so a change to the output geometry does
# not invalidate it — this is the resume path for exactly that case.
if [ "${SKIP_EXPLORE:-0}" = "1" ]; then
    echo "=== [$TAG] $(date +%H:%M:%S) SKIP_EXPLORE — reusing existing manifest ===" | tee -a "$LOG"
else
    echo "=== [$TAG] $(date +%H:%M:%S) explore depth=$DEPTH genome=$GENOME ===" | tee -a "$LOG"
    # No out_dir argument on purpose — see the OUT= comment above.
    # --final-width/--final-height MUST be the real output geometry: the
    # search validates its chains on the frames the exporter would actually
    # render, and frame content depends on the output ASPECT. Validating at
    # a different aspect checks a different crop of the fractal.
    ./target/release/explorer video-zoom-explore "$GENOME" \
        --depth "$DEPTH" --top-winners "$TOPW" \
        --final-width "$W" --final-height "$H" $ANGLE_FLAG >> "$LOG" 2>&1
    echo "=== [$TAG] $(date +%H:%M:%S) explore done rc=$? ===" | tee -a "$LOG"
fi

MANIFEST="$OUT/video_zoom_winners.jsonl"
if [ ! -s "$MANIFEST" ]; then
    echo "[$TAG] NO WINNERS — skipping" | tee -a "$LOG"
    exit 2
fi

NW=$(wc -l < "$MANIFEST")
echo "[$TAG] $NW winners" | tee -a "$LOG"

# Try winners best-first; render the first one that verifies alive.
for RANK in $(seq 0 $((NW - 1))); do
    echo "=== [$TAG] $(date +%H:%M:%S) verifying rank $RANK ===" | tee -a "$LOG"
    # Verify at 1/3 scale but the SAME aspect (1080x1980 -> 360x660): the
    # frame sequence depends on aspect, not size, and a collapsed frame is
    # collapsed at any resolution — so this costs ~9x less than verifying at
    # full res while testing the identical camera path.
    VERDICT=$(./target/release/explorer verify-chain \
        --winners "$MANIFEST" --rank "$RANK" --nn "$GENOME" \
        --render-width $((W / 3)) --render-height $((H / 3)) --render-steps "$STEPS" \
        --stride 60 2>&1 | tee -a "$LOG" | grep -E "^VERDICT" | head -1)
    echo "[$TAG] rank $RANK -> $VERDICT" | tee -a "$LOG"

    # A chain that degrades only near the END is still a usable video: render
    # the clean prefix rather than discarding the whole search. Real case:
    # a 140-minute search produced one surviving chain, good for 2220 of
    # 2388 frames, thrown away for the last 7%.
    CAP=""
    case "$VERDICT" in
        *ALIVE*) ;;
        *NOISE*|*DEAD*)
            # Two separate extractions rather than word-splitting one string:
            # splitting behaviour differs between shells (zsh does not split
            # unquoted expansions), and this must not depend on that.
            FIRST_BAD=$(printf '%s' "$VERDICT" | sed -n 's/.*frame \([0-9]*\) of [0-9]*.*/\1/p')
            TOTAL=$(printf '%s' "$VERDICT" | sed -n 's/.*frame [0-9]* of \([0-9]*\).*/\1/p')
            if [ -n "$FIRST_BAD" ] && [ -n "$TOTAL" ]; then
                # Only worth salvaging if most of it is good; below this the
                # video would be a stub. 10% margin before the bad stretch.
                if [ "$FIRST_BAD" -gt $((TOTAL / 2)) ]; then
                    CAP=$(( FIRST_BAD - FIRST_BAD / 10 ))
                    echo "[$TAG] rank $RANK salvageable: clean to $FIRST_BAD/$TOTAL, capping at $CAP" | tee -a "$LOG"
                fi
            fi
            ;;
    esac

    if [ -n "$CAP" ] || [ -z "${VERDICT##*ALIVE*}" ]; then
        MP4="viewer_output/BATCH_${TAG}_rank${RANK}.mp4"
        echo "=== [$TAG] $(date +%H:%M:%S) rendering $MP4 ${CAP:+(capped $CAP frames)} ===" | tee -a "$LOG"
        ./target/release/explorer verify-chain \
            --winners "$MANIFEST" --rank "$RANK" --nn "$GENOME" \
            --render-video "$MP4" \
            --render-width "$W" --render-height "$H" \
            --render-steps "$STEPS" --render-fps "$FPS" \
            --keyframe-stride "$KFSTRIDE" $ANGLE_FLAG \
            ${CAP:+--max-frames "$CAP"} >> "$LOG" 2>&1
        echo "=== [$TAG] $(date +%H:%M:%S) render done: $MP4 ===" | tee -a "$LOG"
        exit 0
    fi
    echo "[$TAG] rank $RANK DEAD — trying next" | tee -a "$LOG"
done

echo "[$TAG] all $NW winners failed verification" | tee -a "$LOG"
exit 3
