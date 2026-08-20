#!/usr/bin/env bash
# Drives scripts/zoom_batch.sh across the shortlisted genomes, one at a time.
#
# Sequential on purpose: each render already saturates every core, so running
# two would just make both slower and interleave the logs. Failures never stop
# the run — a genome whose winners all verify DEAD is recorded and skipped, so
# an overnight batch keeps producing videos instead of stalling on one bad one.
#
# Usage: scripts/zoom_batch_all.sh <shortlist.tsv>
#   shortlist.tsv: one "<tag>\t<genome.nn>" per line, in processing order.
set -u

LIST="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
SUMMARY="viewer_output/BATCH_SUMMARY.txt"

echo "=== batch driver started $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$SUMMARY"

while IFS=$'\t' read -r TAG GENOME; do
    [ -z "${TAG:-}" ] && continue
    case "$TAG" in \#*) continue ;; esac
    [ -f "$GENOME" ] || { echo "$(date +%H:%M:%S) $TAG MISSING $GENOME" >> "$SUMMARY"; continue; }

    # Already produced a video for this tag? Skip, so the driver can be
    # restarted after an interruption without redoing finished work.
    if ls "viewer_output/BATCH_${TAG}_rank"*.mp4 >/dev/null 2>&1; then
        echo "$(date +%H:%M:%S) $TAG SKIP (video already exists)" >> "$SUMMARY"
        continue
    fi

    START=$(date +%s)
    echo "$(date +%H:%M:%S) $TAG START $GENOME" >> "$SUMMARY"
    ./scripts/zoom_batch.sh "$GENOME" "$TAG"
    RC=$?
    MINS=$(( ($(date +%s) - START) / 60 ))
    case $RC in
        0) MP4=$(ls -t "viewer_output/BATCH_${TAG}_rank"*.mp4 2>/dev/null | head -1)
           echo "$(date +%H:%M:%S) $TAG OK ${MINS}min -> $MP4" >> "$SUMMARY" ;;
        2) echo "$(date +%H:%M:%S) $TAG NO-WINNERS ${MINS}min" >> "$SUMMARY" ;;
        3) echo "$(date +%H:%M:%S) $TAG ALL-DEAD ${MINS}min" >> "$SUMMARY" ;;
        *) echo "$(date +%H:%M:%S) $TAG FAILED rc=$RC ${MINS}min" >> "$SUMMARY" ;;
    esac
done < "$LIST"

echo "=== batch driver finished $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$SUMMARY"
