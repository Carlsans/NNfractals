#!/usr/bin/env bash
# Watch nnfractals-viewer for UI-thread stalls and capture what caused them.
#
# The viewer freezes hard enough that niri's close request goes unanswered
# and only `killall` clears it — which also destroys the evidence. This runs
# alongside a session and dumps stacks WHILE the freeze is happening, so the
# post-mortem doesn't depend on catching it by hand.
#
# Detection: egui only repaints on events, so an idle viewer legitimately
# burns no CPU. The detector therefore only arms while an `explorer` child is
# alive — during a stage the viewer calls request_repaint() on every child
# log line, so the main thread MUST be accumulating CPU. If it stops for
# STALL_SAMPLES consecutive samples, that is a real stall, not idleness.
#
# Usage: scripts/hang_watch.sh [outdir]    (Ctrl-C to stop)
set -u
OUT="${1:-hang_dumps}"
INTERVAL="${INTERVAL:-2}"
STALL_SAMPLES="${STALL_SAMPLES:-3}"   # x INTERVAL seconds of no main-thread CPU
# Matched against the process NAME (pgrep -x), not the command line: -f also
# matches any shell, editor or script whose arguments merely mention the
# viewer — including this script's own invocation, which made the first
# version of this detector fire on itself. Note the kernel truncates a
# process name to 15 chars, hence "nnfractals-view".
# Overridable so the detector can be tested against stand-in processes
# instead of waiting for a real freeze.
VIEWER_PAT="${VIEWER_PAT:-nnfractals-view}"
CHILD_PAT="${CHILD_PAT:-explorer}"
mkdir -p "$OUT"
LOG="$OUT/hang_watch.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

cpu_of() {  # total ticks for one tid; empty if gone
    awk '{print $14+$15}' "/proc/$1/task/$1/stat" 2>/dev/null
}
state_of() { awk '{print $3}' "/proc/$1/task/$1/stat" 2>/dev/null; }

say "watching (interval ${INTERVAL}s, stall after $((INTERVAL*STALL_SAMPLES))s of no main-thread CPU)"
prev=""; stall=0; dumped=0
while true; do
    VPID=$(pgrep -x "$VIEWER_PAT" | head -1)
    if [ -z "${VPID:-}" ]; then prev=""; stall=0; dumped=0; sleep "$INTERVAL"; continue; fi

    # Only arm while a child stage is running — see header.
    if ! pgrep -x "$CHILD_PAT" >/dev/null 2>&1; then prev=""; stall=0; dumped=0; sleep "$INTERVAL"; continue; fi

    cur=$(cpu_of "$VPID")
    if [ -n "$prev" ] && [ -n "$cur" ] && [ "$cur" = "$prev" ]; then
        stall=$((stall + 1))
    else
        if [ "$stall" -ge "$STALL_SAMPLES" ]; then
            say "RECOVERED after $((stall*INTERVAL))s"
        fi
        stall=0; dumped=0
    fi
    prev="$cur"

    # One dump per stall episode: gdb/eu-stack pause the target, and dumping
    # repeatedly would perturb the very thing being measured.
    if [ "$stall" -ge "$STALL_SAMPLES" ] && [ "$dumped" -eq 0 ]; then
        dumped=1
        TS=$(date +%Y%m%d_%H%M%S)
        D="$OUT/stall_$TS"
        mkdir -p "$D"
        say "STALL: viewer pid $VPID main-thread state=$(state_of "$VPID"), no CPU for $((stall*INTERVAL))s -> $D"

        # User-space stacks need ptrace, which yama restricts to ancestors —
        # so this only works if the viewer was started with
        # NNFRACTALS_ALLOW_PTRACE=1. When it wasn't, the per-thread `wchan`
        # below is always readable and still says WHAT each thread is
        # blocked in (futex vs. poll vs. a driver ioctl), which is most of
        # what this needs to answer.
        eu-stack -p "$VPID" > "$D/viewer_stacks.txt" 2>&1
        if grep -q "not permitted" "$D/viewer_stacks.txt"; then
            say "  (no stacks: restart the viewer with NNFRACTALS_ALLOW_PTRACE=1 for those)"
        fi
        EPID=$(pgrep -x "$CHILD_PAT" | head -1)
        [ -n "${EPID:-}" ] && eu-stack -p "$EPID" > "$D/explorer_stacks.txt" 2>&1

        # VRAM is the top open suspect: 12 GB card, ~3 GB already resident,
        # and the search allocates 4095^2 canvases while the viewer holds its
        # own textures. A stalled submission looks exactly like this freeze.
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory \
                   --format=csv > "$D/gpu.txt" 2>&1
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$D/gpu.txt" 2>&1
        {
            echo "== viewer $VPID =="; grep -E "VmRSS|VmSize|Threads" "/proc/$VPID/status" 2>/dev/null
            echo "wchan: $(cat "/proc/$VPID/wchan" 2>/dev/null)"
            echo "== explorer ${EPID:-none} =="
            [ -n "${EPID:-}" ] && grep -E "VmRSS|VmSize|Threads" "/proc/$EPID/status" 2>/dev/null
            echo "== per-thread state (viewer) =="
            # wchan = the kernel function each thread is parked in. Readable
            # without ptrace, and enough to tell a driver/ioctl stall apart
            # from a lock (futex) or an ordinary event wait (poll/epoll).
            for t in /proc/$VPID/task/*; do
                tid=$(basename "$t")
                awk -v w="$(cat "$t/wchan" 2>/dev/null)" \
                    '{printf "tid %s state %s cpu %d %s wchan=%s\n", $1, $3, $14+$15, $2, w}' \
                    "$t/stat" 2>/dev/null || echo "tid $tid gone"
            done
            echo "== system =="; free -m; cat /proc/pressure/{memory,io,cpu} 2>/dev/null
        } > "$D/context.txt" 2>&1
        say "captured"
    fi
    sleep "$INTERVAL"
done
