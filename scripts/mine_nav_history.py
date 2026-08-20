#!/usr/bin/env python3
"""
Mine historical viewer navigation trajectories from already-saved images.

The viewer has no persisted navigation log going back further than today
(see src/bin/viewer.rs's `log_nav_event`/`nav_log.jsonl`, added alongside
this script). But every image saved via the viewer's Save dialog encodes its
own view in the filename (`{stem}_cx{cx:.4}_cy{cy:.4}_z{zoom:.2}_{w}x{h}.png`,
`spawn_save` in viewer.rs) — grouping same-stem saves by mtime reconstructs a
COARSE, checkpoint-level trajectory: only the moments Carl explicitly hit
Save, not the continuous pan/zoom path between them. This is a one-time
bootstrap for the (u, v, log-zoom-ratio) navigation-prediction model — real
volume comes from nav_log.jsonl accumulating over normal use going forward.

Usage:
  python3 scripts/mine_nav_history.py                  scan, segment, write nav_log_mined.jsonl + contact sheet(s)
  python3 scripts/mine_nav_history.py --dirs D1 D2 ...  scan specific directories instead of the defaults
  python3 scripts/mine_nav_history.py --dry-run         print what would be found, write nothing

Output (gitignored, not committed):
  nav_log_mined.jsonl   one {"event":"mined_zoom", "source":"mined", ...} line per
                         (before, after) step in a segment — same shape as
                         nav_log.jsonl's live "nav" events, plus a "segment"
                         id and "step" index so a segment can be reassembled.
  nav_contact_*.png     one contact sheet per candidate segment, for a human
                         (Carl) to review and prune before anything trains on
                         this data — see this project's own established
                         "render and look before trusting a metric" habit.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_DIRS = ["viewer_output", "fractals", "fractals_dag"]
DEFAULT_SEARCH_DIRS = [
    "fractals_1", "fractals_2", "fractals_3", "fractals_4",
    "oldfractals", "Starred", "train_corpus",
]

# `{stem}_cx{cx:.4}_cy{cy:.4}_z{zoom:.2}_{w}x{h}[_{n}].png` — spawn_save, viewer.rs
FNAME_RE = re.compile(
    r"^(?P<stem>.+)_cx(?P<cx>-?\d+\.\d+)_cy(?P<cy>-?\d+\.\d+)_z(?P<zoom>-?\d+\.\d+)"
    r"_(?P<w>\d+)x(?P<h>\d+)(?:_(?P<n>\d+))?\.png$"
)

# Containment tolerance for "does `after` land inside `before`'s frame" — a
# real navigation step, not two unrelated saves under one genome load.
# `after.zoom` must be at least this close to `before.zoom` (>= means same
# depth or deeper; a small slack below 1.0 absorbs float-formatting rounding
# in the 2-decimal `z{:.2}` filename field, not a real zoom-out).
MIN_ZOOM_RATIO = 0.98
# Center must land within this fraction of before's half-extent to count as
# "inside" — 1.0 would be the literal frame edge; a little slack (the
# filename's cx/cy are only 4-decimal-rounded) avoids rejecting a real step
# that lands just past the nominal edge due to rounding.
CONTAINMENT_SLACK = 1.15


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def find_viewer_saves(dirs):
    """Return {stem: [(mtime, cx, cy, zoom, w, h, path), ...]} sorted by mtime,
    collision-suffix duplicates (_2/_3/... of an identical cx/cy/zoom) collapsed
    to their earliest occurrence."""
    by_stem = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.png")):
            m = FNAME_RE.match(p.name)
            if not m:
                continue
            stem = m.group("stem")
            cx, cy, zoom = float(m.group("cx")), float(m.group("cy")), float(m.group("zoom"))
            w, h = int(m.group("w")), int(m.group("h"))
            mtime = p.stat().st_mtime
            by_stem.setdefault(stem, []).append((mtime, cx, cy, zoom, w, h, p))

    for stem, entries in by_stem.items():
        entries.sort(key=lambda e: e[0])
        deduped = []
        seen = {}
        for e in entries:
            key = (round(e[1], 4), round(e[2], 4), round(e[3], 2))
            if key in seen:
                continue  # a _2/_3/... re-save of the identical view — not a new step
            seen[key] = True
            deduped.append(e)
        by_stem[stem] = deduped
    return by_stem


def contained(before, after):
    """before/after: (mtime, cx, cy, zoom, w, h, path). True if `after`'s
    center+depth plausibly lands inside `before`'s own frame — i.e. a real
    zoom-in step, not two unrelated saves under one genome load."""
    _, bcx, bcy, bzoom, bw, bh, _ = before
    _, acx, acy, azoom, _, _, _ = after
    if bzoom <= 0 or azoom / bzoom < MIN_ZOOM_RATIO:
        return False
    aspect = bw / bh if bh else 1.0
    half_x = 2.0 / bzoom * aspect
    half_y = 2.0 / bzoom
    return abs(acx - bcx) <= half_x * CONTAINMENT_SLACK and abs(acy - bcy) <= half_y * CONTAINMENT_SLACK


def segment(entries):
    """Split one stem's sorted, deduped entries into coherent sub-trajectories
    (see `contained`) — a noisy session (several unrelated exploratory
    attempts under one genome load) breaks into its real sub-attempts instead
    of being treated as one trajectory. Returns a list of segments (each a
    list of entries), singletons (length 1) dropped since a segment needs at
    least one (before, after) step to be useful."""
    if not entries:
        return []
    segments = [[entries[0]]]
    for e in entries[1:]:
        if contained(segments[-1][-1], e):
            segments[-1].append(e)
        else:
            segments.append([e])
    return [s for s in segments if len(s) >= 2]


def resolve_nn(stem, search_dirs):
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        p = d / f"{stem}.nn"
        if p.exists():
            return str(p)
        hits = list(d.rglob(f"{stem}.nn"))
        if hits:
            return str(hits[0])
    return None


def label_for_step(before, after):
    """(u, v, log_zoom_ratio) — see src/explore.rs's apply_offset/sweep_positions
    for the same (dx, dy, zoom) parameterization used internally; this is the
    training target a model's output would plug directly into."""
    _, bcx, bcy, bzoom, bw, bh, _ = before
    _, acx, acy, azoom, _, _, _ = after
    aspect = bw / bh if bh else 1.0
    half_x = 2.0 / bzoom * aspect
    half_y = 2.0 / bzoom
    u = (acx - bcx) / half_x if half_x else 0.0
    v = (acy - bcy) / half_y if half_y else 0.0
    zoom_ratio = azoom / bzoom if bzoom else 1.0
    return u, v, math.log(zoom_ratio) if zoom_ratio > 0 else 0.0


def make_contact_sheet(stem, seg_idx, entries, out_path, thumb=180):
    n = len(entries)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 16)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (mtime, cx, cy, zoom, w, h, path) in enumerate(entries):
        try:
            im = Image.open(path).convert("RGB").resize((thumb, thumb))
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (80, 0, 0))
        x = (i % cols) * thumb
        y = (i // cols) * (thumb + 16)
        sheet.paste(im, (x, y + 16))
        draw.rectangle([x, y, x + thumb, y + 14], fill=(0, 0, 0))
        draw.text((x + 2, y + 1), f"{i} z={zoom:.2e}", fill=(255, 255, 0))
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                     help=f"directories to scan for viewer-saved PNGs (default: {DEFAULT_DIRS})")
    ap.add_argument("--out", default="nav_log_mined.jsonl")
    ap.add_argument("--contact-sheet-dir", default=".")
    ap.add_argument("--dry-run", action="store_true", help="report what would be found, write nothing")
    args = ap.parse_args()

    by_stem = find_viewer_saves(args.dirs)
    total_files = sum(len(v) for v in by_stem.values())
    log(f"found {total_files} viewer-saved PNGs (post collision-dedup) across {len(by_stem)} stems in {args.dirs}")

    all_segments = []  # (stem, seg_idx, entries)
    for stem, entries in sorted(by_stem.items()):
        for i, seg in enumerate(segment(entries)):
            all_segments.append((stem, i, seg))

    n_steps = sum(len(seg) - 1 for _, _, seg in all_segments)
    log(f"{len(all_segments)} candidate segments, {n_steps} total (before,after) steps")
    for stem, i, seg in all_segments:
        z0, z1 = seg[0][3], seg[-1][3]
        log(f"  {stem} seg{i}: {len(seg)} checkpoints, zoom {z0:.3g} -> {z1:.3g}")

    if args.dry_run:
        return

    Path(args.contact_sheet_dir).mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        for stem, seg_idx, seg in all_segments:
            nn_path = resolve_nn(stem, DEFAULT_SEARCH_DIRS + args.dirs)
            segment_id = f"{stem}_{seg_idx}"
            for step, (before, after) in enumerate(zip(seg, seg[1:])):
                u, v, log_zoom_ratio = label_for_step(before, after)
                rec = {
                    "event": "mined_zoom", "source": "mined",
                    "segment": segment_id, "step": step,
                    "genome_id": stem, "nn_path": nn_path,
                    "before": {"cx": before[1], "cy": before[2], "zoom": before[3], "path": str(before[6])},
                    "after": {"cx": after[1], "cy": after[2], "zoom": after[3], "path": str(after[6])},
                    "label": {"u": u, "v": v, "log_zoom_ratio": log_zoom_ratio},
                }
                f.write(json.dumps(rec) + "\n")
                written += 1

            sheet_path = Path(args.contact_sheet_dir) / f"nav_contact_{segment_id}.png"
            make_contact_sheet(stem, seg_idx, seg, sheet_path)
    log(f"wrote {written} mined_zoom records to {args.out}")
    log(f"wrote {len(all_segments)} contact sheets (nav_contact_*.png) to {args.contact_sheet_dir} — review before training on this data")


if __name__ == "__main__":
    main()
