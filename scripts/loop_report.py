#!/usr/bin/env python3
"""Loop iteration report: read all saved .nn files and display beauty metrics."""
import json, glob, os, statistics, sys, time

files = sorted(glob.glob("fractals/*.nn"), key=os.path.getmtime)
if not files:
    print("No saved genomes yet.")
    sys.exit(0)

records = []
for f in files:
    try:
        d = json.load(open(f))
        b = d.get("beauty", 0.0)
        records.append({
            "file":     os.path.basename(f),
            "mtime":    os.path.getmtime(f),
            "beauty":   b,
            "boundary": d.get("beauty_boundary", None),
            "edge":     d.get("beauty_edge",     None),
            "entropy":  d.get("beauty_entropy",  None),
            "self_sim": d.get("beauty_self_sim",  None),
            "cool":     d.get("beauty_cool_zone", None),
            "clip":     d.get("clip_score",  None) or None,
            "laion":    d.get("laion_score", None) or None,
        })
    except Exception:
        pass

records.sort(key=lambda r: r["mtime"])

# --- Summary table ---
print()
print("━" * 72)
print(f"  NNfractals Loop Report  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("━" * 72)

have_sub = any(r["boundary"] is not None for r in records)

COLS = "  {:>4}  {:>6}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {}"
HDR  = COLS.format("rank", "beauty", "bound", "edge", "entro", "sim", "cool", "clip", "laion", "file")
print()
print(HDR)
print("  " + "─" * 68)

sorted_r = sorted(records, key=lambda r: r["beauty"], reverse=True)
for rank, r in enumerate(sorted_r[:15], 1):
    def fmt(v): return f"{v:.3f}" if v is not None else "  —  "
    def fmtl(v): return f"{v:.2f}" if v is not None else "  —  "
    flag = " ✓" if r["beauty"] >= 0.42 else ""
    print(COLS.format(
        rank,
        f"{r['beauty']:.4f}",
        fmt(r["boundary"]),
        fmt(r["edge"]),
        fmt(r["entropy"]),
        fmt(r["self_sim"]),
        fmt(r["cool"]),
        fmt(r["clip"]),
        fmtl(r["laion"]),
        r["file"][:20] + flag,
    ))

# --- Aggregate stats ---
beauties = [r["beauty"] for r in records if r["beauty"] > 0]
passed   = [b for b in beauties if b >= 0.42]
print()
print(f"  Total saved : {len(records)}")
print(f"  Above 0.42  : {len(passed)} ({100*len(passed)/max(len(records),1):.0f}%)")
if beauties:
    print(f"  Beauty      : min={min(beauties):.3f}  mean={statistics.mean(beauties):.3f}"
          f"  max={max(beauties):.3f}  stdev={statistics.stdev(beauties):.3f}" if len(beauties)>1
          else f"  Beauty      : {beauties[0]:.3f}")

# --- Sub-score breakdown for records that have them ---
sub_records = [r for r in records if r["boundary"] is not None and r["beauty"] > 0]
if sub_records:
    print()
    print("  Sub-score averages (all saved genomes with breakdown):")
    keys  = ["boundary", "edge", "entropy", "self_sim", "cool"]
    names = ["Boundary ", "Edge     ", "Entropy  ", "Self-sim ", "Cool-zone"]
    weights = [0.20, 0.25, 0.20, 0.15, 0.20]
    for name, key, w in zip(names, keys, weights):
        vals = [r[key] for r in sub_records if r[key] is not None]
        if not vals: continue
        avg = statistics.mean(vals)
        mx  = max(vals)
        bar_full  = int(avg * 20)
        bar       = "█" * bar_full + "░" * (20 - bar_full)
        contrib   = avg * w
        print(f"    {name} [{bar}] avg={avg:.3f}  max={mx:.3f}  w={w:.2f} → {contrib:.3f}")

    # Identify weakest component
    avgs = {key: statistics.mean([r[key] for r in sub_records if r[key] is not None] or [0])
            for key in keys}
    weighted = {k: avgs[k] * w for k, w in zip(keys, weights)}
    weakest = min(weighted, key=weighted.get)
    label_map = {"boundary": "boundary zone", "edge": "edge density",
                 "entropy": "color entropy", "self_sim": "self-similarity",
                 "cool": "cool-zone distribution"}
    print()
    print(f"  ⚠  Weakest component: {label_map[weakest]}  "
          f"(weighted contrib = {weighted[weakest]:.3f} / {weights[keys.index(weakest)]:.2f})")

print()
print("━" * 72)
