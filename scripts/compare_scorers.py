#!/usr/bin/env python3
"""Compare how well each scorer separates the fractal gallery.

Reads all .nn in one or more dirs and, for every score field present, reports:
  n, min, max, mean, std, cv (=std/|mean|), and p5..p95 spread normalised to the
  observed range (spread01) — roughly "how much dynamic range this metric uses".
Also prints Spearman rank correlation of each metric vs laion_score, so you can see
which new metrics add NEW signal (low |corr|) rather than echoing the old one.

A metric that "fluctuates very little" shows up as low std / low spread01 / low cv.

Usage:
  python3 scripts/compare_scorers.py fractals_dag fractals
"""
import sys
import json
import glob
import math

FIELDS = [
    "beauty", "clip_score", "laion_score",           # existing
    "topiq_iaa", "nima", "clipiqa", "clipiqa_plus",  # new: aesthetic
    "musiq", "maniqa",                               # new: technical quality
    "ap25_score", "qalign",                          # new: SigLIP aesthetic / Q-Align
]


def spearman(pairs):
    """Spearman rho via Pearson on ranks (no scipy)."""
    n = len(pairs)
    if n < 3:
        return float("nan")
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def pct(sv, p):
    idx = min(len(sv) - 1, max(0, int(round(p / 100.0 * (len(sv) - 1)))))
    return sv[idx]


def main():
    dirs = sys.argv[1:] or ["fractals_dag"]
    genomes = []
    for d in dirs:
        for f in glob.glob(f"{d}/*.nn"):
            try:
                genomes.append(json.load(open(f)))
            except Exception:
                pass
    print(f"Loaded {len(genomes)} .nn from {', '.join(dirs)}\n")

    hdr = (f"{'metric':14} {'n':>6} {'min':>8} {'max':>8} {'mean':>8} {'std':>8} "
           f"{'cv':>6} {'spread01':>8} {'ρ(laion)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for k in FIELDS:
        vals = [float(g[k]) for g in genomes
                if isinstance(g.get(k), (int, float)) and not isinstance(g.get(k), bool)]
        if len(vals) < 2:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        vmin, vmax = min(vals), max(vals)
        sv = sorted(vals)
        spread01 = (pct(sv, 95) - pct(sv, 5)) / ((vmax - vmin) or 1.0)
        cv = std / abs(mean) if mean != 0 else float("nan")

        rho = float("nan")
        if k != "laion_score":
            pairs = [(float(g[k]), float(g["laion_score"])) for g in genomes
                     if isinstance(g.get(k), (int, float))
                     and isinstance(g.get("laion_score"), (int, float))]
            if len(pairs) >= 3:
                rho = spearman(pairs)

        print(f"{k:14} {n:6d} {vmin:8.3f} {vmax:8.3f} {mean:8.3f} {std:8.3f} "
              f"{cv:6.3f} {spread01:8.3f} {rho:9.3f}")

    print("\nRead:")
    print("  cv / spread01 LOW  → metric barely varies across fractals (the CLIP/LAION problem)")
    print("  cv / spread01 HIGH → metric separates fractals well")
    print("  |ρ(laion)| LOW     → metric carries NEW signal vs the old LAION score")


if __name__ == "__main__":
    main()
