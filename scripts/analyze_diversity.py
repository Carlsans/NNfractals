#!/usr/bin/env python3
"""
Formula diversity analysis for the 3×2h evolution loop.

Measures:
  - Basis frequency entropy (bits): Shannon H over which bases appear in archive
  - Mean pairwise formula distance (L2 in normalised 58-dim basis space)
  - Unique formula signatures (distinct top-3 basis sets)
  - formula_diversity score distribution (from .nn files)
  - Beauty/LAION trends

Usage:
  python3 scripts/analyze_diversity.py [--since EPOCH] [--dir fractals]
"""
import argparse, glob, json, math, os, statistics, time
from collections import Counter

N_BASIS = 58
BASIS_NAMES = {
    0:"z²",1:"z³",2:"z⁴",3:"z⁵",4:"z",5:"1/z",6:"1/z²",7:"c",8:"c²",9:"c³",
    10:"zc",11:"z²c",12:"zc²",13:"z²c²",14:"c/z",15:"(z+c)²",16:"(z−c)²",17:"(zc)²",
    18:"sin",19:"cos",20:"sin(π)",21:"cos(π)",22:"sin(z²)",23:"cos(z²)",24:"sin(z+c)",
    25:"cos(z+c)",26:"sin(zc)",27:"cos(zc)",28:"z·sin",29:"z·cos",30:"tan",31:"sinh",
    32:"cosh",33:"tanh",34:"exp",35:"exp(−z)",36:"exp(zc)",37:"z·exp",38:"exp·c",
    39:"log(z+1)",40:"log(z²+1)",41:"z·log",42:"sin(1/z)",43:"exp(1/z)",44:"|Re|+Im",
    45:"Re+|Im|",46:"|BS|",47:"conj",48:"conj²",49:"z|z|",50:"z/|z|",51:"z/(z²+1)",
    52:"(z²−1)/(z²+1)",53:"z²/(z−1)",54:"1/(z²+c)",55:"z²c/(z+c)",56:"1",57:"i",
}

def basis_vec(genome: dict):
    """Normalised 58-dim basis-weight vector (mirrors genome.rs formula_basis_normalized)."""
    v = [0.0] * N_BASIS
    for t in genome.get("terms", []):
        b = min(t.get("basis", 0), N_BASIS - 1)
        re, im = t.get("re", 0.0), t.get("im", 0.0)
        v[b] += math.sqrt(re*re + im*im)
    norm = math.sqrt(sum(x*x for x in v)) or 1e-9
    return [x / norm for x in v]

def top_bases(genome: dict, k=3):
    mags = {}
    for t in genome.get("terms", []):
        b = min(t.get("basis", 0), N_BASIS - 1)
        re, im = t.get("re", 0.0), t.get("im", 0.0)
        mags[b] = mags.get(b, 0.0) + math.sqrt(re*re + im*im)
    return tuple(sorted(mags, key=lambda b: -mags[b])[:k])

def pairwise_mean_dist(vecs, max_pairs=2000):
    """Mean pairwise L2 distance (sampled if too many)."""
    import random
    if len(vecs) < 2: return 0.0
    pairs = []
    n = len(vecs)
    indices = list(range(n))
    if n*(n-1)//2 > max_pairs:
        random.seed(42)
        sample = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(max_pairs*2)]
        pairs = [(i, j) for i, j in sample if i != j][:max_pairs]
    else:
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    dists = []
    for i, j in pairs:
        a, b = vecs[i], vecs[j]
        dists.append(math.sqrt(sum((x-y)**2 for x, y in zip(a, b))))
    return statistics.mean(dists) if dists else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="fractals")
    ap.add_argument("--since", type=float, default=0.0)
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.dir, "*.nn"))
    recs, since_recs = [], []
    for f in files:
        try:
            d = json.load(open(f))
            mt = os.path.getmtime(f)
            recs.append((mt, d))
            if mt >= a.since:
                since_recs.append(d)
        except Exception:
            pass

    recs.sort(key=lambda x: x[0])
    all_recs = [r for _, r in recs]
    window = since_recs if a.since > 0 else all_recs

    print(f"{'━'*70}")
    print(f"  Formula Diversity Analysis — {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'━'*70}")
    print(f"Archive total: {len(all_recs)}   Window (--since): {len(window)}")

    # ── Basis frequency entropy ────────────────────────────────────────────
    basis_counts = Counter()
    for g in window:
        used = set(min(t.get("basis",0), N_BASIS-1) for t in g.get("terms",[]))
        for b in used: basis_counts[b] += 1
    total = max(len(window), 1)
    probs = [basis_counts[b] / total for b in range(N_BASIS)]
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    H_max = math.log2(N_BASIS)
    print(f"\n── Basis frequency entropy (window) ──")
    print(f"  H = {H:.3f} bits  ({H/H_max*100:.1f}% of max {H_max:.2f} bits)")
    top10 = basis_counts.most_common(10)
    print(f"  Top 10 bases: " + ", ".join(f"{BASIS_NAMES.get(b,str(b))}={c}" for b,c in top10))

    # ── Formula diversity score distribution ────────────────────────────────
    fdivs = [g.get("formula_diversity", 0.0) for g in window if g.get("formula_diversity", 0) > 0]
    if fdivs:
        print(f"\n── formula_diversity scores (window, n={len(fdivs)}) ──")
        fdivs.sort()
        print(f"  min={fdivs[0]:.4f}  mean={statistics.mean(fdivs):.4f}  "
              f"median={statistics.median(fdivs):.4f}  max={fdivs[-1]:.4f}")

    # ── Pairwise formula distance ───────────────────────────────────────────
    vecs = [basis_vec(g) for g in window]
    mpd = pairwise_mean_dist(vecs)
    print(f"\n── Mean pairwise formula distance (window) ──")
    print(f"  {mpd:.5f}  (higher = more diverse formula families)")

    # ── Unique formula signatures ──────────────────────────────────────────
    sigs = Counter(top_bases(g, 3) for g in window)
    print(f"\n── Unique top-3 formula signatures (window) ──")
    print(f"  {len(sigs)} distinct  (out of {len(window)} genomes)")
    print("  Top 5:")
    for sig, cnt in sigs.most_common(5):
        names = "+".join(BASIS_NAMES.get(b, str(b)) for b in sig)
        print(f"    {names:<40} {cnt}")

    # ── Beauty/LAION trend ────────────────────────────────────────────────
    laions  = [g["laion_score"]  for g in window if g.get("laion_score")]
    clips   = [g["clip_score"]   for g in window if g.get("clip_score")]
    recs_sc = [g.get("fractal_recursion", 0) for g in window]
    if laions:
        print(f"\n── Quality metrics (window, n={len(laions)}) ──")
        print(f"  LAION: mean={statistics.mean(laions):.4f}  max={max(laions):.4f}")
        print(f"  CLIP:  mean={statistics.mean(clips):.4f}  max={max(clips):.4f}" if clips else "")
        print(f"  rec:   mean={statistics.mean(recs_sc):.4f}")

    # ── Recommendation ────────────────────────────────────────────────────
    print(f"\n── Diversity assessment ──")
    if H < 3.5:
        print(f"  WARNING: low basis entropy ({H:.2f} bits) — population converging on few bases")
        print(f"  RECOMMEND: increase formula_diversity_weight")
    elif H > 5.0:
        print(f"  Good diversity ({H:.2f} bits). If beauty is dropping, reduce formula_diversity_weight.")
    else:
        print(f"  Healthy diversity ({H:.2f} bits). Maintain or fine-tune.")
    if laions and statistics.mean(laions) < 5.28:
        print(f"  NOTE: LAION mean ({statistics.mean(laions):.3f}) below 5.28 — may need to reduce weight")

if __name__ == "__main__":
    main()
