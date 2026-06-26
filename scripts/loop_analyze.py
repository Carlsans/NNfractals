#!/usr/bin/env python3
"""
Per-loop evolution analysis for the 48h autonomous run.

Summarises the saved fractal pool so each 6h review is data-driven:
 - score distributions (laion, clip, beauty, self_replication)
 - how self_replication relates to beauty (are the prettiest also self-similar?)
 - which basis functions dominate the top performers
 - growth since a given timestamp (--since EPOCH)

Usage:
  python3 scripts/loop_analyze.py [--since EPOCH] [--dir fractals] [--top N]
"""
import argparse, glob, json, os, statistics, time
from collections import Counter

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

def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs.sort()
    return {
        "n": len(xs), "min": xs[0], "max": xs[-1],
        "mean": statistics.mean(xs), "median": statistics.median(xs),
        "p90": xs[int(0.9*(len(xs)-1))],
    }

def fmt(s):
    if not s: return "  (no data)"
    return f"n={s['n']:<4} min={s['min']:.3f} mean={s['mean']:.3f} median={s['median']:.3f} p90={s['p90']:.3f} max={s['max']:.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="fractals")
    ap.add_argument("--since", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.dir, "*.nn"))
    recs = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        mt = os.path.getmtime(f)
        recs.append({
            "file": os.path.basename(f), "mtime": mt,
            "beauty": d.get("beauty", 0.0),
            "clip": d.get("clip_score", 0.0) or 0.0,
            "laion": d.get("laion_score", 0.0) or 0.0,
            "selfrep": d.get("self_replication", 0.0) or 0.0,
            "terms": d.get("terms", []),
        })

    print("━"*74)
    print(f"  NNfractals Evolution Analysis — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("━"*74)
    print(f"Total saved genomes: {len(recs)}")
    if a.since > 0:
        new = [r for r in recs if r["mtime"] >= a.since]
        print(f"New since {time.strftime('%Y-%m-%d %H:%M', time.localtime(a.since))}: {len(new)}")

    print("\n── Score distributions (whole pool) ──")
    print(f"  laion          {fmt(stats([r['laion'] for r in recs]))}")
    print(f"  clip           {fmt(stats([r['clip'] for r in recs]))}")
    print(f"  beauty         {fmt(stats([r['beauty'] for r in recs]))}")
    print(f"  self_replication {fmt(stats([r['selfrep'] for r in recs]))}")

    # Relationship: do high-self-replication genomes also score well?
    scored = [r for r in recs if r["selfrep"] > 0 and r["laion"] > 0]
    if len(scored) >= 4:
        sr = [r["selfrep"] for r in scored]
        la = [r["laion"] for r in scored]
        try:
            corr = statistics.correlation(sr, la)
            print(f"\n  corr(self_replication, laion) = {corr:+.3f}  over {len(scored)} genomes")
        except Exception:
            pass

    # Top by a combined rank (laion/10 + 0.35*selfrep), the seeding objective.
    for r in recs:
        r["rank"] = (r["laion"]/10.0 if r["laion"] else r["beauty"]) + 0.35*r["selfrep"]
    recs.sort(key=lambda r: -r["rank"])

    print(f"\n── Top {a.top} by seed rank (laion/10 + 0.35·self_rep) ──")
    print(f"  {'rank':>5} {'laion':>5} {'clip':>5} {'selfR':>5}  formula")
    for r in recs[:a.top]:
        ops = "+".join(BASIS_NAMES.get(t.get("basis",-1), "?") for t in r["terms"][:6])
        print(f"  {r['rank']:.3f} {r['laion']:>5.2f} {r['clip']:>5.3f} {r['selfrep']:>5.3f}  {ops}")

    # Basis frequency among the top quartile.
    q = max(1, len(recs)//4)
    bc = Counter()
    for r in recs[:q]:
        for t in r["terms"]:
            bc[t.get("basis", -1)] += 1
    print(f"\n── Basis frequency in top quartile ({q} genomes) ──")
    for b, c in bc.most_common(12):
        print(f"  {BASIS_NAMES.get(b,'?'):>14}: {c}")

if __name__ == "__main__":
    main()
