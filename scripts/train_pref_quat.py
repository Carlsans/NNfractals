#!/usr/bin/env python3
"""Train a QUATERNION fractal preference model from pairwise human ratings —
answers Carl's actual question ("what combination of metrics produce good
fractals") directly, by fitting a linear Bradley-Terry model over the ~30
`quat_*` metric fields already computed per genome, instead of an opaque
image embedding (which is what scripts/train_pref.py, the 2D sibling this
mirrors, uses).

Because the features are just numbers already sitting in each .nn file —
no image loading, no vision backbone — this needs nothing but numpy/torch
for the linear algebra. The payoff over the 2D script's approach: the
learned weight on EACH metric is directly readable as "how much this axis
matters, and in which direction" — that's the actual deliverable, not just
a black-box scorer.

Ratings come from the browser's existing "⚖ Rate" pairwise-comparison mode
(browser.rs) — when the loaded folder is a quaternion archive
(`fractals_dag_quat/` or any path containing that name), ratings are
logged to train_corpus_quat/ratings.jsonl instead of the 2D pipeline's
train_corpus/ratings.jsonl, so the two domains' comparisons never mix
(quat genomes don't even have the 2D `beauty*`/`aesthetic_ensemble` fields
populated, and vice versa — training on a mix would be meaningless noise).

Usage:
  python3 scripts/train_pref_quat.py --ratings train_corpus_quat/ratings.jsonl \
      --dirs fractals_dag_quat [--epochs 400] [--holdout 0.2]

  # Apply a trained model to score/rank a directory (writes quat_pref_score,
  # a new sortable gallery column, to every .nn found):
  python3 scripts/train_pref_quat.py --score-only --dirs fractals_dag_quat
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Fixed, documented feature order — every `quat_*` field on Genome as of
# the metrics work this script exists to evaluate (src/genome.rs,
# src/quat_dag_fitness.rs). Order only matters for reading the printed
# weights back against a name; training itself doesn't care.
QUAT_METRIC_FIELDS = [
    "quat_anisotropy", "quat_coverage", "quat_solidity", "quat_shading_richness",
    "quat_color_entropy", "quat_silhouette_irregularity",
    "quat_box_dim", "quat_lacunarity", "quat_convexity", "quat_isoperimetric",
    "quat_bilateral_symmetry", "quat_centroid_offset", "quat_largest_component_frac",
    "quat_shading_gradient", "quat_shading_skewness", "quat_specular_fraction", "quat_crevice_fraction",
    "quat_color_gradient", "quat_color_shading_corr", "quat_color_band_autocorr", "quat_color_range_utilization",
    "quat_cross_view_iou", "quat_cross_view_coverage_delta", "quat_c_sensitivity", "quat_c_coverage_range",
    "quat_node_count", "quat_opcode_diversity", "quat_max_depth",
    "quat_warp_node_count", "quat_warp_opcode_diversity",
]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_metrics(nn_path: str):
    """Returns a (len(QUAT_METRIC_FIELDS),) float32 vector, or None if the
    file is missing/unreadable/has no quat_* fields at all (a 2D genome,
    or a quat genome saved before the metrics existed and never
    rescored — see `quat-dag-rescore`)."""
    try:
        g = json.loads(Path(nn_path).read_text())
    except Exception:
        return None
    if not any(k in g for k in QUAT_METRIC_FIELDS):
        return None
    return np.array([float(g.get(k, 0.0)) for k in QUAT_METRIC_FIELDS], dtype=np.float32)


def score_only(args):
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"no {model_path} — train the model first (drop --score-only).")
    data = np.load(model_path)
    w = np.asarray(data["w"], dtype=np.float32)
    lo, hi = float(data["lo"]), float(data["hi"])
    rng = (hi - lo) or 1.0
    fields = list(data["fields"]) if "fields" in data else QUAT_METRIC_FIELDS
    if fields != QUAT_METRIC_FIELDS:
        log("WARNING: saved model's feature order doesn't match this script's current QUAT_METRIC_FIELDS — scoring anyway, but the model may be stale relative to the metrics code.")

    all_nn = []
    for d in args.dirs:
        all_nn += glob.glob(f"{d}/*.nn")
    log(f"scoring {len(all_nn)} genomes with saved model ({model_path})…")
    written, skipped = 0, 0
    for p in all_nn:
        vec = load_metrics(p)
        if vec is None:
            skipped += 1
            continue
        r = float(np.dot(vec, w))
        score = float(np.clip((r - lo) / rng, 0.0, 1.0))
        try:
            g = json.loads(Path(p).read_text())
            g[args.field] = round(score, 5)
            Path(p).write_text(json.dumps(g, indent=2))
            written += 1
        except Exception:
            skipped += 1
    log(f"wrote {args.field} to {written} .nn files ({skipped} skipped — no quat_* metrics)")
    print(f"DONE wrote {args.field} to {written} file(s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="train_corpus_quat/ratings.jsonl")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--score-only", action="store_true",
                     help="skip training; load the saved model and (re)score every .nn in --dirs")
    ap.add_argument("--model", default="pref_model_quat.npz")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--field", default="quat_pref_score")
    ap.add_argument("--holdout", type=float, default=0.0,
                     help="fraction of comparisons held out to report generalization accuracy")
    ap.add_argument("--holdout-repeats", type=int, default=5,
                     help="number of random holdout splits to average")
    args = ap.parse_args()

    if args.score_only:
        score_only(args)
        return

    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        raise SystemExit(f"no {ratings_path} — rate some quaternion fractals in the browser's ⚖ Rate mode first (open fractals_dag_quat/ so ratings go to train_corpus_quat/, not the 2D pipeline's corpus).")

    comps = []
    for line in ratings_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            comps.append((d["winner"], d["loser"]))
        except Exception:
            pass
    if len(comps) < 5:
        raise SystemExit(f"only {len(comps)} comparisons in {ratings_path} — rate more first (even ~30-50 gives a usable first read for this few-dimensional a model).")
    log(f"{len(comps)} comparisons from {ratings_path}")

    diffs = []
    missing = 0
    for w_path, l_path in comps:
        wv, lv = load_metrics(w_path), load_metrics(l_path)
        if wv is None or lv is None:
            missing += 1
            continue
        diffs.append(wv - lv)
    if missing:
        log(f"  ({missing} comparisons skipped — missing .nn or no quat_* metrics; rescore with quat-dag-rescore if these predate the metrics)")
    if len(diffs) < 5:
        raise SystemExit("too few usable comparisons after filtering — check the rated genomes actually have quat_* fields (quat-dag-rescore backfills them).")

    Xall = torch.tensor(np.stack(diffs), dtype=torch.float32)
    n_features = Xall.shape[1]
    log(f"{len(diffs)} usable comparisons, {n_features} metric features")

    def fit(X):
        w = torch.zeros(X.shape[1], requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        loss = None
        for _ in range(args.epochs):
            opt.zero_grad()
            loss = torch.nn.functional.softplus(-(X @ w)).mean() + args.reg * (w * w).sum()
            loss.backward()
            opt.step()
        return w.detach(), float(loss.item())

    def acc(w, X):
        with torch.no_grad():
            return (X @ w > 0).float().mean().item()

    if args.holdout > 0.0:
        n = Xall.shape[0]
        n_val = max(1, int(n * args.holdout))
        vals = []
        for _ in range(max(1, args.holdout_repeats)):
            perm = torch.randperm(n)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            w_tr, _ = fit(Xall[tr_idx])
            vals.append(acc(w_tr, Xall[val_idx]))
        vals = np.array(vals)
        log(f"holdout {args.holdout:.0%} × {len(vals)} splits: VAL acc {vals.mean()*100:.1f}% ± {vals.std()*100:.1f}%  (chance 50%, {n - n_val} train / {n_val} val pairs each)")

    w, final_loss = fit(Xall)
    train_acc = acc(w, Xall)
    log(f"trained on {Xall.shape[0]} pairs (dim {n_features}); train pairwise accuracy {train_acc*100:.1f}%  (loss {final_loss:.4f})")

    # ── The actual deliverable: which metrics matter, and which way ──
    w_np = w.cpu().numpy()
    order = np.argsort(-np.abs(w_np))
    log("")
    log("Metric weights, ranked by |weight| (positive = higher value → more preferred):")
    for i in order:
        bar_len = int(min(abs(w_np[i]), 3.0) / 3.0 * 30)
        bar = ("+" if w_np[i] >= 0 else "-") * bar_len
        log(f"  {QUAT_METRIC_FIELDS[i]:32s} {w_np[i]:+7.3f}  {bar}")
    log("")

    # ── Score every genome in --dirs, write quat_pref_score ──
    all_nn = []
    for d in args.dirs:
        all_nn += glob.glob(f"{d}/*.nn")
    raw = {}
    for p in all_nn:
        vec = load_metrics(p)
        if vec is not None:
            raw[p] = float(np.dot(vec, w_np))
    if raw:
        vals = np.array(list(raw.values()))
        lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        rng = (hi - lo) or 1.0
        written = 0
        for p, r in raw.items():
            score = float(np.clip((r - lo) / rng, 0.0, 1.0))
            try:
                g = json.loads(Path(p).read_text())
                g[args.field] = round(score, 5)
                Path(p).write_text(json.dumps(g, indent=2))
                written += 1
            except Exception:
                pass
        log(f"wrote {args.field} to {written} .nn files across {args.dirs}")
        print(f"DONE {written}", flush=True)

        out = Path(args.model)
        np.savez(out, w=w_np, lo=lo, hi=hi, fields=np.array(QUAT_METRIC_FIELDS))
        # Plain-JSON twin of the same 4 arrays, next to the .npz — lets
        # quat-dag-evolve apply this model natively in Rust (a 30-float dot
        # product + clip) with no Python round-trip, so evolution can select
        # on Carl's own trained preference instead of the generic
        # NIMA/TOPIQ/AP25 ensemble (Carl: "discontinue the aesthetic scorer
        # unless it is one I trained myself"). Kept in lockstep with the
        # .npz on every save rather than requiring a separate export step.
        out.with_suffix(".json").write_text(json.dumps({
            "fields": QUAT_METRIC_FIELDS, "w": [float(x) for x in w_np], "lo": lo, "hi": hi,
        }, indent=2))
        log(f"saved model → {out}  (lo={lo:.4f} hi={hi:.4f})")


if __name__ == "__main__":
    main()
