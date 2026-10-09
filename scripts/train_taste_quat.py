#!/usr/bin/env python3
"""Train a quaternion-fractal taste model from Carl's pairwise ratings.

Port of scripts/train_pref.py for the quaternion GA, with one structural
change: each genome's feature is pooled over a MULTI-VIEW render set (3
orbit angles x 3 C values, `<stem>_view_00..08.png`, from `explorer
quat-dag-thumbs --views`) instead of a single C=0 still — the "4D" input
Carl asked for, since a single still only shows one cross-section of a
quaternion object. Falls back to the single `<stem>.png` when a genome's
view set is missing (e.g. role-model stubs that predate this render).

Reads ratings.jsonl (browser ⚖ Rate mode, routed to train_corpus_quat/ for
quat genomes — see is_quat_genome_path in browser.rs) plus optional
role-model bootstrap pairs (weighted below human ratings) and starred
genomes (weighted extra positives). Fits a linear Bradley-Terry model:
score(genome) = w . pooled_embed(genome), trained so the winner scores
above the loser. Writes `quat_taste` (browser-sortable column) and saves
the weights as both an .npz (for the sidecar) and a .json twin (for the
Rust side to sanity-check backbone/pooling/n_views at startup without
needing a numpy reader).

Usage:
  python3 scripts/train_taste_quat.py --ratings train_corpus_quat/ratings.jsonl \
      --dirs train_corpus_quat [--pooling mean|mean_max] [--holdout 0.2 --holdout-repeats 25]
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pref import load_backbone, png_for, progress, log  # noqa: E402
from taste_pooling import N_VIEWS, pool_grid  # noqa: E402


def view_paths_for(nn_path):
    """The N_VIEWS (angles x C values) `_view_NN.png` paths for one genome's .nn path, in order."""
    stem = Path(nn_path).with_suffix("")
    return [Path(f"{stem}_view_{i:02}.png") for i in range(N_VIEWS)]


def embed_genomes(embed, nn_paths, device, pooling, batch=32):
    """Returns {nn_path: np.array(D)} (D = 768, or 1536 for mean_max),
    pooling each genome's view set (falling back to its single .png when
    the view set is incomplete). `pooling="grid"` = 6 angles x 4 C values pooled to [mean,max,C-std] (3*D); see taste_pooling.py. `pooling="single"` skips the view set
    entirely and always uses the single C=0 .png — the production default:
    Phase 1b's gate found multi-view pooling UNDERPERFORMS single-view on
    held-out accuracy (87.7% vs 89.4%), most likely because Carl's ratings
    were made looking at exactly that one still, and averaging in 8 views
    he never rated dilutes rather than sharpens the signal. The view
    render path (`render_genome_views`) and mean/mean_max pooling are kept
    for the archive's optional role-model-distance axis (Phase 2c), not
    for the taste score itself."""
    # Collect every distinct image path we'll need, embed them all in one
    # batched pass, then pool per genome — avoids re-embedding shared
    # images and keeps the batching logic identical to train_pref.py's.
    img_paths = []
    per_genome_imgs = {}
    for p in dict.fromkeys(nn_paths):
        if pooling == "single":
            single = png_for(p)
            if single.exists():
                per_genome_imgs[p] = [single]
            else:
                continue
        elif pooling == "grid":
            # Grid pooling needs the complete angle x C set (fixed feature
            # layout); genomes without it are skipped, never padded.
            views = view_paths_for(p)
            if all(v.exists() for v in views):
                per_genome_imgs[p] = views
            else:
                continue
        else:
            views = view_paths_for(p)
            if all(v.exists() for v in views):
                per_genome_imgs[p] = views
            else:
                single = png_for(p)
                if single.exists():
                    per_genome_imgs[p] = [single]
                else:
                    continue
        img_paths.extend(per_genome_imgs[p])
    img_paths = list(dict.fromkeys(img_paths))

    total = len(img_paths)
    vecs = {}
    buf_paths, buf_imgs = [], []

    def flush():
        if not buf_imgs:
            return
        e = embed(buf_imgs).cpu().numpy()
        for pth, vec in zip(buf_paths, e):
            vecs[pth] = vec
        buf_paths.clear()
        buf_imgs.clear()

    for i, ip in enumerate(img_paths):
        try:
            buf_imgs.append(Image.open(ip).convert("RGB"))
            buf_paths.append(ip)
        except Exception:
            continue
        if len(buf_imgs) >= batch:
            flush()
        if (i + 1) % 200 == 0:
            log(f"  embedded {i+1}/{total}…")
    flush()

    out = {}
    for p, imgs in per_genome_imgs.items():
        img_vecs = [vecs[im] for im in imgs if im in vecs]
        if not img_vecs:
            continue
        arr = np.stack(img_vecs)
        if pooling == "grid":
            if len(img_vecs) != N_VIEWS:
                continue
            out[p] = pool_grid(arr)
            continue
        mean = arr.mean(axis=0)
        if pooling == "mean_max" and len(img_vecs) > 1:
            mx = arr.max(axis=0)
            out[p] = np.concatenate([mean, mx])
        elif pooling == "mean_max":
            # Single-view fallback: max == mean, so concat is well-defined.
            out[p] = np.concatenate([mean, mean])
        else:
            out[p] = mean
    return out


ROT3D_WEIGHT = 2.0


def load_comparisons(path, weight=1.0):
    comps = []
    if not path:
        return comps
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            # Ratings made against the live rotating view (browser logs
            # "view":"rot3d") saw what grid pooling sees; the older ones were
            # judged from one still, so they get less trust.
            w = float(d.get("weight", weight)) * (ROT3D_WEIGHT if d.get("view") == "rot3d" else 1.0)
            comps.append((d["winner"], d["loser"], w))
        except Exception:
            pass
    return comps


def score_only(args, device):
    model_path = Path("taste_model_quat.npz")
    if not model_path.exists():
        raise SystemExit("no taste_model_quat.npz — train the taste model first.")
    data = np.load(model_path)
    w_np = np.asarray(data["w"], dtype=np.float32)
    lo = float(data["lo"]) if "lo" in data else 0.0
    hi = float(data["hi"]) if "hi" in data else 1.0
    rng = (hi - lo) or 1.0
    backbone = str(data["backbone"]) if "backbone" in data else args.backbone
    pooling = str(data["pooling"]) if "pooling" in data else args.pooling

    log(f"loading backbone '{backbone}' on {device}…")
    embed = load_backbone(backbone, device)

    all_nn = []
    for d in args.dirs:
        all_nn += glob.glob(f"{d}/*.nn")
    log(f"scoring {len(all_nn)} fractals with saved model (pooling={pooling})…")
    emb_all = embed_genomes(embed, all_nn, device, pooling)

    written = 0
    items = list(emb_all.items())
    n = len(items)
    for k, (p, vec) in enumerate(items):
        r = float(np.dot(vec, w_np))
        score = float(np.clip((r - lo) / rng, 0.0, 1.0))
        try:
            g = json.loads(Path(p).read_text())
            g["quat_taste"] = round(score, 5)
            Path(p).write_text(json.dumps(g, indent=2))
            written += 1
        except Exception:
            pass
        if k % 200 == 0 or k + 1 == n:
            progress("write", k + 1, n)
    log(f"wrote quat_taste to {written} .nn files")
    print(f"DONE {written}", flush=True)


def fit(X, weights, epochs, reg, device):
    w = torch.zeros(X.shape[1], requires_grad=True, device=device)
    opt = torch.optim.Adam([w], lr=0.05)
    loss = None
    for _ in range(epochs):
        opt.zero_grad()
        per_pair = torch.nn.functional.softplus(-(X @ w))
        loss = (per_pair * weights).sum() / weights.sum() + reg * (w * w).sum()
        loss.backward()
        opt.step()
    return w.detach(), float(loss.item())


def acc(w, X):
    with torch.no_grad():
        return (X @ w > 0).float().mean().item()


def build_diffs(comps, emb):
    diffs, weights = [], []
    for w_path, l_path, wt in comps:
        if w_path in emb and l_path in emb:
            diffs.append(emb[w_path] - emb[l_path])
            weights.append(wt)
    return diffs, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--role-model-pairs", help="optional role_model_pairs.jsonl (Phase 1c)")
    ap.add_argument("--role-model-weight", type=float, default=0.3)
    ap.add_argument("--starred", help="optional dir of starred/favorite genomes -> extra positives (Phase 3)")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--backbone", default="siglip", choices=["siglip", "dinov2"])
    ap.add_argument("--pooling", default="single", choices=["single", "mean", "mean_max", "grid"])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.2,
                     help="fraction held out to report generalization accuracy; 0 to skip "
                          "(defaults ON so a production run always has a real number to "
                          "show — see taste_model_quat.json / the launcher GUI's stats panel)")
    ap.add_argument("--holdout-repeats", type=int, default=15)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--human-pairs-only-check", action="store_true",
                     help="also report held-out accuracy on human pairs alone, "
                          "even when role-model pairs are included in training (Phase 1c guard)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.score_only:
        score_only(args, device)
        return

    if not args.ratings:
        raise SystemExit("--ratings is required unless --score-only is given.")

    human_comps = load_comparisons(args.ratings, weight=1.0)
    if len(human_comps) < 5:
        raise SystemExit(f"only {len(human_comps)} comparisons — rate more fractals first.")
    log(f"{len(human_comps)} human comparisons from {args.ratings}")

    role_comps = load_comparisons(args.role_model_pairs, weight=args.role_model_weight) if args.role_model_pairs else []
    if role_comps:
        log(f"{len(role_comps)} role-model bootstrap pairs (weight {args.role_model_weight})")

    starred_comps = []
    if args.starred:
        starred_nn = glob.glob(f"{args.starred}/*.nn")
        pool_nn = [p for d in args.dirs for p in glob.glob(f"{d}/*.nn")]
        import random
        rng = random.Random(0)
        for s in starred_nn:
            for l in rng.sample(pool_nn, min(3, len(pool_nn))):
                starred_comps.append((s, l, 0.5))
        if starred_comps:
            log(f"{len(starred_comps)} starred-genome bootstrap pairs (weight 0.5)")

    all_comps = human_comps + role_comps + starred_comps

    embed = load_backbone(args.backbone, device)

    rated = [p for c in all_comps for p in (c[0], c[1])]

    def run_for_pooling(pooling):
        emb = embed_genomes(embed, rated, device, pooling)
        diffs, weights = build_diffs(all_comps, emb)
        if len(diffs) < 5:
            raise SystemExit("too few usable comparisons (missing pngs/views?).")
        Xall = torch.tensor(np.stack(diffs), dtype=torch.float32, device=device)
        Wall = torch.tensor(np.array(weights, dtype=np.float32), device=device)

        human_diffs, human_weights = build_diffs(human_comps, emb)
        Xhuman = torch.tensor(np.stack(human_diffs), dtype=torch.float32, device=device) if human_diffs else None

        # Reported back to the caller (not just logged) so the production
        # path can write these into taste_model_quat.json and the launcher
        # GUI's stats panel — see project-taste-driven-quat-ga memory for
        # why Carl wanted this visible without scrolling a log.
        holdout_stats = {"all_mean": None, "all_std": None, "human_mean": None, "human_std": None}

        if args.holdout > 0.0:
            n = Xall.shape[0]
            n_val = max(1, int(n * args.holdout))
            vals = []
            for _ in range(max(1, args.holdout_repeats)):
                perm = torch.randperm(n)
                val_idx, tr_idx = perm[:n_val], perm[n_val:]
                w_tr, _ = fit(Xall[tr_idx], Wall[tr_idx], args.epochs, args.reg, device)
                vals.append(acc(w_tr, Xall[val_idx]))
            vals = np.array(vals)
            holdout_stats["all_mean"] = float(vals.mean() * 100)
            holdout_stats["all_std"] = float(vals.std() * 100)
            log(f"[{pooling}] holdout {args.holdout:.0%} x {len(vals)} splits (all pairs): "
                f"VAL acc {vals.mean()*100:.1f}% +/- {vals.std()*100:.1f}%  "
                f"(chance 50%, {n - n_val} train / {n_val} val pairs each)")

            if role_comps and Xhuman is not None:
                # Human-pairs-only held-out check: does mixing in role-model
                # pairs help or hurt generalization on Carl's OWN ratings
                # specifically? (Phase 1c guard.)
                nh = Xhuman.shape[0]
                nh_val = max(1, int(nh * args.holdout))
                hvals = []
                for _ in range(max(1, args.holdout_repeats)):
                    perm = torch.randperm(nh)
                    val_idx = perm[:nh_val]
                    # Train on everything EXCEPT the held-out human pairs.
                    val_set = set(val_idx.tolist())
                    tr_mask = torch.ones(Xall.shape[0], dtype=torch.bool)
                    # Held-out human diffs are a subset of Xall's first len(human_diffs) rows
                    # (build order: human, role, starred) — mask those indices out.
                    tr_mask[torch.tensor(sorted(val_set))] = False
                    w_tr, _ = fit(Xall[tr_mask], Wall[tr_mask], args.epochs, args.reg, device)
                    hvals.append(acc(w_tr, Xhuman[val_idx]))
                hvals = np.array(hvals)
                holdout_stats["human_mean"] = float(hvals.mean() * 100)
                holdout_stats["human_std"] = float(hvals.std() * 100)
                log(f"[{pooling}] holdout on HUMAN PAIRS ONLY (trained w/ role-model pairs mixed in): "
                    f"VAL acc {hvals.mean()*100:.1f}% +/- {hvals.std()*100:.1f}%")
            elif Xhuman is not None:
                # No role-model pairs mixed in this run -> the all-pairs
                # holdout number already IS the human-only number.
                holdout_stats["human_mean"] = holdout_stats["all_mean"]
                holdout_stats["human_std"] = holdout_stats["all_std"]

        w, final_loss = fit(Xall, Wall, args.epochs, args.reg, device)
        log(f"[{pooling}] trained on {Xall.shape[0]} pairs (dim {Xall.shape[1]}); "
            f"train pairwise accuracy {acc(w, Xall)*100:.1f}%  (loss {final_loss:.4f})")
        return w, emb, Xall.shape[1], holdout_stats

    if args.eval:
        # Report all three poolings for the gate comparison. Phase 1b's
        # run found single beats both multi-view variants (89.4% vs
        # 87.7%/87.7%) — single is the production default; the others stay
        # available for the role-model-distance axis and for re-checking
        # the gate if the corpus composition changes a lot.
        run_for_pooling("single")
        run_for_pooling("mean")
        run_for_pooling("mean_max")
        run_for_pooling("grid")
        log("eval mode — not scoring galleries.")
        return

    w, emb, dim, holdout_stats = run_for_pooling(args.pooling)

    all_nn = []
    for d in args.dirs:
        all_nn += glob.glob(f"{d}/*.nn")
    log(f"scoring {len(all_nn)} fractals…")
    emb_all = embed_genomes(embed, all_nn, device, args.pooling)
    w_np = w.detach().cpu().numpy()

    raw = {p: float(np.dot(emb_all[p], w_np)) for p in emb_all}
    lo, hi = 0.0, 1.0
    written = 0
    if raw:
        vals = np.array(list(raw.values()))
        lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        rng = (hi - lo) or 1.0
        items = list(raw.items())
        n = len(items)
        for k, (p, r) in enumerate(items):
            score = float(np.clip((r - lo) / rng, 0.0, 1.0))
            try:
                g = json.loads(Path(p).read_text())
                g["quat_taste"] = round(score, 5)
                Path(p).write_text(json.dumps(g, indent=2))
                written += 1
            except Exception:
                pass
            if k % 200 == 0 or k + 1 == n:
                progress("write", k + 1, n)
        log(f"wrote quat_taste to {written} .nn files")

    out = Path("taste_model_quat.npz")
    np.savez(out, w=w_np, lo=lo, hi=hi, backbone=args.backbone, pooling=args.pooling, n_views=N_VIEWS)
    # The JSON twin carries everything the launcher GUI's stats panel
    # needs to show "current model" without spawning Python or reading the
    # .npz — held-out accuracy, corpus composition, and when this model
    # was trained. Written every production run (not --eval), which is
    # also the reason the CLI defaults --holdout/--holdout-repeats ON now
    # (see argparse defaults above) rather than requiring them to be
    # remembered on every invocation.
    model_json = {
        "backbone": args.backbone, "pooling": args.pooling, "n_views": N_VIEWS,
        "dim": dim, "lo": lo, "hi": hi,
        "n_human_pairs": len(human_comps),
        "n_role_pairs": len(role_comps),
        "n_starred_pairs": len(starred_comps),
        "holdout_all_acc": holdout_stats["all_mean"],
        "holdout_all_std": holdout_stats["all_std"],
        "holdout_human_acc": holdout_stats["human_mean"],
        "holdout_human_std": holdout_stats["human_std"],
        "trained_at": int(time.time()),
    }
    Path("taste_model_quat.json").write_text(json.dumps(model_json, indent=2))
    log(f"saved model -> {out} (+ .json twin)  (lo={lo:.4f} hi={hi:.4f})")

    # One rich summary line — this is what the launcher GUI shows directly
    # under the progress bar (JobMsg::Summary), so it needs to carry the
    # number Carl actually cares about (held-out accuracy), not just a
    # file count.
    if holdout_stats["human_mean"] is not None:
        acc_str = f"held-out {holdout_stats['human_mean']:.1f}%+/-{holdout_stats['human_std']:.1f}% on {len(human_comps)} human pairs"
    else:
        acc_str = f"{len(human_comps)} human pairs (no holdout run)"
    print(f"DONE {written} genomes scored | {acc_str}", flush=True)


if __name__ == "__main__":
    main()
