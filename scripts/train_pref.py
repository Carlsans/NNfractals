#!/usr/bin/env python3
"""Train a fractal-specific aesthetic model from pairwise human ratings.

Reads ratings.jsonl (produced by the browser's ⚖ Rate mode — one
{"winner": <.nn>, "loser": <.nn>} per line), embeds each fractal's .png with a
frozen backbone (SigLIP by default, or DINOv2), and fits a linear Bradley-Terry
preference model:  score(img) = w · embed(img), trained so the winner scores
above the loser. It then applies the model to every .nn in --dirs and writes a
normalised `pref_score` field (browser column), and saves the weights.

Usage:
  python3 scripts/train_pref.py --ratings fractals_dag/ratings.jsonl \
      --dirs fractals_dag fractals [--backbone siglip|dinov2] [--epochs 400]

More ratings → a sharper model. Even ~100 comparisons give a usable ranking.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _pooled(out):
    """Extract a (B, D) embedding tensor from a HF model output (version-robust)."""
    if torch.is_tensor(out):
        return out
    p = getattr(out, "pooler_output", None)
    if p is not None:
        return p
    h = getattr(out, "last_hidden_state", None)
    if h is not None:
        return h.mean(1)
    raise RuntimeError("no embedding tensor in model output")


def load_backbone(name, device):
    from transformers import AutoModel, AutoProcessor
    repo = {"siglip": "google/siglip-base-patch16-224",
            "dinov2": "facebook/dinov2-base"}[name]
    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(device).eval()

    def embed(pils):
        inp = proc(images=pils, return_tensors="pt").to(device)
        with torch.no_grad():
            # Use the vision tower's pooled output — stable across transformers
            # versions (get_image_features return type has changed over releases).
            vision = getattr(model, "vision_model", model)
            out = vision(pixel_values=inp["pixel_values"])
        e = _pooled(out)
        return torch.nn.functional.normalize(e.float(), dim=-1)

    return embed


def png_for(nn_path):
    p = Path(nn_path)
    return p.with_suffix(".png")


def embed_paths(embed, paths, device, batch=32):
    """Return {path: np.array(D)} for paths whose png exists."""
    uniq = [p for p in dict.fromkeys(paths)]
    out = {}
    buf_paths, buf_imgs = [], []

    def flush():
        if not buf_imgs:
            return
        e = embed(buf_imgs).cpu().numpy()
        for pth, vec in zip(buf_paths, e):
            out[pth] = vec
        buf_paths.clear(); buf_imgs.clear()

    for i, p in enumerate(uniq):
        png = png_for(p)
        if not png.exists():
            continue
        try:
            buf_imgs.append(Image.open(png).convert("RGB"))
            buf_paths.append(p)
        except Exception:
            continue
        if len(buf_imgs) >= batch:
            flush()
        if (i + 1) % 500 == 0:
            log(f"  embedded {i+1}/{len(uniq)}…")
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--backbone", default="siglip", choices=["siglip", "dinov2"])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--field", default="pref_score")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction of comparisons held out to report generalization accuracy")
    ap.add_argument("--holdout-repeats", type=int, default=1,
                    help="number of random holdout splits to average (embed once, refit each)")
    ap.add_argument("--eval", action="store_true",
                    help="only train + report accuracy; do not score/write the galleries")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load comparisons ──
    comps = []
    for line in open(args.ratings):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            comps.append((d["winner"], d["loser"]))
        except Exception:
            pass
    if len(comps) < 5:
        raise SystemExit(f"only {len(comps)} comparisons — rate more fractals first.")
    log(f"{len(comps)} comparisons from {args.ratings}")

    embed = load_backbone(args.backbone, device)

    # ── Embed rated images, build training diffs ──
    rated = [p for c in comps for p in c]
    emb = embed_paths(embed, rated, device)
    diffs = []
    for w, l in comps:
        if w in emb and l in emb:
            diffs.append(emb[w] - emb[l])
    if len(diffs) < 5:
        raise SystemExit("too few usable comparisons (missing pngs?).")
    Xall = torch.tensor(np.stack(diffs), dtype=torch.float32, device=device)

    def fit(X):
        w = torch.zeros(X.shape[1], requires_grad=True, device=device)
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

    # ── Optional held-out generalization check (averaged over random splits) ──
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
        log(f"holdout {args.holdout:.0%} × {len(vals)} splits: "
            f"VAL acc {vals.mean()*100:.1f}% ± {vals.std()*100:.1f}%  "
            f"(chance 50%, {n - n_val} train / {n_val} val pairs each)")

    # ── Fit on ALL comparisons for the final model ──
    w, final_loss = fit(Xall)
    log(f"trained on {Xall.shape[0]} pairs (dim {Xall.shape[1]}); "
        f"train pairwise accuracy {acc(w, Xall)*100:.1f}%  (loss {final_loss:.4f})")

    if args.eval:
        log("eval mode — not scoring galleries.")
        return

    # ── Apply to every .nn in --dirs ──
    all_nn = []
    for d in args.dirs:
        all_nn += glob.glob(f"{d}/*.nn")
    log(f"scoring {len(all_nn)} fractals…")
    emb_all = embed_paths(embed, all_nn, device)
    w_np = w.detach().cpu().numpy()

    raw = {p: float(np.dot(emb_all[p], w_np)) for p in emb_all}
    lo, hi = 0.0, 1.0
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
        log(f"wrote {args.field} to {written} .nn files")

    # ── Save weights + normalisation for the sidecar to reuse ──
    out = Path("pref_model.npz")
    np.savez(out, w=w_np, lo=lo, hi=hi, backbone=args.backbone)
    log(f"saved model → {out}  (lo={lo:.4f} hi={hi:.4f})")


if __name__ == "__main__":
    main()
