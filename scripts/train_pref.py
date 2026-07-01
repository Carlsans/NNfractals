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
    X = torch.tensor(np.stack(diffs), dtype=torch.float32, device=device)
    log(f"training on {X.shape[0]} pairs, dim={X.shape[1]}")

    # ── Fit Bradley-Terry (logistic on embedding differences) ──
    w = torch.zeros(X.shape[1], requires_grad=True, device=device)
    opt = torch.optim.Adam([w], lr=0.05)
    for ep in range(args.epochs):
        opt.zero_grad()
        logits = X @ w                       # want > 0 (winner beats loser)
        loss = torch.nn.functional.softplus(-logits).mean() + args.reg * (w * w).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (X @ w > 0).float().mean().item()
    log(f"train pairwise accuracy: {acc*100:.1f}%  (final loss {loss.item():.4f})")

    # ── Apply to every .nn in --dirs ──
    all_nn = []
    for d in args.dirs:
        all_nn += glob.glob(f"{d}/*.nn")
    log(f"scoring {len(all_nn)} fractals…")
    emb_all = embed_paths(embed, all_nn, device)
    w_np = w.detach().cpu().numpy()

    raw = {p: float(np.dot(emb_all[p], w_np)) for p in emb_all}
    if raw:
        vals = np.array(list(raw.values()))
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
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

    # ── Save weights for reuse ──
    out = Path("pref_model.npz")
    np.savez(out, w=w_np, backbone=args.backbone)
    log(f"saved model → {out}")


if __name__ == "__main__":
    main()
