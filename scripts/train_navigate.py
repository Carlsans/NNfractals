#!/usr/bin/env python3
"""
Train a model to predict where Carl would zoom next, from his own real
navigation history — see the "nav-imitation model" project memory.

Rather than hand-tuning another visual-richness heuristic (entropy/
edge_density/intricacy/aesthetic ensemble all tried this project, each with
real blind spots confirmed on real fractals), this learns directly from
Carl's actual choices: frozen backbone (SigLIP/DINOv2, `load_backbone` from
train_pref.py — same infra, imported not reimplemented) + a small trainable
head predicting `(u, v, log_zoom_ratio)` — where within the CURRENT view's
own frame the next zoom lands (u, v roughly in [-1,1], since the viewer's
own drag-zoom/zoom-in actions can only ever land inside or near the current
frame) and by what log-ratio the zoom deepens. This is the exact
`(dx, dy, zoom)` parameterization `src/explore.rs`'s `apply_offset`/
`sweep_positions` already use internally, so a trained head's output plugs
straight into that geometry with no conversion layer.

Data — two manifests, read identically here:
  nav_manifest.jsonl    live viewer navigation, rendered on demand
                        (`explorer prep-nav-data`, since only Rust renders
                        fractals in this project)
  nav_log_mined.jsonl   historical saved-image trajectories, images already
                        exist (`scripts/mine_nav_history.py`)

Usage:
  python3 scripts/train_navigate.py --holdout 0.15
  python3 scripts/train_navigate.py --model-path nav_model.npz --head-path nav_head.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_pref import load_backbone, embed_paths
from autoencoder import load_ae_backbone
from train_pca import load_pca_backbone


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    """Machine-readable progress line, same convention as train_pref.py/train_novelty.py."""
    print(f"PROGRESS {phase} {done} {total}", flush=True)


def load_manifest_records(path):
    """Yields (image_path, u, v, log_zoom_ratio, target_entropy). Handles
    both manifest shapes transparently: `nav_manifest.jsonl` is flat
    (`{"path","u","v","log_zoom_ratio",...}`); `nav_log_mined.jsonl` nests
    the image under `before.path` and the label under `label` (see
    `mine_nav_history.py`) — same information, different producer, no
    reason to force one shape before training on both. `target_entropy`
    (added 2026-08-04 by `explorer prep-nav-data`/`score-mined-targets` —
    see the nav-imitation-model project memory's Phase 9) is `None` for
    any record produced before that scoring pass existed; callers that
    filter/weight by it must treat `None` as "unscored," not zero."""
    p = Path(path)
    if not p.exists():
        log(f"  (no {path}, skipping)")
        return
    n = 0
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "label" in d:
            lab = d["label"]
            yield d["before"]["path"], lab["u"], lab["v"], lab["log_zoom_ratio"], d.get("target_entropy")
        else:
            yield d["path"], d["u"], d["v"], d["log_zoom_ratio"], d.get("target_entropy")
        n += 1
    log(f"  {n} records from {path}")


class NavHead(nn.Module):
    """Small regression head on top of a frozen backbone embedding — NOT
    `train_novelty.py`'s `ProjectionHead` (uses BatchNorm1d at
    batch_size=256, unstable at this project's current data scale — tens to
    low hundreds of examples, not thousands) and NOT `train_pref.py`'s bare
    linear projection (fit for a 1-D binary preference target, an easier
    problem than 3-D continuous regression). One small hidden layer,
    dropout for regularization, no normalization layer that needs a real
    batch to be stable."""
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 3),  # u, v, log_zoom_ratio
        )

    def forward(self, x):
        return self.net(x)


def fit(X, Y, epochs, lr, weight_decay, device, log_zoom_weight):
    head = NavHead(X.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    # u/v live roughly in [-1,1]; log_zoom_ratio's natural range is much
    # wider and varies far more across examples (a small drag-zoom vs. a
    # mined multi-step jump) — down-weight it rather than let it dominate
    # the loss purely because of scale, not because it matters more.
    w = torch.tensor([1.0, 1.0, log_zoom_weight], device=device)
    final_loss = None
    for _ in range(epochs):
        opt.zero_grad()
        pred = head(X)
        loss = (loss_fn(pred, Y) * w).mean()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    head.eval()
    return head, final_loss


def mean_abs_err(pred, target):
    with torch.no_grad():
        return (pred - target).abs().mean(dim=0).cpu().numpy()


custom_defaults = {
    "ae": "ae_model.pt", "vae": "vae_model.pt", "dae": "dae_model.pt",
    "resnet_ae": "resnet_ae_model.pt", "resnet_vae": "resnet_vae_model.pt",
    "inception_ae": "inception_ae_model.pt", "inception_vae": "inception_vae_model.pt",
    "pca": "pca_model.npz", "simclr": "simclr_model.pt", "mae": "mae_model.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="nav_manifest.jsonl")
    ap.add_argument("--mined", default="nav_log_mined.jsonl")
    ap.add_argument("--backbone", default="dinov2",
                     choices=["siglip", "dinov2", *custom_defaults.keys()])
    ap.add_argument("--ae-weights", default=None,
                     help="path to a trained checkpoint for a custom (non siglip/dinov2) "
                          "backbone (default: <backbone>_model.pt / _model.npz)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--log-zoom-weight", type=float, default=0.3,
                     help="loss weight on the log-zoom-ratio term relative to the "
                          "u/v offset terms (1.0 each) — tune against real held-out "
                          "error printed below, not asserted; same calibrate-against-"
                          "real-data discipline this project uses for every other "
                          "threshold.")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--holdout-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                     help="seeds the holdout splits so separate runs (e.g. comparing "
                          "backbones) see the EXACT same train/val partitions — without "
                          "this, split-to-split noise (real at this data volume: 40 val "
                          "examples/split) confounds any cross-run comparison.")
    ap.add_argument("--model-path", default="nav_model.npz")
    ap.add_argument("--head-path", default="nav_head.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    log("loading manifests…")
    records = list(load_manifest_records(args.manifest)) + list(load_manifest_records(args.mined))
    log(f"{len(records)} total labeled (image, u, v, log_zoom_ratio) records")
    if len(records) < 20:
        raise SystemExit(
            f"only {len(records)} records — need more navigation data first "
            f"(nav logging is always-on; just keep exploring, then re-run)."
        )

    log(f"loading backbone '{args.backbone}' on {device}…")
    ae_weights = args.ae_weights or custom_defaults.get(args.backbone, "")
    if args.backbone in ("ae", "vae", "dae", "resnet_ae", "resnet_vae", "inception_ae", "inception_vae"):
        embed = load_ae_backbone(ae_weights, device)
    elif args.backbone == "pca":
        embed = load_pca_backbone(ae_weights, device)
    elif args.backbone == "simclr":
        from train_contrastive import load_simclr_backbone
        embed = load_simclr_backbone(ae_weights, device)
    elif args.backbone == "mae":
        from train_mae import load_mae_backbone
        embed = load_mae_backbone(ae_weights, device)
    else:
        embed = load_backbone(args.backbone, device)

    paths = [r[0] for r in records]
    emb = embed_paths(embed, paths, device, on_progress=lambda d, t: progress("embed", d, t))

    X, Y = [], []
    for p, u, v, lz, _target_entropy in records:
        if p in emb:
            X.append(emb[p])
            Y.append([u, v, lz])
    log(f"{len(X)}/{len(records)} records had a usable embedding (image found + loaded)")
    if len(X) < 20:
        raise SystemExit(f"only {len(X)} usable records after embedding — too few to train.")

    Xall = torch.tensor(np.stack(X), dtype=torch.float32, device=device)
    Yall = torch.tensor(np.array(Y, dtype=np.float32), device=device)
    in_dim = Xall.shape[1]
    n = Xall.shape[0]

    if args.holdout > 0.0:
        n_val = max(1, int(n * args.holdout))
        model_errs, baseline_errs = [], []
        # Dedicated generator for the split draw, seeded independently of
        # the global RNG `fit()`'s head-init/dropout consume from — those
        # draw a DIFFERENT number of randoms depending on `in_dim` (768 for
        # DINOv2/SigLIP, 256 here for ae/vae), which would desync the
        # global stream and silently give repeat 2+ a different split per
        # backbone even with the same --seed. This keeps every repeat's
        # split identical across backbones — the whole point of --seed is
        # a valid cross-backbone comparison, not just run-to-run repeatability.
        split_gen = torch.Generator().manual_seed(args.seed)
        for _ in range(max(1, args.holdout_repeats)):
            perm = torch.randperm(n, generator=split_gen)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            head_tr, _ = fit(Xall[tr_idx], Yall[tr_idx], args.epochs, args.lr,
                              args.weight_decay, device, args.log_zoom_weight)
            model_errs.append(mean_abs_err(head_tr(Xall[val_idx]), Yall[val_idx]))
            # Baseline: predict the TRAINING mean for every val example, no
            # image information at all — if the model isn't meaningfully
            # better than this, it hasn't learned anything from the images,
            # just the dataset's average navigation action.
            mean_y = Yall[tr_idx].mean(dim=0, keepdim=True)
            baseline_errs.append(mean_abs_err(mean_y.expand(n_val, -1), Yall[val_idx]))
        model_errs = np.array(model_errs)
        baseline_errs = np.array(baseline_errs)
        log(f"holdout {args.holdout:.0%} x {len(model_errs)} splits "
            f"({n - n_val} train / {n_val} val each):")
        log(f"  model    mean abs err: u={model_errs[:,0].mean():.3f} "
            f"v={model_errs[:,1].mean():.3f} log_zoom={model_errs[:,2].mean():.3f}")
        log(f"  baseline mean abs err: u={baseline_errs[:,0].mean():.3f} "
            f"v={baseline_errs[:,1].mean():.3f} log_zoom={baseline_errs[:,2].mean():.3f}  "
            f"(predict-the-training-mean, zero image information)")

    head, final_loss = fit(Xall, Yall, args.epochs, args.lr, args.weight_decay, device, args.log_zoom_weight)
    log(f"trained on {n} records (embedding dim {in_dim}); final train loss {final_loss:.4f}")

    torch.save(head.state_dict(), args.head_path)
    with open(args.model_path, "wb") as f:
        np.savez(f, backbone=args.backbone, ae_weights=ae_weights if args.backbone in custom_defaults else "",
                 in_dim=in_dim, hidden=args.hidden_dim, out_dim=3)
    log(f"saved head -> {args.head_path}, model meta -> {args.model_path}")


if __name__ == "__main__":
    main()
