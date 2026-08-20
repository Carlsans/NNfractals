#!/usr/bin/env python3
"""Bounding-box-aware alternative to train_navigate.py's embedding+MLP
pipeline — Carl's specific request after seeing that u/v prediction got
ZERO signal (never beat the predict-the-mean baseline) across 16
completely different embedding-based backbones (PCA, DINOv2, plain/
residual/inception conv AE/VAE, denoising, contrastive — see the
nav-imitation-model project memory's Phase 6). The one thing all 16 share:
every one of them GLOBAL-POOLS the image down to a single vector before
any (u, v) prediction happens. That pooling step is a prime suspect for
throwing away exactly the spatial-location information (u, v) — a soft-
argmax over a spatial feature map keeps that information intact.

Architecture: a trunk (any autoencoder.py AutoEncoder — conv/resnet/
inception, AE or VAE, doesn't matter which) run up to its LAST spatial
feature map (B, 256, 8, 8) via `encode_spatial()` — no flatten, no Linear-
to-latent. `SpatialHead` puts a 1x1 conv "interest heatmap" over that grid,
softmaxes it into an attention distribution, and reads off `(u, v)` as the
EXPECTED grid coordinate under that distribution (soft-argmax) — i.e. "where
would Carl zoom" is answered by directly attending over space, the same
shape of answer a bounding-box/detection head gives, not by hoping an MLP
can recover a location from a pooled summary. `log_zoom` is read off the
same attended location (a second 1x1 conv, same soft-argmax weighting) — a
scalar that plausibly still depends on what's THERE, not just a global
summary.

Trunk defaults to FROZEN (matches this project's established pattern:
every other backbone here is frozen + a small trainable head) — spatial
features are precomputed ONCE, same efficiency trick train_navigate.py's
embed_paths uses. `--finetune-backbone` unfreezes it for genuine end-to-end
training, at real overfitting risk given only ~273 labeled examples — off
by default for exactly that reason.

Usage:
  python3 scripts/train_navigate_spatial.py --trunk-weights vae_model.pt
  python3 scripts/train_navigate_spatial.py --trunk-weights resnet_vae_model.pt --finetune-backbone
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from autoencoder import Encoder, RES, load_ae_model
from train_navigate import load_manifest_records


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    print(f"PROGRESS {phase} {done} {total}", flush=True)


class SpatialHead(nn.Module):
    """`(u, v)` = soft-argmax of a learned interest heatmap over the
    trunk's (grid x grid) feature map — the bounding-box-style answer to
    "where," not a regression from a pooled vector. `log_zoom` is the same
    attention weights applied to a second per-cell scalar map.

    `temperature` divides the heatmap logits before softmax (< 1.0
    sharpens). Added 2026-08-04 after catching the shipped model's
    attention had collapsed to near-uniform (max cell weight ~0.019 vs
    ~0.016 for a flat distribution, across every one of 40 sampled real
    images, each landing within std=0.004 of the SAME near-(0,0)
    prediction regardless of what was in the image) — i.e. it wasn't
    "choosing" a low-entropy zone so much as barely attending anywhere and
    defaulting near the grid's own center. A plain softmax's logits only
    need to be LARGE to peak, and `--weight-decay` (applied to every
    parameter by default, `heat`/`zoom` included) directly fights growing
    those logits — the "small weights" solution IS the collapsed-uniform
    one. `temperature < 1` sharpens the same logits without relying on
    their magnitude alone; `main()` also excludes the head from weight
    decay entirely (see its optimizer param groups) for the same reason
    from the other direction."""
    def __init__(self, in_ch, grid, temperature=1.0):
        super().__init__()
        self.grid = grid
        self.temperature = temperature
        self.heat = nn.Conv2d(in_ch, 1, 1)
        self.zoom = nn.Conv2d(in_ch, 1, 1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, grid), torch.linspace(-1, 1, grid), indexing="ij")
        self.register_buffer("gx", xs.reshape(1, -1))
        self.register_buffer("gy", ys.reshape(1, -1))

    def forward(self, feat):
        b = feat.shape[0]
        w = F.softmax(self.heat(feat).reshape(b, -1) / self.temperature, dim=-1)  # attention over the grid
        u = (w * self.gx).sum(dim=-1, keepdim=True)
        v = (w * self.gy).sum(dim=-1, keepdim=True)
        log_zoom = (w * self.zoom(feat).reshape(b, -1)).sum(dim=-1, keepdim=True)
        return torch.cat([u, v, log_zoom], dim=-1), w.reshape(b, self.grid, self.grid)


def load_spatial_model(path, device):
    """Loads a combined checkpoint saved by this script's `main()` — trunk
    (with whatever weights it actually ended training with, fine-tuned or
    not — NEVER the original --trunk-weights file, which `--finetune-
    backbone` runs have since diverged from) + `SpatialHead`. Used by
    `main()`'s own training-time eval AND by nav_predict_sidecar.py at
    inference time — one loader, two callers, no drift between them."""
    from autoencoder import AutoEncoder
    ckpt = torch.load(path, map_location=device, weights_only=False)
    trunk = AutoEncoder(latent_dim=ckpt["trunk_latent_dim"], variational=ckpt["trunk_variational"],
                         arch=ckpt["trunk_arch"]).to(device)
    trunk.load_state_dict(ckpt["trunk_state_dict"])
    trunk.eval()
    head = SpatialHead(ckpt["feat_ch"], ckpt["feat_grid"], temperature=ckpt.get("temperature", 1.0)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return trunk, head


def load_images(paths, device):
    """Loads+resizes every path ONCE into one (N, 3, RES, RES) tensor —
    same reason train_navigate.py's embed_paths caches embeddings once:
    this project's whole labeled set is small enough (hundreds of images)
    that "just keep it all in memory" beats a DataLoader's bookkeeping."""
    tf = transforms.Compose([transforms.Resize((RES, RES)), transforms.ToTensor()])
    imgs, kept = [], []
    for p in paths:
        try:
            imgs.append(tf(Image.open(p).convert("RGB")))
            kept.append(p)
        except Exception:
            continue
    return torch.stack(imgs).to(device), kept


def fit(feats, Y, epochs, lr, weight_decay, device, log_zoom_weight, in_ch, grid,
        trunk=None, finetune=False, temperature=1.0):
    """`feats` is EITHER precomputed (N, C, grid, grid) spatial features
    (frozen-trunk path) OR raw (N, 3, RES, RES) images (finetune path, in
    which case `trunk` must be given and gradients flow through it too).

    ONLY `head.heat` (the softmax logits) gets weight_decay=0 — see
    `SpatialHead`'s docstring for why: weight decay directly fights those
    logits growing large enough to peak, which is exactly what stops the
    attention collapsing to near-uniform. `head.zoom` goes through no
    softmax (its output is just per-cell-averaged directly) — nothing
    about IT structurally needs that exemption, and an earlier version of
    this function exempted the whole head, which measurably hurt log_zoom
    specifically (removing regularization it did benefit from) without
    helping u/v any further than exempting `heat` alone already does. The
    (optional, finetuned) trunk keeps the requested `weight_decay` too —
    the collapse risk is specific to a raw 1x1-conv-into-softmax, not
    general."""
    head = SpatialHead(in_ch, grid, temperature=temperature).to(device)
    param_groups = [
        {"params": head.heat.parameters(), "weight_decay": 0.0},
        {"params": head.zoom.parameters(), "weight_decay": weight_decay},
    ]
    if finetune:
        param_groups.append({"params": trunk.parameters(), "weight_decay": weight_decay})
    opt = torch.optim.Adam(param_groups, lr=lr)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    w = torch.tensor([1.0, 1.0, log_zoom_weight], device=device)
    final_loss = None
    for _ in range(epochs):
        opt.zero_grad()
        f = trunk.encode_spatial(feats) if finetune else feats
        pred, _ = head(f)
        loss = (loss_fn(pred, Y) * w).mean()
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    head.eval()
    return head, final_loss


def predict(head, feats, trunk, finetune):
    with torch.no_grad():
        f = trunk.encode_spatial(feats) if finetune else feats
        pred, _ = head(f)
    return pred


def mean_abs_err(pred, target):
    with torch.no_grad():
        return (pred - target).abs().mean(dim=0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="nav_manifest.jsonl")
    ap.add_argument("--mined", default="nav_log_mined.jsonl")
    ap.add_argument("--trunk-weights", required=True,
                     help="any autoencoder.py checkpoint (ae/vae/dae/resnet/inception "
                          ".pt from train_autoencoder.py) to supply spatial features")
    ap.add_argument("--finetune-backbone", action="store_true",
                     help="unfreeze the trunk and train it jointly with the head — real "
                          "overfitting risk at ~273 labels, off by default")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--log-zoom-weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=1.0,
                     help="softmax temperature on the attention heatmap; <1.0 sharpens. "
                          "See SpatialHead's docstring — the shipped v1 head's attention had "
                          "collapsed to near-uniform (effectively always predicting near the "
                          "grid center), which is the direct mechanism behind Carl's "
                          "low-entropy-zone report, not just noisy training targets.")
    ap.add_argument("--min-target-entropy", type=float, default=0.0,
                     help="drop training records whose TARGET (after-view) png_compression_"
                          "entropy is below this — teaches the model fewer 'zooming into "
                          "blank space is fine' examples. Records with no target_entropy "
                          "(pre-2026-08-04 manifests, or mined records whose genome couldn't "
                          "be resolved) are always KEPT, never dropped by this filter — "
                          "'unscored' isn't evidence of 'bad.'")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--holdout-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="nav_spatial_model.pt",
                     help="ONE self-contained checkpoint: trunk arch/state (as actually "
                          "trained here — including fine-tuned weights, if any) + head "
                          "state. Deliberately not split into two files: with "
                          "--finetune-backbone the trunk's weights diverge from whatever "
                          "was loaded via --trunk-weights, so a caller must not mix this "
                          "head with the original trunk checkpoint.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    log("loading manifests…")
    records = list(load_manifest_records(args.manifest)) + list(load_manifest_records(args.mined))
    log(f"{len(records)} total labeled (image, u, v, log_zoom_ratio) records")
    if len(records) < 20:
        raise SystemExit(f"only {len(records)} records — need more navigation data first.")

    if args.min_target_entropy > 0.0:
        n_before = len(records)
        n_unscored = sum(1 for r in records if r[4] is None)
        records = [r for r in records if r[4] is None or r[4] >= args.min_target_entropy]
        log(f"--min-target-entropy {args.min_target_entropy}: kept {len(records)}/{n_before} "
            f"records ({n_unscored} unscored records always kept)")
        if len(records) < 20:
            raise SystemExit(f"only {len(records)} records survive the entropy filter — lower --min-target-entropy.")

    log(f"loading trunk '{args.trunk_weights}' on {device}…")
    trunk = load_ae_model(args.trunk_weights, device)
    if not args.finetune_backbone:
        for p in trunk.parameters():
            p.requires_grad_(False)

    paths = [r[0] for r in records]
    imgs, kept = load_images(paths, device)
    kept_set = {p: i for i, p in enumerate(kept)}
    Y_list, idx = [], []
    for p, u, v, lz, _target_entropy in records:
        if p in kept_set:
            Y_list.append([u, v, lz]); idx.append(kept_set[p])
    log(f"{len(idx)}/{len(records)} records had a loadable image")
    if len(idx) < 20:
        raise SystemExit(f"only {len(idx)} usable records — too few to train.")

    imgs = imgs[idx]
    Yall = torch.tensor(np.array(Y_list, dtype=np.float32), device=device)
    n = imgs.shape[0]
    in_ch, grid = Encoder.FEAT_CH, Encoder.FEAT_GRID

    if args.finetune_backbone:
        feats_or_imgs = imgs
        log("finetune-backbone: trunk trains jointly with the head (spatial features recomputed every step)")
    else:
        with torch.no_grad():
            feats_or_imgs = trunk.encode_spatial(imgs)
        log(f"frozen trunk: precomputed {tuple(feats_or_imgs.shape)} spatial features once")

    if args.holdout > 0.0:
        n_val = max(1, int(n * args.holdout))
        model_errs, baseline_errs = [], []
        split_gen = torch.Generator().manual_seed(args.seed)
        for _ in range(max(1, args.holdout_repeats)):
            perm = torch.randperm(n, generator=split_gen)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            head_tr, _ = fit(feats_or_imgs[tr_idx], Yall[tr_idx], args.epochs, args.lr,
                              args.weight_decay, device, args.log_zoom_weight, in_ch, grid,
                              trunk=trunk, finetune=args.finetune_backbone, temperature=args.temperature)
            pred = predict(head_tr, feats_or_imgs[val_idx], trunk, args.finetune_backbone)
            model_errs.append(mean_abs_err(pred, Yall[val_idx]))
            mean_y = Yall[tr_idx].mean(dim=0, keepdim=True)
            baseline_errs.append(mean_abs_err(mean_y.expand(n_val, -1), Yall[val_idx]))
        model_errs = np.array(model_errs)
        baseline_errs = np.array(baseline_errs)
        log(f"holdout {args.holdout:.0%} x {len(model_errs)} splits ({n - n_val} train / {n_val} val each):")
        log(f"  model    mean abs err: u={model_errs[:,0].mean():.3f} "
            f"v={model_errs[:,1].mean():.3f} log_zoom={model_errs[:,2].mean():.3f}")
        log(f"  baseline mean abs err: u={baseline_errs[:,0].mean():.3f} "
            f"v={baseline_errs[:,1].mean():.3f} log_zoom={baseline_errs[:,2].mean():.3f}  "
            f"(predict-the-training-mean, zero image information)")

    head, final_loss = fit(feats_or_imgs, Yall, args.epochs, args.lr, args.weight_decay,
                            device, args.log_zoom_weight, in_ch, grid,
                            trunk=trunk, finetune=args.finetune_backbone, temperature=args.temperature)
    log(f"trained on {n} records; final train loss {final_loss:.4f}")

    # Attention-collapse sanity check — printed every run, not just when
    # someone remembers to check by hand. Uniform-over-64-cells is
    # 1/64=0.0156; the v1 shipped model averaged ~0.019 here (effectively
    # uniform) and it showed up as a real usage complaint (Carl: Auto-
    # Select often landing on low-entropy zones) before anyone measured
    # it. A healthy head should look meaningfully peaked, not just barely
    # above the flat floor.
    with torch.no_grad():
        f = trunk.encode_spatial(imgs)
        _, attn = head(f)
        max_w = attn.reshape(attn.shape[0], -1).max(dim=1).values
        log(f"attention concentration: mean max-cell-weight={max_w.mean():.4f} "
            f"(uniform floor={1.0/(grid*grid):.4f}, 1.0=one-hot) — "
            f"{'LOOKS COLLAPSED (near-uniform)' if max_w.mean() < 3.0/(grid*grid) else 'looks peaked'}")

    torch.save({
        "trunk_state_dict": trunk.state_dict(),
        "trunk_latent_dim": trunk.latent_dim,
        "trunk_variational": trunk.variational,
        "trunk_arch": trunk.arch,
        "head_state_dict": head.state_dict(),
        "feat_ch": in_ch,
        "feat_grid": grid,
        "temperature": args.temperature,
        "finetuned": args.finetune_backbone,
        "min_target_entropy": args.min_target_entropy,
    }, args.out)
    log(f"saved combined trunk+head checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
