#!/usr/bin/env python3
"""PCA / linear-projection embedding — not a network at all, the classic
cheap baseline for this kind of task. Included specifically so the AE/VAE/
ResNet/contrastive results have a trivial floor to beat: if a trained
encoder can't outperform a linear projection, that's an important, humbling
thing to know before trusting it (same "always compare against a dumb
baseline" discipline train_navigate.py already applies with its
predict-the-mean check).

Fits truncated PCA (via torch.pca_lowrank, no new dependency) on flattened,
resized, mean-centered pixels from the same fractal corpus the AE/VAE train
on. `load_pca_backbone` matches every other backbone's `embed(pils) ->
normalized tensor` interface.

Usage:
  python3 scripts/train_pca.py --latent-dim 256
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train_autoencoder import gather_paths
from autoencoder import RES


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=[
        "fractals", "fractals_dag", "fractals_1", "train_corpus", "viewer_output", "nav_train_cache",
    ])
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--max-images", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pca_model.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = gather_paths(args.dirs, args.max_images, args.seed)
    log(f"loading {len(paths)} images at {RES}x{RES}…")

    tf = transforms.Compose([transforms.Resize((RES, RES)), transforms.ToTensor()])
    vecs = []
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
            vecs.append(tf(im).flatten())
        except Exception:
            continue
        if (i + 1) % 2000 == 0:
            log(f"  loaded {i+1}/{len(paths)}…")
    X = torch.stack(vecs).to(device)
    log(f"fitting PCA: {X.shape[0]} images x {X.shape[1]} pixels -> {args.latent_dim} components…")

    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    # q a bit above latent_dim improves accuracy of the top components (torch.pca_lowrank convention).
    U, S, V = torch.pca_lowrank(Xc, q=min(args.latent_dim + 32, min(X.shape) - 1), niter=4)
    components = V[:, :args.latent_dim]  # (pixels, latent_dim)

    explained = (S[:args.latent_dim] ** 2).sum() / (S ** 2).sum()
    log(f"top {args.latent_dim} components explain {explained*100:.1f}% of variance (of the q={V.shape[1]} computed)")

    np.savez(args.out,
              mean=mean.squeeze(0).cpu().numpy(),
              components=components.cpu().numpy(),
              latent_dim=args.latent_dim, res=RES)
    log(f"saved -> {args.out}")


def load_pca_backbone(weights_path, device):
    """Mirrors every other backbone's `embed(pils) -> normalized (B, D)
    tensor` interface: flatten -> center -> project onto the saved
    principal components."""
    data = np.load(weights_path)
    mean = torch.tensor(data["mean"], dtype=torch.float32, device=device)
    components = torch.tensor(data["components"], dtype=torch.float32, device=device)
    res = int(data["res"])
    tf = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor()])

    def embed(pils):
        X = torch.stack([tf(im.convert("RGB")).flatten() for im in pils]).to(device)
        z = (X - mean) @ components
        return F.normalize(z, dim=-1)

    return embed


if __name__ == "__main__":
    main()
