#!/usr/bin/env python3
"""SimCLR-style contrastive encoder for fractal images — the other common
paradigm for label-free representation learning, and a genuinely different
objective than autoencoder.py's reconstruction loss: no decoder at all,
just an encoder trained so two randomly-augmented views of the SAME image
land close together in embedding space while different images land far
apart (NT-Xent / InfoNCE loss). Reconstruction optimizes for pixel
fidelity (has to spend capacity on exact color/texture); this optimizes
for "is this the same underlying structure," which is plausibly closer to
what train_navigate.py's downstream task actually needs.

Reuses autoencoder.py's plain conv `Encoder` as the trunk (barebone,
consistent with the AE/VAE) with `variational=False` — its raw output
(before any projection head) is what gets used downstream, per the
original SimCLR paper's own recommendation (the projection head is a
training-time-only scaffold, discarded afterward).

Augmentations are deliberately fractal-informed, not generic ImageNet
defaults: horizontal flip is a REAL symmetry for Mandelbrot-family
fractals (conjugate mirror across the real axis — the same fact
`explore.rs::dihedral_variants` already exploits for fingerprinting), so
it's a meaningful invariance to teach, not just standard practice.

Usage:
  python3 scripts/train_contrastive.py
"""
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from autoencoder import Encoder, RES
from train_autoencoder import gather_paths


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    print(f"PROGRESS {phase} {done} {total}", flush=True)


def aug_pipeline():
    return transforms.Compose([
        transforms.RandomResizedCrop(RES, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),  # real Mandelbrot-family symmetry, not a generic default
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
    ])


class PairDataset(Dataset):
    """Returns two INDEPENDENTLY-augmented views of the same image —
    that's the entire supervisory signal SimCLR needs."""
    def __init__(self, paths, tf):
        self.paths = paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB")
            return self.tf(im), self.tf(im)
        except Exception:
            z = torch.zeros(3, RES, RES)
            return z, z


class ProjectionHead(nn.Module):
    """Training-time-only scaffold (standard SimCLR practice) — discarded
    after training; `load_simclr_backbone` only ever loads the encoder."""
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        return self.net(x)


def nt_xent(z1, z2, temperature):
    """Standard NT-Xent (normalized temperature-scaled cross-entropy):
    each of the 2B views' positive is its OTHER augmented view of the same
    image; every other view in the batch (2B-2 of them) is a negative."""
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=[
        "fractals", "fractals_dag", "fractals_1", "train_corpus", "viewer_output", "nav_train_cache",
    ])
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-images", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="simclr_model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"gathering image corpus from {args.dirs}…")
    paths = gather_paths(args.dirs, args.max_images, args.seed)
    if len(paths) < 200:
        raise SystemExit(f"only {len(paths)} images found across {args.dirs} — need far more to train an encoder.")
    log(f"{len(paths)} images (res {RES}, latent_dim {args.latent_dim}, proj_dim {args.proj_dim}, temp {args.temperature})")

    dl = DataLoader(PairDataset(paths, aug_pipeline()), batch_size=args.batch_size, shuffle=True,
                     num_workers=args.workers, drop_last=True)

    encoder = Encoder(args.latent_dim, variational=False).to(device)
    proj = ProjectionHead(args.latent_dim, args.proj_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr)

    steps_per_epoch = len(dl)
    for epoch in range(args.epochs):
        encoder.train(); proj.train()
        running = 0.0
        for i, (x1, x2) in enumerate(dl):
            x1, x2 = x1.to(device), x2.to(device)
            z1 = proj(encoder(x1))
            z2 = proj(encoder(x2))
            loss = nt_xent(z1, z2, args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            if (i + 1) % 20 == 0:
                progress(f"train_epoch{epoch}", i + 1, steps_per_epoch)
        log(f"epoch {epoch+1}/{args.epochs}  nt_xent_loss={running/len(dl):.4f}")

    torch.save({"encoder_state_dict": encoder.state_dict(), "latent_dim": args.latent_dim, "res": RES}, args.out)
    log(f"saved encoder -> {args.out}")


def load_simclr_backbone(weights_path, device):
    """Mirrors every other backbone's `embed(pils) -> normalized (B, D)
    tensor` interface — loads ONLY the encoder (the projection head was a
    training-time scaffold, per standard SimCLR practice)."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = Encoder(ckpt["latent_dim"], variational=False).to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.eval()

    tf = transforms.Compose([transforms.Resize((ckpt["res"], ckpt["res"])), transforms.ToTensor()])

    def embed(pils):
        batch = torch.stack([tf(im.convert("RGB")) for im in pils]).to(device)
        with torch.no_grad():
            e = model(batch)
        return F.normalize(e.float(), dim=-1)

    return embed


if __name__ == "__main__":
    main()
