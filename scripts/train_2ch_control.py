#!/usr/bin/env python3
"""Fair ablation control for the complex autoencoder (see
complex_autoencoder.py / project-complex-autoencoder memory): the SAME
`{stem}_re.png`/`{stem}_im.png` data, stacked as a plain 2-channel REAL
tensor, through the project's own proven `Encoder512`/`Decoder512`
(real Conv2d/BatchNorm2d/ReLU — `autoencoder.py`, unmodified) instead of
genuinely complex layers.

This answers the actual scientific question a complex-valued architecture
needs to justify: does treating (re, im) as a true complex number (with
complex multiplication's forced rotation+scaling structure, and a
phase-preserving activation) capture something a real network with the
same information as two ordinary channels does NOT? Match latent_dim,
depth, epochs, batch size, corpus, and everything else between this
script and train_complex_autoencoder.py's runs — the only variable that
should differ is complex vs. real arithmetic.

Usage:
  python3 scripts/train_2ch_control.py --dirs explorer_out/weekend_complex_corpus/mandelbrot \
      --epochs 80 --latent-dim 128 --out 2ch_control.pt
"""
import argparse
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from autoencoder import AutoEncoder, RES_512
from train_complex_autoencoder import gather_pairs
import torch.utils.data as tud
from PIL import Image


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    print(f"PROGRESS {phase} {done} {total}", flush=True)


class TwoChannelDataset(tud.Dataset):
    """Same re/im PAIRS `ComplexPairDataset` loads, but stacked into one
    (2, H, W) real tensor instead of returned as a (re, im) tuple —
    exactly what a real Conv2d with in_ch=2 expects."""
    def __init__(self, pairs, tf):
        self.pairs = pairs
        self.tf = tf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        re_path, im_path = self.pairs[i]
        try:
            re = self.tf(Image.open(re_path).convert("L"))
            im = self.tf(Image.open(im_path).convert("L"))
            return torch.cat([re, im], dim=0)
        except Exception:
            return torch.zeros(2, RES_512, RES_512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--arch", choices=["conv", "resnet", "inception"], default="conv")
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=20000)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-images", type=int, default=50)
    ap.add_argument("--min-val", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="2ch_control.pt")
    ap.add_argument("--contact-sheet", default="2ch_control_recon.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"gathering re/im pairs from {args.dirs}…")
    pairs = gather_pairs(args.dirs, args.max_images, args.seed)
    if len(pairs) < args.min_images:
        raise SystemExit(f"only {len(pairs)} re/im pairs found across {args.dirs} — need at least {args.min_images}.")
    n_val = max(args.min_val, int(len(pairs) * args.val_fraction))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    batch_size = max(1, min(args.batch_size, len(train_pairs)))
    log(f"{len(train_pairs)} train / {len(val_pairs)} val pairs (res {RES_512}, 2ch REAL control, "
        f"latent_dim {args.latent_dim}, arch {args.arch})")

    tf = transforms.Compose([transforms.Resize((RES_512, RES_512)), transforms.ToTensor()])
    train_dl = tud.DataLoader(TwoChannelDataset(train_pairs, tf), batch_size=batch_size,
                               shuffle=True, num_workers=args.workers, drop_last=True)
    val_dl = tud.DataLoader(TwoChannelDataset(val_pairs, tf), batch_size=batch_size,
                             shuffle=False, num_workers=args.workers)

    model = AutoEncoder(latent_dim=args.latent_dim, variational=False, arch=args.arch,
                         res=RES_512, in_ch=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_dl)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, x in enumerate(train_dl):
            x = x.to(device)
            recon, _, _ = model(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            if (i + 1) % 20 == 0:
                progress(f"train_epoch{epoch}", i + 1, steps_per_epoch)
        log(f"epoch {epoch+1}/{args.epochs}  loss={running/len(train_dl):.5f}")

    model.eval()
    val_total, val_n = 0.0, 0
    with torch.no_grad():
        for x in val_dl:
            x = x.to(device)
            recon, _, _ = model(x)
            val_total += F.mse_loss(recon, x, reduction="sum").item()
            val_n += x.numel()
    val_mse = val_total / val_n
    log(f"held-out val 2ch reconstruction MSE (per-pixel): {val_mse:.6f}")
    print(f"RECON_MSE {val_mse:.6f}", flush=True)

    with torch.no_grad():
        sample = next(iter(val_dl))[:8].to(device)
        recon, _, _ = model(sample)
        re, im = sample[:, 0:1], sample[:, 1:2]
        recon_re, recon_im = recon[:, 0:1], recon[:, 1:2]
        mag = (re * re + im * im).sqrt()
        recon_mag = (recon_re * recon_re + recon_im * recon_im).sqrt().clamp(0, 1)
        rows = torch.cat([mag, recon_mag, re, recon_re, im, recon_im], dim=0)
        save_image(rows, args.contact_sheet, nrow=sample.size(0))
    log(f"wrote contact sheet -> {args.contact_sheet}")

    torch.save({
        "state_dict": model.state_dict(), "latent_dim": args.latent_dim,
        "arch": args.arch, "res": RES_512, "in_ch": 2, "variational": False,
    }, args.out)
    log(f"saved model -> {args.out}")


if __name__ == "__main__":
    main()
