#!/usr/bin/env python3
"""Trains SaliencyNet (saliency_model.py) on a dataset built by `explorer
saliency-data` (src/bin/explorer.rs): JSONL rows of
{canvas, px, py, label, pool, zone_stem, is_manual_mark}, where `label` is
a real VAE reconstruction-error (recon_mse) — either already measured by a
prior vae-explore run (`is_manual_mark=False`), or freshly measured for a
zone Carl marked by hand in the viewer (`is_manual_mark=True` — see
`explorer_out/saliency_manual_marks/`). (px, py) is that zone's normalized
position within the freshly-rendered `canvas` image.

Sparse-point regression: for each example the model's full spatial heatmap
is bilinearly sampled (F.grid_sample) at just the one labeled (px, py),
and only that sampled value is compared to the label — the network never
sees a dense per-pixel target, which is why the dataset randomizes each
zone's position within its canvas (see cmd_saliency_data's doc comment in
explorer.rs) rather than always centering it: across many examples, every
relative position gets covered by SOME label, which is what lets the net
learn real spatial discrimination instead of shortcutting to "always
predict the center."

Manual marks are oversampled to a target fraction of total training mass
(`--mark-target-fraction`, default 50% — "half and half," Carl's
follow-up request) — added 2026-08-10 after Carl reported "the behavior
didn't change much" from marking zones by hand.
Root cause: with plain uniform sampling, a handful of marks against
thousands of pool examples (e.g. 5 vs 7375 — 0.07%) are statistically
invisible to SGD regardless of how "correct" their signal is. Marks are
also NEVER held out for validation (all go into train) — there are
usually too few for a meaningful validation split, and the whole point is
maximizing their influence on the trained weights, not evaluating on them.

Usage:
  python3 scripts/train_saliency.py --data explorer_out/saliency_dataset --epochs 30
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from saliency_model import SaliencyNet, save_saliency_model

# recon_mse spans ~5 orders of magnitude in the real corpus (1.7e-6 to
# 0.24) — a plain MSE on raw values would be dominated entirely by the
# handful of largest examples. log-space regression compresses that into
# a well-conditioned range; EPS floors it well below the smallest observed
# real value so log() never sees exactly zero.
LABEL_EPS = 1e-6


def log_label(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x + LABEL_EPS)


def random_dihedral(x: torch.Tensor, px: float, py: float) -> tuple[torch.Tensor, float, float]:
    """Applies a uniformly random one of the 8 square symmetries (D4) to
    `x` (C,H,W) and the matching (px,py) label point — escape-time fields
    have no canonical orientation, so this is a free, safe way to multiply
    effective training diversity. Added alongside 50% mark oversampling
    (2026-08-10, Carl's "half and half" request): at that target fraction,
    152 real marks against ~7375 pool examples means each mark canvas gets
    repeated roughly 45x per epoch — without augmentation the model would
    see the literal SAME 152 images that often and could start memorizing
    them instead of learning a generalizable correction. The 3 independent
    coin flips (transpose, then hflip, then vflip) cover the group exactly
    once each, composed in that order so the coordinate updates stay
    simple (transpose swaps which axis px/py refer to; the two flips
    complement whichever axis they now refer to)."""
    if random.random() < 0.5:
        x = x.transpose(-2, -1)
        px, py = py, px
    if random.random() < 0.5:
        x = x.flip(-1)
        px = 1.0 - px
    if random.random() < 0.5:
        x = x.flip(-2)
        py = 1.0 - py
    return x, px, py


class SaliencyDataset(Dataset):
    def __init__(self, data_dir: Path, rows: list[dict], augment: bool = False):
        self.data_dir = data_dir
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(self.data_dir / row["canvas"]).convert("L")
        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float() / 255.0
        x = x.view(1, img.height, img.width)
        px, py = float(row["px"]), float(row["py"])
        if self.augment:
            x, px, py = random_dihedral(x, px, py)
        return x, px, py, float(row["label"])


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    px = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    py = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    labels = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return xs, px, py, labels


def oversample_factor(n_marks: int, n_pool: int, target_fraction: float, cap: int) -> int:
    """How many times to repeat each mark so `n_marks * factor` reaches
    `target_fraction` of `n_pool + n_marks * factor` — solving
    `factor = target_fraction * n_pool / (n_marks * (1 - target_fraction))`
    directly rather than searching for it. Capped so a handful of marks
    (or even just one) can't blow up an epoch's size unboundedly."""
    if n_marks == 0:
        return 0
    target_total = target_fraction * n_pool / max(1e-6, 1 - target_fraction)
    factor = max(1, round(target_total / n_marks))
    return min(factor, cap)


def sample_heatmap(heatmap: torch.Tensor, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Bilinear-samples `heatmap` (B,1,Hh,Wh) at each example's own (px,py)
    in [0,1] canvas-fraction coordinates. `F.grid_sample` wants [-1,1] with
    (x,y) order and align_corners=False matching how a resize/crop would
    sample a continuous image — px/py -> 2*p-1 is the direct mapping."""
    b = heatmap.shape[0]
    grid = torch.stack([px * 2 - 1, py * 2 - 1], dim=-1).view(b, 1, 1, 2)
    sampled = F.grid_sample(heatmap, grid, mode="bilinear", align_corners=False)
    return sampled.view(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("explorer_out/saliency_dataset"))
    ap.add_argument("--out", type=Path, default=Path("explorer_out/saliency_dataset/saliency_model.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--mark-target-fraction", type=float, default=0.5,
                     help="Manually-marked zones are repeated (oversampled) until they reach "
                          "roughly this fraction of the TRAINING set's mass, regardless of how "
                          "few of them exist — plain uniform sampling makes a handful of marks "
                          "against thousands of pool examples statistically invisible. 0.5 "
                          "('half and half', Carl's request 2026-08-10) means the model sees a "
                          "mark as often as it sees the entire rest of the corpus combined — "
                          "see augment_canvas's doc comment for how the resulting heavy per-mark "
                          "repetition (152 marks at 0.5 vs. ~7375 pool examples is ~45x each) is "
                          "kept from just memorizing those exact images.")
    ap.add_argument("--mark-oversample-cap", type=int, default=300,
                     help="Upper bound on how many times any single mark gets repeated (guards "
                          "against a huge factor when there's only 1-2 marks total).")
    args = ap.parse_args()

    dataset_jsonl = args.data / "saliency_dataset.jsonl"
    rows = []
    with open(dataset_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        print(f"no rows in {dataset_jsonl} — run `explorer saliency-data` first.", file=sys.stderr)
        sys.exit(1)

    mark_rows = [r for r in rows if r.get("is_manual_mark", False)]
    pool_rows = [r for r in rows if not r.get("is_manual_mark", False)]

    random.seed(0)
    random.shuffle(pool_rows)
    n_val = max(1, int(len(pool_rows) * args.val_fraction))
    val_rows, train_rows = pool_rows[:n_val], pool_rows[n_val:]

    if mark_rows:
        factor = oversample_factor(len(mark_rows), len(pool_rows), args.mark_target_fraction, args.mark_oversample_cap)
        train_rows = train_rows + mark_rows * factor
        random.shuffle(train_rows)
        mark_share = (len(mark_rows) * factor) / len(train_rows)
        print(f"{len(mark_rows)} manual mark(s) found — oversampled {factor}x "
              f"({len(mark_rows) * factor} rows, ~{mark_share:.0%} of training mass); "
              f"never held out for validation", flush=True)
    else:
        print("no manual marks in this dataset (explorer_out/saliency_manual_marks/ was empty or absent)", flush=True)

    print(f"{len(train_rows)} train / {len(val_rows)} val examples from {dataset_jsonl}")

    train_ds = SaliencyDataset(args.data, train_rows, augment=True)
    val_ds = SaliencyDataset(args.data, val_rows, augment=False)  # unaugmented so val_loss stays comparable run-to-run
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SaliencyNet(base_ch=args.base_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, px, py, label in train_dl:
            x, px, py, label = x.to(device), px.to(device), py.to(device), label.to(device)
            heatmap = model(x)
            pred = sample_heatmap(heatmap, px, py)
            loss = F.mse_loss(pred, log_label(label))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, px, py, label in val_dl:
                x, px, py, label = x.to(device), px.to(device), py.to(device), label.to(device)
                heatmap = model(x)
                pred = sample_heatmap(heatmap, px, py)
                val_loss += F.mse_loss(pred, log_label(label)).item() * x.size(0)
        val_loss /= len(val_ds)

        print(f"PROGRESS epoch={epoch+1}/{args.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            save_saliency_model(model, args.out)
            print(f"  saved (best val so far) -> {args.out}", flush=True)

    print(f"done. best val_loss={best_val:.5f}, checkpoint at {args.out}")


if __name__ == "__main__":
    main()
