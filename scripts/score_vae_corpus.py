#!/usr/bin/env python3
"""Batch VAE reconstruction-error scorer — one JSONL manifest line per
image. `src/bin/explorer.rs::cmd_vae_explore` spawns this as a blocking
subprocess after each retrain, then parses the manifest itself (not this
script's stdout) to compute the authoritative per-iteration mean error.

Mirrors `train_novelty.py`'s `--score-only` mode in SPIRIT (load a model
once, iterate a corpus, write results) but is its own small script since
this feature's manifest schema is specific to it and `train_autoencoder.py`
is a shared, general-purpose script other pipelines already depend on
unmodified.

Usage:
  python3 scripts/score_vae_corpus.py --dirs explorer_out/mandelbrot_vae \
      --model-path explorer_out/mandelbrot_vae/vae_model.pt \
      --out explorer_out/mandelbrot_vae/vae_recon_manifest.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from autoencoder import load_ae_model
from train_autoencoder import gather_paths, ImageFolderFlat


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    print(f"PROGRESS {phase} {done} {total}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading model '{args.model_path}' on {device}…")
    model = load_ae_model(args.model_path, device)
    log(f"model: arch={model.arch} res={model.res} in_ch={model.in_ch} variational={model.variational}")

    paths = gather_paths(args.dirs, max_images=-1, seed=0)
    # Only score the actual VAE training input (raw escape-time crops), not
    # the human-browsable colormapped preview that lives alongside it —
    # vae_explore.rs saves BOTH `zone_NNNN_raw.png` (the real input) and
    # `zone_NNNN.png` (preview) per zone.
    paths = [p for p in paths if p.endswith("_raw.png")]
    log(f"{len(paths)} raw-field images found across {args.dirs}")
    if not paths:
        raise SystemExit("no *_raw.png images found — nothing to score.")

    tf = transforms.Compose([transforms.Resize((model.res, model.res)), transforms.ToTensor()])
    ds = ImageFolderFlat(paths, tf, res=model.res, channels=model.in_ch)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    results = []
    i = 0
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            recon = model.reconstruct_deterministic(batch)
            per_image_mse = F.mse_loss(recon, batch, reduction="none").flatten(1).mean(dim=1)
            for mse in per_image_mse.tolist():
                p = paths[i]
                stem = Path(p).stem.removesuffix("_raw")
                results.append({"stem": stem, "path": p, "recon_mse": mse})
                i += 1
            if i % 200 < args.batch_size:
                progress("score", i, len(paths))

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    mean = sum(r["recon_mse"] for r in results) / len(results)
    log(f"scored {len(results)} images -> {args.out}")
    print(f"RECON_MSE {mean:.6f}", flush=True)
    print(f"DONE {len(results)}", flush=True)


if __name__ == "__main__":
    main()
