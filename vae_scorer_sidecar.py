#!/usr/bin/env python3
"""VAE reconstruction-error scorer sidecar for vae_explore.

A per-formula VAE (trained from scratch on raw escape-time fields — see
scripts/autoencoder.py/train_autoencoder.py) scores a candidate zone by how
well it reconstructs it — high error means the VAE hasn't learned this
region well yet (novel/undermodeled), which is the signal
src/vae_explore.rs's recursive drill chases by default (--select-by
max-error). Mirrors novelty_scorer.py's/nav_predict_sidecar.py's exact
process/IPC shape (same READY-handshake, one-path-per-line protocol).

Protocol (stdin/stdout):
  startup  -> prints "READY\n" once the model is loaded
  request  -> one image file path per line on stdin
  response -> "<recon_mse>\n" or "ERROR: <msg>\n" on failure

Usage: python3 vae_scorer_sidecar.py --model-path PATH
  No shared default — the VAE is per-formula by design (unlike
  novelty_scorer.py, which has both a shared production model and
  per-pool ones). The checkpoint changes every vae-explore outer
  iteration; src/vae_score.rs's caller drops and respawns this sidecar
  after each retrain rather than expecting a hot-reload.
"""
import argparse
import contextlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from autoencoder import load_ae_model


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", type=Path, required=True)
    try:
        args = parser.parse_args()
    except SystemExit:
        print("ERROR: usage: vae_scorer_sidecar.py --model-path PATH", flush=True)
        raise
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.model_path.exists():
        print(f"ERROR: {args.model_path} not found — train one first (scripts/train_autoencoder.py).", flush=True)
        sys.exit(1)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = load_ae_model(args.model_path, device)
            log(f"loaded VAE (arch={model.arch} res={model.res} in_ch={model.in_ch}) on {device}")
    except Exception as e:
        print(f"ERROR: failed to load model: {e}", flush=True)
        sys.exit(1)

    mode = "L" if model.in_ch == 1 else "RGB"
    tf = transforms.Compose([transforms.Resize((model.res, model.res)), transforms.ToTensor()])
    print("READY", flush=True)

    EMPTY_CACHE_EVERY = 100
    n_requests = 0
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            img = Image.open(path).convert(mode)
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                recon = model.reconstruct_deterministic(x)
                mse = F.mse_loss(recon, x, reduction="mean").item()
            print(f"{mse:.6f}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        n_requests += 1
        if device == "cuda" and n_requests % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
