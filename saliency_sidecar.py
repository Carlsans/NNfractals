#!/usr/bin/env python3
"""Saliency-heatmap inference sidecar for the fully-convolutional net in
scripts/saliency_model.py — mirrors vae_scorer_sidecar.py's exact process/
IPC shape (READY handshake, one path per line), except the response is a
whole spatial heatmap (JSON-encoded), not a single float: this model
predicts WHERE in a canvas is interesting, not how interesting one already-
cropped image is.

Protocol (stdin/stdout):
  startup  -> prints "READY\n" once the model is loaded
  request  -> one image file path per line on stdin (the wide canvas, any
              resolution — the net is fully convolutional, no fixed input
              size)
  response -> one JSON object per line: {"h": H, "w": W, "data": [H*W
              floats, row-major, raw (unscaled) log-space score — higher
              means the net predicts a higher VAE reconstruction error if
              drilled into there]} or {"error": "<msg>"} on failure

Usage: python3 saliency_sidecar.py --model-path PATH
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from saliency_model import load_saliency_model


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", type=Path, required=True)
    try:
        args = parser.parse_args()
    except SystemExit:
        print(json.dumps({"error": "usage: saliency_sidecar.py --model-path PATH"}), flush=True)
        raise
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.model_path.exists():
        print(json.dumps({"error": f"{args.model_path} not found — train one first (scripts/train_saliency.py)."}), flush=True)
        sys.exit(1)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = load_saliency_model(args.model_path, device)
            log(f"loaded SaliencyNet (base_ch={model.base_ch}) on {device}")
    except Exception as e:
        print(json.dumps({"error": f"failed to load model: {e}"}), flush=True)
        sys.exit(1)

    print("READY", flush=True)

    EMPTY_CACHE_EVERY = 100
    n_requests = 0
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            img = Image.open(path).convert("L")
            x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float() / 255.0
            x = x.view(1, 1, img.height, img.width).to(device)
            with torch.no_grad():
                heatmap = model(x)[0, 0]
            h, w = heatmap.shape
            data = heatmap.flatten().tolist()
            print(json.dumps({"h": h, "w": w, "data": data}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)
        n_requests += 1
        if device == "cuda" and n_requests % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
