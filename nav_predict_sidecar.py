#!/usr/bin/env python3
"""Navigation-imitation predictor sidecar for NNfractals.

Bounding-box-style prediction (see scripts/train_navigate_spatial.py and
the nav-imitation-model project memory's Phase 7): a conv trunk's spatial
feature map feeds a soft-argmax "interest heatmap" head that reads off
`(u, v, log_zoom_ratio)` directly as an attended grid location — where
within the CURRENT view's frame Carl would zoom next (u, v roughly in
[-1,1], always exactly in [-1,1] here since a softmax-weighted average of
grid points can't leave the grid's own span) and by what log-ratio the
zoom deepens. Same `(dx, dy, zoom)` parameterization `src/explore.rs`'s
`apply_offset`/`sweep_positions` already use internally (see
`src/nav_predict.rs`, the Rust-side caller, for the conversion).

Promoted to production 2026-08-04: across 19 backbone/architecture
combinations tried, this was the only one to beat a predict-the-mean
baseline on BOTH u and v at once (previous embedding+global-pool+MLP
approach — train_navigate.py/NavHead — never beat baseline on u; kept
importable there for comparison, just no longer what this sidecar runs).

Protocol (stdin/stdout), same shape as aesthetic_scorer.py/novelty_scorer.py:
  startup  -> prints "READY\n" once the trunk+head are loaded
  request  -> one image file path per line on stdin
  response -> "<u>,<v>,<log_zoom_ratio>\n" or "ERROR: <msg>\n" on failure

Usage: python3 nav_predict_sidecar.py [--model-path PATH]
  Defaults to nav_spatial_model.pt (scripts/train_navigate_spatial.py's
  default output) — gitignored, not committed, trained from Carl's own
  local navigation history.
"""
import argparse
import contextlib
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from train_navigate_spatial import load_spatial_model
from autoencoder import RES


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", type=Path, default=Path("nav_spatial_model.pt"))
    # Accepted-but-unused: src/nav_predict.rs's NavPredictor::new() forwards a
    # (model_path, head_path) pair when given Some(..) — never exercised in
    # practice (every call site passes None), but keep the flag so an old
    # invocation doesn't hard-fail on an unrecognized argument.
    parser.add_argument("--head-path", type=Path, default=None)
    try:
        args = parser.parse_args()
    except SystemExit:
        print("ERROR: usage: nav_predict_sidecar.py [--model-path PATH]", flush=True)
        raise
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.model_path.exists():
        print(f"ERROR: {args.model_path} not found — train one first "
              f"(scripts/train_navigate_spatial.py).", flush=True)
        sys.exit(1)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            trunk, head = load_spatial_model(args.model_path, device)
            log(f"loaded spatial nav-predict model (trunk arch={trunk.arch}, "
                f"variational={trunk.variational}) on {device}")
    except Exception as e:
        print(f"ERROR: failed to load model: {e}", flush=True)
        sys.exit(1)

    tf = transforms.Compose([transforms.Resize((RES, RES)), transforms.ToTensor()])
    print("READY", flush=True)

    EMPTY_CACHE_EVERY = 100
    n_requests = 0
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            img = Image.open(path).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = trunk.encode_spatial(x)
                pred, _attn = head(feat)
            u, v, log_zoom_ratio = (float(pred[0, 0]), float(pred[0, 1]), float(pred[0, 2]))
            print(f"{u:.6f},{v:.6f},{log_zoom_ratio:.6f}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        n_requests += 1
        if device == "cuda" and n_requests % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
