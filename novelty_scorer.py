#!/usr/bin/env python3
"""Image-based novelty scorer sidecar for NNfractals.

Frozen DINOv2 backbone + a small VICReg-trained projection head (see
scripts/train_novelty.py) score how visually different a rendered fractal
is from fractals already created — average L2 distance to its k nearest
neighbors in the pool's learned embedding space, the same definition
scripts/train_novelty.py's backfill uses, just computed live and
incrementally instead of batched.

Protocol (stdin/stdout), same shape as aesthetic_scorer.py:
  startup  -> prints "READY\n" once the backbone+head+archive are loaded
  request  -> one image file path per line on stdin
  response -> "<novelty_score>|<v0>,<v1>,...,<v127>\n" (score, then the
              L2-normalized 128-d projection-head embedding used to compute
              it — the same `vec` that scores archive novelty, exposed so a
              caller can also do PAIRWISE latent-distance comparisons
              between its OWN candidates, not just each one's distance to
              the archive) or "ERROR: <msg>\n" on failure

Usage: python3 novelty_scorer.py <save_dir> [--model-path PATH] [--head-path PATH]
  <save_dir> is the pool directory (e.g. ./fractals_1) whose
  .novelty_cache.npz (if present) seeds the starting archive — if it's
  absent (a pool with no trained/backfilled model yet), starts empty and
  degrades to "different from what's been scored so far this run", which
  is correct behavior, not an error.
  --model-path/--head-path default to novelty_model.npz/novelty_head.pt
  (the production model) — pass a custom pair to score against a
  formula-specific model instead, e.g. one trained by
  scripts/train_novelty.py against a single-formula exploration pool.
"""
import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from train_pref import load_backbone
from train_novelty import ProjectionHead

K_NEIGHBORS = 15
# Cap on the live archive size (FIFO — oldest dropped first past this). Not
# just a memory bound: average k-NN distance shrinks as an archive densifies
# for purely geometric reasons, so letting it grow without limit over a
# long-running instance would make novelty_score drift down over time even
# with visual diversity holding steady. This keeps the archive's density —
# and so novelty_score's scale — roughly stationary across a long run.
MAX_ARCHIVE = 20000
CHUNK = 4000


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_archive(save_dir):
    """Return (n_loaded, mat: (N,D) float32 or None) from
    <save_dir>/.novelty_cache.npz, or (0, None) if absent/unreadable."""
    cache_path = Path(save_dir) / ".novelty_cache.npz"
    if not cache_path.exists():
        log(f"no {cache_path} — starting with an empty novelty archive.")
        return 0, None
    try:
        with np.load(cache_path) as z:
            mat = np.asarray(z["vecs"], dtype=np.float32)
        log(f"loaded {len(mat)} archived embeddings from {cache_path}")
        return len(mat), mat
    except Exception as e:
        log(f"  [WARN] {cache_path} unreadable ({e}); starting empty.")
        return 0, None


def knn_novelty_one(vec, mat, k=K_NEIGHBORS, chunk=CHUNK):
    """vec: (D,) L2-normalized. mat: (N,D) L2-normalized or None/empty.
    Average euclidean distance to the k nearest rows of mat (no
    self-exclusion needed — vec isn't in mat yet when this is called)."""
    if mat is None or len(mat) == 0:
        return 0.0
    k = min(k, len(mat))
    sims = np.empty(len(mat), dtype=np.float32)
    for start in range(0, len(mat), chunk):
        end = min(start + chunk, len(mat))
        sims[start:end] = mat[start:end] @ vec
    part = np.partition(sims, -k)[-k:]
    dist = np.sqrt(np.clip(2.0 - 2.0 * part, 0.0, None))
    return float(dist.mean())


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("save_dir")
    parser.add_argument("--model-path", type=Path, default=Path("novelty_model.npz"))
    parser.add_argument("--head-path", type=Path, default=Path("novelty_head.pt"))
    try:
        args = parser.parse_args()
    except SystemExit:
        print("ERROR: usage: novelty_scorer.py <save_dir> [--model-path PATH] [--head-path PATH]", flush=True)
        raise
    save_dir = args.save_dir
    model_path, head_path = args.model_path, args.head_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_path.exists() or not head_path.exists():
        print(f"ERROR: {model_path}/{head_path} not found — train a head first "
              f"(scripts/train_novelty.py).", flush=True)
        sys.exit(1)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            meta = np.load(model_path)
            backbone_name = str(meta["backbone"])
            head = ProjectionHead(
                in_dim=int(meta["in_dim"]), hidden=int(meta["hidden"]), out_dim=int(meta["out_dim"])
            ).to(device)
            head.load_state_dict(torch.load(head_path, map_location=device))
            head.eval()
            log(f"loaded novelty head ({backbone_name} backbone) on {device}")
            embed = load_backbone(backbone_name, device)
            _, archive = load_archive(save_dir)
    except Exception as e:
        print(f"ERROR: failed to load models: {e}", flush=True)
        sys.exit(1)

    print("READY", flush=True)

    # Same rationale as aesthetic_scorer.py: release torch's cached allocator
    # blocks periodically so a long-running instance doesn't look like it's
    # slowly leaking GPU memory.
    EMPTY_CACHE_EVERY = 100
    n_requests = 0
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                e = embed([img])  # (1, D) backbone embedding, L2-normalized
                z = torch.nn.functional.normalize(head(e), dim=-1)
            vec = z.cpu().numpy()[0].astype(np.float32)

            score = knn_novelty_one(vec, archive)
            vec_str = ",".join(f"{x:.6f}" for x in vec)
            print(f"{score:.5f}|{vec_str}", flush=True)

            # Grow the live archive so novelty stays meaningful as the run
            # scores more candidates, not just what was present at startup —
            # capped (oldest dropped first) so density stays roughly
            # stationary over a long run (see MAX_ARCHIVE).
            archive = vec[None, :] if archive is None or len(archive) == 0 \
                else np.concatenate([archive, vec[None, :]], axis=0)
            if len(archive) > MAX_ARCHIVE:
                archive = archive[-MAX_ARCHIVE:]
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        n_requests += 1
        if device == "cuda" and n_requests % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
