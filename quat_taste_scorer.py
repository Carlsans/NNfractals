#!/usr/bin/env python3
"""Lightweight taste-scoring sidecar for the quaternion GA.

SigLIP backbone + `taste_model_quat.npz` linear head only — NOT the 2D
side's 6-model aesthetic ensemble — so it fits in well under 1GB VRAM
alongside a GA GPU render process. Mirrors aesthetic_scorer.py's and
novelty_scorer.py's line-based stdin/stdout sidecar protocol structurally,
but the request line carries a command + a stable cache key, not just a
bare image path.

Protocol:
  startup  -> "READY\n" once SigLIP + the taste head are loaded
  request  -> "<CMD>\t<key>\t<png1>[\t<png2>...]\n"   CMD in {SCORE, EMBED, DIST}
              <key> is the genome's stable content hash (see
              QuatIndividual::content_hash in explorer.rs) — used as the
              on-disk embedding cache key (<cache-dir>/<key>.npy) so an
              elite that survives many generations is never re-embedded.
  response ->
    SCORE  -> "<taste>\n"
    EMBED  -> "<taste>|<v0>,<v1>,...,<v767>\n"   (backbone embedding,
              mean-pooled across the request's paths, L2-normalized)
    DIST   -> "<cosine distance to nearest role-model centroid>\n", or
              "ERROR: no role_model_centroids.npz" if that file is
              missing — loaded lazily so the sidecar starts fine without
              it (Phase 2c's optional archive axis).
  or "ERROR: <msg>\n" on failure for any command.

Hot-reload: before each SCORE/EMBED request, checks the model file's
mtime; if it changed since last load (a retrain finished — see
scripts/train_taste_quat.py), reloads the linear head's weights only —
the SigLIP backbone stays resident the whole time, so a reload is cheap
and never re-pays the ~1-2s model-load cost.

Usage: python3 quat_taste_scorer.py [--model taste_model_quat.npz]
  [--cache-dir embed_cache] [--role-centroids role_model_centroids.npz]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from train_pref import load_backbone  # noqa: E402


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class TasteHead:
    def __init__(self, model_path):
        self.model_path = model_path
        self.mtime = None
        self.w = None
        self.lo = 0.0
        self.hi = 1.0
        self.backbone = "siglip"
        self.reload_if_changed(force=True)

    def reload_if_changed(self, force=False):
        try:
            mtime = self.model_path.stat().st_mtime
        except FileNotFoundError:
            if force:
                raise
            return False
        if not force and mtime == self.mtime:
            return False
        data = np.load(self.model_path)
        self.w = np.asarray(data["w"], dtype=np.float32)
        self.lo = float(data["lo"]) if "lo" in data else 0.0
        self.hi = float(data["hi"]) if "hi" in data else 1.0
        self.backbone = str(data["backbone"]) if "backbone" in data else "siglip"
        self.mtime = mtime
        log(f"[taste] model {'loaded' if force else 'reloaded'} from {self.model_path} (mtime={mtime:.0f})")
        return True

    def score(self, vec):
        rng = (self.hi - self.lo) or 1.0
        r = float(np.dot(vec, self.w))
        return float(np.clip((r - self.lo) / rng, 0.0, 1.0))


def load_centroids(path):
    if not path.exists():
        return None
    data = np.load(path)
    return np.asarray(data["centroids"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("taste_model_quat.npz"))
    ap.add_argument("--cache-dir", type=Path, default=Path("embed_cache"))
    ap.add_argument("--role-centroids", type=Path, default=Path("role_model_centroids.npz"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.model.exists():
        print(f"ERROR: {args.model} not found -- train the taste model first "
              f"(scripts/train_taste_quat.py).", flush=True)
        sys.exit(1)

    try:
        head = TasteHead(args.model)
        log(f"loading backbone '{head.backbone}' on {device}...")
        embed = load_backbone(head.backbone, device)
    except Exception as e:
        print(f"ERROR: failed to load models: {e}", flush=True)
        sys.exit(1)

    args.cache_dir.mkdir(exist_ok=True)
    centroids = load_centroids(args.role_centroids)

    print("READY", flush=True)

    # Same rationale as aesthetic_scorer.py/novelty_scorer.py: periodically
    # release torch's cached allocator blocks so a long-running instance
    # doesn't look like it's slowly leaking GPU memory.
    EMPTY_CACHE_EVERY = 100
    n_requests = 0
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            print("ERROR: malformed request (need CMD\\tkey\\tpath...)", flush=True)
            continue
        cmd, key, paths = parts[0], parts[1], parts[2:]
        try:
            cache_path = args.cache_dir / f"{key}.npy"
            if cache_path.exists():
                vec = np.load(cache_path)
            else:
                imgs = [Image.open(p).convert("RGB") for p in paths]
                with torch.no_grad():
                    e = embed(imgs).cpu().numpy()
                vec = e.mean(axis=0).astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec = vec / norm
                np.save(cache_path, vec)

            if cmd == "SCORE":
                head.reload_if_changed()
                print(f"{head.score(vec):.5f}", flush=True)
            elif cmd == "EMBED":
                head.reload_if_changed()
                vec_str = ",".join(f"{x:.6f}" for x in vec)
                print(f"{head.score(vec):.5f}|{vec_str}", flush=True)
            elif cmd == "DIST":
                if centroids is None:
                    centroids = load_centroids(args.role_centroids)
                if centroids is None:
                    print("ERROR: no role_model_centroids.npz", flush=True)
                else:
                    sims = centroids @ vec
                    dist = float(np.sqrt(np.clip(2.0 - 2.0 * sims.max(), 0.0, None)))
                    print(f"{dist:.5f}", flush=True)
            else:
                print(f"ERROR: unknown command {cmd!r}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        n_requests += 1
        if device == "cuda" and n_requests % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
