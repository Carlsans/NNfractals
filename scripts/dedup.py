#!/usr/bin/env python3
"""
NNFractals duplicate finder and cleaner.

Finds near-duplicate fractals in the fractals/ folder using multi-scale
grayscale DCT cosine similarity. Operates on 512×512 re-renders so the
comparison is consistent regardless of original save resolution.

Standard image size for all .nn-paired images: 512×512.

Usage:
  python3 scripts/dedup.py --test        Show similar pairs (no deletion)
  python3 scripts/dedup.py --rerender    Re-render all non-512×512 paired images
  python3 scripts/dedup.py --run         Full dedup loop (deletes losers)

Options:
  --threshold F   Similarity cutoff, 0–1 (default 0.94)
  --dir PATH      Fractals directory (default ./fractals)
  --binary PATH   Path to nnfractals binary (default ./target/release/nnfractals)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct as sdct

TARGET_SIZE = (512, 512)

# Multi-scale DCT parameters: (thumbnail side, DCT keep rows/cols)
_SCALES = [(16, 5), (32, 8), (64, 12)]
# Total feature length: sum of keep*keep for each scale
_FEAT_LEN = sum(k * k for _, k in _SCALES)


# ── DCT helpers ───────────────────────────────────────────────────────────────

def _dct2(a):
    return sdct(sdct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def feature_vec(img):
    """
    Multi-scale grayscale DCT feature vector (float32, L2-normalised).

    Resize to each scale → grayscale → DCT → keep top-left keep×keep
    coefficients → concatenate → normalise.

    Invariant to colour / brightness shifts; captures structural similarity
    at coarse, medium, and fine scale simultaneously.
    """
    parts = []
    for size, keep in _SCALES:
        gray = np.array(
            img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32
        )
        d = _dct2(gray.astype(float))[:keep, :keep].flatten().astype(np.float32)
        parts.append(d)
    v = np.concatenate(parts)
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


# ── File discovery ─────────────────────────────────────────────────────────────

def find_pairs(fractals_dir):
    """Return sorted list of (stem, nn_path, png_path) with both files present."""
    fractals_dir = Path(fractals_dir)
    pairs = []
    for nn_path in sorted(fractals_dir.rglob("*.nn")):
        png_path = nn_path.with_suffix(".png")
        if png_path.exists():
            pairs.append((nn_path.stem, nn_path, png_path))
    return pairs


# Comparison is O(n^2) (full pairwise cosine-similarity matrix). As an archive
# grows past several thousand files this becomes minutes-to-hours (observed:
# 17k-file archive pinned a CPU core for 2h12m+, longer than the periodic 2h
# interval that triggers this script, so runs started overlapping back-to-back).
# Cap the COMPARISON scope to the most-recently-modified files each round —
# recent saves are also the ones most likely to be near-duplicates of each
# other (the population explores similar regions in bursts), and this bounds
# runtime to a fixed size regardless of how large the total archive grows.
#
# 4000 turned out to still be insufficient under a high-duplicate-density burst
# (a population that converges on one fecund-but-repetitive formula family can
# flood the recent window with near-duplicate pairs, needing MANY dedup_round()
# passes to fully clean — observed a single invocation running ~2h45m+, right
# back to the severity this cap was meant to fix). Lowered to 2000 (~4x cheaper
# per round) and — more importantly — the round count itself is now hard-capped
# below, since no fixed per-round file limit can bound the TOTAL runtime if an
# unbounded number of rounds are needed.
MAX_COMPARE_FILES = 2000
# Hard cap on rounds per invocation: dedup is an ongoing, resumable hygiene
# task, not a one-shot operation that must reach "fully clean" every time it
# runs — better to do a bounded amount of cleanup now and pick up the rest at
# the next scheduled interval than to let one invocation run indefinitely.
MAX_ROUNDS = 15


def find_recent_pairs(fractals_dir, limit=MAX_COMPARE_FILES):
    """Like find_pairs, but capped to the `limit` most-recently-modified pairs."""
    pairs = find_pairs(fractals_dir)
    if len(pairs) <= limit:
        return pairs
    pairs.sort(key=lambda p: p[2].stat().st_mtime, reverse=True)
    return pairs[:limit]


# ── Beauty score ──────────────────────────────────────────────────────────────

def beauty_score(nn_path):
    """laion_score > clip_score > beauty (first non-zero wins)."""
    try:
        with open(nn_path) as f:
            d = json.load(f)
        for key in ("laion_score", "clip_score", "beauty"):
            v = d.get(key, 0.0)
            if v and v > 0:
                return float(v)
    except Exception:
        pass
    return 0.0


# ── Re-rendering ──────────────────────────────────────────────────────────────

def needs_rerender(png_path):
    try:
        return Image.open(png_path).size != TARGET_SIZE
    except Exception:
        return True


def rerender(nn_path, binary):
    cmd = [
        str(binary), "--render", str(nn_path),
        "--width",  str(TARGET_SIZE[0]),
        "--height", str(TARGET_SIZE[1]),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"  [WARN] rerender failed {nn_path.name}: {r.stderr.strip()[:80]}")
        return False
    return True


def rerender_all(pairs, binary):
    todo = [(s, nn, png) for s, nn, png in pairs if needs_rerender(png)]
    if not todo:
        print(f"All {len(pairs)} paired images are already {TARGET_SIZE[0]}×{TARGET_SIZE[1]}.")
        return
    print(f"Re-rendering {len(todo)}/{len(pairs)} images to {TARGET_SIZE[0]}×{TARGET_SIZE[1]}…")
    for i, (stem, nn, png) in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {stem}", end="\r", flush=True)
        rerender(nn, binary)
    print(f"\nDone. {len(todo)} images updated.")


# ── Similarity index ──────────────────────────────────────────────────────────

def build_vector_matrix(pairs):
    """
    Load all paired images, compute feature_vec for each.
    Returns (stems list, matrix N×_FEAT_LEN) with L2-normalised rows.
    """
    stems, vecs = [], []
    n = len(pairs)
    for i, (stem, nn_path, png_path) in enumerate(pairs):
        if i % 200 == 0:
            print(f"  Loading {i}/{n}…", end="\r", flush=True)
        try:
            img = Image.open(png_path)
            # Re-render on the fly if not the right size (cheap check)
            if img.size != TARGET_SIZE:
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
            vecs.append(feature_vec(img))
            stems.append(stem)
        except Exception as e:
            print(f"  [WARN] {png_path.name}: {e}")
    print(f"  Loaded {len(stems)}/{n} images.          ")
    return stems, np.stack(vecs)   # (N, _FEAT_LEN)


def find_duplicate_pairs(stems, mat, threshold):
    """
    O(n²) cosine similarity via matrix multiply.
    Returns list of (sim, idx_a, idx_b) sorted by sim descending.
    mat rows are already L2-normalised → dot product = cosine similarity.
    """
    sim_mat = (mat @ mat.T).astype(np.float32)
    upper = np.triu_indices(len(stems), k=1)
    sims  = sim_mat[upper]
    mask  = sims >= threshold
    pairs = list(zip(
        sims[mask].tolist(),
        upper[0][mask].tolist(),
        upper[1][mask].tolist(),
    ))
    pairs.sort(key=lambda x: -x[0])
    return pairs


# ── Dedup round ───────────────────────────────────────────────────────────────

def dedup_round(pairs, threshold):
    """
    One pass: find near-duplicate pairs, delete the lower-beauty one.
    Returns number of deletions (0 = done).
    """
    print(f"Building feature vectors for {len(pairs)} images…")
    stems, mat = build_vector_matrix(pairs)
    stem_to_paths = {s: (nn, png) for s, nn, png in pairs}

    print(f"Comparing all pairs (threshold ≥ {threshold:.3f})…")
    hits = find_duplicate_pairs(stems, mat, threshold)
    print(f"Found {len(hits)} near-duplicate pair(s).")

    if not hits:
        return 0

    deleted = set()
    deletions = 0
    for sim, ia, ib in hits:
        sa, sb = stems[ia], stems[ib]
        if sa in deleted or sb in deleted:
            continue
        nn_a, png_a = stem_to_paths[sa]
        nn_b, png_b = stem_to_paths[sb]
        score_a = beauty_score(nn_a)
        score_b = beauty_score(nn_b)
        if score_a >= score_b:
            winner, loser_nn, loser_png, loser_s, winner_s = sa, nn_b, png_b, score_b, score_a
            loser = sb
        else:
            winner, loser_nn, loser_png, loser_s, winner_s = sb, nn_a, png_a, score_a, score_b
            loser = sa
        print(
            f"  sim={sim:.4f}  KEEP {winner[:16]} ({winner_s:.4f})"
            f"  DROP {loser[:16]} ({loser_s:.4f})"
        )
        loser_nn.unlink(missing_ok=True)
        loser_png.unlink(missing_ok=True)
        deleted.add(loser)
        deletions += 1

    return deletions


def run_dedup_loop(fractals_dir, threshold, binary):
    round_num = 0
    while True:
        round_num += 1
        if round_num > MAX_ROUNDS:
            print(f"\nHit MAX_ROUNDS={MAX_ROUNDS} for this invocation — stopping here; "
                  f"remaining near-duplicates (if any) will be cleaned at the next "
                  f"scheduled interval instead of extending this run indefinitely.")
            break
        print(f"\n── Round {round_num}/{MAX_ROUNDS} ──")
        pairs = find_pairs(fractals_dir)
        if not pairs:
            print("No paired images left.")
            break
        # Re-render any non-512×512 images before comparing (cheap size check,
        # scoped to the whole archive — correctness matters here, not speed).
        rerender_all(pairs, binary)
        # The O(n^2) comparison itself is capped to the most-recently-modified
        # files (see find_recent_pairs) so runtime stays bounded as the total
        # archive keeps growing across the project's life.
        recent = find_recent_pairs(fractals_dir)
        if len(recent) < len(pairs):
            print(f"Comparing the {len(recent)} most recent of {len(pairs)} total "
                  f"paired images (capped to bound O(n²) runtime).")
        n = dedup_round(recent, threshold)
        if n == 0:
            print(f"Pool is clean (threshold {threshold:.3f}). Done.")
            break
        print(f"Deleted {n} lookalike(s) this round.")


# ── Test mode ─────────────────────────────────────────────────────────────────

def run_test(pairs, n_examples=10):
    """
    Show pairs spanning the similarity spectrum. No deletion.
    Shows top 5 (most similar) + 5 from the 0.70–0.90 range.
    """
    print(f"\nBuilding feature vectors for {len(pairs)} images…")
    stems, mat = build_vector_matrix(pairs)
    stem_to_paths = {s: (nn, png) for s, nn, png in pairs}

    print("Computing all pairwise similarities…")
    sim_mat = (mat @ mat.T).astype(np.float32)
    upper   = np.triu_indices(len(stems), k=1)
    sims    = sim_mat[upper]
    order   = np.argsort(-sims)

    top_pairs = []
    mid_pairs = []
    seen = set()
    for idx in order:
        ia, ib = int(upper[0][idx]), int(upper[1][idx])
        sa, sb  = stems[ia], stems[ib]
        key = (min(sa, sb), max(sa, sb))
        if key in seen:
            continue
        seen.add(key)
        sim = float(sims[idx])
        entry = (
            sim, sa, sb,
            stem_to_paths[sa][0], stem_to_paths[sa][1],
            stem_to_paths[sb][0], stem_to_paths[sb][1],
        )
        if sim >= 0.95 and len(top_pairs) < 5:
            top_pairs.append(entry)
        elif 0.70 <= sim < 0.90 and len(mid_pairs) < 5:
            mid_pairs.append(entry)
        if len(top_pairs) >= 5 and len(mid_pairs) >= 5:
            break

    def print_entry(rank, sim, sa, sb, nn_a, png_a, nn_b, png_b):
        score_a = beauty_score(nn_a)
        score_b = beauty_score(nn_b)
        print(f"  [{rank}] similarity = {sim:.4f}")
        print(f"        A: {sa}  score={score_a:.4f}  {png_a}")
        print(f"        B: {sb}  score={score_b:.4f}  {png_b}")
        print()

    print(f"\n─── Very similar (likely duplicates, sim ≥ 0.95) ───")
    for i, e in enumerate(top_pairs, 1):
        print_entry(i, *e)

    print(f"─── Mid-range (borderline) ───")
    for i, e in enumerate(mid_pairs, len(top_pairs) + 1):
        print_entry(i, *e)


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="NNFractals near-duplicate cleaner")
    ap.add_argument("--test",       action="store_true",
                    help="Show top similar pairs without deleting")
    ap.add_argument("--rerender",   action="store_true",
                    help="Re-render non-512×512 paired images to 512×512")
    ap.add_argument("--run",        action="store_true",
                    help="Run full dedup loop (destructive)")
    ap.add_argument("--threshold",  type=float, default=0.94,
                    help="Similarity cutoff 0–1 (default 0.94)")
    ap.add_argument("--dir",        default="./fractals",
                    help="Fractals directory (default ./fractals)")
    ap.add_argument("--binary",     default="./target/release/nnfractals",
                    help="Path to nnfractals binary")
    args = ap.parse_args()

    if not any([args.test, args.rerender, args.run]):
        ap.print_help()
        sys.exit(0)

    fractals_dir = Path(args.dir)
    if not fractals_dir.is_dir():
        print(f"Fractals directory not found: {fractals_dir}")
        sys.exit(1)

    binary = Path(args.binary)

    if args.rerender:
        if not binary.exists():
            print(f"Binary not found: {binary}  (run: cargo build --release)")
            sys.exit(1)
        pairs = find_pairs(fractals_dir)
        print(f"Found {len(pairs)} paired images.")
        rerender_all(pairs, args.binary)

    if args.test:
        pairs = find_pairs(fractals_dir)
        print(f"Found {len(pairs)} paired images.")
        run_test(pairs)

    if args.run:
        if not binary.exists():
            print(f"Binary not found: {binary}  (run: cargo build --release)")
            sys.exit(1)
        print(f"Starting dedup loop (threshold ≥ {args.threshold:.3f})…")
        run_dedup_loop(fractals_dir, args.threshold, binary)


if __name__ == "__main__":
    main()
