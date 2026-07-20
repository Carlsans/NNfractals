#!/usr/bin/env python3
"""
NNFractals duplicate finder and cleaner.

Finds near-duplicate fractals in the fractals/ folder using multi-scale
grayscale DCT cosine similarity. Operates on 512×512 re-renders so the
comparison is consistent regardless of original save resolution.

Standard image size for all .nn-paired images: 512×512.

Usage:
  python3 scripts/dedup.py --test              Show similar pairs (no deletion)
  python3 scripts/dedup.py --rerender          Re-render all non-512×512 paired images
  python3 scripts/dedup.py --run               Full dedup pass (deletes losers)
  python3 scripts/dedup.py --run --dry-run     Report how many would be deleted; no writes

Options:
  --threshold F     Similarity cutoff, 0–1 (default 0.94)
  --dir PATH        Fractals directory (default ./fractals)
  --binary PATH     Path to nnfractals binary (default ./target/release/nnfractals)
  --max-compare N   Cap comparison to the N most-recently-modified files
                     (default: no cap, the full archive is compared — see
                     the vector-cache note below). Only useful for bounding
                     a manual/one-off run; the periodic 2h job never passes
                     this.
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

def progress(phase, done, total):
    """Machine-readable progress line the launcher parses to drive its progress bar."""
    print(f"PROGRESS {phase} {done} {total}", flush=True)


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


def find_recent_pairs(fractals_dir, limit):
    """Like find_pairs, but capped to the `limit` most-recently-modified pairs.

    Only used when the caller explicitly passes --max-compare (a manual/
    one-off escape hatch) — the periodic dedup job compares the whole
    archive via the vector cache instead (see build_vectors_cached)."""
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


# ── Vector cache ─────────────────────────────────────────────────────────────
#
# Comparison used to be capped to the MAX_COMPARE_FILES most-recently-modified
# files each round, to bound the O(n^2) cost of an archive that had grown
# past a few thousand files (observed: a 17k-file archive pinned a CPU core
# for 2h12m+ in the old full-rescan design). That cap fixed the runtime
# problem but broke correctness: once an archive is larger than the cap
# (which all of this project's pools now are, at 20k-40k+ files each), any
# file more than `limit` saves old permanently exits the comparison window
# and is never compared to anything again — near-duplicates saved hours
# apart, on either side of that boundary, are simply never checked against
# each other. That's the "duplicates never get cleaned up" bug.
#
# Fix: cache each file's feature vector (keyed by stem + mtime, invalidated
# automatically if the file changes, e.g. a re-render) in a sidecar file
# next to the pool. A given run then only has to *compute* vectors for
# files that are new or changed since the last run — usually a small
# fraction of the archive — and compares those against the FULL cached
# corpus (chunked to bound peak memory), not a fixed-size recent window.
# Every pair that existed as of a given run is fully resolved by that run,
# so nothing ever ages out; each subsequent run only has new work to do.

CACHE_NAME = ".dedup_cache.npz"
# Column-chunk size for new-vs-corpus similarity search: bounds peak memory
# to roughly (batch_size * CHUNK * 4) bytes regardless of total archive size.
CHUNK = 4000


def _cache_path(fractals_dir):
    return Path(fractals_dir) / CACHE_NAME


def _load_cache(fractals_dir):
    """Return dict stem -> (mtime, vec, resolved_threshold), or {} if no/corrupt cache.

    `resolved_threshold` is the loosest (lowest) similarity threshold a stem
    has been through a *completed, non-dry-run* comparison pass at, or None
    if never resolved. Resolving at a loose threshold implies resolution at
    any stricter one too (every pair that would match at a higher threshold
    would also have matched at a lower one, so it was already seen) — but
    NOT the reverse: resolving at threshold 0.97 says nothing about pairs in
    [0.80, 0.97), so a later request at 0.80 must still recheck. A file only
    counts as already-checked for the CURRENT run's threshold T if
    resolved_threshold <= T. (Earlier versions stored a bare bool, which
    silently broke this: resolving once at 0.97 marked everything "done",
    so a later dry-run preview at 0.80 found "0 unresolved" and reported 0
    duplicates regardless of threshold — legacy bool caches are treated as
    unresolved below so they self-heal into the new format on next use.)

    Distinct from "vector is cached": --test and --run --dry-run both
    happily cache vectors (expensive to recompute, harmless to reuse)
    without ever advancing resolved_threshold, since neither of them
    actually deletes anything — if they did, a real --run afterward could
    see "already resolved" and skip comparing, silently leaving known
    duplicates on disk forever.
    """
    path = _cache_path(fractals_dir)
    if not path.exists():
        return {}
    try:
        with np.load(path) as z:
            stems, mtimes, vecs, resolved = z["stems"], z["mtimes"], z["vecs"], z["resolved"]
        if resolved.dtype == bool:
            # Legacy format: no record of *which* threshold. Keep the
            # (expensive) vectors, treat as never-resolved so the next run
            # rebuilds resolution state at whatever threshold it uses.
            return {s: (float(m), v, None) for s, m, v in zip(stems, mtimes, vecs)}
        return {
            s: (float(m), v, (None if np.isinf(r) else float(r)))
            for s, m, v, r in zip(stems, mtimes, vecs, resolved)
        }
    except Exception as e:
        print(f"  [WARN] dedup vector cache unreadable ({e}); rebuilding.")
        return {}


def _save_cache(fractals_dir, cache):
    if not cache:
        return
    stems = np.array(list(cache.keys()))
    mtimes = np.array([m for m, _, _ in cache.values()], dtype=np.float64)
    vecs = np.stack([v for _, v, _ in cache.values()]).astype(np.float32)
    resolved = np.array(
        [(np.inf if t is None else t) for _, _, t in cache.values()], dtype=np.float64
    )
    path = _cache_path(fractals_dir)
    tmp = path.with_name(path.name + ".tmp")
    # np.savez auto-appends ".npz" to a path-like target that doesn't already
    # end in it (which would silently write to "<tmp>.npz" instead of <tmp>)
    # — pass an open file object instead so it writes exactly where told.
    with open(tmp, "wb") as f:
        np.savez(f, stems=stems, mtimes=mtimes, vecs=vecs, resolved=resolved)
    tmp.replace(path)  # atomic swap, avoids a half-written cache on crash


def build_vectors_cached(pairs, fractals_dir, threshold, on_progress=None):
    """
    Load cached feature vectors, compute only for new/changed files (absent
    from the cache, or whose png mtime moved — e.g. a fresh render), persist
    the vector-cache update, and return everything needed for comparison.

    Returns (stems, mat, unresolved_indices, cache): stems/mat cover the
    *entire* current archive (aligned by row); unresolved_indices are the
    positions not yet resolved *at this threshold or looser* (freshly
    computed this call, never resolved, or only ever resolved at a
    stricter threshold than the one being requested now) — the only ones
    that can newly cross `threshold` against the rest of the corpus.
    `cache` is returned so the caller can advance resolved_threshold after
    a successful non-dry-run comparison pass and persist that.
    """
    cache = _load_cache(fractals_dir)
    current_stems = {s for s, _, _ in pairs}
    cache = {s: v for s, v in cache.items() if s in current_stems}  # drop deleted files

    to_compute = []
    for stem, nn_path, png_path in pairs:
        mtime = png_path.stat().st_mtime
        cached = cache.get(stem)
        if cached is None or cached[0] != mtime:
            to_compute.append((stem, png_path, mtime))

    n_new = len(to_compute)
    if n_new:
        print(f"Computing feature vectors for {n_new} new/changed image(s) "
              f"({len(pairs) - n_new} reused from cache)…")
    for i, (stem, png_path, mtime) in enumerate(to_compute):
        if i % 200 == 0:
            if on_progress:
                on_progress(i, n_new)
            else:
                print(f"  Vectorizing {i}/{n_new}…", end="\r", flush=True)
        try:
            img = Image.open(png_path)
            if img.size != TARGET_SIZE:
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
            cache[stem] = (mtime, feature_vec(img), None)  # unresolved until a real run confirms it
        except Exception as e:
            print(f"  [WARN] {png_path.name}: {e}")
    if on_progress:
        on_progress(n_new, n_new)
    elif n_new:
        print(f"  Vectorized {n_new}/{n_new}.          ")

    _save_cache(fractals_dir, cache)

    stems = [s for s, _, _ in pairs if s in cache]
    mat = np.stack([cache[s][1] for s in stems]).astype(np.float32)
    unresolved_indices = [
        i for i, s in enumerate(stems)
        if cache[s][2] is None or cache[s][2] > threshold
    ]
    return stems, mat, unresolved_indices, cache


def resolve_duplicates(mat, unresolved_indices, scores, threshold, chunk=CHUNK):
    """
    Resolve every unresolved file against the *entire* corpus and decide
    winner/loser on the fly, instead of collecting every matching edge into
    a list/dict first. That collect-then-sort approach is what actually
    caused a real OOM kill in production: one archive turned out to have a
    single file with 13,000+ near-duplicates (a population that converged
    hard on one repetitive formula family), which alone produces tens of
    millions of edges — a plain Python dict of that size runs into tens of
    GB of overhead. Here peak memory is bounded to O(chunk * corpus size)
    no matter how dense the duplicate cluster is, because a decision is
    made and the block discarded before moving to the next one.

    Processes unresolved files in beauty-score descending order: by the
    time a file is reached, every higher-scored *unresolved* file has
    already had a chance to kill it (symmetric similarity), so a survivor
    only needs its remaining live neighbors killed, never re-litigated.
    Already-resolved (previously kept) files stay eligible to be killed
    too — a new, higher-scoring duplicate arriving in a later run must
    still beat out an old resolved one (this is the fix for the original
    "duplicates saved hours apart never get compared" bug: nothing is ever
    permanently exempt from being challenged by a better duplicate).

    Returns {loser_idx: (winner_idx, sim)}.
    """
    if not unresolved_indices:
        return {}
    n = len(mat)
    alive = np.ones(n, dtype=bool)
    order = sorted(unresolved_indices, key=lambda i: -scores[i])
    verdict = {}

    for start in range(0, len(order), chunk):
        batch_idx = np.array(order[start:start + chunk])
        live_mask = alive[batch_idx]
        if not live_mask.any():
            continue
        active = batch_idx[live_mask]
        sims = mat[active] @ mat.T  # (len(active), n) — bounded, discarded after this batch
        for row, i in enumerate(active.tolist()):
            if not alive[i]:
                continue  # killed by an earlier row within this same batch
            neighbor_mask = (sims[row] >= threshold) & alive
            neighbor_mask[i] = False
            for j in np.nonzero(neighbor_mask)[0].tolist():
                if not alive[j]:
                    continue
                sim = float(sims[row, j])
                if scores[i] >= scores[j]:
                    alive[j] = False
                    verdict[j] = (i, sim)
                else:
                    alive[i] = False
                    verdict[i] = (j, sim)
                    break
    return verdict


_SCAN_CAP = 2000  # candidate pool cap per bucket — plenty for a top-5/mid-5 preview


def _scan_block(sims2d, row_offset, col_offset, top, mid):
    """Collect (sim, gi, gj) from a 2D similarity block into `top`
    (sim >= 0.95) and `mid` (0.70 <= sim < 0.90) — np.nonzero on the 2D
    block gives coordinate arrays directly, so this only ever iterates
    actual matches, never every cell. Stops touching a bucket once it's
    past _SCAN_CAP: a densely-duplicated archive can have a single file
    with 10k+ near-duplicates, so an uncapped accumulation here has the
    same runaway-memory failure mode as the old edge-collecting dedup
    path did (see resolve_duplicates's docstring) — this is only a
    preview, a bounded candidate pool is all --test needs."""
    if len(top) < _SCAN_CAP:
        hi_r, hi_c = np.nonzero(sims2d >= 0.95)
        for r, c in zip(hi_r.tolist(), hi_c.tolist()):
            top.append((float(sims2d[r, c]), row_offset + r, col_offset + c))
            if len(top) >= _SCAN_CAP:
                break
    if len(mid) < _SCAN_CAP:
        mi_r, mi_c = np.nonzero((sims2d >= 0.70) & (sims2d < 0.90))
        for r, c in zip(mi_r.tolist(), mi_c.tolist()):
            mid.append((float(sims2d[r, c]), row_offset + r, col_offset + c))
            if len(mid) >= _SCAN_CAP:
                break


def scan_similarity_spectrum(mat, chunk=CHUNK):
    """
    Full self-comparison across the whole corpus (needed for --test's "show
    me the most similar pairs" query — there's no way around O(n^2) work for
    a true global ranking), chunked over the column axis so a dense N×N
    matrix (multiple GB at 40k+ files) is never materialized, and every
    block comparison stays a vectorized numpy op rather than a per-pair
    Python loop (which at 40k files would be ~800M iterations — far too
    slow). Returns (top_candidates, mid_candidates) as (sim, i, j) lists,
    sorted by similarity descending.
    """
    n = len(mat)
    top, mid = [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = mat[start:end]
        if start > 0:
            # Rows before this chunk vs. this chunk's columns: every row
            # index here is already < every column index, no mask needed.
            _scan_block(mat[:start] @ block.T, 0, start, top, mid)
        # Within-chunk pairs need the i<j restriction — mask self + the
        # lower triangle out of range before scanning.
        diag = block @ block.T
        diag[np.tril_indices(end - start)] = -1.0
        _scan_block(diag, start, start, top, mid)
    top.sort(key=lambda x: -x[0])
    mid.sort(key=lambda x: -x[0])
    return top, mid


# ── Dedup pass ────────────────────────────────────────────────────────────────

def dedup_round(pairs, threshold, fractals_dir, dry_run=False, on_progress=None):
    """
    One full pass over the archive: reuse cached vectors for unchanged
    files, compute vectors only for new/changed ones, and compare
    still-unresolved files against the full corpus. Deletes (or, if
    dry_run, reports) the lower-beauty file of each matched pair.

    Only a completed non-dry-run pass advances resolved_threshold in the
    cache — a dry-run must never do that, or a later real run at the same
    (or looser) threshold would see nothing unresolved and silently skip
    re-checking (and thus never delete) the exact duplicates the dry-run
    just reported.
    """
    stems, mat, unresolved, cache = build_vectors_cached(
        pairs, fractals_dir, threshold, on_progress=on_progress
    )
    stem_to_paths = {s: (nn, png) for s, nn, png in pairs}
    scores = np.array([beauty_score(stem_to_paths[s][0]) for s in stems], dtype=np.float64)

    print(f"Comparing {len(unresolved)} unresolved image(s) against the full "
          f"{len(stems)}-file archive (threshold ≥ {threshold:.3f})…")
    verdict = resolve_duplicates(mat, unresolved, scores, threshold)
    print(f"Found {len(verdict)} near-duplicate pair(s).")

    deleted = set()
    examples = []
    if verdict:
        tag = "WOULD DROP" if dry_run else "DROP"
        # Report highest-similarity first; a densely-duplicated archive can
        # produce thousands of deletions in one pass, so only print the
        # first VERBOSE_LIMIT lines in full and summarize the rest — this
        # is purely a log-readability cap, every deletion below still
        # happens (or would, in dry-run).
        VERBOSE_LIMIT = 50
        ranked = sorted(verdict.items(), key=lambda kv: -kv[1][1])
        for i, (loser_i, (winner_i, sim)) in enumerate(ranked):
            loser, winner = stems[loser_i], stems[winner_i]
            loser_nn, loser_png = stem_to_paths[loser]
            winner_s, loser_s = float(scores[winner_i]), float(scores[loser_i])
            if i < VERBOSE_LIMIT:
                print(
                    f"  sim={sim:.4f}  KEEP {winner[:16]} ({winner_s:.4f})"
                    f"  {tag} {loser[:16]} ({loser_s:.4f})"
                )
            elif i == VERBOSE_LIMIT:
                print(f"  … and {len(ranked) - VERBOSE_LIMIT} more (suppressed for log size)")
            if not dry_run:
                loser_nn.unlink(missing_ok=True)
                loser_png.unlink(missing_ok=True)
            deleted.add(loser)
            if len(examples) < 25:
                examples.append((sim, winner, loser, winner_s, loser_s))

    if not dry_run:
        # The full corpus considered this round is now mutually resolved at
        # `threshold` (or looser, if it already had a lower resolved_threshold
        # from an earlier run): drop deleted stems from the cache and advance
        # every survivor's resolved_threshold so the next run at this
        # threshold or higher has only new-since-now files left to check —
        # but a future run requesting a LOOSER threshold still correctly
        # sees everyone as unresolved again (see build_vectors_cached).
        for s in deleted:
            cache.pop(s, None)
        for s in stems:
            if s in cache and s not in deleted:
                mtime, vec, prev_rt = cache[s]
                new_rt = threshold if prev_rt is None else min(prev_rt, threshold)
                cache[s] = (mtime, vec, new_rt)
        _save_cache(fractals_dir, cache)

    return len(deleted), examples


def run_dedup_loop(fractals_dir, threshold, binary, dry_run=False, max_compare=None):
    """
    Run one full dedup pass over the archive. Every file is eligible for
    comparison via the on-disk vector cache — nothing ever ages out the way
    it did under the old recency-windowed approach. `max_compare`, if given,
    caps how many of the most-recently-modified files are considered at all;
    only meant for bounding a manual/one-off run — the periodic 2h job never
    passes it.
    """
    progress("round", 1, 1)
    pairs = find_pairs(fractals_dir)
    total_in_dir = len(pairs)
    if not pairs:
        print("No paired images left.")
        return 0, 0, 0, []

    if not dry_run:
        # Re-render any non-512×512 images before comparing (skipped in
        # dry-run: previewing must not touch disk). Re-scan afterward since
        # this can change mtimes, which build_vectors_cached uses to decide
        # what needs a fresh feature vector.
        rerender_all(pairs, binary)
        pairs = find_pairs(fractals_dir)
        total_in_dir = len(pairs)

    if max_compare and len(pairs) > max_compare:
        pairs = find_recent_pairs(fractals_dir, max_compare)
        print(f"--max-compare set: limiting to the {len(pairs)} most recent of "
              f"{total_in_dir} paired images.")

    considered = len(pairs)
    deleted, examples = dedup_round(
        pairs, threshold, fractals_dir, dry_run=dry_run,
        on_progress=lambda done, tot: progress("vectorize", done, tot),
    )
    if deleted:
        print(f"{'Would delete' if dry_run else 'Deleted'} {deleted} lookalike(s).")
    else:
        print(f"Pool is clean (threshold {threshold:.3f}).")
    return deleted, considered, total_in_dir, examples


# ── Test mode ─────────────────────────────────────────────────────────────────

def run_test(pairs, fractals_dir):
    """
    Show pairs spanning the similarity spectrum. No deletion, no disk writes
    beyond the vector cache (safe/idempotent — same cache --run would use).
    Shows top 5 (most similar) + 5 from the 0.70-0.90 range.
    """
    print(f"\nBuilding feature vectors for {len(pairs)} images (cached)…")
    # threshold=0.0 is a placeholder — run_test does its own full self-scan
    # via scan_similarity_spectrum below regardless of resolved_threshold,
    # so the unresolved-indices list (discarded here) is never consulted.
    stems, mat, _, _ = build_vectors_cached(pairs, fractals_dir, 0.0)
    stem_to_paths = {s: (nn, png) for s, nn, png in pairs}

    print("Scanning pairwise similarities…")
    top_hits, mid_hits = scan_similarity_spectrum(mat)
    top_pairs = [(sim, stems[ia], stems[ib]) for sim, ia, ib in top_hits[:5]]
    mid_pairs = [(sim, stems[ia], stems[ib]) for sim, ia, ib in mid_hits[:5]]

    def print_entry(rank, sim, sa, sb):
        nn_a, png_a = stem_to_paths[sa]
        nn_b, png_b = stem_to_paths[sb]
        score_a, score_b = beauty_score(nn_a), beauty_score(nn_b)
        print(f"  [{rank}] similarity = {sim:.4f}")
        print(f"        A: {sa}  score={score_a:.4f}  {png_a}")
        print(f"        B: {sb}  score={score_b:.4f}  {png_b}")
        print()

    print(f"\n─── Very similar (likely duplicates, sim ≥ 0.95) ───")
    for i, (sim, sa, sb) in enumerate(top_pairs, 1):
        print_entry(i, sim, sa, sb)

    print(f"─── Mid-range (borderline) ───")
    for i, (sim, sa, sb) in enumerate(mid_pairs, len(top_pairs) + 1):
        print_entry(i, sim, sa, sb)


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="NNFractals near-duplicate cleaner")
    ap.add_argument("--test",       action="store_true",
                    help="Show top similar pairs without deleting")
    ap.add_argument("--rerender",   action="store_true",
                    help="Re-render non-512×512 paired images to 512×512")
    ap.add_argument("--run",        action="store_true",
                    help="Run full dedup pass (destructive, unless --dry-run)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="With --run: don't delete or re-render anything, just "
                         "report how many fractals would be deleted")
    ap.add_argument("--threshold",  type=float, default=0.94,
                    help="Similarity cutoff 0–1 (default 0.94)")
    ap.add_argument("--dir",        default="./fractals",
                    help="Fractals directory (default ./fractals)")
    ap.add_argument("--binary",     default="./target/release/nnfractals",
                    help="Path to nnfractals binary")
    ap.add_argument("--max-compare", type=int, default=None,
                    help="Cap comparison to the N most-recently-modified files "
                         "(default: no cap, compares the full archive via the "
                         "vector cache)")
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
        run_test(pairs, fractals_dir)

    if args.run:
        if not args.dry_run and not binary.exists():
            print(f"Binary not found: {binary}  (run: cargo build --release)")
            sys.exit(1)
        mode = "Previewing (dry-run)" if args.dry_run else "Starting"
        print(f"{mode} dedup pass (threshold ≥ {args.threshold:.3f})…")
        deleted, considered, total_in_dir, examples = run_dedup_loop(
            fractals_dir, args.threshold, binary,
            dry_run=args.dry_run, max_compare=args.max_compare,
        )
        if args.dry_run:
            scope = (
                f"{considered} compared" if considered == total_in_dir
                else f"{considered} of {total_in_dir} compared (--max-compare)"
            )
            print(f"\n{deleted} of {considered} fractal(s) would be deleted "
                  f"({scope}, threshold {args.threshold:.3f}).")
            print(f"DONE would delete {deleted} of {considered} fractal(s) "
                  f"— {scope}, threshold {args.threshold:.3f}")
        else:
            print(f"DONE deleted {deleted} fractal(s) ({considered} compared, "
                  f"threshold {args.threshold:.3f})")


if __name__ == "__main__":
    main()
