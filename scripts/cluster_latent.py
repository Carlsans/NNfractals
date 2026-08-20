#!/usr/bin/env python3
"""Splits a fractal-zone pool into visual-style GROUPS using the complex
VAE's latent space, via HDBSCAN — the "classify into groups" reading of
Carl's latent-space request (2026-08-08), complementary to
`select_diverse_latent.py`'s "pick N mutually-different zones" reading.

Same embedding as `select_diverse_latent.py`: `model.encode()`'s
`(mu_re, mu_im)` posterior means, concatenated into one real vector per
zone. HDBSCAN over scikit-learn's kd-tree/ball-tree backend (this
project's `hdbscan` package, not sklearn's built-in — same algorithm,
predates sklearn's copy) rather than k-means: no need to guess a cluster
COUNT up front, and points that don't belong to any dense group come back
labeled `-1` ("noise") instead of being force-fit into the nearest
cluster — the density-based equivalent of "doesn't look like anything
else here," useful signal in its own right for a fractal corpus that's
likely a mix of a few common styles plus one-off outliers.

Two things this script tunes automatically rather than leaving to guesswork
(2026-08-08, after the first pass across all 7 pools needed hand-picked
`--min-cluster-size` per pool and still left some pools at 84%+ noise):
1. **Dimensionality reduction: UMAP by default, not PCA.** PCA is linear —
   it only captures directions of maximum VARIANCE, which needn't align
   with cluster-relevant structure. UMAP preserves local neighborhood
   relationships non-linearly and is the standard, purpose-built
   companion to HDBSCAN (this is the pairing the HDBSCAN docs themselves
   recommend). `--reduction pca` kept as a fallback/comparison option.
2. **`--min-cluster-size` auto-swept and picked by DBCV, not guessed.**
   `hdbscan`'s `relative_validity_` (computed via `gen_min_span_tree=True`)
   is a fast approximation of DBCV — density-based clustering's analog of
   silhouette score, which assumes convex/globular clusters HDBSCAN
   doesn't produce and so isn't appropriate here. Sweeps a range of
   candidates scaled to the pool size and keeps whichever scores highest,
   rather than the previous per-pool hand-tuning (20 for large pools, all
   the way down to 5 for the smallest before accepting zero clusters).
   Pass `--min-cluster-size` explicitly to skip the sweep and force one
   value.

For each cluster, renders the `--reps-per-cluster` zones CLOSEST to that
cluster's centroid (the most "typical" examples of that style) — same
`explorer vae-curate` scratch-dir-symlink render path
`select_diverse_latent.py`/`curate_combined.py` already established, one
`vae-curate` call per cluster so each lands in its own subdirectory
(`out_dir/cluster_00/`, `cluster_01/`, ...). Noise points get their own
`out_dir/noise/` (an arbitrary sample, not "most typical" since noise by
definition isn't a coherent group) so Carl can see what got excluded.

Usage:
  python3 scripts/cluster_latent.py \
      --model-path explorer_out/complex_ae/complex_vae_tuned_et.pt \
      --dirs explorer_out/weekend_complex_corpus/mandelbrot \
      --pool-dir explorer_out/mandelbrot_vae \
      --out-dir explorer_out/mandelbrot_vae/clusters \
      --min-cluster-size 20 --reps-per-cluster 5 --res 512
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import hdbscan
import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import transforms

from complex_autoencoder import ComplexAutoEncoder, RES
from train_complex_autoencoder import gather_pairs, ComplexPairDataset


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def default_candidates(n):
    """Scaled `min_cluster_size` candidates to sweep, roughly 2%-16% of
    the pool plus a couple of small absolute floors — wide enough to
    cover everything from this project's largest pools (mcs=20 worked)
    down to its smallest (needed mcs=5, or genuinely had zero clusters
    at any size)."""
    fracs = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16]
    cands = sorted({max(5, int(n * f)) for f in fracs} | {5, 8})
    return [c for c in cands if 4 < c < n]


def fit_hdbscan(cluster_space, min_cluster_size, min_samples, method):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                 metric="euclidean", cluster_selection_method=method,
                                 gen_min_span_tree=True)
    labels = clusterer.fit_predict(cluster_space)
    n_clusters = len(set(labels) - {-1})
    try:
        validity = clusterer.relative_validity_ if n_clusters > 0 else -1.0
    except Exception:
        validity = -1.0
    return clusterer, labels, n_clusters, validity


def sweep_min_cluster_size(cluster_space, candidates, min_samples, method):
    """Fits HDBSCAN at each candidate `min_cluster_size`, scores each by
    `relative_validity_` (DBCV approximation), returns the best-scoring
    fit. Ties/all-noise fits score -1.0 and lose to any real clustering."""
    best = None
    for mcs in candidates:
        clusterer, labels, n_clusters, validity = fit_hdbscan(cluster_space, mcs, min_samples, method)
        log(f"  min_cluster_size={mcs}: {n_clusters} cluster(s), relative_validity={validity:.4f}")
        if best is None or validity > best[4]:
            best = (mcs, clusterer, labels, n_clusters, validity)
    return best


def render_group(stems, pool_dir, out_dir, scratch, res, explorer_bin, join_nn):
    """Symlinks `stems` (in the given order) into a fresh scratch dir with
    a synthetic manifest, renders via `explorer vae-curate`, optionally
    copies each source `.nn` alongside its render — same pattern
    `select_diverse_latent.py` uses, factored out here since this script
    calls it once per cluster instead of once total."""
    scratch = Path(scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    pool_dir = Path(pool_dir)
    missing = 0
    with open(scratch / "vae_recon_manifest.jsonl", "w") as mf:
        for rank_i, stem in enumerate(stems):
            nn_src = pool_dir / f"{stem}.nn"
            if not nn_src.exists():
                missing += 1
                continue
            (scratch / f"{stem}.nn").symlink_to(nn_src.resolve())
            mf.write(json.dumps({"stem": stem, "path": str(nn_src), "recon_mse": -rank_i}) + "\n")
    if missing:
        log(f"  warning: {missing} zones had no matching .nn in {pool_dir}, skipped")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [explorer_bin, "vae-curate", str(scratch), str(len(stems)), str(out_dir), str(res)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    if join_nn:
        for rank_i, stem in enumerate(stems):
            nn_src = pool_dir / f"{stem}.nn"
            if nn_src.exists():
                shutil.copy2(nn_src, out_dir / f"zone_{rank_i + 1:02d}.nn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-cluster-size", type=int, default=None,
                     help="fixed HDBSCAN min_cluster_size. Omit (default) to auto-sweep a range "
                          "scaled to pool size and pick the highest-relative_validity_ result")
    ap.add_argument("--min-cluster-size-candidates", type=int, nargs="+", default=None,
                     help="override the auto-derived sweep candidate list")
    ap.add_argument("--min-samples", type=int, default=None,
                     help="HDBSCAN min_samples (default: same as min_cluster_size, HDBSCAN's own default)")
    ap.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="leaf",
                     help="HDBSCAN's own default 'eom' (excess of mass) prefers a few large, stable "
                          "clusters and collapsed this project's fractal-zone corpora into 1-2 giant "
                          "blobs; 'leaf' (this script's default instead) takes the clustering tree's "
                          "leaves, giving finer-grained groups — confirmed on Mandelbrot (2026-08-08): "
                          "3 visually coherent, distinct styles vs. eom's 1 dominant blob + 1 tiny one")
    ap.add_argument("--reduction", choices=["umap", "pca", "none"], default="umap",
                     help="dimensionality reduction before clustering — umap (default, the standard "
                          "HDBSCAN companion, preserves local neighborhood structure non-linearly) or "
                          "pca (linear, this script's original approach — variance isn't necessarily "
                          "cluster-relevant) or none")
    ap.add_argument("--umap-dims", type=int, default=10)
    ap.add_argument("--pca-dims", type=int, default=50,
                     help="reduce the 1024-dim concatenated (mu_re,mu_im) embedding via PCA before "
                          "clustering, only used with --reduction pca. Density-based clustering "
                          "degenerates in high dimensions (distances concentrate, HDBSCAN can't find "
                          "dense cores) — some reduction is needed either way, fit fresh per pool.")
    ap.add_argument("--reps-per-cluster", type=int, default=5)
    ap.add_argument("--noise-sample", type=int, default=10,
                     help="how many noise (-1 label) zones to render for inspection, 0 to skip")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scratch", default="/tmp/cluster_latent_scratch")
    ap.add_argument("--explorer-bin", default="./target/release/explorer")
    ap.add_argument("--join-nn", action="store_true", default=True)
    ap.add_argument("--no-join-nn", dest="join_nn", action="store_false")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading model '{args.model_path}' on {device}…")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if not ckpt.get("variational", False):
        raise SystemExit("this model isn't a VAE (no variational latent) — clustering needs a "
                          "genuine posterior mean, pass a --variant vae checkpoint.")
    in_ch = ckpt.get("in_ch", 1)
    model = ComplexAutoEncoder(latent_dim=ckpt["latent_dim"], in_ch=in_ch,
                                norm=ckpt.get("norm", "groupnorm"),
                                residual=ckpt.get("residual", False),
                                variational=True).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log(f"model: latent_dim={ckpt['latent_dim']} norm={ckpt.get('norm', 'groupnorm')} in_ch={in_ch}")

    include_et = in_ch == 2
    pairs = gather_pairs(args.dirs, max_images=-1, seed=0, include_escape_time=include_et)
    log(f"{len(pairs)} pairs found across {args.dirs}")
    if not pairs:
        raise SystemExit("no matching pairs found.")

    tf = transforms.Compose([transforms.Resize((RES, RES)), transforms.ToTensor()])
    ds = ComplexPairDataset(pairs, tf, include_escape_time=include_et)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    stems, embeddings = [], []
    with torch.no_grad():
        i = 0
        for re, im in dl:
            re, im = re.to(device), im.to(device)
            mu_re, mu_im = model.encode((re, im))
            embeddings.append(torch.cat([mu_re, mu_im], dim=1).cpu().numpy())
            for _ in range(re.size(0)):
                stems.append(Path(pairs[i][0]).stem.removesuffix("_re"))
                i += 1
            if i % 200 < args.batch_size:
                log(f"embedded {i}/{len(pairs)}")
    embeddings = np.concatenate(embeddings, axis=0)
    log(f"embeddings shape: {embeddings.shape}")

    cluster_space = embeddings
    if args.reduction == "umap":
        n_neighbors = max(2, min(15, len(embeddings) - 1))
        reducer = umap.UMAP(n_components=args.umap_dims, n_neighbors=n_neighbors, random_state=0)
        cluster_space = reducer.fit_transform(embeddings)
        log(f"UMAP {embeddings.shape[1]} -> {args.umap_dims} dims (n_neighbors={n_neighbors})")
    elif args.reduction == "pca" and args.pca_dims > 0 and args.pca_dims < embeddings.shape[1]:
        pca = PCA(n_components=args.pca_dims, random_state=0)
        cluster_space = pca.fit_transform(embeddings)
        log(f"PCA {embeddings.shape[1]} -> {args.pca_dims} dims "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

    if args.min_cluster_size is not None:
        _, labels, n_clusters, validity = fit_hdbscan(
            cluster_space, args.min_cluster_size, args.min_samples, args.cluster_selection_method)
        log(f"min_cluster_size={args.min_cluster_size} (fixed): {n_clusters} cluster(s), "
            f"relative_validity={validity:.4f}")
    else:
        candidates = args.min_cluster_size_candidates or default_candidates(len(embeddings))
        log(f"sweeping min_cluster_size candidates: {candidates}")
        best = sweep_min_cluster_size(cluster_space, candidates, args.min_samples, args.cluster_selection_method)
        if best is None:
            raise SystemExit("no valid min_cluster_size candidates for this pool size — pass "
                              "--min-cluster-size-candidates explicitly.")
        mcs, _, labels, n_clusters, validity = best
        log(f"best: min_cluster_size={mcs} ({n_clusters} cluster(s), relative_validity={validity:.4f})")

    unique_labels = sorted(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    log(f"HDBSCAN found {len(unique_labels)} cluster(s), {n_noise}/{len(labels)} points labeled noise")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    # Largest clusters first — the corpus's dominant styles up front.
    unique_labels.sort(key=lambda lb: -int((labels == lb).sum()))
    for lb in unique_labels:
        idx = np.where(labels == lb)[0]
        centroid = cluster_space[idx].mean(axis=0, keepdims=True)
        dist = ((cluster_space[idx] - centroid) ** 2).sum(axis=1)
        order = idx[np.argsort(dist)][:args.reps_per_cluster]
        rep_stems = [stems[j] for j in order]
        cluster_dir = out_dir / f"cluster_{lb:02d}"
        log(f"cluster {lb:02d}: {len(idx)} zones, rendering {len(rep_stems)} representative(s) -> {cluster_dir}")
        render_group(rep_stems, args.pool_dir, cluster_dir,
                     f"{args.scratch}_c{lb}", args.res, args.explorer_bin, args.join_nn)
        summary.append({"cluster": int(lb), "size": int(len(idx)), "representatives": rep_stems})

    if args.noise_sample > 0 and n_noise > 0:
        noise_idx = np.where(labels == -1)[0][:args.noise_sample]
        noise_stems = [stems[j] for j in noise_idx]
        noise_dir = out_dir / "noise"
        log(f"noise: {n_noise} zones total, rendering {len(noise_stems)} sample(s) -> {noise_dir}")
        render_group(noise_stems, args.pool_dir, noise_dir,
                     f"{args.scratch}_noise", args.res, args.explorer_bin, args.join_nn)
        summary.append({"cluster": -1, "size": int(n_noise), "representatives": noise_stems})

    with open(out_dir / "cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"done -> {out_dir} ({len(unique_labels)} clusters + {'noise sample' if args.noise_sample else 'no noise sample'})")


if __name__ == "__main__":
    main()
