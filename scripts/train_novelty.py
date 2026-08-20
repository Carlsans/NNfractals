#!/usr/bin/env python3
"""
Train a fractal-specific novelty/similarity embedding and backfill
`novelty_score` + `novelty_cluster` onto saved .nn files.

Frozen DINOv2 backbone (self-supervised ViT, no text-alignment — CLIP/LAION
were tried for aesthetic scoring and dropped for barely varying on fractals,
see aesthetic_scorer.py) feeds a small trainable projection head trained
with VICReg. VICReg's variance loss term explicitly forbids the same
"embeddings barely vary" collapse that sank CLIP/LAION here, rather than
just hoping a contrastive setup avoids it. Only the head is trained; the
backbone's per-image embeddings are computed once and cached.

Discovery/validation only for now — these fields are not read by fitness,
selection, or seeding. Inspect novelty_score/novelty_cluster in the browser
before deciding whether this is worth wiring into the GA.

Usage:
  # Train a head on fractals_1's images, then backfill fractals_1:
  python3 scripts/train_novelty.py --dirs fractals_1

  # Re-backfill (or backfill additional folders) with an already-trained head:
  python3 scripts/train_novelty.py --dirs fractals_1 fractals_2 --score-only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from scipy.cluster.vq import kmeans2

from train_pref import load_backbone
from dedup import find_pairs


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    """Machine-readable progress line, same convention as train_pref.py/dedup.py."""
    print(f"PROGRESS {phase} {done} {total}", flush=True)


# ── Augmentation: a fixed, cheap view pool per image ────────────────────────
# Mirrors the D4 dihedral-invariance idea already used by dihedral_variants()
# in src/fractal.rs, plus a mild crop/zoom and a colormap-invariance proxy
# (brightness/contrast/saturation jitter) — all pure PIL ops on the existing
# 512x512 PNG, no re-render needed.

_DIHEDRAL = [
    lambda im: im,
    lambda im: im.transpose(Image.ROTATE_90),
    lambda im: im.transpose(Image.ROTATE_180),
    lambda im: im.transpose(Image.ROTATE_270),
    lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
]

N_VIEWS = 4  # original + 3 augmented


def _view_dihedral(img, rng):
    return _DIHEDRAL[rng.integers(0, len(_DIHEDRAL))](img)


def _view_crop(img, rng):
    w, h = img.size
    scale = rng.uniform(0.8, 1.0)
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
    x0 = int(rng.integers(0, w - cw + 1))
    y0 = int(rng.integers(0, h - ch + 1))
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)


def _view_color(img, rng):
    out = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.15))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.7, 1.3))
    return out


def make_views(img, rng):
    return [img, _view_dihedral(img, rng), _view_crop(img, rng), _view_color(img, rng)]


# ── Per-view backbone-embedding cache (training-time only) ──────────────────
# Same stem+mtime keyed sidecar-cache idiom as scripts/dedup.py's
# .dedup_cache.npz: the expensive backbone forward pass runs once per view
# and is reused across epochs / reruns.

EMBED_CACHE_NAME = ".novelty_embed_cache.npz"


def _embed_cache_path(d):
    return Path(d) / EMBED_CACHE_NAME


def load_embed_cache(d):
    path = _embed_cache_path(d)
    if not path.exists():
        return {}
    try:
        with np.load(path) as z:
            stems, mtimes, vecs = z["stems"], z["mtimes"], z["vecs"]
        return {s: (float(m), v) for s, m, v in zip(stems, mtimes, vecs)}
    except Exception as e:
        log(f"  [WARN] novelty embed cache unreadable ({e}); rebuilding.")
        return {}


def save_embed_cache(d, cache):
    if not cache:
        return
    stems = np.array(list(cache.keys()))
    mtimes = np.array([m for m, _ in cache.values()], dtype=np.float64)
    vecs = np.stack([v for _, v in cache.values()]).astype(np.float32)
    path = _embed_cache_path(d)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, stems=stems, mtimes=mtimes, vecs=vecs)
    tmp.replace(path)


def build_view_embeddings(pairs, fractals_dir, embed, seed=0, batch=32, on_progress=None):
    """pairs: [(stem, nn_path, png_path)]. Returns {stem: (N_VIEWS, D) float32}.
    Cached views are reused as-is; new/changed files (by png mtime) get all
    N_VIEWS recomputed."""
    rng = np.random.default_rng(seed)
    cache = load_embed_cache(fractals_dir)
    current_stems = {s for s, _, _ in pairs}
    cache = {s: v for s, v in cache.items() if s in current_stems}

    to_compute = []
    for stem, nn_path, png_path in pairs:
        mtime = png_path.stat().st_mtime
        cached = cache.get(stem)
        if cached is None or cached[0] != mtime:
            to_compute.append((stem, png_path, mtime))

    total = len(to_compute)
    if total:
        log(f"Embedding {total} new/changed image(s) x {N_VIEWS} views "
            f"({len(pairs) - total} reused from cache)…")

    buf_stems, buf_mtimes, buf_imgs = [], [], []
    stems_per_flush = max(1, batch // N_VIEWS)

    def flush():
        if not buf_imgs:
            return
        e = embed(buf_imgs).cpu().numpy()
        d = e.shape[1]
        e = e.reshape(-1, N_VIEWS, d)
        for s, m, v in zip(buf_stems, buf_mtimes, e):
            cache[s] = (m, v.astype(np.float32))
        buf_stems.clear(); buf_mtimes.clear(); buf_imgs.clear()

    done = 0
    for stem, png_path, mtime in to_compute:
        try:
            img = Image.open(png_path).convert("RGB")
        except Exception:
            continue
        buf_stems.append(stem)
        buf_mtimes.append(mtime)
        buf_imgs.extend(make_views(img, rng))
        if len(buf_stems) >= stems_per_flush:
            flush()
        done += 1
        if done % 200 == 0 and on_progress:
            on_progress(done, total)
    flush()
    if on_progress:
        on_progress(total, total)

    save_embed_cache(fractals_dir, cache)
    return {s: v for s, (m, v) in cache.items()}


# ── VICReg head ───────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(za, zb, inv_w=25.0, var_w=25.0, cov_w=1.0, gamma=1.0, eps=1e-4):
    """Standard VICReg: invariance (MSE between paired views) + variance
    (hinge forcing each embedding dim's std >= gamma across the batch — the
    explicit anti-collapse term) + covariance (decorrelate dims)."""
    inv = torch.nn.functional.mse_loss(za, zb)

    def var_term(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(torch.relu(gamma - std))

    var = var_term(za) + var_term(zb)

    def cov_term(z):
        n, d = z.shape
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (n - 1)
        return _off_diagonal(cov).pow(2).sum() / d

    cov = cov_term(za) + cov_term(zb)
    total = inv_w * inv + var_w * var + cov_w * cov
    return total, {"inv": inv.item(), "var": var.item(), "cov": cov.item()}


def train_head(view_embeds, epochs, device, lr=1e-3, batch_size=256,
                hidden=256, out_dim=128, seed=0, inv_w=25.0, var_w=25.0, cov_w=1.0):
    stems = list(view_embeds.keys())
    mat = np.stack([view_embeds[s] for s in stems])  # (N, N_VIEWS, D)
    mat_t = torch.tensor(mat, dtype=torch.float32, device=device)
    n, n_views, d = mat_t.shape
    log(f"Training VICReg head on {n} images x {n_views} cached views (dim {d}), "
        f"inv_w={inv_w} var_w={var_w} cov_w={cov_w}…")

    head = ProjectionHead(in_dim=d, hidden=hidden, out_dim=out_dim).to(device)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    log_every = max(1, epochs // 20)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        totals = {"inv": 0.0, "var": 0.0, "cov": 0.0}
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            va = rng.integers(0, n_views, size=len(idx))
            vb = (va + 1 + rng.integers(0, n_views - 1, size=len(idx))) % n_views
            va_t = torch.tensor(va, device=device, dtype=torch.long)
            vb_t = torch.tensor(vb, device=device, dtype=torch.long)
            xa = mat_t[idx_t, va_t]
            xb = mat_t[idx_t, vb_t]
            za, zb = head(xa), head(xb)
            loss, parts = vicreg_loss(za, zb, inv_w=inv_w, var_w=var_w, cov_w=cov_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in totals:
                totals[k] += parts[k]
            n_batches += 1
        if n_batches and (epoch % log_every == 0 or epoch == epochs - 1):
            log(f"  epoch {epoch + 1}/{epochs}  "
                f"inv={totals['inv'] / n_batches:.4f}  "
                f"var={totals['var'] / n_batches:.4f}  "
                f"cov={totals['cov'] / n_batches:.4f}")

    head.eval()
    return head


# ── Backfill: embed originals, project, score, cluster, write ──────────────

NOVELTY_CACHE_NAME = ".novelty_cache.npz"
CHUNK = 4000


def _novelty_cache_path(d):
    return Path(d) / NOVELTY_CACHE_NAME


def embed_originals(pairs, embed, batch=32, on_progress=None):
    """{stem: (D,) backbone embedding} for each pair's ORIGINAL (unaugmented) image."""
    out = {}
    buf_stems, buf_imgs = [], []
    total = len(pairs)

    def flush():
        if not buf_imgs:
            return
        e = embed(buf_imgs).cpu().numpy()
        for s, v in zip(buf_stems, e):
            out[s] = v
        buf_stems.clear(); buf_imgs.clear()

    for i, (stem, nn_path, png_path) in enumerate(pairs):
        try:
            buf_imgs.append(Image.open(png_path).convert("RGB"))
            buf_stems.append(stem)
        except Exception:
            continue
        if len(buf_imgs) >= batch:
            flush()
        if (i + 1) % 200 == 0 and on_progress:
            on_progress(i + 1, total)
    flush()
    if on_progress:
        on_progress(total, total)
    return out


def project(head, backbone_embeds, device, batch=512):
    stems = list(backbone_embeds.keys())
    mat = np.stack([backbone_embeds[s] for s in stems]).astype(np.float32)
    out_dim = head.net[-1].out_features
    out = np.zeros((len(stems), out_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(stems), batch):
            chunk = torch.tensor(mat[start:start + batch], device=device)
            z = head(chunk)
            z = torch.nn.functional.normalize(z, dim=-1)
            out[start:start + batch] = z.cpu().numpy()
    return stems, out


def knn_novelty(mat, k, chunk=CHUNK):
    """mat: (N, D) L2-normalized. Average euclidean distance to the k nearest
    neighbors (excluding self) — same sparseness metric as classic novelty
    search, computed chunked (dedup.py's resolve_duplicates idiom) so memory
    stays O(chunk * n) regardless of archive size."""
    n = len(mat)
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    k = min(k, n - 1)
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = mat[start:end]
        sims = block @ mat.T
        rows = np.arange(end - start)
        sims[rows, np.arange(start, end)] = -np.inf  # exclude self
        part = np.partition(sims, -k, axis=1)[:, -k:]
        dist = np.sqrt(np.clip(2.0 - 2.0 * part, 0.0, None))  # unit-vector cosine -> euclidean
        out[start:end] = dist.mean(axis=1)
    return out


def cluster_ids(mat, k, seed=0):
    n = len(mat)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    k = min(k, n)
    _, labels = kmeans2(mat.astype(np.float64), k, seed=seed, minit="++")
    return labels.astype(np.int32)


def backfill(args, device, head, backbone_name):
    embed = load_backbone(backbone_name, device)
    for d in args.dirs:
        pairs = find_pairs(d)
        if not pairs:
            log(f"{d}: no paired .nn/.png files, skipping.")
            continue
        log(f"{d}: embedding {len(pairs)} original images…")
        backbone_embeds = embed_originals(
            pairs, embed,
            on_progress=lambda done, tot: progress("embed", done, tot),
        )
        stems, proj = project(head, backbone_embeds, device)
        log(f"{d}: scoring novelty (k={args.k_neighbors}) and "
            f"clustering (k={args.k_clusters})…")
        novelty = knn_novelty(proj, args.k_neighbors)
        clusters = cluster_ids(proj, args.k_clusters, seed=args.seed)

        stem_to_paths = {s: (nn, png) for s, nn, png in pairs}
        mtimes = np.array([stem_to_paths[s][1].stat().st_mtime for s in stems], dtype=np.float64)
        cpath = _novelty_cache_path(d)
        tmp = cpath.with_name(cpath.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, stems=np.array(stems), mtimes=mtimes, vecs=proj)
        tmp.replace(cpath)

        written = 0
        n = len(stems)
        for i, s in enumerate(stems):
            nn_path, _ = stem_to_paths[s]
            try:
                g = json.loads(nn_path.read_text())
                g[args.field_score] = round(float(novelty[i]), 5)
                g[args.field_cluster] = int(clusters[i])
                nn_path.write_text(json.dumps(g, indent=2))
                written += 1
            except Exception:
                pass
            if i % 200 == 0 or i + 1 == n:
                progress("write", i + 1, n)
        log(f"{d}: wrote {args.field_score}/{args.field_cluster} to {written} .nn files")
        if n:
            log(f"{d}: {args.field_score}  min={novelty.min():.4f}  "
                f"median={np.median(novelty):.4f}  max={novelty.max():.4f}  "
                f"std={novelty.std():.4f}")
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                     help="Folders whose .nn files get novelty_score/novelty_cluster written")
    ap.add_argument("--train-dir", default=None,
                     help="Folder to train the VICReg head on (default: first of --dirs)")
    ap.add_argument("--score-only", action="store_true",
                     help="skip training; load a saved head and just backfill --dirs")
    ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "siglip"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--k-neighbors", type=int, default=15)
    ap.add_argument("--k-clusters", type=int, default=16)
    ap.add_argument("--field-score", default="novelty_score")
    ap.add_argument("--field-cluster", default="novelty_cluster")
    ap.add_argument("--model-path", default="novelty_model.npz")
    ap.add_argument("--head-path", default="novelty_head.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inv-weight", type=float, default=25.0,
                     help="VICReg invariance weight — how strongly augmented views of the "
                          "same image are pulled together. Lower = the head cares less about "
                          "exact pose/color consistency, more about raw content.")
    ap.add_argument("--var-weight", type=float, default=25.0,
                     help="VICReg variance weight — anti-collapse floor on each embedding "
                          "dimension's spread across the batch.")
    ap.add_argument("--cov-weight", type=float, default=1.0,
                     help="VICReg covariance weight — decorrelates embedding dimensions so "
                          "each one captures a genuinely different axis of variation instead "
                          "of redundant ones. This is the main DIVERSITY lever: a higher "
                          "cov-weight spreads the latent space out more, which is what a "
                          "downstream farthest-point diversity selection benefits from.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.score_only:
        if not Path(args.model_path).exists() or not Path(args.head_path).exists():
            raise SystemExit(
                f"no {args.model_path}/{args.head_path} — train a head first (omit --score-only)."
            )
        meta = np.load(args.model_path)
        backbone_name = str(meta["backbone"])
        head = ProjectionHead(
            in_dim=int(meta["in_dim"]), hidden=int(meta["hidden"]), out_dim=int(meta["out_dim"])
        ).to(device)
        head.load_state_dict(torch.load(args.head_path, map_location=device))
        head.eval()
        log(f"loaded head ({backbone_name} backbone) from {args.head_path}")
        backfill(args, device, head, backbone_name)
        return

    train_dir = args.train_dir or args.dirs[0]
    log(f"loading backbone '{args.backbone}' on {device}…")
    embed = load_backbone(args.backbone, device)

    pairs = find_pairs(train_dir)
    if not pairs:
        raise SystemExit(f"no paired .nn/.png files in {train_dir}")
    log(f"{train_dir}: found {len(pairs)} paired images.")

    view_embeds = build_view_embeddings(
        pairs, train_dir, embed, seed=args.seed,
        on_progress=lambda done, tot: progress("embed", done, tot),
    )
    if len(view_embeds) < 20:
        raise SystemExit(f"only {len(view_embeds)} embeddable images — need more to train a head.")

    in_dim = next(iter(view_embeds.values())).shape[-1]
    head = train_head(
        view_embeds, args.epochs, device, lr=args.lr, batch_size=args.batch_size,
        hidden=args.hidden_dim, out_dim=args.out_dim, seed=args.seed,
        inv_w=args.inv_weight, var_w=args.var_weight, cov_w=args.cov_weight,
    )

    torch.save(head.state_dict(), args.head_path)
    with open(args.model_path, "wb") as f:
        np.savez(f, backbone=args.backbone, in_dim=in_dim, hidden=args.hidden_dim, out_dim=args.out_dim)
    log(f"saved head → {args.head_path}, model meta → {args.model_path}")

    backfill(args, device, head, args.backbone)


if __name__ == "__main__":
    main()
