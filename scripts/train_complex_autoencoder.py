#!/usr/bin/env python3
"""Trains `ComplexAutoEncoder` (see complex_autoencoder.py) on
`{stem}_re.png`/`{stem}_im.png` pairs from `explorer complex-export` —
the bailout z value, not the escape-time tensor `train_autoencoder.py`
trains on. Same CLI shape/conventions as `train_autoencoder.py` (this
project's own script, not a from-scratch design) minus the flags that
don't apply here (--variant/--arch/--res/--channels/--kl-weight: this
model is AE-only, one architecture, fixed 512px/1 complex channel).

Usage:
  python3 scripts/train_complex_autoencoder.py \
      --dirs explorer_out/complex_export_sample explorer_out/novelty_complex_export \
      --epochs 15 --out complex_ae_model.pt
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud
from PIL import Image
from scipy.spatial import cKDTree
from torchvision import transforms
from torchvision.utils import save_image

from complex_autoencoder import (ComplexAutoEncoder, complex_mse_loss, complex_kl_loss,
                                  complex_triplet_loss, multiscale_complex_mse_loss, RES)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    print(f"PROGRESS {phase} {done} {total}", flush=True)


def gather_pairs(dirs, max_images, seed, include_escape_time=False):
    """Every `*_re.png` with a matching `*_im.png` alongside it — same
    directory layout `explorer complex-export` writes
    (`{stem}_tensor/re/im/mag.png`), just picking out the two files this
    model actually trains on.

    `include_escape_time=True` also requires the matching `{stem}_tensor.png`
    (the plain escape-time field `explorer complex-export` already writes
    alongside re/im/mag — see `render_escape_times` in Rust) and returns
    3-tuples instead of pairs. Carl's diagnosis (2026-08-08): the complex
    bailout value `(zx,zy)` can be near-uniform across a broad "already
    escaped" region even where escape-TIME (iteration count) varies a lot —
    that's literally what the classic escape-time color bands visualize —
    so a latent space built from `(zx,zy)` alone is blind to exactly that
    structure. Packed as a second complex channel with a zero imaginary
    part (see `ComplexPairDataset`), not a separate real trunk — reuses
    every existing complex layer unchanged since they already generalize
    over `in_ch`."""
    stems = []
    for d in dirs:
        for re_path in glob.glob(f"{d}/*_re.png"):
            im_path = re_path[:-len("_re.png")] + "_im.png"
            if not Path(im_path).exists():
                continue
            if include_escape_time:
                et_path = re_path[:-len("_re.png")] + "_tensor.png"
                if not Path(et_path).exists():
                    continue
                stems.append((re_path, im_path, et_path))
            else:
                stems.append((re_path, im_path))
    random.Random(seed).shuffle(stems)
    if max_images > 0:
        stems = stems[:max_images]
    return stems


def _load_triple(triple, tf, include_escape_time):
    """Loads one zone's re/im(+escape-time) tensors — factored out of
    `ComplexPairDataset.__getitem__` so `TripletPairDataset` can load an
    anchor's positive partner with the exact same logic without
    duplicating it."""
    try:
        if include_escape_time:
            re_path, im_path, et_path = triple
            re = tf(Image.open(re_path).convert("L"))
            im = tf(Image.open(im_path).convert("L"))
            et = tf(Image.open(et_path).convert("L"))
            return torch.cat([re, et], dim=0), torch.cat([im, torch.zeros_like(et)], dim=0)
        re_path, im_path = triple
        re = tf(Image.open(re_path).convert("L"))
        im = tf(Image.open(im_path).convert("L"))
        return re, im
    except Exception:
        ch = 2 if include_escape_time else 1
        z = torch.zeros(ch, RES, RES)
        return z, z


class ComplexPairDataset(tud.Dataset):
    """`include_escape_time=True`: `re`/`im` come back as 2-channel tensors,
    channel 0 = the z-value part (unchanged), channel 1 = escape-time
    packed into the REAL part with the IMAGINARY part fixed at zero (a
    genuinely real scalar embedded as a complex number with `im=0` —
    architecturally free since `ComplexConv2d` et al. already generalize
    over `in_ch`, at the cost of a minor semantic wart: half of that
    channel's complex "bandwidth" is structurally unused)."""
    def __init__(self, pairs, tf, include_escape_time=False):
        self.pairs = pairs
        self.tf = tf
        self.include_escape_time = include_escape_time

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return _load_triple(self.pairs[i], self.tf, self.include_escape_time)


def _stem(triple):
    return Path(triple[0]).stem.removesuffix("_re")


def _match_pool_dir(re_path, dirs, pool_dirs):
    """Which `--pool-dir` a zone belongs to, by which `--dirs` entry its
    `re_path` sits under (positionally matched, same order/length as
    `--dirs`/`--pool-dir`)."""
    rp = str(Path(re_path).resolve())
    for d, p in zip(dirs, pool_dirs):
        if rp.startswith(str(Path(d).resolve())):
            return str(Path(p).resolve())
    return None


def compute_positive_pairs(pairs, dirs, pool_dirs):
    """For each zone, finds its nearest same-POOL neighbor by
    `(cx, cy, log10(zoom))` parameter distance — the free, self-supervised
    positive-pair signal for `--triplet-weight` (2026-08-08, Carl's
    request for a triplet-loss VAE variant). Every zone in a pool is a
    crop of the SAME single genome (`cmd_pool` builds one genome, drills
    many `(cx,cy,zoom)` views from it), so parameter-adjacent crops are a
    reasonable proxy for "probably visually similar" — no manual labeling
    or extra rendering needed, purely derived from each zone's already-
    saved `.nn` file. zoom compared in log-space since it spans orders of
    magnitude; all three axes z-scored before the nearest-neighbor search
    so none dominates purely from having a larger numeric range.

    The full training corpus spans MULTIPLE pools (one per formula/
    genome) with the same `.nn` stem convention in each (`zone_0000`
    exists in every pool) — searching for a "nearest neighbor" across
    pools would be meaningless (different formulas' `cx`/`cy`/`zoom`
    ranges aren't comparable), and a bare-stem lookup would silently
    collide (Mandelbrot's `zone_0000` overwriting Burning Ship's).
    Grouped by pool here, and keyed `(pool_dir, stem)` throughout — never
    bare `stem` — for exactly that reason."""
    groups = {}
    for triple in pairs:
        pd = _match_pool_dir(triple[0], dirs, pool_dirs)
        if pd is not None:
            groups.setdefault(pd, []).append(triple)

    positive_map = {}
    for pd, group_pairs in groups.items():
        pool_path = Path(pd)
        stems, feats = [], []
        for triple in group_pairs:
            stem = _stem(triple)
            nn_path = pool_path / f"{stem}.nn"
            if not nn_path.exists():
                continue
            try:
                g = json.loads(nn_path.read_text())
                cx, cy, zoom = float(g["view_cx"]), float(g["view_cy"]), float(g["view_zoom"])
            except Exception:
                continue
            stems.append(stem)
            feats.append([cx, cy, np.log10(max(zoom, 1e-9))])
        if len(stems) < 2:
            continue
        feats = np.array(feats)
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
        tree = cKDTree(feats)
        _, idxs = tree.query(feats, k=2)  # k=1 is self, k=2 is nearest other
        for i, stem in enumerate(stems):
            positive_map[(pd, stem)] = (pd, stems[idxs[i, 1]])
    return positive_map


class TripletPairDataset(tud.Dataset):
    """Wraps `_load_triple` to additionally return each anchor's positive
    partner (from `compute_positive_pairs`) — used only when
    `--triplet-weight > 0`. `stem_to_triple` is built over the FULL pool
    (before the train/val split) so a positive can still be located even
    if it happens to fall in the val split — loading its image doesn't
    leak any label/target information, it's the same kind of input data
    either way. Keyed `(pool_dir, stem)`, see `compute_positive_pairs`."""
    def __init__(self, pairs, dirs, pool_dirs, stem_to_triple, positive_map, tf, include_escape_time=False):
        self.pairs = pairs
        self.dirs = dirs
        self.pool_dirs = pool_dirs
        self.stem_to_triple = stem_to_triple
        self.positive_map = positive_map
        self.tf = tf
        self.include_escape_time = include_escape_time

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        triple = self.pairs[i]
        anchor_re, anchor_im = _load_triple(triple, self.tf, self.include_escape_time)
        pd = _match_pool_dir(triple[0], self.dirs, self.pool_dirs)
        key = (pd, _stem(triple)) if pd is not None else None
        pos_key = self.positive_map.get(key) if key is not None else None
        pos_triple = self.stem_to_triple.get(pos_key) if pos_key is not None else None
        if pos_triple is not None:
            pos_re, pos_im = _load_triple(pos_triple, self.tf, self.include_escape_time)
        else:
            pos_re, pos_im = anchor_re, anchor_im
        return anchor_re, anchor_im, pos_re, pos_im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--norm", choices=["groupnorm", "whitened"], default="groupnorm",
                     help="groupnorm = proven-working default; whitened = proper covariance-whitening "
                          "complex BatchNorm (Trabelsi et al.), an open comparison")
    ap.add_argument("--loss", choices=["mse", "multiscale"], default="mse",
                     help="mse = plain per-pixel complex MSE (default); multiscale = also penalize "
                          "error at downsampled resolutions, counters plain-MSE's bias toward blur. "
                          "Held-out val MSE is always reported via plain mse regardless, for "
                          "comparability across runs — this flag only changes the TRAINING loss.")
    ap.add_argument("--residual", action="store_true",
                     help="add a ComplexResBlock before each down/up-sample stage")
    ap.add_argument("--include-escape-time", action="store_true",
                     help="pack the escape-time field in as a 2nd complex channel (im=0) alongside "
                          "the bailout z value — see gather_pairs' docstring for why")
    ap.add_argument("--variant", choices=["ae", "vae"], default="ae",
                     help="ae = plain deterministic bottleneck (default); vae = two independent "
                          "real-Gaussian latents (real part, imaginary part) — see ComplexEncoder's "
                          "docstring for the design rationale")
    ap.add_argument("--kl-weight", type=float, default=1e-3, help="vae only; ignored for --variant ae")
    ap.add_argument("--triplet-weight", type=float, default=0.0,
                     help="0 (default) disables. >0 adds a triplet-margin loss on the (mu_re,mu_im) "
                          "embedding, pulling parameter-adjacent zones together in latent space — "
                          "see compute_positive_pairs' docstring. Needs --pool-dir.")
    ap.add_argument("--triplet-margin", type=float, default=1.0)
    ap.add_argument("--pool-dir", nargs="+", default=None,
                     help="directory/directories holding each zone's .nn, positionally matched to "
                          "--dirs (same order/length) — for --triplet-weight's positive-pair search, "
                          "kept per-pool since formulas' cx/cy/zoom ranges aren't comparable")
    ap.add_argument("--max-images", type=int, default=20000)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Sized for a small feasibility-check corpus (a handful of exported
    # genomes so far, not vae-explore's thousands of zones) — same
    # rationale as train_autoencoder.py's own --min-images/--min-val, just
    # smaller defaults since this corpus starts smaller.
    ap.add_argument("--min-images", type=int, default=4)
    ap.add_argument("--min-val", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--out", default="complex_ae_model.pt")
    ap.add_argument("--contact-sheet", default="complex_ae_recon.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 2 if args.include_escape_time else 1
    log(f"gathering re/im{'+escape-time' if args.include_escape_time else ''} pairs from {args.dirs}…")
    pairs = gather_pairs(args.dirs, args.max_images, args.seed, include_escape_time=args.include_escape_time)
    if len(pairs) < args.min_images:
        raise SystemExit(f"only {len(pairs)} pairs found across {args.dirs} — need at least {args.min_images} (--min-images) to train an encoder.")
    n_val = max(args.min_val, int(len(pairs) * args.val_fraction))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    batch_size = max(1, min(args.batch_size, len(train_pairs)))
    if batch_size != args.batch_size:
        log(f"--batch-size {args.batch_size} > {len(train_pairs)} train pairs — clamped to {batch_size}")
    log(f"{len(train_pairs)} train / {len(val_pairs)} val pairs (res {RES}, latent_dim {args.latent_dim}, in_ch {in_ch})")

    use_triplet = args.triplet_weight > 0
    positive_map, stem_to_triple = {}, {}
    if use_triplet:
        if not args.pool_dir:
            raise SystemExit("--triplet-weight > 0 needs --pool-dir (to read each zone's .nn "
                              "for parameter-proximity positive pairs)")
        if len(args.pool_dir) != len(args.dirs):
            raise SystemExit(f"--pool-dir has {len(args.pool_dir)} entries but --dirs has "
                              f"{len(args.dirs)} — they must be positionally matched, one pool-dir per corpus dir")
        stem_to_triple = {}
        for t in pairs:
            pd = _match_pool_dir(t[0], args.dirs, args.pool_dir)
            if pd is not None:
                stem_to_triple[(pd, _stem(t))] = t
        positive_map = compute_positive_pairs(pairs, args.dirs, args.pool_dir)
        log(f"triplet: {len(positive_map)}/{len(pairs)} zones matched a positive partner via parameter proximity")

    tf = transforms.Compose([transforms.Resize((RES, RES)), transforms.ToTensor()])
    if use_triplet:
        train_ds = TripletPairDataset(train_pairs, args.dirs, args.pool_dir, stem_to_triple, positive_map, tf,
                                       include_escape_time=args.include_escape_time)
    else:
        train_ds = ComplexPairDataset(train_pairs, tf, include_escape_time=args.include_escape_time)
    train_dl = tud.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.workers, drop_last=True)
    val_dl = tud.DataLoader(ComplexPairDataset(val_pairs, tf, include_escape_time=args.include_escape_time),
                             batch_size=batch_size, shuffle=False, num_workers=args.workers)

    variational = args.variant == "vae"
    model = ComplexAutoEncoder(latent_dim=args.latent_dim, in_ch=in_ch, norm=args.norm,
                                residual=args.residual, variational=variational).to(device)
    if args.init_from:
        try:
            ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
            compatible = (ckpt.get("latent_dim") == args.latent_dim and ckpt.get("res", RES) == RES
                          and ckpt.get("in_ch", 1) == in_ch)
            if compatible:
                model.load_state_dict(ckpt["state_dict"])
                log(f"warm-started from {args.init_from}")
            else:
                log(f"--init-from {args.init_from}: architecture mismatch, ignoring (random init)")
        except Exception as e:
            log(f"--init-from {args.init_from}: failed to load ({e}), ignoring (random init)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_dl)
    for epoch in range(args.epochs):
        model.train()
        running, running_recon, running_kl, running_trip = 0.0, 0.0, 0.0, 0.0
        for i, batch in enumerate(train_dl):
            if use_triplet:
                re, im, pos_re, pos_im = batch
                pos_re, pos_im = pos_re.to(device), pos_im.to(device)
            else:
                re, im = batch
            re, im = re.to(device), im.to(device)
            recon, mu_re, logvar_re, mu_im, logvar_im = model((re, im))
            recon_loss = multiscale_complex_mse_loss(recon, (re, im)) if args.loss == "multiscale" else complex_mse_loss(recon, (re, im))
            if variational:
                kl = complex_kl_loss(mu_re, logvar_re, mu_im, logvar_im)
                loss = recon_loss + args.kl_weight * kl
                anchor_mu_re, anchor_mu_im = mu_re, mu_im
            else:
                kl = torch.tensor(0.0)
                loss = recon_loss
                anchor_mu_re, anchor_mu_im = None, None

            if use_triplet:
                if anchor_mu_re is None:  # plain AE: forward() doesn't expose the bottleneck, encode() does
                    anchor_mu_re, anchor_mu_im = model.encode((re, im))
                pos_mu_re, pos_mu_im = model.encode((pos_re, pos_im))
                trip = complex_triplet_loss(anchor_mu_re, anchor_mu_im, pos_mu_re, pos_mu_im, margin=args.triplet_margin)
                loss = loss + args.triplet_weight * trip
            else:
                trip = torch.tensor(0.0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item(); running_recon += recon_loss.item(); running_kl += kl.item(); running_trip += trip.item()
            if (i + 1) % 20 == 0:
                progress(f"train_epoch{epoch}", i + 1, steps_per_epoch)
        n = len(train_dl)
        log(f"epoch {epoch+1}/{args.epochs}  loss={running/n:.5f}  recon={running_recon/n:.5f}  "
            f"kl={running_kl/n:.5f}  trip={running_trip/n:.5f}")

    # ── Held-out reconstruction error — reconstruct_deterministic bypasses
    # the VAE's stochastic sample, so this is comparable across AE/VAE and
    # doesn't flip between identical calls from sampling noise ──
    model.eval()
    val_total, val_n = 0.0, 0
    with torch.no_grad():
        for re, im in val_dl:
            re, im = re.to(device), im.to(device)
            recon = model.reconstruct_deterministic((re, im))
            val_total += complex_mse_loss(recon, (re, im)).item() * re.size(0)
            val_n += re.size(0)
    val_mse = val_total / val_n
    log(f"held-out val complex reconstruction MSE: {val_mse:.6f}")
    print(f"RECON_MSE {val_mse:.6f}", flush=True)

    # ── Contact sheet: magnitude + re + im (channel 0, the z value only —
    # channel 1 if present is escape-time, shown as its own row below),
    # original vs reconstruction ──
    with torch.no_grad():
        re, im = next(iter(val_dl))
        re, im = re[:8].to(device), im[:8].to(device)
        recon_re, recon_im = model.reconstruct_deterministic((re, im))
        z_re, z_im = re[:, 0:1], im[:, 0:1]
        recon_z_re, recon_z_im = recon_re[:, 0:1], recon_im[:, 0:1]
        mag = (z_re * z_re + z_im * z_im).sqrt()
        recon_mag = (recon_z_re * recon_z_re + recon_z_im * recon_z_im).sqrt().clamp(0, 1)
        rows = [mag, recon_mag, z_re, recon_z_re, z_im, recon_z_im]
        if args.include_escape_time:
            rows += [re[:, 1:2], recon_re[:, 1:2]]
        save_image(torch.cat(rows, dim=0), args.contact_sheet, nrow=re.size(0))
    row_desc = "orig mag, recon mag, orig re, recon re, orig im, recon im" + \
        (", orig escape-time, recon escape-time" if args.include_escape_time else "")
    log(f"wrote contact sheet -> {args.contact_sheet} (rows: {row_desc})")

    torch.save({
        "state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "res": RES,
        "norm": args.norm,
        "loss": args.loss,
        "residual": args.residual,
        "variational": variational,
        "in_ch": in_ch,
        "triplet_weight": args.triplet_weight,
        "triplet_margin": args.triplet_margin,
    }, args.out)
    log(f"saved model -> {args.out}")


if __name__ == "__main__":
    main()
