#!/usr/bin/env python3
"""Selects the N most MUTUALLY DIFFERENT zones in a corpus, using the
complex VAE's latent space as the feature representation instead of
reconstruction error (score_complex_corpus.py's job — "how hard is this
to reconstruct" — is a genuinely different question from "how different
is this from everything else already picked").

Each zone's embedding is `(mu_re, mu_im)` — the VAE's deterministic
posterior mean (never the noisy reparameterization sample, same
`model.encode()` convention every other complex-model script this
session uses), concatenated into one real vector of size `2*latent_dim`.
Concatenation rather than a genuinely complex distance metric is a
deliberate simplification: it loses nothing (every component of the
complex embedding is still present) and Euclidean distance on the
concatenated vector is a standard, well-understood diversity metric —
whether a Hermitian/complex-aware distance would select differently is
an open question, not answered here, see project-complex-nn-weekend-
research.md for the "why concatenation, not complex algebra" reasoning
already established for `curate_combined.py`'s analogous choice.

Selection is greedy farthest-point sampling (a.k.a. max-min diversity
sampling): start from the zone closest to the corpus centroid (a
reasonable, deterministic starting point — the "most average" zone),
then repeatedly add whichever remaining zone is farthest from its
nearest already-selected neighbor. Classic, simple, O(N * top_n).

Reuses `explorer vae-curate` for the actual high-res render (same
scratch-dir symlink + synthetic-manifest trick `curate_combined.py`
already uses) rather than reimplementing rendering.

Usage:
  python3 scripts/select_diverse_latent.py \
      --model-path explorer_out/complex_ae/complex_vae_tuned.pt \
      --dirs explorer_out/weekend_complex_corpus/mandelbrot \
      --pool-dir explorer_out/mandelbrot_vae \
      --out-dir explorer_out/mandelbrot_vae/diverse_selection \
      --top-n 30
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from complex_autoencoder import ComplexAutoEncoder, RES
from train_complex_autoencoder import gather_pairs, ComplexPairDataset


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def farthest_point_sample(embeddings, top_n):
    """Greedy max-min diversity sampling. `embeddings`: (N, D) array.
    Returns indices of the selected `top_n` rows, in selection order
    (index 0 = the seed/"most average" zone, not necessarily the most
    interesting — the LAST few picks are the ones farthest from
    everything else, often the most visually distinct)."""
    n = embeddings.shape[0]
    top_n = min(top_n, n)
    centroid = embeddings.mean(axis=0, keepdims=True)
    seed = int(np.argmin(((embeddings - centroid) ** 2).sum(axis=1)))

    selected = [seed]
    min_dist = np.full(n, np.inf)
    min_dist = np.minimum(min_dist, ((embeddings - embeddings[seed]) ** 2).sum(axis=1))
    min_dist[seed] = -np.inf  # never re-select

    for _ in range(top_n - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d = ((embeddings - embeddings[nxt]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[nxt] = -np.inf
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="complex-export re/im pair directories")
    ap.add_argument("--pool-dir", required=True, help="directory holding the matching zone_NNNN.nn files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--res", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scratch", default="/tmp/select_diverse_latent_scratch")
    ap.add_argument("--explorer-bin", default="./target/release/explorer")
    ap.add_argument("--join-nn", action="store_true", default=True,
                     help="copy each selected zone's .nn genome alongside its render in --out-dir, "
                          "named to match (zone_01.png + zone_01.nn) — on by default")
    ap.add_argument("--no-join-nn", dest="join_nn", action="store_false")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading model '{args.model_path}' on {device}…")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if not ckpt.get("variational", False):
        raise SystemExit("this model isn't a VAE (no variational latent) — diversity selection "
                          "needs a genuine posterior mean, pass a --variant vae checkpoint.")
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
            emb = torch.cat([mu_re, mu_im], dim=1).cpu().numpy()
            embeddings.append(emb)
            for _ in range(re.size(0)):
                re_path = pairs[i][0]
                stems.append(Path(re_path).stem.removesuffix("_re"))
                i += 1
            if i % 200 < args.batch_size:
                log(f"embedded {i}/{len(pairs)}")
    embeddings = np.concatenate(embeddings, axis=0)
    log(f"embeddings shape: {embeddings.shape}")

    selected_idx = farthest_point_sample(embeddings, args.top_n)
    selected_stems = [stems[i] for i in selected_idx]
    log(f"selected {len(selected_stems)} maximally-diverse zones (seed: {selected_stems[0]})")

    scratch = Path(args.scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    pool_dir = Path(args.pool_dir)
    missing = 0
    with open(scratch / "vae_recon_manifest.jsonl", "w") as mf:
        for rank_i, stem in enumerate(selected_stems):
            nn_src = pool_dir / f"{stem}.nn"
            if not nn_src.exists():
                missing += 1
                continue
            (scratch / f"{stem}.nn").symlink_to(nn_src.resolve())
            mf.write(json.dumps({"stem": stem, "path": str(nn_src), "recon_mse": -rank_i}) + "\n")
    if missing:
        log(f"warning: {missing} selected zones had no matching .nn file in {pool_dir}, skipped")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.explorer_bin, "vae-curate", str(scratch), str(len(selected_stems)), str(out_dir), str(args.res)]
    log(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if args.join_nn:
        # vae-curate names output zone_{rank+1:02d}.png — copy each
        # selected zone's .nn alongside its render under the SAME name, so
        # out_dir is self-contained (image + genome pairs) instead of the
        # genome only existing in the throwaway --scratch dir.
        joined = 0
        for rank_i, stem in enumerate(selected_stems):
            nn_src = pool_dir / f"{stem}.nn"
            if not nn_src.exists():
                continue
            shutil.copy2(nn_src, out_dir / f"zone_{rank_i + 1:02d}.nn")
            joined += 1
        log(f"joined {joined} .nn files into {out_dir}")

    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
