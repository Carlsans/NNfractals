#!/usr/bin/env python3
"""Batch complex-AE reconstruction-error scorer — mirrors
`score_vae_corpus.py`'s job (load a model once, score a corpus, write a
JSONL manifest of `{stem, path, recon_mse}`) but for the complex model
(`complex_autoencoder.py`) on `{stem}_re.png`/`{stem}_im.png` pairs
instead of the real-valued escape-time `*_raw.png` tensor.

Built for downstream validation (project-complex-nn-weekend-research.md
item #7): now that the complex AE's reconstruction quality is understood
(sharp on structured content, blurry on high-entropy noise — a real
floor, not a bug), the open question is whether ITS reconstruction-error
ranking is actually USEFUL as a novelty/interest signal — the same role
the real-valued VAE's reconstruction error already plays in
`explorer vae-explore`'s zone selection. Rank a real corpus by complex
reconstruction error and look at the top/bottom results the same way
`vae-curate` does for the real pipeline.

Usage:
  python3 scripts/score_complex_corpus.py --dirs explorer_out/weekend_complex_corpus/mandelbrot \
      --model-path explorer_out/complex_ae/complex_ae_full_corpus.pt \
      --out explorer_out/complex_ae/mandelbrot_complex_manifest.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from complex_autoencoder import ComplexAutoEncoder, RES
from train_complex_autoencoder import gather_pairs, ComplexPairDataset


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def progress(phase, done, total):
    print(f"PROGRESS {phase} {done} {total}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading model '{args.model_path}' on {device}…")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    in_ch = ckpt.get("in_ch", 1)
    model = ComplexAutoEncoder(latent_dim=ckpt["latent_dim"], in_ch=in_ch,
                                norm=ckpt.get("norm", "groupnorm"),
                                residual=ckpt.get("residual", False),
                                variational=ckpt.get("variational", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log(f"model: latent_dim={ckpt['latent_dim']} norm={ckpt.get('norm', 'groupnorm')} in_ch={in_ch} "
        f"loss={ckpt.get('loss', 'mse')} variational={ckpt.get('variational', False)}")

    include_et = in_ch == 2
    pairs = gather_pairs(args.dirs, max_images=-1, seed=0, include_escape_time=include_et)
    log(f"{len(pairs)} pairs found across {args.dirs}")
    if not pairs:
        raise SystemExit("no matching pairs found — nothing to score.")

    tf = transforms.Compose([transforms.Resize((RES, RES)), transforms.ToTensor()])
    ds = ComplexPairDataset(pairs, tf, include_escape_time=include_et)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    results = []
    i = 0
    with torch.no_grad():
        for re, im in dl:
            re, im = re.to(device), im.to(device)
            recon_re, recon_im = model.reconstruct_deterministic((re, im))
            per_image_mse = ((recon_re - re) ** 2 + (recon_im - im) ** 2).flatten(1).mean(dim=1)
            for mse in per_image_mse.tolist():
                re_path = pairs[i][0]
                stem = Path(re_path).stem.removesuffix("_re")
                results.append({"stem": stem, "path": re_path, "recon_mse": mse})
                i += 1
            if i % 200 < args.batch_size:
                progress("score", i, len(pairs))

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    mean = sum(r["recon_mse"] for r in results) / len(results)
    log(f"scored {len(results)} images -> {args.out}")
    print(f"RECON_MSE {mean:.6f}", flush=True)
    print(f"DONE {len(results)}", flush=True)


if __name__ == "__main__":
    main()
