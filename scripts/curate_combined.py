#!/usr/bin/env python3
"""Combined-ranking curation: merges a real-valued VAE manifest and a
complex-AE manifest for the SAME pool via RANK-based combination (see
project-complex-nn-weekend-research.md item #7's documented gotcha — a
naive z-score combination silently buried the complex model's genuine
top pick because the two signals' error distributions have different
shapes; rank-based combination is robust to that).

Reuses the existing `explorer vae-curate` Rust command for the actual
high-res re-render (rather than reimplementing rendering in Python):
symlinks just the selected top-N `.nn` files into a scratch directory
alongside a SYNTHETIC `vae_recon_manifest.jsonl` (same schema, combined
score standing in for `recon_mse`), then calls `vae-curate` against that
scratch directory. Doesn't touch or overwrite either real manifest.

Usage:
  python3 scripts/curate_combined.py \
      --real-manifest explorer_out/mandelbrot_vae/vae_recon_manifest.jsonl \
      --complex-manifest explorer_out/complex_ae/mandelbrot_complex_manifest_ld256.jsonl \
      --pool-dir explorer_out/mandelbrot_vae \
      --out-dir explorer_out/mandelbrot_vae/curated_combined \
      --top-n 30
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_manifest(path):
    d = {}
    for line in open(path):
        r = json.loads(line)
        d[r["stem"]] = r["recon_mse"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-manifest", required=True)
    ap.add_argument("--complex-manifest", required=True)
    ap.add_argument("--pool-dir", required=True, help="directory holding the zone_NNNN.nn files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--res", type=int, default=4000)
    ap.add_argument("--scratch", default="/tmp/curate_combined_scratch")
    ap.add_argument("--explorer-bin", default="./target/release/explorer")
    args = ap.parse_args()

    real = load_manifest(args.real_manifest)
    complex_ = load_manifest(args.complex_manifest)
    common = sorted(set(real) & set(complex_))
    if not common:
        raise SystemExit("no stems in common between the two manifests — wrong pool/formula pairing?")
    log(f"{len(common)} zones in both manifests")

    real_rank = {s: i for i, s in enumerate(sorted(common, key=lambda s: real[s], reverse=True))}
    complex_rank = {s: i for i, s in enumerate(sorted(common, key=lambda s: complex_[s], reverse=True))}
    # best rank in EITHER signal — a zone that's #1 by either measure
    # ranks #0 combined, regardless of how the other signal treats it.
    combined_rank = {s: min(real_rank[s], complex_rank[s]) for s in common}
    ranked = sorted(common, key=lambda s: combined_rank[s])[:args.top_n]
    log(f"top {len(ranked)} by combined (best-of-either-rank): "
        + ", ".join(f"{s}(r{real_rank[s]}/c{complex_rank[s]})" for s in ranked[:5]) + " ...")

    scratch = Path(args.scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    pool_dir = Path(args.pool_dir)
    missing = 0
    with open(scratch / "vae_recon_manifest.jsonl", "w") as mf:
        for rank_i, stem in enumerate(ranked):
            nn_src = pool_dir / f"{stem}.nn"
            if not nn_src.exists():
                missing += 1
                continue
            (scratch / f"{stem}.nn").symlink_to(nn_src.resolve())
            # Synthetic recon_mse: monotonically decreasing by rank so
            # vae-curate's own MaxError sort reproduces this exact order.
            mf.write(json.dumps({"stem": stem, "path": str(nn_src), "recon_mse": -rank_i}) + "\n")
    if missing:
        log(f"warning: {missing} ranked zones had no matching .nn file in {pool_dir}, skipped")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.explorer_bin, "vae-curate", str(scratch), str(len(ranked)), str(out_dir), str(args.res)]
    log(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
