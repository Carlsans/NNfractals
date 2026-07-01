#!/usr/bin/env python3
"""Add extra aesthetic / quality scores to NNfractals .nn genome files.

Each .nn is JSON. We score its sibling .png with a chosen model and write a new
top-level field (e.g. "ap25_score", "topiq_iaa"). Rust's serde ignores unknown
fields, so this is safe; the schema-agnostic browser surfaces each new field as a
sortable column automatically.

The script is idempotent / resumable: a file that already has a non-zero value
for the target field is skipped unless --overwrite is given.

Backends
  pyiqa:<metric>   any pyiqa metric, e.g. pyiqa:topiq_iaa, pyiqa:clipiqa+,
                   pyiqa:nima, pyiqa:musiq, pyiqa:maniqa, pyiqa:qalign
  ap25             Aesthetic Predictor v2.5 (SigLIP)  → field ap25_score

Usage
  python3 scripts/score_extra.py --dir fractals_dag --model pyiqa:topiq_iaa
  python3 scripts/score_extra.py --dir fractals      --model ap25
  python3 scripts/score_extra.py --dir fractals_dag --model pyiqa:qalign --limit 400
"""
import argparse
import json
import sys
import time
from pathlib import Path


def build_scorer(model: str):
    """Return (score_fn(png_path)->float, default_field_name)."""
    if model.startswith("pyiqa:"):
        import torch
        import pyiqa
        name = model.split(":", 1)[1]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        metric = pyiqa.create_metric(name, device=dev)
        metric.eval() if hasattr(metric, "eval") else None
        field = name.replace("+", "_plus").replace("-", "_")

        def score(path):
            with torch.no_grad():
                return float(metric(str(path)).item())

        return score, field

    if model == "ap25":
        import torch
        from PIL import Image
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        m, prep = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        m = m.to(dtype).to(dev).eval()

        def score(path):
            img = Image.open(path).convert("RGB")
            pv = prep(images=img, return_tensors="pt").pixel_values.to(dtype).to(dev)
            with torch.inference_mode():
                return float(m(pv).logits.squeeze().float().cpu().item())

        return score, "ap25_score"

    raise SystemExit(f"unknown model: {model!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--field", default=None, help="override the output field name")
    ap.add_argument("--limit", type=int, default=0, help="stop after N newly scored")
    ap.add_argument("--overwrite", action="store_true", help="re-score even if present")
    args = ap.parse_args()

    score, default_field = build_scorer(args.model)
    field = args.field or default_field
    print(f"[score_extra] model={args.model} → field '{field}'  dir={args.dir}",
          file=sys.stderr, flush=True)

    nns = sorted(Path(args.dir).glob("*.nn"))
    done = skipped = nopng = failed = 0
    t0 = time.time()
    for p in nns:
        png = p.with_suffix(".png")
        if not png.exists():
            nopng += 1
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            failed += 1
            continue
        if not args.overwrite and d.get(field) not in (None, 0, 0.0):
            skipped += 1
            continue
        try:
            v = score(png)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  fail {png.name}: {e}", file=sys.stderr, flush=True)
            continue
        d[field] = round(v, 5)
        p.write_text(json.dumps(d, indent=2))
        done += 1
        if done % 200 == 0:
            rate = done / max(1e-9, time.time() - t0)
            print(f"  {done} scored ({rate:.1f}/s)…", file=sys.stderr, flush=True)
        if args.limit and done >= args.limit:
            break

    dt = time.time() - t0
    print(f"[score_extra] DONE {args.model}: {done} scored, {skipped} already had value, "
          f"{nopng} missing png, {failed} failed — of {len(nns)} .nn in {args.dir} ({dt:.0f}s)",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
