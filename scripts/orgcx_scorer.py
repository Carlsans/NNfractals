#!/usr/bin/env python3
"""Sidecar for the GA terms `orgcx_rel` / `orgcx_omega` (explorer.rs, `orgcx_scores`).

Scores a field with Carl's organised-complexity measure (/home/carl/claude_complexity, package
`orgcx`, read-only; ORGCX_REPO overrides the path): Omega (orgcx.core.omega) and the
twin-relative score (orgcx.relative.omega_rel: Omega with the organised-novelty part Pv
replaced by its excess over IAAFT twins, i.e. random fields with the same power spectrum and
histogram).

Protocol (one line each way): after start it prints READY. Each request is
    <path to raw little-endian float32 file> <n1> <n2> ... [twins=N] [seed=S]
and the reply is one JSON object {"omega":…, "rel":…, "d_pv":…, "d_omega":…} or {"error": "…"}.
Several such specs joined by " ; " are scored on parallel threads (torch releases the GIL in
its kernels; the IAAFT twins run on CUDA when there is room) and answered with a JSON list
of those objects, in request order.
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.environ.get("ORGCX_REPO", "/home/carl/claude_complexity"))
import numpy as np
import torch
from orgcx.relative import omega_rel

torch.set_num_threads(int(os.environ.get("ORGCX_THREADS", max(1, min(8, os.cpu_count() or 1)))))


def score(parts):
    try:
        path, rest = parts[0], parts[1:]
        shape = [int(p) for p in rest if "=" not in p]
        opts = dict(p.split("=", 1) for p in rest if "=" in p)
        x = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(shape)
        r = omega_rel(torch.as_tensor(x)[None], n_twins=int(opts.get("twins", 3)), seed=int(opts.get("seed", 0)))
        return {"omega": float(r.omega), "rel": float(r.rel), "d_pv": float(r.d_pv), "d_omega": float(r.d_omega)}
    except Exception as e:  # one bad field must not kill the GA's scorer
        return {"error": f"{type(e).__name__}: {e}"}


pool = ThreadPoolExecutor(max_workers=int(os.environ.get("ORGCX_BATCH_THREADS", 3)))
print("READY", flush=True)
for line in sys.stdin:
    specs = [s.split() for s in line.split(";") if s.split()]
    if not specs:
        continue
    out = score(specs[0]) if len(specs) == 1 else list(pool.map(score, specs))
    print(json.dumps(out), flush=True)
