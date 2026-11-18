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
"""
import json, os, sys

sys.path.insert(0, os.environ.get("ORGCX_REPO", "/home/carl/claude_complexity"))
import numpy as np
import torch
from orgcx.relative import omega_rel

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
print("READY", flush=True)
for line in sys.stdin:
    parts = line.split()
    if not parts:
        continue
    try:
        path, rest = parts[0], parts[1:]
        shape = [int(p) for p in rest if "=" not in p]
        opts = dict(p.split("=", 1) for p in rest if "=" in p)
        x = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(shape)
        r = omega_rel(torch.as_tensor(x)[None], n_twins=int(opts.get("twins", 3)), seed=int(opts.get("seed", 0)))
        out = {"omega": float(r.omega), "rel": float(r.rel), "d_pv": float(r.d_pv), "d_omega": float(r.d_omega)}
    except Exception as e:  # one bad field must not kill the GA's scorer
        out = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out), flush=True)
