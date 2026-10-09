"""Shared view-grid pooling for the quat taste model (trainer + scorer sidecar).

The stored view set is `N_ANGLES` orbit angles x `N_C` C values, file index
angle-major (`idx = angle * N_C + c`) -- must match
`nnfractals::quat_live_view::{VIEW_ANGLES, VIEW_C_VALUES}` and
`explorer.rs::render_genome_views`.
"""
import numpy as np

N_ANGLES = 6
N_C = 4
N_VIEWS = N_ANGLES * N_C


def pool_grid(arr):
    """arr: (N_VIEWS, D) per-view embeddings -> (3*D,) feature:
    [mean over all views, max over all views, mean over angles of the std
    across the C axis]. The last block is the "time" (C) variation: how much
    the look changes as C sweeps, which a mean/max alone cannot see."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] != N_VIEWS:
        raise ValueError(f"grid pooling needs {N_VIEWS} views, got {arr.shape[0]}")
    grid = arr.reshape(N_ANGLES, N_C, -1)
    c_std = grid.std(axis=1).mean(axis=0)
    return np.concatenate([arr.mean(axis=0), arr.max(axis=0), c_std]).astype(np.float32)


def grid_mean_block(vec):
    """The L2-normalised mean block of a pooled grid feature -- used by the
    sidecar for EMBED/DIST, whose consumers expect a plain backbone-sized
    embedding."""
    d = len(vec) // 3
    m = np.asarray(vec[:d], dtype=np.float32)
    n = np.linalg.norm(m)
    return m / n if n > 1e-8 else m
