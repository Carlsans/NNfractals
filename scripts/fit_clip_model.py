#!/usr/bin/env python3
"""Fit a FORMULA-ONLY linear model that predicts a genome's measured
clip_score from its formula alone (no rendering). Trained on the archive
of saved .nn genomes, each of which carries a measured clip_score plus the
sparse term set it was produced by.

Output: clip_model.json — read by the Rust GA (src/recursion_model.rs, same
struct) to cheaply bias evolution toward aesthetically preferred formula families
every generation, without any CLIP inference overhead.

Feature vector (length 61), MUST match Genome::recursion_features() in Rust:
  [0..58)  basis_mag[i] = Σ over terms with basis==i of hypot(re, im)
  58       num_terms
  59       total_mag    = Σ over all terms of hypot(re, im)
  60       c_zpow       = basis_mag[7] * (basis_mag[0]+basis_mag[1]+basis_mag[2]+basis_mag[3])
"""
import json, glob, math, sys, time
import numpy as np

N_BASIS  = 58
FEAT_DIM = N_BASIS + 3

def features(terms):
    f = np.zeros(FEAT_DIM, dtype=np.float64)
    for t in terms:
        b = int(t["basis"])
        if 0 <= b < N_BASIS:
            f[b] += math.hypot(t["re"], t["im"])
    f[58] = len(terms)
    f[59] = f[:N_BASIS].sum()
    f[60] = f[7] * (f[0] + f[1] + f[2] + f[3])
    return f

def load():
    X, y = [], []
    for p in glob.glob("fractals/*.nn"):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        terms     = d.get("terms")
        clip_val  = d.get("clip_score", 0.0)
        if not terms or clip_val <= 0.0:
            continue
        X.append(features(terms))
        y.append(float(clip_val))
    return np.array(X), np.array(y)

def ridge_fit(Xz, yc, lam):
    d = Xz.shape[1]
    A = Xz.T @ Xz + lam * np.eye(d)
    return np.linalg.solve(A, Xz.T @ yc)

def pearson(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def main():
    X, y = load()
    n = len(y)
    if n < 30:
        print(f"too few samples ({n}); aborting", file=sys.stderr)
        sys.exit(1)

    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Xz   = (X - mean) / std
    ymean = y.mean()
    yc    = y - ymean

    # CLIP scores vary in [0.51, 0.53] — much tighter range than recursion [0, 1].
    # Use stronger regularization to avoid overfitting to noise.
    lam = 10.0

    # 5-fold CV
    rng   = np.random.default_rng(0)
    idx   = rng.permutation(n)
    folds = np.array_split(idx, 5)
    preds = np.zeros(n)
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        w  = ridge_fit(Xz[tr], yc[tr], lam)
        preds[te] = Xz[te] @ w + ymean
    cv_r   = pearson(preds, y)
    cv_mae = float(np.abs(preds - y).mean())

    # Final fit on all data
    w          = ridge_fit(Xz, yc, lam)
    train_pred = Xz @ w + ymean
    train_r    = pearson(train_pred, y)

    model = {
        "version":      1,
        "trained_at":   time.time(),
        "n_samples":    n,
        "feature_dim":  FEAT_DIM,
        "lambda":       lam,
        "cv_pearson":   cv_r,
        "cv_mae":       cv_mae,
        "train_pearson": train_r,
        "label_mean":   float(ymean),
        "label_std":    float(y.std()),
        "mean":         mean.tolist(),
        "std":          std.tolist(),
        "weight":       w.tolist(),
        "bias":         float(ymean),
    }
    json.dump(model, open("clip_model.json", "w"), indent=1)

    top = sorted(range(N_BASIS), key=lambda i: -abs(w[i]))[:8]
    print(f"n={n}  cv_pearson={cv_r:.3f}  cv_mae={cv_mae:.4f}  train_pearson={train_r:.3f}")
    print(f"label: clip_score  mean={ymean:.4f}  std={y.std():.4f}  range=[{y.min():.4f},{y.max():.4f}]")
    print("top formula drivers of CLIP score (standardized weight):")
    for i in top:
        print(f"  basis {i:2d}  w={w[i]:+.4f}")
    print(f"  num_terms w={w[58]:+.4f}   total_mag w={w[59]:+.4f}   c·z_pow w={w[60]:+.4f}")

    if cv_r < 0.20:
        print(f"\nWARNING: cv_pearson={cv_r:.3f} is low — formula structure may not predict CLIP well.")
        print("Consider setting clip_pred_weight=0 in config.toml to disable the criterion.")
    else:
        print(f"\nModel is useful (cv_pearson={cv_r:.3f}). clip_pred_weight=0.50 in config.toml is active.")

if __name__ == "__main__":
    main()
