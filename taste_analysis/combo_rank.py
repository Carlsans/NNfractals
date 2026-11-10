"""Rank candidate fitness recipes by pairwise accuracy on Carl's S3 ratings.
Terms come from `explorer quat-metric-terms` (taste_analysis/s3_terms.csv). Gate-failing
genomes get the GA's floor (-5). Fitted weights are scored with 5-fold CV over genomes."""
import csv, json, os, itertools, numpy as np
rows = {r['file']: r for r in csv.DictReader(open('taste_analysis/s3_terms.csv'))}
T = [k for k in next(iter(rows.values())) if k not in ('file', 'gate')]
X = {f: np.array([float(r[k]) for k in T]) for f, r in rows.items()}
gate = {f: r['gate'] == '1' for f, r in rows.items()}
pairs = []
for l in open('train_corpus_quat_s3/ratings.jsonl'):
    r = json.loads(l); w, lo = os.path.basename(r['winner']), os.path.basename(r['loser'])
    if w in X and lo in X: pairs.append((w, lo))
print(len(rows), 'genomes', len(pairs), 'pairs', sum(gate.values()), 'pass gate')
idx = {k: i for i, k in enumerate(T)}
# z-scale so weights are comparable
A = np.array(list(X.values())); mu, sd = A.mean(0), A.std(0) + 1e-9
def fit(spec, f, use_gate=True):
    if use_gate and any(k in spec for k in ('image_entropy_band', 'image_motion')) and not gate[f]: return -5.0
    return sum(w * X[f][idx[k]] for k, w in spec.items())
def acc(spec, ps=pairs, use_gate=True):
    ok = tot = 0
    for w, l in ps:
        a, b = fit(spec, w, use_gate), fit(spec, l, use_gate)
        if abs(a - b) < 1e-12: ok += .5
        else: ok += a > b
        tot += 1
    return ok / tot
W = json.load(open('taste_analysis/weights.json'))
def half(rw):
    t = sum(abs(v) for v in rw.values()); d = {'image_entropy_band': t, 'image_motion': t * .5}; d.update(rw); return d
s3 = W['s3']; union = dict(s3); union.update({k: v for k, v in W['pool'].items() if k not in union})
LR = {'organization_continuity':0.74,'sphericity':-0.92,'organization_entropy':-0.27,'bilateral_symmetry':-9.56,'color_gradient':-0.88,'color_shading_corr':2.96,'silhouette_irregularity':0.27,'centroid_offset':-15.69,'color_band_autocorr':0.66,'cross_view_coverage_delta':2.44}
C = {'taste_only (reference, in-sample)': {'taste': 1}, 'entropy_only': {'image_entropy_band': 1, 'image_motion': .5},
     'H1_s3_rho25': half(s3), 'H2_union_rho25': half(union), 'H3_lr': half({k: v / sum(abs(x) for x in LR.values()) for k, v in LR.items()})}
# rho sign from univariate pairwise accuracy
uni = {k: acc({k: 1}, use_gate=False) for k in T if k != 'taste'}
for k, a in sorted(uni.items(), key=lambda x: -abs(x[1] - .5)):
    if abs(a - .5) >= .07 and k not in ('image_entropy_band', 'image_motion'):
        sgn = 1 if a > .5 else -1
        C[f'ent+{k}'] = {'image_entropy_band': 1, 'image_motion': .5, k: sgn * 1.5 / sd[idx[k]]}
# top-k univariate sum (z-scaled) with entropy half
top = [k for k, a in sorted(uni.items(), key=lambda x: -abs(x[1] - .5)) if k not in ('image_entropy_band', 'image_motion')]
for n in (3, 5, 8):
    rw = {k: (1 if uni[k] > .5 else -1) / sd[idx[k]] / n for k in top[:n]}
    C[f'half_top{n}'] = half(rw)
# CV logistic regression on pair differences
def lr_cv(keys, entropy_half):
    files = sorted(X); rng = np.random.default_rng(0); folds = rng.permutation(len(files)) % 5
    fo = dict(zip(files, folds)); ok = tot = 0
    for k in range(5):
        tr = [(w, l) for w, l in pairs if fo[w] != k and fo[l] != k]; te = [(w, l) for w, l in pairs if fo[w] == k and fo[l] == k]
        if not te: continue
        D = np.array([(X[w][[idx[q] for q in keys]] - X[l][[idx[q] for q in keys]]) / sd[[idx[q] for q in keys]] for w, l in tr])
        D = np.vstack([D, -D]); y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        wv = np.zeros(len(keys))
        for _ in range(500):
            p = 1 / (1 + np.exp(-D @ wv)); wv -= .1 * (D.T @ (p - y) / len(y) + .05 * wv)
        rw = {q: wv[i] / sd[idx[q]] for i, q in enumerate(keys)}
        spec = half(rw) if entropy_half else rw
        ok += acc(spec, te) * len(te); tot += len(te)
    return ok / tot
res = sorted(((acc(s), n) for n, s in C.items()), reverse=True)
for a, n in res: print(f'{a:.3f}  {n}')
keys = top[:10]
print(f'{lr_cv(keys, False):.3f}  LR-cv top10 metrics, no entropy')
print(f'{lr_cv(keys, True):.3f}  LR-cv top10 metrics + entropy half')
print(f'{lr_cv(keys + ["image_entropy_band", "image_motion"], False):.3f}  LR-cv top10 + entropy terms fitted')
json.dump({n: s for n, s in C.items()}, open('taste_analysis/combo_specs.json', 'w'), indent=1)
print('--- no gate, no motion term ---')
def nomot(s): return {k: v for k, v in s.items() if k != 'image_motion'}
res = sorted(((acc(nomot(s), use_gate=False), n) for n, s in C.items()), reverse=True)
for a, n in res[:14]: print(f'{a:.3f}  {n}')
print('univariate (no gate):', ', '.join(f'{k}={a:.2f}' for k, a in sorted(uni.items(), key=lambda x: -abs(x[1]-.5))[:12]))
print('--- static gate (blank/noise only), no motion term ---')
res = sorted(((acc(nomot(s)), n) for n, s in C.items()), reverse=True)
for a, n in res[:10]: print(f'{a:.3f}  {n}')
print('half_top3 =', nomot(C['half_top3']))
