#!/usr/bin/env python3
"""Controlled comparison of metric recipes: SEEDS short S3 GAs per recipe at the loop's
budget; keeps the top genome of EVERY run in OUT/<recipe>/ (not just the best)."""
import json, os, re, shutil, subprocess, tempfile, glob
OUT = os.environ.get('OUT', 'fractals_dag_s3_metric_eval')
COLLAPSE = os.environ.get('COLLAPSE', 's3')
SEEDS = int(os.environ.get('SEEDS', 4))
C = json.load(open('taste_analysis/combo_specs.json'))
nomot = lambda s: {k: v for k, v in s.items() if k != 'image_motion'}
R = {'half_top3': nomot(C['half_top3']), 'half_top3_motion': dict(C['half_top3'], image_motion=C['half_top3']['image_entropy_band'] * .5),
     'ent_sphericity': nomot(C['ent+sphericity']), 'half_top8': nomot(C['half_top8']),
     'H1_nomotion': nomot(C['H1_s3_rho25']), 'H1_current': C['H1_s3_rho25'], 'entropy_only': nomot(C['entropy_only'])}
H1 = C['H1_s3_rho25']
R['H1_cont'] = dict({k: v for k, v in H1.items() if k != 'image_motion'}, t_cont_band=H1['image_motion'])
R['H1_cont_bulb'] = dict(R['H1_cont'])  # same spec; binary with the BULB op (2026-10-18)
R['H1_cont_bulbseed'] = dict(R['H1_cont'])  # + population seeded from fractals_dag_bulb_seed
EXTRA = {'H1_cont_bulbseed': ['--pool-dir', 'fractals_dag_bulb_seed']}
# 2026-10-18 slice tournament: plane terms (taste_analysis/planes/principles.md) and kbc.
PL = {'planes_smooth': 1.8, 'planes_open': 1.0, 'planes_asym': 0.8}
R['S_A_control'] = dict(R['H1_cont'])
R['S_B_planes'] = dict(R['H1_cont'], **PL)
R['S_C_kbc'] = dict(R['H1_cont'])
R['S_D_planes_kbc'] = dict(R['H1_cont'], **PL)
EXTRA.update({'S_A_control': ['--pool-dir', 'fractals_dag_mixed_seed', '--no-kbc'],
              'S_B_planes': ['--pool-dir', 'fractals_dag_mixed_seed', '--no-kbc'],
              'S_C_kbc': ['--pool-dir', 'fractals_dag_mixed_kbc_seed'],
              'S_D_planes_kbc': ['--pool-dir', 'fractals_dag_mixed_kbc_seed']})
import sys
if sys.argv[1:]: R = {k: v for k, v in R.items() if k in sys.argv[1:]}
code = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip() + '+uncommitted'
mf = f'{OUT}/manifest.json'; os.makedirs(OUT, exist_ok=True)
man = json.load(open(mf)) if os.path.exists(mf) else {}
for s in range(SEEDS):
    for name, spec in R.items():
        if sum(1 for v in man.values() if v['recipe'] == name) > s: continue  # resume
        seed = int.from_bytes(os.urandom(8), 'little'); tmp = tempfile.mkdtemp(prefix='tourn_', dir='/tmp')
        arg = ','.join(f'{k}:{round(w, 4)}' for k, w in spec.items())
        cmd = ['target/release/explorer', 'quat-dag-evolve', '--gpu', '--fitness-metric', arg, '--collapse', COLLAPSE, '--eval-phases', '3', '--out-dir', tmp,
               '--seed', str(seed), '--population', '32', '--survivors', '8', '--generations', '10', '--save-top', '1', '--stop-flat', '6',
               '--immigrant-frac', '0.15', '--crossover-frac', '0.45'] + EXTRA.get(name, [])
        print(name, 'seed', seed, flush=True)
        subprocess.run(cmd, stdout=open(f'{tmp}/log.txt', 'w'), stderr=subprocess.STDOUT)
        best = None
        for f in glob.glob(f'{tmp}/*.nn'):
            m = re.search(r'value=([-0-9.]+)', json.load(open(f)).get('formula_readable', '')); v = float(m.group(1)) if m else float('-inf')
            if best is None or v > best[0]: best = (v, f)
        if best:
            d = f'{OUT}/{name}'; os.makedirs(d, exist_ok=True); stem = f'{name}_s{s}_' + os.path.basename(best[1])[:-3].split('_')[-1]
            shutil.copy(best[1], f'{d}/{stem}.nn')
            if os.path.exists(best[1][:-3] + '.png'): shutil.copy(best[1][:-3] + '.png', f'{d}/{stem}.png')
            man[stem] = {'recipe': name, 'spec': spec, 'value': best[0], 'seed': seed, 'code': code}
            json.dump(man, open(mf, 'w'), indent=1); print('DONE', stem, best[0], flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
