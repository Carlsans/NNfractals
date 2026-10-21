#!/usr/bin/env python3
"""Short S3 GA runs driven by metric combinations (no taste model). For each combo
run RUNS short GAs with fresh random seeds, keep ONLY the best genome of the combo
(by the GA's own fitness value) in OUT, named <metric-tag>__<id>.nn/.png."""
import json, os, re, shutil, subprocess, sys, tempfile, glob
OUT = os.environ.get('OUT', 'fractals_dag_s3_metric_combos')
COLLAPSE = os.environ.get('COLLAPSE', 's3')   # e.g. COLLAPSE=slice
POOL = os.environ.get('POOL')                 # optional --pool-dir for the GA's seeds/immigrants
RUNS = int(os.environ.get('RUNS', 3))
def _rev():
    try:
        r = subprocess.run(['git','rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip()
        d = subprocess.run(['git','status','--porcelain','src'],capture_output=True,text=True).stdout.strip()
        return r + ('+uncommitted' if d else '')
    except Exception: return 'unknown'
CODE = _rev()
SCORING = 'v4: entropy=0.5*deflate-band(.11)+0.5*shuffle-band(.5), T-continuity band, low-res gate; slice T sampled linearly over [-h,h)'
AB = {'organization_continuity':'cont','organization_chaoticity':'chao','sphericity':'sph','bilateral_symmetry':'sym','color_gradient':'cgrad','shading_gradient':'sgrad','image_entropy_band':'ent','image_motion':'mot','t_cont_band':'tcont','organization_compression':'comp',
      'color_shading_corr':'cshade','box_dim':'box','silhouette_irregularity':'sil','color_band_autocorr':'cband','color_entropy':'cent',
      'centroid_offset':'cent_off','organization_entropy':'oent','cross_view_coverage_delta':'xcov','solidity':'sol','isoperimetric':'iso','c_sensitivity':'csens'}
W = json.load(open('taste_analysis/weights.json'))
def half_entropy(rating_w):
    """50% image-entropy band (deflate ratio of the rendered still, peak 0.11; blank and noise score 0), 50% rating-derived metrics."""
    tot = sum(abs(v) for v in rating_w.values())
    d = {'image_entropy_band': round(tot, 3), 'image_motion': round(tot * 0.5, 3)}; d.update(rating_w)
    return d
s3 = W['s3']; union = dict(s3); union.update({k: v for k, v in W['pool'].items() if k not in union})
LR = {'organization_continuity':0.74,'sphericity':-0.92,'organization_entropy':-0.27,'bilateral_symmetry':-9.56,'color_gradient':-0.88,'color_shading_corr':2.96,'silhouette_irregularity':0.27,'centroid_offset':-15.69,'color_band_autocorr':0.66,'cross_view_coverage_delta':2.44}
LRn = {k: v / sum(abs(x) for x in LR.values()) for k, v in LR.items()}
# 2026-10-17 tournament (fractals_dag_s3_metric_eval/report.md): H1 won the visual check
# (3.1/5 vs ≤2.25 for the rest); H2/H3 untested there, so the loop runs H1 only.
COMBOS = {'H1_s3_rho25': half_entropy(s3)}
# 2026-10-18: T-continuity band in place of image_motion (no dotted noise, smooth motion).
if os.environ.get('RECIPE', 'cont') == 'cont':
    _h = COMBOS['H1_s3_rho25']
    COMBOS = {'H1_cont': dict({k: v for k, v in _h.items() if k != 'image_motion'}, t_cont_band=_h['image_motion'])}
GUARD = {}
def tag(spec):
    return '_'.join(f"{AB.get(k,k)}{'+' if w>0 else '-'}{abs(w):.2g}" if abs(w)!=1 else f"{AB.get(k,k)}{'+' if w>0 else '-'}" for k,w in spec.items())
# Guard terms: without them the organization metrics are maxed by a blank/solid ball.
for _n in COMBOS:
    for _k,_w in GUARD.items(): COMBOS[_n].setdefault(_k,_w)
os.makedirs(OUT, exist_ok=True)
manifest = json.load(open(f'{OUT}/manifest.json')) if os.path.exists(f'{OUT}/manifest.json') else {}
only = sys.argv[1:]
LOOP = os.environ.get('LOOP') == '1'
STOP = 'STOP_combos'   # touch this file to end the loop after the current run
while True:
  for name, spec in COMBOS.items():
      if os.path.exists(STOP): sys.exit(0)
      if only and name not in only: continue
      best = None
      for run in range(RUNS):
          if os.path.exists(STOP): sys.exit(0)   # checked per run, not just per combo
          seed = int.from_bytes(os.urandom(8), 'little')
          tmp = tempfile.mkdtemp(prefix='combo_', dir='/tmp')
          arg = ','.join(f'{k}:{w}' for k, w in spec.items())
          cmd = ['target/release/explorer','quat-dag-evolve','--gpu','--fitness-metric',arg,'--collapse',COLLAPSE,'--eval-phases','3',
                 '--out-dir',tmp,'--seed',str(seed),'--population','32','--survivors','8','--generations','10','--save-top','1','--stop-flat','6',
                 '--immigrant-frac','0.15','--crossover-frac','0.45'] + (['--pool-dir', POOL] if POOL else [])
          print(name, 'run', run, 'seed', seed, flush=True)
          subprocess.run(cmd, stdout=open(f'{tmp}/log.txt','w'), stderr=subprocess.STDOUT)
          for f in glob.glob(f'{tmp}/*.nn'):
              g = json.load(open(f)); m = re.search(r'value=([-0-9.]+)', g.get('formula_readable',''))
              v = float(m.group(1)) if m else float('-inf')
              if best is None or v > best[0]: best = (v, f, seed)
          # keep tmp of current best until the combo is done
      if best:
          v, f, seed = best
          stem = os.path.splitext(os.path.basename(f))[0]
          dst = f"{OUT}/{name}__{tag(spec)[:110]}__{stem.split("_")[-1]}"
          shutil.copy(f, dst + '.nn')
          for ext in ('.png',):
              if os.path.exists(f.replace('.nn', ext)): shutil.copy(f.replace('.nn', ext), dst + ext)
          manifest[os.path.basename(dst)] = {'combo': name, 'spec': spec, 'value': v, 'seed': seed, 'code': CODE, 'scoring': SCORING, 'collapse': COLLAPSE, 'pool': POOL}
          json.dump(manifest, open(f'{OUT}/manifest.json','w'), indent=1)
          print('WINNER', name, v, dst, flush=True)
      for d in glob.glob('/tmp/combo_*'): shutil.rmtree(d, ignore_errors=True)
  if not LOOP: break
