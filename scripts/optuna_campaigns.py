#!/usr/bin/env python3
"""Optuna campaigns over the quat GA's own hyperparameters (Carl's RewriteFractal method).

Each campaign is a FRESH Optuna study of TRIALS trials (default 30); one trial = one
short quat-dag-evolve run with the suggested operator settings, scored by its best
fitness value (absolute, same recipe within a study, so trials are comparable). After
TRIALS the study is dropped and a new one starts, so results stay optimised without
always converging to the same settings. The top KEEP winners of each campaign are
copied into OUT (manifest records params, value, campaign); every trial's winner is
kept under OUT/_campaigns/<campaign>/ until the campaign ends.

Population and generations are fixed so a trial cannot win by buying more search.

Env: OUT, COLLAPSE (slice), POOL, TRIALS (30), KEEP (8), RECIPE (cont). Stop with
`touch STOP_campaigns` (checked before every trial).
"""
import glob, json, os, re, shutil, subprocess, sys, tempfile, time
import optuna
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optuna_tuner import score_run

OUT = os.environ.get('OUT', 'fractals_dag_slice_metric_loop')
COLLAPSE = os.environ.get('COLLAPSE', 'slice')
POOL = os.environ.get('POOL', 'fractals_dag_mixed_kbc_seed')
TRIALS = int(os.environ.get('TRIALS', 30))
KEEP = int(os.environ.get('KEEP', 8))
STOP = 'STOP_campaigns'
POP, GENS = 32, 10

C = json.load(open('taste_analysis/combo_specs.json'))
H1 = C['H1_s3_rho25']
SPEC = dict({k: v for k, v in H1.items() if k != 'image_motion'}, t_cont_band=H1['image_motion'])
# Campaign 2+: penalise winners whose sheet views are mostly unlit (the "empty crescents").
SPEC['front_cover_band'] = H1['image_motion']
# Campaign 3+: penalise chrome/silver (grey share of lit sheet pixels > 0.3; Carl's ratings: less grey wins 57-64%).
SPEC['grey_band'] = H1['image_motion']
ARG = ','.join(f'{k}:{round(w, 4)}' for k, w in SPEC.items())
CODE = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip() + '+uncommitted'
os.makedirs(OUT, exist_ok=True)
MF = f'{OUT}/manifest.json'


def run_with_watchdog(cmd, log_path, grace=60, limit=2400):
    """Run one GA. A run that has written its final contact sheet but is still alive
    after `grace` s is killed (2026-10-18: one run hung at exit for 12 h, a thread
    spinning after the save, and stalled the whole campaign); any run is killed
    after `limit` s. The saved genomes are complete either way."""
    proc = subprocess.Popen(cmd, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    start, done_at = time.time(), None
    while proc.poll() is None:
        time.sleep(5)
        if done_at is None:
            try:
                if 'wrote contact sheet' in open(log_path, errors='replace').read()[-400:]:
                    done_at = time.time()
            except OSError:
                pass
        if (done_at and time.time() - done_at > grace) or time.time() - start > limit:
            print(f'WATCHDOG killing hung run (pid {proc.pid}, {"after save" if done_at else "time limit"})', flush=True)
            proc.kill()
            proc.wait()


def run_trial(trial, camp_dir):
    p = {
        'survivors': trial.suggest_int('survivors', 4, 16),
        'immigrant_frac': trial.suggest_float('immigrant_frac', 0.0, 0.3),
        'crossover_frac': trial.suggest_float('crossover_frac', 0.1, 0.7),
        'crossover_mode': trial.suggest_categorical('crossover_mode', ['legacy', 'subtree']),
        'min_edits': trial.suggest_int('min_edits', 1, 2),
        'extra_edits': trial.suggest_int('extra_edits', 0, 3),
        'const_scale': trial.suggest_float('const_scale', 0.02, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'w_perturb': trial.suggest_float('w_perturb', 0.05, 1.0),
        'w_swap': trial.suggest_float('w_swap', 0.05, 1.0),
        'w_rewire': trial.suggest_float('w_rewire', 0.05, 1.0),
        'w_grow': trial.suggest_float('w_grow', 0.05, 1.0),
        'w_prune': trial.suggest_float('w_prune', 0.05, 1.0),
        'bulb_prob': trial.suggest_float('bulb_prob', 0.0, 0.3),
        'kbc_prob': trial.suggest_float('kbc_prob', 0.0, 0.3),
    }
    seed = int.from_bytes(os.urandom(8), 'little')
    tmp = tempfile.mkdtemp(prefix='optc_', dir='/tmp')
    cmd = ['target/release/explorer', 'quat-dag-evolve', '--gpu', '--fitness-metric', ARG, '--collapse', COLLAPSE,
           '--eval-phases', '3', '--out-dir', tmp, '--seed', str(seed), '--population', str(POP),
           '--survivors', str(p['survivors']), '--generations', str(GENS), '--save-top', '5', '--stop-flat', '6',
           '--immigrant-frac', f"{p['immigrant_frac']:.4f}", '--crossover-frac', f"{p['crossover_frac']:.4f}",
           '--crossover-mode', p['crossover_mode'], '--mutation-min-edits', str(p['min_edits']),
           '--mutation-max-edits', str(p['min_edits'] + p['extra_edits']), '--mutation-const-scale', f"{p['const_scale']:.4f}",
           '--mutation-max-depth', str(p['max_depth']),
           '--edit-weights', ','.join(f"{p[k]:.4f}" for k in ('w_perturb', 'w_swap', 'w_rewire', 'w_grow', 'w_prune')),
           '--bulb-prob', f"{p['bulb_prob']:.4f}", '--kbc-prob', f"{p['kbc_prob']:.4f}"]
    if POOL:
        cmd += ['--pool-dir', POOL]
    run_with_watchdog(cmd, f'{tmp}/log.txt')
    # Score the trial by the best genome the GA actually EVOLVED: an unchanged seed or
    # immigrant from the pool says nothing about the settings (campaign 1 had two pool
    # genomes win 4 trials each). If the top 5 are all unchanged pool genomes, the
    # settings failed to improve on the pool: score the worst of them.
    r = score_run(tmp, camp_dir, f'trial{trial.number:02d}')
    if r['value'] is None:
        shutil.rmtree(tmp, ignore_errors=True)
        raise optuna.TrialPruned('run saved nothing')
    v, stem = r['value'], r['stem']
    trial.set_user_attr('evolved_in_top5', r['evolved'])
    trial.set_user_attr('stem', stem)
    trial.set_user_attr('seed', seed)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'TRIAL {trial.number} value {v:.3f} {json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in p.items()})}', flush=True)
    return v


while True:
    if os.path.exists(STOP):
        sys.exit(0)
    camp = time.strftime('c%Y%m%d_%H%M%S')
    camp_dir = f'{OUT}/_campaigns/{camp}'
    os.makedirs(camp_dir, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=int.from_bytes(os.urandom(4), 'little'), n_startup_trials=8)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=camp)
    print(f'CAMPAIGN {camp} start ({TRIALS} trials)', flush=True)

    def stop_cb(study, trial):
        if os.path.exists(STOP):
            study.stop()

    study.optimize(lambda t: run_trial(t, camp_dir), n_trials=TRIALS, callbacks=[stop_cb], catch=(Exception,))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        continue
    json.dump([{'n': t.number, 'value': t.value, 'params': t.params, 'stem': t.user_attrs.get('stem'),
                'seed': t.user_attrs.get('seed')} for t in done], open(f'{camp_dir}/trials.json', 'w'), indent=1)
    man = json.load(open(MF)) if os.path.exists(MF) else {}
    for t in sorted(done, key=lambda t: -t.value)[:KEEP]:
        src = f"{camp_dir}/{t.user_attrs['stem']}"
        dst = f"{OUT}/{camp}_{t.user_attrs['stem']}"
        shutil.copy(src + '.nn', dst + '.nn')
        if os.path.exists(src + '.png'):
            shutil.copy(src + '.png', dst + '.png')
        man[os.path.basename(dst)] = {'combo': 'H1_cont', 'spec': SPEC, 'value': t.value, 'seed': t.user_attrs.get('seed'),
                                      'code': CODE, 'collapse': COLLAPSE, 'pool': POOL, 'campaign': camp,
                                      'trial': t.number, 'hp': t.params,
                                      'scoring': 'v7: optuna campaign (30 trials, reset), H1_cont + front_cover_band + grey_band, evolved-only trial score, slice T linear'}
        print(f'WINNER {camp} trial {t.number} value {t.value:.3f} {dst}', flush=True)
    json.dump(man, open(MF, 'w'), indent=1)
    vals = [t.value for t in done]
    print(f'CAMPAIGN {camp} done: best {max(vals):.3f}, first-8 mean {sum(vals[:8]) / max(1, len(vals[:8])):.3f}, '
          f'last-8 mean {sum(vals[-8:]) / max(1, len(vals[-8:])):.3f}; best params {study.best_params}', flush=True)
