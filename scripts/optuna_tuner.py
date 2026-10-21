#!/usr/bin/env python3
"""Ask/tell bridge between the launcher's campaigns and Optuna. Every call is stateless:
the study lives in <campaign dir>/study.db (SQLite), JSON goes in on stdin and out on stdout.

  ask         stdin {optimizer, params:[{name,kind,min,max,log,locked,value,choices}], override?}
              -> {trial, params:{name: value}}   (override = the user's edited values: they are
              enqueued so the study records exactly what will run)
  tell        stdin {trial, value|null}          -> {ok}
  importances                                    -> {name: importance}   ({} until enough trials)
  score       --dir RUN_DIR [--keep DEST --prefix P]
              -> {value, stem, evolved}          (trial value of one finished GA run)

Scoring rule (shared with scripts/optuna_campaigns.py): the best genome the GA actually
evolved (origin not seed/immigrant) in the saved top genomes; if every saved genome is an
unchanged pool genome, the worst of them; None if nothing was saved.
"""
import argparse, glob, json, os, re, shutil, sys
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)


def score_run(run_dir, keep=None, prefix='trial'):
    saved = []
    for f in glob.glob(f'{run_dir}/*.nn'):
        fr = json.load(open(f)).get('formula_readable', '')
        m = re.search(r'value=([-0-9.]+)', fr)
        o = re.search(r'origin=(\w+)', fr)
        saved.append((float(m.group(1)) if m else float('-inf'), f, o.group(1) if o else '?'))
    if not saved:
        return {'value': None, 'stem': None, 'evolved': 0}
    evolved = [x for x in saved if x[2] not in ('seed', 'immigrant')]
    v, f, _ = max(evolved) if evolved else min(saved)
    stem = f'{prefix}_' + os.path.basename(f)[:-3].split('_')[-1]
    if keep:
        os.makedirs(keep, exist_ok=True)
        shutil.copy(f, f'{keep}/{stem}.nn')
        if os.path.exists(f[:-3] + '.png'):
            shutil.copy(f[:-3] + '.png', f'{keep}/{stem}.png')
    return {'value': v, 'stem': stem, 'evolved': len(evolved)}


def make_sampler(name):
    seed = int.from_bytes(os.urandom(4), 'little')
    S = optuna.samplers
    if name == 'tpe':
        return S.TPESampler(seed=seed, n_startup_trials=8)
    if name == 'random':
        return S.RandomSampler(seed=seed)
    if name == 'qmc':
        return S.QMCSampler(seed=seed)
    if name == 'cmaes':
        try:
            import cmaes  # noqa: F401
        except ImportError:
            sys.exit('CMA-ES needs the "cmaes" package: pip install cmaes')
        return S.CmaEsSampler(seed=seed)
    if name == 'gp':
        return S.GPSampler(seed=seed)
    sys.exit(f'unknown optimizer {name}')


def study_for(campaign, sampler):
    os.makedirs(campaign, exist_ok=True)
    return optuna.create_study(study_name='campaign', storage=f'sqlite:///{campaign}/study.db',
                               sampler=sampler, direction='maximize', load_if_exists=True)


def distributions(params):
    D = optuna.distributions
    out = {}
    for p in params:
        locked = p.get('locked')
        if p['kind'] == 'Choice':
            ch = p['choices']
            out[p['name']] = D.CategoricalDistribution([ch[int(p['value'])]] if locked else ch)
        elif p['kind'] == 'Int':
            lo, hi = (p['value'], p['value']) if locked else (p['min'], p['max'])
            out[p['name']] = D.IntDistribution(int(lo), int(hi))
        else:
            lo, hi = (p['value'], p['value']) if locked else (p['min'], p['max'])
            out[p['name']] = D.FloatDistribution(float(lo), float(hi), log=bool(p['log']) and lo > 0 and not locked)
    return out


def cmd_ask(a):
    req = json.load(sys.stdin)
    study = study_for(a.campaign, make_sampler(req['optimizer']))
    dists = distributions(req['params'])
    if req.get('override'):
        study.enqueue_trial(req['override'], skip_if_exists=False)
    trial = study.ask(fixed_distributions=dists)
    print(json.dumps({'trial': trial.number, 'params': trial.params}))


def cmd_tell(a):
    req = json.load(sys.stdin)
    study = study_for(a.campaign, optuna.samplers.RandomSampler())
    if req.get('value') is None:
        study.tell(req['trial'], state=optuna.trial.TrialState.FAIL)
    else:
        study.tell(req['trial'], req['value'])
    print(json.dumps({'ok': True}))


def cmd_importances(a):
    study = study_for(a.campaign, optuna.samplers.RandomSampler())
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        imp = {}
    print(json.dumps(imp))


def cmd_score(a):
    print(json.dumps(score_run(a.dir, a.keep, a.prefix)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('ask', cmd_ask), ('tell', cmd_tell), ('importances', cmd_importances)):
        s = sub.add_parser(name); s.add_argument('--campaign', required=True); s.set_defaults(fn=fn)
    s = sub.add_parser('score'); s.add_argument('--dir', required=True); s.add_argument('--keep'); s.add_argument('--prefix', default='trial'); s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)
