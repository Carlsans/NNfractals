#!/usr/bin/env python3
"""Optuna hyperparameter search over the VAE architecture — the follow-up
explicitly deferred when vae_explore shipped (Carl's own decision at the
time: ship a fixed architecture first, tune it properly once the pipeline
itself is proven — see the "Explicitly out of scope" section of the
project-vae-explore memory). That decision is still the right shape here:
this script doesn't touch train_autoencoder.py's training loop at all —
each trial just spawns it as a subprocess with a suggested set of flags
and reads back the `RECON_MSE` line it already prints. Zero duplicated
training logic, and the winning config is immediately usable as-is (same
flags, same script, no translation step).

Searches `arch`/`latent_dim`/`lr`/`kl_weight` (VAE only) — the exact
parameter surface `train_autoencoder.py`/`AutoEncoder` already expose.
Objective: minimize held-out reconstruction MSE.

"The VAE is unique to a single fractal formula but the ideal VAE
structure is shared" (Carl's original framing): run ONE study against a
representative corpus, not per-formula — the winning arch/latent_dim/
kl_weight/lr becomes a new shared default, not a per-formula search.
`--res`/`--channels` select which regime to search (128/RGB for the
nav-imitation-style corpus, 512/1 for vae_explore's raw escape-time
zones) — same flags `train_autoencoder.py` itself takes.

Trials default to a much smaller `--epochs`/`--max-images` than a real
training run — this is a SEARCH (relative comparison across configs),
not a final model; re-run the winning config through
`train_autoencoder.py` directly, at full scale, once a study picks it.

Usage:
  python3 scripts/tune_autoencoder.py --dirs explorer_out/mandelbrot_vae \
      --variant vae --res 512 --channels 1 --n-trials 20
  python3 scripts/tune_autoencoder.py --dirs fractals fractals_dag \
      --variant vae --res 128 --channels 3 --n-trials 20
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import optuna


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def parse_recon_mse(text):
    m = re.search(r"RECON_MSE ([\d.eE+-]+)", text)
    return float(m.group(1)) if m else None


def make_objective(args):
    def objective(trial):
        arch = trial.suggest_categorical("arch", ["conv", "resnet", "inception"])
        latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256, 512])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        kl_weight = trial.suggest_float("kl_weight", 1e-5, 1e-1, log=True) if args.variant == "vae" else 1e-3

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "trial_model.pt"
            contact_path = Path(tmp) / "trial_recon.png"
            cmd = [
                sys.executable, "scripts/train_autoencoder.py",
                "--dirs", *args.dirs,
                "--variant", args.variant,
                "--arch", arch,
                "--res", str(args.res),
                "--channels", str(args.channels),
                "--latent-dim", str(latent_dim),
                "--lr", str(lr),
                "--epochs", str(args.epochs),
                "--max-images", str(args.max_images),
                "--min-images", str(args.min_images),
                "--min-val", str(args.min_val),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--seed", str(args.seed),
                "--out", str(out_path),
                "--contact-sheet", str(contact_path),
            ]
            if args.variant == "vae":
                cmd += ["--kl-weight", str(kl_weight)]

            log(f"[trial {trial.number}] arch={arch} latent_dim={latent_dim} lr={lr:.2e}"
                + (f" kl_weight={kl_weight:.2e}" if args.variant == "vae" else ""))
            result = subprocess.run(cmd, capture_output=True, text=True)
            mse = parse_recon_mse(result.stderr) or parse_recon_mse(result.stdout)
            if mse is None:
                log(f"[trial {trial.number}] FAILED — no RECON_MSE found. stderr tail:\n{result.stderr[-800:]}")
                raise optuna.TrialPruned()
            log(f"[trial {trial.number}] RECON_MSE={mse:.6f}")
            return mse
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--variant", choices=["ae", "vae"], default="vae")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--channels", type=int, default=3, choices=[1, 3])
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=5,
                     help="kept small — this is a SEARCH (relative comparison), not a final training run")
    ap.add_argument("--max-images", type=int, default=3000)
    ap.add_argument("--min-images", type=int, default=20)
    ap.add_argument("--min-val", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--study-name", default="vae_arch_search")
    ap.add_argument("--out", default="tuned_vae_config.json")
    args = ap.parse_args()

    study = optuna.create_study(direction="minimize", study_name=args.study_name)
    study.optimize(make_objective(args), n_trials=args.n_trials, show_progress_bar=False)

    log(f"\nbest trial: #{study.best_trial.number}  RECON_MSE={study.best_value:.6f}")
    log(f"best params: {study.best_params}")

    result = {
        "variant": args.variant, "res": args.res, "channels": args.channels,
        "best_recon_mse": study.best_value, **study.best_params,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    log(f"saved best config -> {args.out}")

    history_path = Path(args.out).with_suffix(".trials.json")
    with open(history_path, "w") as f:
        json.dump([{"number": t.number, "value": t.value, "params": t.params, "state": t.state.name}
                    for t in study.trials], f, indent=2)
    log(f"saved trial history -> {history_path}")


if __name__ == "__main__":
    main()
