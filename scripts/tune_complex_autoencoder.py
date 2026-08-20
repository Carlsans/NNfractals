#!/usr/bin/env python3
"""Optuna hyperparameter search over the complex VAE architecture
(complex_autoencoder.py) — mirrors scripts/tune_autoencoder.py's proven
approach for the real-valued AE/VAE: each trial spawns
train_complex_autoencoder.py fresh as a subprocess with Optuna-suggested
flags and reads back the RECON_MSE line it already prints. Zero duplicated
training logic, winning config immediately reusable as-is.

Searches `latent_dim`/`norm`/`residual`/`lr`/`kl_weight` — the exact
surface train_complex_autoencoder.py already exposes. Fixed to
`--variant vae` throughout (Carl asked for "the best vae structure"
specifically, not a comparison against the plain AE — that comparison
already happened manually this session, see project-complex-nn-weekend-
research.md). Objective: minimize held-out reconstruction MSE (plain
per-pixel complex MSE, reported the same way regardless of training loss
or variant — directly comparable across every trial).

Trials use a SUBSET of the full corpus and fewer epochs than a real
training run (this is a SEARCH — relative comparison across configs —
not a final model); re-run the winning config through
train_complex_autoencoder.py directly, at full scale, once the study
picks it (this script does that automatically at the end, same as
tune_autoencoder.py's own convention of leaving that as a manual step —
except here we go one further and actually launch the confirmation run,
since Carl asked to see the best result, not just the winning config).

Usage:
  python3 scripts/tune_complex_autoencoder.py \
      --dirs explorer_out/weekend_complex_corpus/mandelbrot explorer_out/weekend_complex_corpus/burning_ship ... \
      --n-trials 25
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
        latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256, 512])
        norm = trial.suggest_categorical("norm", ["groupnorm", "whitened"])
        residual = trial.suggest_categorical("residual", [False, True])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        kl_weight = trial.suggest_float("kl_weight", 1e-5, 1e-2, log=True)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "trial_model.pt"
            contact_path = Path(tmp) / "trial_recon.png"
            cmd = [
                sys.executable, "scripts/train_complex_autoencoder.py",
                "--dirs", *args.dirs,
                "--variant", "vae",
                "--latent-dim", str(latent_dim),
                "--norm", norm,
                "--lr", str(lr),
                "--kl-weight", str(kl_weight),
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
            if residual:
                cmd.append("--residual")

            log(f"[trial {trial.number}] latent_dim={latent_dim} norm={norm} residual={residual} "
                f"lr={lr:.2e} kl_weight={kl_weight:.2e}")
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
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=12,
                     help="kept small — this is a SEARCH (relative comparison), not a final training run")
    ap.add_argument("--max-images", type=int, default=1200)
    ap.add_argument("--min-images", type=int, default=50)
    ap.add_argument("--min-val", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--study-name", default="complex_vae_arch_search")
    ap.add_argument("--out", default="tuned_complex_vae_config.json")
    # Confirmation run settings (full-scale, using the winning config)
    ap.add_argument("--confirm-epochs", type=int, default=80)
    ap.add_argument("--confirm-out", default="explorer_out/complex_ae/complex_vae_tuned.pt")
    ap.add_argument("--confirm-contact-sheet", default="explorer_out/complex_ae/complex_vae_tuned_recon.png")
    ap.add_argument("--skip-confirm", action="store_true", help="only run the search, skip the full-scale confirmation training")
    args = ap.parse_args()

    study = optuna.create_study(direction="minimize", study_name=args.study_name)
    study.optimize(make_objective(args), n_trials=args.n_trials, show_progress_bar=False)

    log(f"\nbest trial: #{study.best_trial.number}  RECON_MSE={study.best_value:.6f}")
    log(f"best params: {study.best_params}")

    result = {"variant": "vae", "best_recon_mse": study.best_value, **study.best_params}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    log(f"saved best config -> {args.out}")

    history_path = Path(args.out).with_suffix(".trials.json")
    with open(history_path, "w") as f:
        json.dump([{"number": t.number, "value": t.value, "params": t.params, "state": t.state.name}
                    for t in study.trials], f, indent=2)
    log(f"saved trial history -> {history_path}")

    if args.skip_confirm:
        return

    log(f"\n=== running full-scale confirmation training with the winning config ===")
    p = study.best_params
    cmd = [
        sys.executable, "scripts/train_complex_autoencoder.py",
        "--dirs", *args.dirs,
        "--variant", "vae",
        "--latent-dim", str(p["latent_dim"]),
        "--norm", str(p["norm"]),
        "--lr", str(p["lr"]),
        "--kl-weight", str(p["kl_weight"]),
        "--epochs", str(args.confirm_epochs),
        "--batch-size", str(args.batch_size),
        "--min-images", str(args.min_images),
        "--min-val", str(args.min_val),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--out", args.confirm_out,
        "--contact-sheet", args.confirm_contact_sheet,
    ]
    if p["residual"]:
        cmd.append("--residual")
    log(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    log(f"confirmation run done -> {args.confirm_out}")


if __name__ == "__main__":
    main()
