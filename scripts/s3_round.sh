#!/bin/zsh
# One S3 taste-rating round: scripts/s3_round.sh N   -> fractals_dag_s3_range_rN
# --fitness-metric taste is REQUIRED: without it the GA silently uses the old
# geometric+aesthetic blend and never consults the taste model.
N=$1; SEED=$(od -An -N8 -tu8 /dev/urandom | tr -d ' '); echo "seed $SEED"
exec target/release/explorer quat-dag-evolve --gpu --fitness-metric taste --collapse s3 --eval-phases 6 --smooth-weight 0.2 --auto-range --out-dir fractals_dag_s3_range_r$N --seed $SEED --population 64 --survivors 16 --gens 30 --save-top 40 --embed-diversity 1.0 --immigrant-frac 0.15 --crossover-frac 0.45 --stop-flat 12
