#!/bin/bash
# Sweep 1 — user_embedding on Llama, both k10 pools (10 jobs).
# Robustness only on seed 0 of each (pool) cell.
# RUN THIS LAST — user_embedding has slow HF generate at eval (~2-3h/job).
set -euo pipefail
cd "$(dirname "$0")/.."

POOLS=(k10_1attrs k10_2attrs)
SEEDS=(0 1 2 3 4)

n=0
for pool in "${POOLS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [[ $seed -eq 0 ]]; then
            cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=llama31_8b pool=${pool} method=user_embedding +world_seeds=[${seed}] +robustness=full"
        else
            cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=llama31_8b pool=${pool} method=user_embedding +world_seeds=[${seed}]"
        fi
        echo "$cmd"; eval "$cmd"; n=$((n+1))
    done
done
echo "Submitted ${n} jobs."
