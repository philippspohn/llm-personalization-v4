#!/bin/bash
# Sweep 7 — full user_embedding comparison: SimPO + DPO, llama, both k10 pools.
#   - SimPO (user_embedding): seeds 5..9 (seeds 0..4 already in sweep 1)  -> 10 jobs
#   - DPO (user_embedding_dpo): seeds 0..9                                -> 20 jobs
# Total: 30 jobs. Slow (~2-3h each due to HF generate at eval).
set -euo pipefail
cd "$(dirname "$0")/.."

POOLS=(k10_1attrs k10_2attrs)

n=0
# SimPO continuation (seeds 5..9; sweep 1 covered 0..4)
for pool in "${POOLS[@]}"; do
    for seed in 5 6 7 8 9; do
        cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=llama31_8b pool=${pool} method=user_embedding +world_seeds=[${seed}]"
        echo "$cmd"; eval "$cmd"; n=$((n+1))
    done
done

# DPO from scratch (seeds 0..9)
for pool in "${POOLS[@]}"; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=llama31_8b pool=${pool} method=user_embedding_dpo +world_seeds=[${seed}]"
        echo "$cmd"; eval "$cmd"; n=$((n+1))
    done
done

echo "Submitted ${n} jobs."
