#!/bin/bash
# Sweep 3 — re-run sweep A+B with seeds 5..9 to get 10 seeds total per cell.
# 7 methods x 2 pools x 2 models x 5 new seeds = 140 jobs. No robustness.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(qwen3p5_27b llama31_8b)
POOLS=(k10_1attrs k10_2attrs)
METHODS=(routing_margin routing_oracle rag_simple rag_advanced routing_abs routing_abs_two_sided routing_regression)
SEEDS=(5 6 7 8 9)

n=0
for gm in "${MODELS[@]}"; do
    for pool in "${POOLS[@]}"; do
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=${gm} pool=${pool} method=${method} +world_seeds=[${seed}]"
                echo "$cmd"; eval "$cmd"; n=$((n+1))
            done
        done
    done
done
echo "Submitted ${n} jobs."
