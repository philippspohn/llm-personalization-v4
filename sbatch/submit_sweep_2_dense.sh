#!/bin/bash
# Sweep 2 — dense world matrix, Qwen, k10_1+k10_2, 4 methods, 5 seeds (40 jobs).
set -euo pipefail
cd "$(dirname "$0")/.."

POOLS=(k10_1attrs k10_2attrs)
METHODS=(routing_margin routing_oracle rag_simple rag_advanced)
SEEDS=(0 1 2 3 4)

n=0
for pool in "${POOLS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=qwen3p5_27b pool=${pool} method=${method} world_matrix_type=dense +world_seeds=[${seed}]"
            echo "$cmd"; eval "$cmd"; n=$((n+1))
        done
    done
done
echo "Submitted ${n} jobs."
