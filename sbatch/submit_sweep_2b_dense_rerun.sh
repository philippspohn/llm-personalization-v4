#!/bin/bash
# Sweep 2b — re-run sweep 2 (dense matrix) with seeds 5..9 to reach 10 seeds
# total per cell. Same setup as sweep 2: qwen, both k10 pools, 4 methods.
set -euo pipefail
cd "$(dirname "$0")/.."

POOLS=(k10_1attrs k10_2attrs)
METHODS=(routing_margin routing_oracle rag_simple rag_advanced)
SEEDS=(5 6 7 8 9)

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
