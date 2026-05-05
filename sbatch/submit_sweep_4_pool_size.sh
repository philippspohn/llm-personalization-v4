#!/bin/bash
# Sweep 4 — pool size scaling. Qwen, k={2,4,6,8}_1attrs, 4 methods, 5 seeds (80 jobs).
set -euo pipefail
cd "$(dirname "$0")/.."

POOLS=(k2_1attrs k4_1attrs k6_1attrs k8_1attrs)
METHODS=(routing_margin routing_oracle rag_simple rag_advanced)
SEEDS=(0 1 2 3 4)

n=0
for pool in "${POOLS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=qwen3p5_27b pool=${pool} method=${method} +world_seeds=[${seed}]"
            echo "$cmd"; eval "$cmd"; n=$((n+1))
        done
    done
done
echo "Submitted ${n} jobs."
