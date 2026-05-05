#!/bin/bash
# Sweep B — routing-variant comparison (60 jobs, no robustness).
#   methods: routing_abs, routing_abs_two_sided, routing_regression
#   pools:   k10_1attrs, k10_2attrs
#   models:  qwen3p5_27b, llama31_8b
#   seeds:   0..4
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(qwen3p5_27b llama31_8b)
POOLS=(k10_1attrs k10_2attrs)
METHODS=(routing_abs routing_abs_two_sided routing_regression)
SEEDS=(0 1 2 3 4)

n=0
for gm in "${MODELS[@]}"; do
    for pool in "${POOLS[@]}"; do
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=${gm} pool=${pool} method=${method} +world_seeds=[${seed}]"
                echo "$cmd"
                eval "$cmd"
                n=$((n + 1))
            done
        done
    done
done
echo "Submitted ${n} jobs."
