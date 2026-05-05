#!/bin/bash
# Sweep A — main grid (80 jobs).
#   methods: routing_margin, routing_oracle, rag_simple, rag_advanced
#   pools:   k10_1attrs, k10_2attrs
#   models:  qwen3p5_27b, llama31_8b
#   seeds:   0..4
# +robustness=full added only on seed 0 of each (method, pool, model) cell.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(qwen3p5_27b llama31_8b)
POOLS=(k10_1attrs k10_2attrs)
METHODS=(routing_margin routing_oracle rag_simple rag_advanced)
SEEDS=(0 1 2 3 4)

n=0
for gm in "${MODELS[@]}"; do
    for pool in "${POOLS[@]}"; do
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if [[ $seed -eq 0 ]]; then
                    cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=${gm} pool=${pool} method=${method} +world_seeds=[${seed}] +robustness=full"
                else
                    cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=${gm} pool=${pool} method=${method} +world_seeds=[${seed}]"
                fi
                echo "$cmd"
                eval "$cmd"
                n=$((n + 1))
            done
        done
    done
done
echo "Submitted ${n} jobs."
