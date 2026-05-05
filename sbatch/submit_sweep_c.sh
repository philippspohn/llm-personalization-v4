#!/bin/bash
# Sweep C — history-length ablation (40 jobs, no robustness).
#   method:  routing_margin only
#   pools:   k10_1attrs, k10_2attrs
#   models:  qwen3p5_27b, llama31_8b
#   history: 1, 4 (vs default 2 in Sweep A)
#   seeds:   0..4
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(qwen3p5_27b llama31_8b)
POOLS=(k10_1attrs k10_2attrs)
HIST_LENGTHS=(1 4)
SEEDS=(0 1 2 3 4)

n=0
for gm in "${MODELS[@]}"; do
    for pool in "${POOLS[@]}"; do
        for hist in "${HIST_LENGTHS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=${gm} pool=${pool} method=routing_margin cache.history_max_len=${hist} +world_seeds=[${seed}]"
                echo "$cmd"
                eval "$cmd"
                n=$((n + 1))
            done
        done
    done
done
echo "Submitted ${n} jobs."
