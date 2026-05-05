#!/bin/bash
# Sweep 5 — history length ablation, Qwen, k10_1attrs, 4 methods, 5 seeds.
# Only h={1, 4} since h=2 is the default already covered by main + sweep 3 (40 jobs).
set -euo pipefail
cd "$(dirname "$0")/.."

METHODS=(routing_margin routing_oracle rag_simple rag_advanced)
HIST=(1 4)
SEEDS=(0 1 2 3 4)

n=0
for hist in "${HIST[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=qwen3p5_27b pool=k10_1attrs method=${method} cache.history_max_len=${hist} +world_seeds=[${seed}]"
            echo "$cmd"; eval "$cmd"; n=$((n+1))
        done
    done
done
echo "Submitted ${n} jobs."
