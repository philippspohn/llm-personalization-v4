#!/bin/bash
# Sweep 6 — train-size scaling. Qwen, k10_1attrs, 4 methods, 5 seeds (80 jobs).
# train_size in {500, 1000, 2000, 4000}; we override `cache.train_limit` =
# train_size + val_size (val_size stays at 500). The train_size=4000 column
# overlaps with main + sweep 3 (defaults to train_limit=4500); kept here for
# self-contained ablation but can be deduped at analysis time.
set -euo pipefail
cd "$(dirname "$0")/.."

# (train_size, train_limit) pairs
TRAIN_SIZES=(500 1000 2000 4000)
TRAIN_LIMITS=(1000 1500 2500 4500)

METHODS=(routing_margin routing_oracle rag_simple rag_advanced)
SEEDS=(0 1 2 3 4)

n=0
for i in "${!TRAIN_SIZES[@]}"; do
    limit="${TRAIN_LIMITS[$i]}"
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=qwen3p5_27b pool=k10_1attrs method=${method} cache.train_limit=${limit} +world_seeds=[${seed}]"
            echo "$cmd"; eval "$cmd"; n=$((n+1))
        done
    done
done
echo "Submitted ${n} jobs."
