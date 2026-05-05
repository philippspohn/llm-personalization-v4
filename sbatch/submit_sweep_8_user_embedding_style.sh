#!/bin/bash
# Sweep 8 — style-aware embedder for user_embedding, on llama, 5 seeds, both
# k10 pools. Two variants:
#   user_embedding_style          : Style-Embedding frozen (only projector + LoRA train)
#   user_embedding_style_finetune : Style-Embedding fine-tuned jointly
# Total: 2 methods x 2 pools x 5 seeds = 20 jobs.
set -euo pipefail
cd "$(dirname "$0")/.."

POOLS=(k10_1attrs k10_2attrs)
METHODS=(user_embedding_style user_embedding_style_finetune)
SEEDS=(0 1 2 3 4)

n=0
for method in "${METHODS[@]}"; do
    for pool in "${POOLS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cmd="sbatch sbatch/run_benchmark_v3.sbatch gen_model=llama31_8b pool=${pool} method=${method} +world_seeds=[${seed}]"
            echo "$cmd"; eval "$cmd"; n=$((n+1))
        done
    done
done
echo "Submitted ${n} jobs."
