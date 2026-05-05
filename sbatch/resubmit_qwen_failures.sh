#!/bin/bash
# Re-submit the Qwen-cache-related failures:
#   - 4 cells from sweep 3 missing-seed gap (rag_advanced k10_2attrs seeds 5,7,8,9)
#   - 40 cells from sweep 2b (dense matrix, all 4 methods x 2 pools x seeds 5..9)
# Pre-requisite: run the Qwen3.5-27B snapshot prefetch fix first:
#   HF_HOME=/lustre/groups/eml/projects/pspohn/llm-personalization-v2/tmp/hf_home \
#   HF_TOKEN="$(cat ~/.hf_token | tr -d '[:space:]')" \
#   python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('Qwen/Qwen3.5-27B', allow_patterns=['*.json','tokenizer*','*.txt','*.py','*.md'])"
set -euo pipefail
cd "$(dirname "$0")/.."

# Sweep 3 missing-seed Qwen cells (rag_advanced k10_2attrs seeds 5,7,8,9)
GAP_CMDS=(
    "gen_model=qwen3p5_27b pool=k10_2attrs method=rag_advanced +world_seeds=[5]"
    "gen_model=qwen3p5_27b pool=k10_2attrs method=rag_advanced +world_seeds=[7]"
    "gen_model=qwen3p5_27b pool=k10_2attrs method=rag_advanced +world_seeds=[8]"
    "gen_model=qwen3p5_27b pool=k10_2attrs method=rag_advanced +world_seeds=[9]"
)

n=0
for cmd in "${GAP_CMDS[@]}"; do
    echo "sbatch sbatch/run_benchmark_v3.sbatch $cmd"
    sbatch sbatch/run_benchmark_v3.sbatch $cmd
    n=$((n+1))
done

# Sweep 2b: dense matrix, qwen, both k10 pools, 4 methods, seeds 5..9
POOLS=(k10_1attrs k10_2attrs)
METHODS=(routing_margin routing_oracle rag_simple rag_advanced)
SEEDS=(5 6 7 8 9)
for pool in "${POOLS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cmd="gen_model=qwen3p5_27b pool=${pool} method=${method} world_matrix_type=dense +world_seeds=[${seed}]"
            echo "sbatch sbatch/run_benchmark_v3.sbatch $cmd"
            sbatch sbatch/run_benchmark_v3.sbatch $cmd
            n=$((n+1))
        done
    done
done

echo "Submitted ${n} jobs."
