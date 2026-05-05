#!/bin/bash
# Re-submit the 12 (gm, pool, method, seed) cells still missing after sweep 3.
# All previous attempts failed due to vLLM/HF DNS timeouts. The sbatch now sets
# HF_HUB_OFFLINE=1, so these should succeed.
set -euo pipefail
cd "$(dirname "$0")/.."

CMDS=(
    "gen_model=qwen3p5_27b pool=k10_2attrs method=rag_advanced +world_seeds=[5]"
    "gen_model=qwen3p5_27b pool=k10_2attrs method=rag_advanced +world_seeds=[7]"
    "gen_model=qwen3p5_27b pool=k10_2attrs method=rag_advanced +world_seeds=[8]"
    "gen_model=qwen3p5_27b pool=k10_2attrs method=rag_advanced +world_seeds=[9]"
    "gen_model=llama31_8b pool=k10_1attrs method=routing_margin +world_seeds=[5]"
    "gen_model=llama31_8b pool=k10_1attrs method=routing_margin +world_seeds=[6]"
    "gen_model=llama31_8b pool=k10_1attrs method=routing_margin +world_seeds=[7]"
    "gen_model=llama31_8b pool=k10_1attrs method=routing_margin +world_seeds=[8]"
    "gen_model=llama31_8b pool=k10_1attrs method=routing_margin +world_seeds=[9]"
    "gen_model=llama31_8b pool=k10_1attrs method=routing_oracle +world_seeds=[6]"
    "gen_model=llama31_8b pool=k10_1attrs method=routing_oracle +world_seeds=[7]"
    "gen_model=llama31_8b pool=k10_1attrs method=routing_oracle +world_seeds=[8]"
)

for cmd in "${CMDS[@]}"; do
    echo "sbatch sbatch/run_benchmark_v3.sbatch $cmd"
    sbatch sbatch/run_benchmark_v3.sbatch $cmd
done
echo "Submitted ${#CMDS[@]} re-runs."
