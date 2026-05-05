#!/bin/bash
# Submit one job per (model, dataset) for unpersonalized response generation.
# 10 datasets x 2 models = 20 jobs, each handling ~1k users on its own.
#
# Usage:
#   ./sbatch/submit_all_unpersonalized_responses.sh           # all 10 datasets x 2 models
#   ./sbatch/submit_all_unpersonalized_responses.sh qwen3p5_27b
#   ./sbatch/submit_all_unpersonalized_responses.sh llama31_8b k10_1attrs k4_1attrs
#
# Datasets here are test-only (the benchmark only consumes _test).
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_MODELS=(qwen3p5_27b llama31_8b)
ALL_DATASETS=(k2_1attrs k2_2attrs k4_1attrs k4_2attrs k6_1attrs k6_2attrs k8_1attrs k8_2attrs k10_1attrs k10_2attrs)

if [[ $# -ge 1 ]]; then
    MODELS=("$1"); shift
else
    MODELS=("${ALL_MODELS[@]}")
fi
if [[ $# -ge 1 ]]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

for m in "${MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
        echo "Submitting: llm=${m} dataset=${d}_test"
        sbatch sbatch/generate_unpersonalized_responses.sbatch llm=${m} dataset=${d}_test
    done
done
