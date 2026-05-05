#!/bin/bash
# Submit one job per (model, pool) for unpersonalized response judging.
# 10 pools x 2 models = 20 jobs.
#
# Usage:
#   ./sbatch/submit_all_unpersonalized_judging.sh
#   ./sbatch/submit_all_unpersonalized_judging.sh qwen3p5_27b
#   ./sbatch/submit_all_unpersonalized_judging.sh llama31_8b k10_1attrs
#
# `attributes=kN` is auto-derived from the pool tag.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_MODELS=(qwen3p5_27b llama31_8b)
ALL_POOLS=(k2_1attrs k2_2attrs k4_1attrs k4_2attrs k6_1attrs k6_2attrs k8_1attrs k8_2attrs k10_1attrs k10_2attrs)

if [[ $# -ge 1 ]]; then
    MODELS=("$1"); shift
else
    MODELS=("${ALL_MODELS[@]}")
fi
if [[ $# -ge 1 ]]; then
    POOLS=("$@")
else
    POOLS=("${ALL_POOLS[@]}")
fi

for m in "${MODELS[@]}"; do
    for p in "${POOLS[@]}"; do
        k=${p%%_*}                              # k10 from k10_1attrs
        echo "Submitting: dataset=${m}_${p}_test attributes=${k}"
        sbatch sbatch/judge_unpersonalized_responses.sbatch \
            dataset=${m}_${p}_test attributes=${k}
    done
done
