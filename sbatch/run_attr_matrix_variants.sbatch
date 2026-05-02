#!/bin/bash
# Routing-only sweep over 3 world-matrix variants (1 seed each, 4000 train / 1000 test).
# task 0 -> dense_n1, task 1 -> dense_n2, task 2 -> perm_n2.
#SBATCH --job-name=llm-pers-matrix-variants
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --constraint=h100_80gb
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-2

SCRATCH_DIR="/lustre/groups/eml/projects/pspohn/llm-personalization-v2/tmp"
HOME_DIR="/home/eml/philipp.spohn"
PROJECTS_DIR="/lustre/groups/eml/projects/pspohn/llm-personalization-v4"

source ${HOME_DIR}/.bashrc
cd "${PROJECTS_DIR}"

export TRANSFORMERS_CACHE="${SCRATCH_DIR}/hf_cache"
export HF_HOME="${SCRATCH_DIR}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

export PYTHONPATH="${PROJECTS_DIR}/src:${PROJECTS_DIR}:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -f "${HOME_DIR}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME_DIR}/.hf_token" | tr -d '[:space:]')"
elif [ -f "${HOME}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token" | tr -d '[:space:]')"
fi

conda activate llm-pers
pip install -r requirements.txt
pip install -e .

TASK_ID=${SLURM_ARRAY_TASK_ID}
case ${TASK_ID} in
    0) CONFIG_NAME="config_medium_attrs_k10_1attrs_dense_n1" ;;
    1) CONFIG_NAME="config_medium_attrs_k10_1attrs_dense_n2" ;;
    2) CONFIG_NAME="config_medium_attrs_k10_1attrs_perm_n2" ;;
    *) echo "Unknown TASK_ID=${TASK_ID}" && exit 1 ;;
esac

SEED=0

echo "========================================"
echo "Task ${TASK_ID}: config=${CONFIG_NAME} seed=${SEED}"
echo "========================================"

set -x
HYDRA_FULL_ERROR=1 python experiments/run_benchmark/scripts/run_benchmark.py \
    --config-name "${CONFIG_NAME}" \
    num_worlds=1 \
    "+world_seeds=[${SEED}]" \
    attribute_personalization_dataset.train_limit=4000 \
    attribute_personalization_dataset.test_limit=1000
set +x

echo "========================================"
echo "Done! task=${TASK_ID} config=${CONFIG_NAME}"
echo "========================================"
