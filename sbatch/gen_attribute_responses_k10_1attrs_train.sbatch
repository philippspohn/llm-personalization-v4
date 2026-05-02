#!/bin/bash
#SBATCH --job-name=llm-pers-attr-resp-k10-1attrs-train
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --constraint=h100_80gb
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Generate Nemotron attribute responses for the
# gpt-oss-120b_k10_1attrs_16000_1000_train.jsonl synthetic-conversations file.
# Always submit as a 4-way array, e.g.:
#   sbatch --array=0-3 sbatch/gen_attribute_responses_k10_1attrs_train.sbatch
# Each task uses tensor_parallel_size=2 across the 2 GPUs of the job.

SCRATCH_DIR="/lustre/groups/eml/projects/pspohn/llm-personalization-v2/tmp"
HOME_DIR="/home/eml/philipp.spohn"
PROJECTS_DIR="/lustre/groups/eml/projects/pspohn/llm-personalization-v4"

source ${HOME_DIR}/.bashrc

cd "${PROJECTS_DIR}"

export TRANSFORMERS_CACHE="${SCRATCH_DIR}/hf_cache"
export HF_HOME="${SCRATCH_DIR}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

# Use the on-disk HF cache only; cluster nodes intermittently fail to
# resolve huggingface.co and abort vLLM model loading.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTHONPATH="${PROJECTS_DIR}/src:${PROJECTS_DIR}:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export VLLM_CACHE_ROOT="${SCRATCH_DIR}/vllm_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/triton_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_DIR}/inductor_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

if [ -f "${HOME_DIR}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME_DIR}/.hf_token" | tr -d '[:space:]')"
elif [ -f "${HOME}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token" | tr -d '[:space:]')"
fi

conda activate llm-pers
pip install -r requirements.txt
pip install -e .

echo "========================================"
echo "Generating attribute responses (k=10, 1attrs, train)"
echo "  array task: ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
echo "========================================"
echo "Args: $@"

set -x
python experiments/attribute_responses/scripts/generate_attribute_responses.py \
    attributes=k10 \
    llm.tensor_parallel_size=2 \
    input_path=data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_train.jsonl \
    split_name=train \
    output_dir=data/attribute_responses/gpt-oss-120b_k10_1attrs_nemotron \
    input_limit=9000 \
    "$@"
set +x

echo "========================================"
echo "Done!"
echo "========================================"
