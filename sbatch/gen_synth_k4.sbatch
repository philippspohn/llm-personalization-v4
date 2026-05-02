#!/bin/bash
#SBATCH --job-name=llm-pers-gen-k4
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%A.out
#SBATCH --time=1-00:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --constraint=h100_80gb
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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

echo "========================================"
echo "Generating synthetic conversations (k4)"
echo "========================================"

set -x
python experiments/synthetic_conversations/scripts/generate_conversations_chunked.py \
    attributes=user_prompt_attributes_k4 \
    llm.tensor_parallel_size=4 \
    train_output_path=data/synthetic_conversations/gpt-oss-120b_k4_2attrs_16000_1000_train.jsonl \
    test_output_path=data/synthetic_conversations/gpt-oss-120b_k4_2attrs_16000_1000_test.jsonl
set +x

echo "========================================"
echo "Done!"
echo "========================================"
