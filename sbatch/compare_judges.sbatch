#!/bin/bash
#SBATCH --job-name=llm-pers
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%A.out
#SBATCH --time=1-00:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --constraint=h100_80gb
#SBATCH --gres=gpu:2
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

# Python path
export PYTHONPATH="${PROJECTS_DIR}/src:${PROJECTS_DIR}:${PYTHONPATH}"

# vLLM multiprocessing fix (avoid CUDA fork issues)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Hugging Face token
if [ -f "${HOME_DIR}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME_DIR}/.hf_token" | tr -d '[:space:]')"
elif [ -f "${HOME}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token" | tr -d '[:space:]')"
fi

# Source venv (update path if needed)
conda activate llm-pers-exp
pip install -e .

# transformers==5.5.3 needed for gemma 4 + huggingface_hub>=1.x compatibility
# (huggingface_hub 1.x removed is_offline_mode which transformers 4.x still imports)
pip install transformers==5.5.3 --no-deps
pip install huggingface-hub==1.10.1 --no-deps


echo "========================================"
echo "Running Compare Judges"
echo "========================================"
echo "Args: $@"

set -x
HYDRA_FULL_ERROR=1 python experiments/compare_judges/scripts/compare_judges.py
set +x

echo "========================================"
echo "Done!"
echo "========================================"
