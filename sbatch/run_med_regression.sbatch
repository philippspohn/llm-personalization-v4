#!/bin/bash
#SBATCH --job-name=llm-pers-regression
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
conda activate llm-pers
pip install -r requirements.txt
pip install -e .

echo "========================================"
echo "Running LLM Personalization Training"
echo "========================================"
echo "Args: $@"

set -x
python experiments/run_benchmark/scripts/run_benchmark.py --config-name config_medium \
    personalization_system.attribute_selection=regression \
    persona_personalization_system.attribute_selection=regression \
    "output_dir='data/run_benchmark/{timestamp}_medium_regression'"
set +x

echo "========================================"
echo "Done!"
echo "========================================"
