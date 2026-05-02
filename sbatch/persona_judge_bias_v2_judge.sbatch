#!/bin/bash
#SBATCH --job-name=pjb-v2-judge
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

export PYTHONPATH="${PROJECTS_DIR}/src:${PROJECTS_DIR}:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -f "${HOME_DIR}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME_DIR}/.hf_token" | tr -d '[:space:]')"
fi

conda activate llm-pers
pip install -r requirements.txt
pip install -e .

python experiments/persona_judge_bias_v2/scripts/judge.py
