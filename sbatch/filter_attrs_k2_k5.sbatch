#!/bin/bash
#SBATCH --job-name=llm-pers-filter-attrs
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%A.out
#SBATCH --time=1-00:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
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

if [ -f "${HOME_DIR}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME_DIR}/.hf_token" | tr -d '[:space:]')"
elif [ -f "${HOME}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token" | tr -d '[:space:]')"
fi

conda activate llm-pers
pip install -r requirements.txt
pip install -e .

echo "========================================"
echo "Filtering synthetic conversation datasets"
echo "========================================"

set -x
python experiments/synthetic_conversations/scripts/filter_by_user_attributes.py --input_path data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_hf --attributes_file experiments/synthetic_conversations/configs/attributes/user_prompt_attributes_k2.yaml --output_path data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_first2_hf
python experiments/synthetic_conversations/scripts/filter_by_user_attributes.py --input_path data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_hf --attributes_file experiments/synthetic_conversations/configs/attributes/user_prompt_attributes_k5.yaml --output_path data/synthetic_conversations/gpt-oss-120b_k10_1attrs_16000_1000_first5_hf
set +x

echo "========================================"
echo "Done!"
echo "========================================"
