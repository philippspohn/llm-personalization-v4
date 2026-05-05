#!/bin/bash
#SBATCH --job-name=llm-pers-v3
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%A.out
#SBATCH --time=1-00:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --constraint=h100_80gb
#SBATCH --gres=gpu:1
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

# All required models are fully cached — skip online lookups that intermittently
# fail on cluster nodes with flaky DNS to huggingface.co.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# gpt-oss-120b (MXFP4) on a single H100 needs the expandable allocator to
# avoid OOM during _swizzle_mxfp4 right after weight load. See judge sbatch.
export PYTORCH_ALLOC_CONF=expandable_segments:True

export VLLM_CACHE_ROOT="${SCRATCH_DIR}/vllm_cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/triton_cache/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_DIR}/torchinductor_cache/${SLURM_JOB_ID}"
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
echo "Running LLM Personalization Benchmark v3"
echo "========================================"
echo "Args: $@"

set -x
python experiments/run_benchmark_v3/scripts/run_benchmark.py "$@"
set +x

echo "========================================"
echo "Done!"
echo "========================================"
