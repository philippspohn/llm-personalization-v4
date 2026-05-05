#!/bin/bash
#SBATCH --job-name=llm-pers-bench-v2
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --constraint=h100_80gb
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Generic v2-benchmark sbatch. Method, dataset and any other Hydra overrides
# come from the CLI:
#   sbatch sbatch/benchmark_v2.sbatch method=oracle_routing
#
# 2 GPUs are sized for Qwen3.5-27B (TP=2, BF16) generation. The judge
# (gpt-oss-120b TP=1) loads sequentially after generation finishes. Bump
# --gres=gpu:N + the LLM config's tensor_parallel_size if you swap in a
# bigger generation model (e.g. 70B family wants 4).
#
# By default we hit huggingface.co (so first-time model downloads work).
# Once the generation + judge models are both cached locally, set the env
# var HF_OFFLINE=1 when submitting to avoid the intermittent DNS failures
# we sometimes see on compute nodes:
#   HF_OFFLINE=1 sbatch sbatch/benchmark_v2.sbatch method=baseline

SCRATCH_DIR="/lustre/groups/eml/projects/pspohn/llm-personalization-v2/tmp"
HOME_DIR="/home/eml/philipp.spohn"
PROJECTS_DIR="/lustre/groups/eml/projects/pspohn/llm-personalization-v4"

source ${HOME_DIR}/.bashrc
cd "${PROJECTS_DIR}"

export TRANSFORMERS_CACHE="${SCRATCH_DIR}/hf_cache"
export HF_HOME="${SCRATCH_DIR}/hf_home"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

# Opt-in HF offline mode (set HF_OFFLINE=1 in the submission environment).
# Useful once weights are cached, to dodge the intermittent compute-node DNS
# failures we sometimes see when reaching huggingface.co.
if [ "${HF_OFFLINE:-0}" = "1" ]; then
    echo "[sbatch] HF offline mode enabled (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

export PYTHONPATH="${PROJECTS_DIR}/src:${PROJECTS_DIR}:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True

export VLLM_CACHE_ROOT="${SCRATCH_DIR}/vllm_cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/triton_cache/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_DIR}/inductor_cache/${SLURM_JOB_ID}"
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
echo "Running run_benchmark_v2"
echo "  args: $@"
echo "========================================"

set -x
python experiments/run_benchmark_v2/scripts/run_benchmark.py "$@"
set +x

echo "========================================"
echo "Done!"
echo "========================================"
