#!/bin/bash
#SBATCH --job-name=llm-pers-soft-token-small
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%A.out
#SBATCH --time=12:00:00
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

# Use the experimental env which supports Gemma 4 (newer transformers)
conda activate llm-pers-exp
pip install -e .
pip install peft --no-deps

# transformers==5.5.3 needed for gemma 4 + huggingface_hub>=1.x compatibility
pip install transformers==5.5.3 --no-deps
pip install huggingface-hub==1.10.1 --no-deps

# Ensure FlashInfer / vLLM MoE JIT can find cublasLt.h and cccl headers.
# These wheels ship headers under site-packages/nvidia/<lib>/include.
pip install nvidia-cublas-cu12 nvidia-cuda-cccl-cu12 >/dev/null
CUBLAS_INC="$(python -c 'import nvidia.cublas, os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), "include"))')"
CUBLAS_LIB="$(python -c 'import nvidia.cublas, os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib"))')"
CCCL_INC="$(python -c 'import nvidia.cuda_cccl, os; print(os.path.join(os.path.dirname(nvidia.cuda_cccl.__file__), "include"))')"
export CPATH="${CUBLAS_INC}:${CCCL_INC}:${CPATH}"
export LIBRARY_PATH="${CUBLAS_LIB}:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUBLAS_LIB}:${LD_LIBRARY_PATH}"
echo "CPATH=${CPATH}"
echo "LIBRARY_PATH=${LIBRARY_PATH}"

echo "========================================"
echo "Running LLM Personalization Soft-Token SMOKE TEST (tiny data)"
echo "========================================"
echo "Args: $@"

# Smoke test: 10 train, 10 test, 1 world, no persona, no robustness.
set -x
HYDRA_FULL_ERROR=1 python experiments/run_benchmark/scripts/run_benchmark.py \
    --config-name config_medium_soft_token \
    attribute_personalization_dataset.train_limit=10 \
    attribute_personalization_dataset.test_limit=10 \
    num_worlds=1 \
    personalization_system.dpo_train_kwargs.epochs=1 \
    personalization_system.dpo_train_kwargs.batch_size=1 \
    personalization_system.dpo_train_kwargs.grad_accum_steps=1 \
    personalization_system.dpo_train_kwargs.log_every=1 \
    personalization_system.dpo_train_kwargs.max_length=1024 \
    personalization_system.generation_kwargs.max_new_tokens=64 \
    personalization_system.generation_kwargs.batch_size=1 \
    unpersonalized_llm_helper.gpu_memory_utilization=0.75 \
    "output_dir='data/run_benchmark/{timestamp}_soft_token_smoke'" \
    "~persona_personalization_dataset" \
    "~persona_personalization_system" \
    "~persona_judge" \
    "~robustness"
set +x

echo "========================================"
echo "Done!"
echo "========================================"
