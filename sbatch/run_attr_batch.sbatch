#!/bin/bash
# Batched attribute-only benchmark over 3 personalization systems x 5 seeds.
# Slurm array indexes 0..14:
#   task // 5  -> system  (0=soft_token, 1=rag, 2=attrs_k10)
#   task %  5  -> seed    (0..4)
# All runs use 4000 train / 1000 test users on the k10_1attrs dataset.
#SBATCH --job-name=llm-pers-attr-batch
#SBATCH --output=/lustre/groups/eml/projects/pspohn/llm-personalization-v4/outputs/%x-%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --constraint=h100_80gb
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-14

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

# Map array task -> (system, seed)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SYSTEM_IDX=$(( TASK_ID / 5 ))
SEED=$(( TASK_ID % 5 ))

case ${SYSTEM_IDX} in
    0) SYSTEM="soft_token" ;;
    1) SYSTEM="rag" ;;
    2) SYSTEM="attrs_k10" ;;
    *) echo "Unknown SYSTEM_IDX=${SYSTEM_IDX}" && exit 1 ;;
esac

echo "========================================"
echo "Task ${TASK_ID}: system=${SYSTEM} seed=${SEED}"
echo "========================================"

# Per-system env + config setup. soft_token needs the experimental env (Gemma 4
# requires newer transformers); the others use the standard env.
COMMON_OVERRIDES=(
    "num_worlds=1"
    "+world_seeds=[${SEED}]"
    "attribute_personalization_dataset.train_limit=4000"
    "attribute_personalization_dataset.test_limit=1000"
)

case ${SYSTEM} in
    soft_token)
        conda activate llm-pers-exp
        pip install -e .
        pip install peft --no-deps
        pip install transformers==5.5.3 --no-deps
        pip install huggingface-hub==1.10.1 --no-deps

        # FlashInfer/vLLM MoE JIT needs cublasLt.h and cccl headers.
        pip install nvidia-cublas-cu12 nvidia-cuda-cccl-cu12 >/dev/null
        CUBLAS_INC="$(python -c 'import nvidia.cublas, os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), "include"))')"
        CUBLAS_LIB="$(python -c 'import nvidia.cublas, os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib"))')"
        CCCL_INC="$(python -c 'import nvidia.cuda_cccl, os; print(os.path.join(os.path.dirname(nvidia.cuda_cccl.__file__), "include"))')"
        export CPATH="${CUBLAS_INC}:${CCCL_INC}:${CPATH}"
        export LIBRARY_PATH="${CUBLAS_LIB}:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${CUBLAS_LIB}:${LD_LIBRARY_PATH}"

        CONFIG_NAME="config_medium_soft_token"
        EXTRA_OVERRIDES=(
            "~persona_personalization_dataset"
            "~persona_personalization_system"
            "~persona_judge"
            "~robustness"
            "output_dir='data/run_benchmark/{timestamp}_attr_batch_soft_token_seed${SEED}'"
        )
        ;;
    rag)
        conda activate llm-pers
        pip install -r requirements.txt
        pip install -e .

        CONFIG_NAME="config_medium_rag"
        EXTRA_OVERRIDES=(
            "~persona_personalization_dataset"
            "~persona_personalization_system"
            "~persona_judge"
            "~robustness"
            "output_dir='data/run_benchmark/{timestamp}_attr_batch_rag_seed${SEED}'"
        )
        ;;
    attrs_k10)
        conda activate llm-pers
        pip install -r requirements.txt
        pip install -e .

        # config_medium_attrs_k10_1attrs already has 4000/1000 + persona/robustness=null.
        CONFIG_NAME="config_medium_attrs_k10_1attrs"
        EXTRA_OVERRIDES=(
            "output_dir='data/run_benchmark/{timestamp}_attr_batch_attrs_k10_seed${SEED}'"
        )
        ;;
esac

set -x
HYDRA_FULL_ERROR=1 python experiments/run_benchmark/scripts/run_benchmark.py \
    --config-name ${CONFIG_NAME} \
    "${COMMON_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
set +x

echo "========================================"
echo "Done! task=${TASK_ID} system=${SYSTEM} seed=${SEED}"
echo "========================================"
