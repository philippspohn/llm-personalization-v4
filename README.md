# llm-personalization-v4

## Setup

### 1. Install Miniconda (once, no sudo needed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /data/pspohn/miniconda3
/data/pspohn/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2. Create and activate environment

```bash
conda create -n llm-pers python=3.12 -y
conda activate llm-pers
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit -y
pip install -r requirements.txt
pip install -e .
```

> **Note:** Until vLLM 0.17.0 is released, install vLLM from nightly (required for Qwen3.5 support),
> then reinstall transformers since the nightly will downgrade it:
> ```bash
> pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
> pip install "transformers>=5.2.0" "huggingface-hub>=1.5.0"
> ```

See [CLUSTER_NOTES.md](CLUSTER_NOTES.md) for cluster-specific issues and workarounds.

### 3. Cache configuration

All caches live on the data volume. Add to `~/.bashrc`:

```bash
export HF_HOME=/data/pspohn/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/pspohn/.cache/huggingface/hub
export VLLM_CACHE_ROOT=/data/pspohn/.cache/vllm
export FLASHINFER_CACHE_DIR=/data/pspohn/.cache/flashinfer

# Needed for FlashInfer JIT compilation to find libcuda.so (compile-time and runtime)
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```
