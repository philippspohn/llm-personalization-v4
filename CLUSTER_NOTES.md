# Cluster Notes (eml-tum-server-1)

Issues encountered and workarounds for the current cluster environment.

## Missing system packages (no sudo)

### python3.12-dev not installed
Some packages try to compile C extensions at runtime and need Python development headers (`Python.h`). The system does not have `python3.12-dev` installed.

**Workaround:** Use Miniconda — its Python distribution includes the headers, so no system package is needed.

### CUDA toolkit (nvcc) not installed
FlashInfer JIT-compiles CUDA kernels at runtime (e.g. for Qwen3.5's gated delta network architecture). This requires `nvcc`, which is not on the system PATH and `/usr/local/cuda` does not exist. Only the CUDA runtime libraries are installed system-wide.

**Workaround:** Install the CUDA toolkit into the conda environment (no sudo):
```bash
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit -y
```

Additionally, the conda linker cannot find `libcuda.so` at compile time. `LIBRARY_PATH` controls compile-time library search (used by `ld`), while `LD_LIBRARY_PATH` controls runtime. Both are needed. Add to `~/.bashrc`:
```bash
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```
# sudo apt install cuda-toolkit-12-8

