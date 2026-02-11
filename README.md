# Sparse Kernel Extension

> **Note:** This repository is included as a **git submodule** in [OpenBMB/sglang (minicpm_sala branch)](https://github.com/OpenBMB/sglang/tree/minicpm_sala).
> For the latest setup instructions and usage guide, please refer to the main repository.

## Install

```shell
python3 setup.py install
```

--------------------------------------------------------------------------------

# MiniCPM-SALA Inference Environment Setup

## Requirements

- CUDA 12.x or higher
- `gcc` / `g++` compiler
- `uv` package manager (script will check)

## Quick Start

### Installation

```bash
# Clone repository
git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
cd sglang

# One-click installation (creates venv and compiles all dependencies)
bash install_minicpm_sala.sh

# Or specify PyPI mirror
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

The installation script performs the following steps:

1. Creates `sglang_minicpm_sala_env` virtual environment (Python 3.12)
2. Clones dependencies to `3rdparty/` (infllmv2) and initializes submodules (sparse_kernel)
3. Installs MiniCPM-SALA (current repo)
4. Compiles and installs `infllmv2_cuda_impl`
5. Compiles and installs `sparse_kernel`
6. Installs `tilelang` & `flash-linear-attention`

### Usage

```bash
# Activate environment
source sglang_minicpm_sala_env/bin/activate

# Launch Inference Server (Replace MODEL_PATH with actual path)
MODEL_PATH=/path/to/your/model

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse
```

| Parameter | Description |
|-----------|-------------|
| `--trust-remote-code` | Allow custom code in model |
| `--disable-radix-cache` | Disable RadixAttention prefix cache |
| `--attention-backend minicpm_flashinfer` | Use MiniCPM FlashInfer backend |
| `--chunked-prefill-size 8192` | Chunked prefill size |
| `--max-running-requests 32` | Max concurrent requests |
| `--skip-server-warmup` | Skip server warmup |
| `--port 31111` | Server port |
| `--dense-as-sparse` | Use dense-as-sparse mode |

> **Tip:** For best generation quality, we recommend setting `temperature=0.9` when sending requests to the server.

## Manual Installation

If the script doesn't work for you, follow these steps:

```bash
# 0. Ensure uv is installed
pip install uv

# 1. Create venv
uv venv --python 3.12 sglang_minicpm_sala_env
source sglang_minicpm_sala_env/bin/activate

# 2. Install SGLang
uv pip install --upgrade pip setuptools wheel
uv pip install -e ./python[all]

# 3. Compile CUDA Extensions
# (Ensure dependencies are cloned to 3rdparty/)
cd 3rdparty/infllmv2_cuda_impl && python setup.py install && cd ../..
cd 3rdparty/sparse_kernel && python setup.py install && cd ../..

# 4. Install extra deps
uv pip install tilelang flash-linear-attention
```

## Q&A

**Q: CUDA extension compilation failed?**

- Ensure CUDA 12+ is installed (`nvcc --version`).
- Ensure `gcc` / `g++` are available.
- If `CXX` is set to `clang++ -pthread`, manually `export CXX=g++`.
