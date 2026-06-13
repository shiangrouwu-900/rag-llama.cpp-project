# Google Colab T4 GPU setup

Use `Runtime > Change runtime type > T4 GPU` before running these cells.

## 1. Check GPU and CUDA

```bash
!nvidia-smi
!nvcc --version
```

## 2. Clone this branch

```bash
!git clone -b codex/generalized-cpu-rag https://github.com/shiangrouwu-900/rag-llama.cpp-project.git
%cd rag-llama.cpp-project
```

If the directory already exists:

```bash
%cd /content/rag-llama.cpp-project
!git fetch origin codex/generalized-cpu-rag
!git checkout codex/generalized-cpu-rag
!git pull
```

## 3. Install CUDA-enabled llama-cpp-python and RAG dependencies

Colab T4 usually runs on a CUDA 12.x image. Pick the wheel URL that matches `nvcc --version`; for example, CUDA 12.4 uses `cu124`.

```bash
!python -m pip install -U pip setuptools wheel
!python -m pip uninstall -y llama-cpp-python
!CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python -m pip install --no-cache-dir --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
!python -m pip install -U numpy scipy scikit-learn sentence-transformers huggingface-hub tqdm
```

If the CUDA wheel is not available for the current Colab image, build from source instead. It takes longer, but is the most reliable fallback:

```bash
!python -m pip uninstall -y llama-cpp-python
!CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python -m pip install --no-cache-dir --force-reinstall llama-cpp-python
```

## 4. Download Qwen3-1.7B Q4_K_M GGUF

```python
from huggingface_hub import hf_hub_download
from pathlib import Path

Path("models").mkdir(exist_ok=True)
model_path = hf_hub_download(
    repo_id="unsloth/Qwen3-1.7B-GGUF",
    filename="Qwen3-1.7B-Q4_K_M.gguf",
    local_dir="models",
)
print(model_path)
```

## 5. Configure runtime parameters

`N_GPU_LAYERS=-1` offloads all possible layers to the T4 GPU. For Qwen3-1.7B Q4_K_M this should fit comfortably on T4 VRAM.

```python
import os

os.environ["MODEL_PATH"] = "models/Qwen3-1.7B-Q4_K_M.gguf"
os.environ["N_GPU_LAYERS"] = "-1"
os.environ["N_CTX"] = "3072"
os.environ["N_THREADS"] = "4"
os.environ["TOP_K"] = "4"
os.environ["MAX_TOKENS"] = "192"
os.environ["TEMPERATURE"] = "0.0"
os.environ["TOP_P"] = "1.0"
```

## 6. Verify GPU offload

```python
import os
from llama_cpp import Llama

llm = Llama(
    model_path=os.environ["MODEL_PATH"],
    n_gpu_layers=int(os.environ["N_GPU_LAYERS"]),
    n_ctx=1024,
    verbose=True,
)
print("llama-cpp-python loaded")
```

In the output log, look for CUDA / GPU offload messages. If it says all layers are running on CPU, reinstall `llama-cpp-python` with the CUDA wheel or the source-build fallback above.

## 7. Build or refresh the index

The branch already includes `storage/chunks.json` and `storage/embeddings.npy`, so this is optional. Run it when you edit source data or want to verify that indexing works in Colab.

```bash
!REBUILD_INDEX=1 python -m rag.build_index
```

## 8. Run main.py

Interactive mode:

```bash
!python main.py
```

One-shot smoke test:

```bash
!printf "AORUS MASTER 16 BZH 的 GPU memory 是多少？\nq\n" | python main.py
```

## 9. Run evaluation.py

```bash
!python evaluation.py
```

## 10. Optional VRAM test

```bash
!python vram_test.py --csv storage/colab_t4_vram_samples.csv
```

