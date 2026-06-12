# LLM-Agent-System - Dependencies Guide

## Core Dependencies (Required)

Installed via `requirements.txt`:
- `llama-cpp-python` - Local LLM inference
- `lancedb` - Vector storage
- `sentence-transformers` - Embeddings
- `rich` - Terminal UI
- `pyyaml` - Configuration
- `pydantic` - Data validation

## Document Processing

- `ebooklib` - EPUB reading
- `PyPDF2` / `pdfplumber` - PDF processing
- `python-docx` - Word documents
- `Pillow` - Image processing

## GPU Support

### llama-cpp-python with CUDA (NVIDIA)
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### llama-cpp-python with ROCm (AMD)
```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### llama-server (Alternative)
If llama-cpp-python fails, use llama-server backend:
```yaml
# config/models.yaml
oracle:
  backend: "llama-server"
  llama_server_path: "~/.local/bin/llama-server"
  llama_server_port: 8080
```

Build llama.cpp:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir -p build && cd build
cmake .. -DGGML_CUDA=on
cmake --build . --config Release -j
```

## Recommended Models

### Lightweight (7-8GB VRAM)
- **Qwen2.5-7B-Instruct Q8_0** (~7GB) - Excellent reasoning
  ```
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
  ```
- **Gemma-2-9B-Instruct Q6_K** (~7GB) - Fast inference
  ```
  https://huggingface.co/bartowski/gemma-2-9b-it-GGUF
  ```
- **Granite-8B-Instruct Q8_0** (~8GB) - Good for code
  ```
  https://huggingface.co/ibm-granite/granite-8b-code-instruct-GGUF
  ```
- **Llama-3.1-8B-Instruct Q8_0** (~8GB) - Well-rounded
  ```
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
  ```

### Larger (16GB+ VRAM)
- **Qwen2.5-14B-Instruct Q6_K** (~12GB) - Better reasoning
- **Llama-3.1-70B Q4_K_M** (~40GB) - Top performance

## Vector Store

LanceDB is the vector store backend:
```bash
pip install lancedb pyarrow
```

## Installation Summary

```bash
# Quick install
python install.py

# Manual install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install lancedb pyarrow
```
