# LLM-Agent-System

**A small demo of multi-agent LLM orchestration — locally hosted, terminal-first**

![version](https://img.shields.io/badge/version-0.2.0-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> Pre-release staging branch. Public release: `https://github.com/TP19/LLM-Agent-System`

---

## Overview

LLM-Agent-System is a **terminal-first demo of multi-agent LLM orchestration** running on
your own hardware via [llama.cpp](https://github.com/ggerganov/llama.cpp). A central
**Oracle** routes requests to specialized agents; results stream back through a persistent
Console session.

This release is best understood as a **working reference** for the orchestration patterns,
with two components doing real production-grade work:

- **Knowledge agent** — RAG-backed Q&A over your own documents (LanceDB + embeddings +
  optional reranker). Solid and useful out of the box.
- **Enhanced Summarization agent** — long-document summarization with hierarchical
  chunking. Genuinely good at digesting big PDFs / transcripts / books.

The remaining agents (Oracle, Triage, Operator, Security, Coder) are **functional
reference implementations** of the orchestration patterns. They wire together correctly
and demonstrate how a multi-agent system fits together, but their per-agent functionality
is intentionally basic for this release. More advanced versions exist in an internal
branch and may surface later.

**Who is this for?**
- People who want a working **RAG / summarization** console they can run locally
- Developers learning multi-agent LLM orchestration patterns
- Anyone who wants local inference without sending data to the cloud

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Console Hub                         │
│  (session management, slash commands, task registry)    │
└───────────────────────┬─────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │     Oracle Agent   │  ← natural language router
              │   + Triage Agent   │  ← task classification
              └──┬──┬──┬──┬──┬────┘
                 │  │  │  │  │
    ┌────────────┘  │  │  │  └──────────────┐
    ▼               ▼  ▼  ▼                 ▼
Operator        Coder  Knowledge  Enhanced   Request
Agent           Agent  Agent      Summarizer Analyzer
(shell/SSH)    (code)  (RAG Q&A)  (docs)    (multi-task)

                 ┌──────────────────────┐
                 │   RAG Pipeline       │
                 │  LanceDB + Embeddings│
                 │  + Optional Reranker │
                 └──────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/TP19/LLM-Agent-System.git
cd LLM-Agent-System

# 2. Install (interactive — downloads llama-server, optional model)
python install.py

# 3. Configure your model
#    Edit config/models.yaml — set the path to your .gguf model

# 4. Start
python start_console.py
```

### Minimal manual install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install llama-cpp-python with GPU support:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
# Edit config/models.yaml, then:
python start_console.py
```

---

## Agents

| Agent | File | Role | Status |
|-------|------|------|--------|
| **Knowledge** | `agents/knowledge_agent.py` | RAG-backed document Q&A | ✓ Production — primary working component |
| **Enhanced Summarization** | `agents/enhanced_summarization.py` | Long-document summarization (hierarchical chunking) | ✓ Production — primary working component |
| **Oracle** | `agents/oracle_agent.py` | Natural-language router — classifies and delegates | Reference / demo |
| **Triage** | `agents/triage_agent.py` | Lightweight pre-classification | Reference / demo |
| **Operator** | `agents/operator_agent.py` | Shell commands, locally or over SSH | Reference / demo |
| **Security** | `agents/security_agent.py` | Collaborative command suggestion (works *with* Operator, not as a gatekeeper) | Reference / demo |
| **Coder** | `agents/coder_agent.py` | Code generation, analysis, and review | Reference / demo |

> `agents/models.py` is a typing/datamodel module (shared data structures), not a standalone agent.

**Production agents** (Knowledge, Summarization) are the ones to lean on for real work.
**Reference agents** demonstrate the orchestration architecture — they execute correctly
end-to-end but the per-agent intelligence is intentionally basic for this release. See
[ARCHITECTURE.md](docs/ARCHITECTURE.md) for how they fit together.

### Ingesting documents for Knowledge

`/agent knowledge` searches the LanceDB store at `~/.llm_engine/lance_db/`.
Populate it with `scripts/manage_rag.py`:

```bash
python scripts/manage_rag.py index ~/notes/report.pdf
python scripts/manage_rag.py index-folder ~/notes --recursive
python scripts/manage_rag.py stats
```

See [docs/MANAGE_RAG.md](docs/MANAGE_RAG.md) for the full ingestion workflow,
supported formats (PDF, EPUB, DOCX, MD, code, …), and troubleshooting.

---

## Console Interface

The Console Hub provides a persistent terminal session with slash commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/status` | Current session status |
| `/agents` | List available agents |
| `/agent <name>` | Chat directly with a specific agent |
| `/project <name>` | Switch to or create a project |
| `/new <name>` | Create a new session |
| `/sessions` | List sessions in current project |
| `/bg <request>` | Run a request in the background (non-blocking) |
| `/tasks` | List background tasks |
| `/what [task]` | Ask Oracle about task progress |
| `/cancel <task>` | Cancel a running background task |
| `/chunks` | Open the chunk viewer (RAG data browser) |
| `/resources` | Show CPU/GPU/memory usage |
| `/fast` | Toggle fast/ephemeral mode |
| `/diag` | Diagnostics and log export |
| `/exit` | Exit the console |

Any input that does not start with `/` is routed to Oracle as a natural language request.

---

## Configuration

### `config/models.yaml`

Defines model paths and backend settings for each agent. Generated by `install.py` based
on detected GPU.

Key fields:
- `model_path`: Path to the `.gguf` model file
- `backend`: `"llama-server"` or `"llama-cpp-python"`
- `llama_server_path`: Path to `llama-server` binary (overridden by `$LLM_ENGINE_LLAMA_SERVER`)
- `n_gpu_layers`: GPU layer offload (0 = CPU-only, 99 = full GPU)

### `config/rag_config.yaml`

Controls the RAG (Retrieval-Augmented Generation) pipeline:
- Embedding model (HuggingFace ID or local `.gguf` path)
- LanceDB storage paths
- Reranker model (disabled by default — see below)
- Chunking strategy

Set `$LLM_ENGINE_LLAMA_SERVER` to override the llama-server binary path used by the
embedding engine and reranker.

#### Model downloads (read this before first use)

The Knowledge and Summarization agents use a small RAG pipeline that depends on
two transformer models:

| Model | Default | Size | When fetched |
|---|---|---|---|
| Embedding | `Qwen/Qwen3-Embedding-0.6B` | ~1.2 GB | First Knowledge or Summarizer call |
| Reranker | (disabled) | ~1.2 GB if enabled | Only if you set `reranking.enable: true` |

Both defaults are HuggingFace identifiers. On first use, `sentence-transformers`
auto-downloads them to `~/.cache/huggingface/` and caches them locally; subsequent
runs are fully offline. **No data is sent anywhere** — these are one-time model-weight
fetches from `huggingface.co`.

If you want zero HF traffic from the start, point `embedding.model` at a local GGUF
file in `config/rag_config.yaml`. The GGUF path uses `llama-server` for inference, no
HF round-trip needed.

Reranking is **off by default** in v0.2 to keep the first-run experience tight. It's a
quality enhancement over plain vector search — enable it once you've decided what
reranker model you want.

---

## Requirements

- **Python**: 3.10 or higher
- **llama-server**: From [llama.cpp](https://github.com/ggerganov/llama.cpp) — install
  via `python install.py` (downloads a pre-built binary; Linux only) or build from source
- **GPU**: Optional — NVIDIA (CUDA) or AMD (ROCm). CPU inference works but is slow.
- **RAM**: 8 GB minimum; 16 GB+ recommended for 7B+ models
- **Platform**: Tested on Linux. The manual `python -m venv` install path should work on
  macOS / Windows too — you'll just need to install llama.cpp's `llama-server` yourself
  for those platforms.

### Python dependencies (key)

| Package | Purpose |
|---------|---------|
| `lancedb` | Vector store |
| `sentence-transformers` | Embeddings |
| `rich` | Terminal UI |
| `pyyaml` | Configuration |
| `psutil` | System monitoring |
| `torch` | ML backend |

Full list in `requirements.txt`.

---

## License

MIT — see [LICENSE](LICENSE).
