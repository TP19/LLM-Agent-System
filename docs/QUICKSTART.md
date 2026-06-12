# Quick Start Guide

> **First-run model downloads.** The Knowledge and Summarization agents need a small
> embedding model to do their RAG work. The default (`Qwen/Qwen3-Embedding-0.6B`,
> ~1.2 GB on disk in fp16) is fetched from HuggingFace the first time those agents run, and cached
> under `~/.cache/huggingface/`. After that, everything is fully offline.
>
> Reranking is **disabled by default** — when enabled, it adds a ~1.2 GB download of
> the configured reranker model. See `config/rag_config.yaml` for both knobs.
>
> For zero HuggingFace traffic from day one, point `embedding.model` at a local
> GGUF file you already have — the GGUF backend uses `llama-server` (no HF round-trip).

## Installation

```bash
# Clone and install
git clone https://github.com/TP19/LLM-Agent-System.git
cd LLM-Agent-System
./install.sh
```

## Configuration

Edit `config/models.yaml` with your model paths:

```yaml
models:
  oracle:
    model_path: "/path/to/your/model.gguf"
    n_ctx: 16384
    n_gpu_layers: 99
  operator:
    model_path: "/path/to/your/model.gguf"
    n_ctx: 8192
    n_gpu_layers: 99
```

**Recommended Models:**
- IBM Granite 3.3 8B Instruct (Q8_0) - General agents
- Nemotron-Nano-12B (Q8_0) - Good balance of speed/quality
- Any 8B+ instruct model with good reasoning

## Usage

### Start Console
```bash
./llm-agent-system
# or
python start_console.py
```

### Basic Commands
| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/agents` | List available agents |
| `/agent <name>` | Chat directly with an agent |
| `/resources` | Show system resources |
| `/exit` | Exit console |

### Natural Language
Just type naturally to chat with Oracle:
```
> check disk space
> summarize this document
> what processes are using the most memory?
```

### Agent Direct Chat
```
/agent operator
> list files in /tmp

/agent coder
> write a python function to calculate fibonacci
```

## Available Agents

| Agent | Purpose |
|-------|---------|
| **Oracle** | Natural language interface, task routing |
| **Operator** | System commands, file operations |
| **Coder** | Code generation, analysis |
| **Knowledge** | RAG-based Q&A over documents |
| **Summarizer** | Document summarization (hierarchical) |
| **Triage** | Request classification |

## RAG System

Index documents for knowledge retrieval:
```bash
python scripts/manage_rag.py index /path/to/file.pdf
python scripts/manage_rag.py index-folder /path/to/docs --recursive
python scripts/manage_rag.py stats
```

Then ask Knowledge:
```
> /agent knowledge
Knowledge>: what does the authentication module do?
```

The Knowledge agent prints sources + a verification panel; press `y` to open
the **rich chunk viewer** with vim-style navigation (`j`/`k`/`+`/`-`/`g<n>`).

See **[Ingesting documents (manage_rag)](MANAGE_RAG.md)** for the full
ingestion workflow, supported file formats, and troubleshooting.

## Tips

1. **Use /fast mode** for quick command execution without Oracle processing
2. **Sessions**: `/new myproject` creates named session, `/sessions` lists all
3. **How Oracle decides when to route**: Oracle uses a keyword-based triage
   pre-check on your message. Short conceptual questions ("what is X?", "how
   do I learn Y?") land in Oracle's chat directly. Inputs that look like
   tasks — shell commands, file paths, words like `python`, `build`,
   `deploy`, `migrate` — trigger the friendly "Triage and dispatch? [y/N]"
   prompt. So whether Oracle routes or chats depends on what your message
   looks like; the inconsistency is by design for this release.
