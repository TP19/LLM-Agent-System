# LLM-Agent-System

**A small demo of multi-agent LLM orchestration — locally hosted, terminal-first.**

This release is best understood as a **working reference** for the orchestration patterns,
with two components doing real production-grade work:

- **Knowledge agent** — RAG-backed Q&A over your own documents. Solid and useful out of the box.
- **Enhanced Summarization agent** — long-document summarization with hierarchical chunking. Genuinely good at digesting big PDFs, transcripts, or books.

Everything else (Oracle, Triage, Operator, Security, Coder) demonstrates the multi-agent
orchestration architecture — functional reference implementations rather than polished
production agents. More advanced versions exist in an internal branch and may surface later.

## Get started

- **[Quick Start](QUICKSTART.md)** — install in a few minutes, point at a `.gguf` model, launch the console.
- **[Dependencies](DEPENDENCIES.md)** — core + optional packages, GPU notes, doc/audio/image processing extras.
- **[Architecture](ARCHITECTURE.md)** — how the agents fit together, the oracle → security → operator flow, the RAG pipeline.

## What's in the box

| Agent | Status | What it does |
|---|---|---|
| Knowledge | ✓ Production | RAG Q&A over indexed documents |
| Enhanced Summarization | ✓ Production | Long-form / hierarchical summarization |
| Oracle | Reference | Natural-language router; classifies + delegates |
| Triage | Reference | Lightweight task pre-classification |
| Operator | Reference | Shell commands, locally or over SSH |
| Security | Reference | Collaborative command suggestion (paired with Operator) |
| Coder | Reference | Code generation, analysis |

`agents/models.py` is shared dataclasses, not a standalone agent.

## Source

- GitHub: [TP19/LLM-Agent-System](https://github.com/TP19/LLM-Agent-System)
- License: MIT
- Platform: Tested on Linux; manual venv install should work on macOS / Windows.
