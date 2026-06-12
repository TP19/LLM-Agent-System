# Architecture

This document explains how LLM-Agent-System fits together. It's intentionally short —
a map you can read in five minutes before exploring the code.

## High-level shape

```
┌─────────────────────────────────────────────────────────┐
│                     Console Hub                         │
│   sessions, slash commands, background task registry    │
└───────────────────────┬─────────────────────────────────┘
                        │  (user input)
              ┌─────────▼──────────┐
              │     Oracle Agent   │  ← natural-language router
              │   + Triage Agent   │  ← lightweight pre-classifier
              └──┬───┬───┬───┬─────┘
                 │   │   │   │
   ┌─────────────┘   │   │   └──────────────┐
   ▼                 ▼   ▼                  ▼
 Security        Operator  Knowledge       Enhanced
 Agent           Agent     Agent           Summarization
 (suggest)       (execute) (RAG Q&A)       (long-doc)
                                                │
                       ┌────────────────────────┘
                       ▼
                ┌──────────────────────┐
                │   RAG Pipeline       │
                │ LanceDB + Embeddings │
                │  + Optional Reranker │
                └──────────────────────┘
```

The Console Hub is the durable surface — sessions, projects, background tasks, history.
Oracle is the brain that decides *which* agent should handle a given request. The
specialist agents do one thing each.

## Request lifecycle

```
User types in Console
        │
        ▼
Oracle receives the raw request
        │
        ▼
Triage classifies it (sysadmin / fileops / network / development /
                       content / security / coding / unknown)
        │
        ▼
Oracle picks the agent (or agent pair) for that category
        │
        ▼
Specialist agent runs; result flows back to Oracle
        │
        ▼
Oracle synthesizes / formats the reply for the Console
```

For most categories Oracle delegates to a single agent. The interesting case is
**security work**, where Oracle uses a two-agent collaboration pattern:

## The security → operator collaboration

For security-flagged requests (audit user permissions, check failed logins, review
firewall rules, scan for outdated packages, etc.), Oracle runs a **two-stage** flow:

```
1. SecurityAgent.suggest_approach(request)
        └── returns SecuritySuggestion(
                commands=[...],          # shell commands that would gather evidence
                reasoning="...",          # why these commands help
                approach="...",           # the overall strategy
                next_steps=[...],         # what to do after results land
                confidence=0.8,
            )

2. OperatorAgent.execute_task(
       user_request=...,
       security_suggestion=<the dict above>,
   )
        └── plans concrete commands from the suggestion, runs them
            (locally or via ssh), captures output, analyzes results.
```

Note the deliberate framing: **the security agent suggests, it does not block**.
It's a collaborator, not a gatekeeper. This avoids the common "safety theater"
anti-pattern where security tooling refuses to act and instead just produces
a thinking partner whose ideas flow into the executor.

This same collaboration pattern is the orchestration template for adding more
specialized agents in the future.

## Agent base class

Every agent inherits from `core.base_agent.BaseAgent`, which provides:

- A pluggable `model_manager` for inference backend (llama-server HTTP or
  llama-cpp-python in-process)
- Lazy model loading — agents don't load their model until first invoked
- Standard stats tracking (`tasks_processed`, `avg_processing_time`, etc.)
- Logging conventions (one logger per agent)
- A `_create_*_prompt()` hook for the agent's system prompt

This makes adding a new agent cheap: subclass `BaseAgent`, define a prompt, write
the handful of methods the orchestration code calls on it.

## The RAG pipeline (Knowledge + Summarization)

Both production agents lean on a shared retrieval stack:

```
docs → chunker → embedding → LanceDB (vector store)
                                  │
                                  ▼
query → embed → search → (optional rerank) → top-K → agent prompt
```

Key modules:

- `rag/embedding/` — embedding engine. Backends: GGUF via llama-server,
  HuggingFace sentence-transformers, or HTTP service.
- `rag/vector_stores/` — LanceDB-backed store (the chromadb path was removed in
  v0.2). Dual-store manager keeps private + public collections separate.
- `rag/retrieval/` — query parsing, top-K search, optional reranker.
- `rag/multimodal/` — PDF / EPUB / audio / OCR ingestion pipelines for indexing
  non-text sources.
- `utilities/semantic_chunker.py`, `utilities/chapter_chunker.py`,
  `utilities/adaptive_chunker.py` — different chunking strategies the indexer
  picks between based on document shape.

Knowledge agent uses retrieval-then-generate over this stack. Summarization
uses hierarchical chunking (chapter → section → page → paragraph) to fit
long inputs into the model context, then composes per-chunk summaries into
a coherent whole.

### Models the RAG pipeline depends on

- **Embedding model** — by default, the HuggingFace identifier
  `Qwen/Qwen3-Embedding-0.6B`. Auto-downloaded (~1.2 GB) and cached under
  `~/.cache/huggingface/` on first use. After that, fully offline. Swap for a local
  GGUF in `config/rag_config.yaml` if you want zero HF traffic.
- **Reranker** — disabled by default in v0.2. Enable it for higher-quality retrieval
  on long result sets; the default model is the IBM Granite reranker (~1.2 GB HF
  download on first use). See `config/rag_config.yaml` for both options.

No user data ever leaves the machine — these are one-time model-weight pulls.

## Console Hub

`console/console_hub.py` is the long-lived orchestrator. It owns:

- **Sessions** — per-project conversations with their own history, model
  context, working directory.
- **Slash commands** — `/help`, `/status`, `/agent`, `/project`, `/sessions`,
  `/bg`, `/tasks`, etc. Routed by `console/command_router.py`.
- **Background task registry** — `/bg <request>` runs an Oracle pipeline
  detached; you check progress with `/tasks` and `/what`. State persists
  across exits (`console/task_registry.py`).
- **Resource monitoring** — `/resources` reports CPU / GPU / RAM via psutil
  and GPUtil.

Anything not starting with `/` is treated as a natural-language request and
goes straight to Oracle.

## What's deferred

Several pieces from the internal version did not ship in v0.2:

- **Request Analyzer** — split compound requests into sub-tasks. Removed from
  this release; Oracle currently treats each input as a single task.
- **Collaborative / Harmony sessions** — multi-agent conversational modes with
  shared blackboard. Deferred — were too coupled to features that aren't
  shipping yet.
- **Workflow Executor** — long-running multi-step task runner with persisted
  state. The original module had cross-package dependencies that aren't part
  of this release, so it's stripped; Oracle returns a clear "unavailable"
  message if anything asks for it.

These may surface in a later release once they're cleaner.

## Where to start reading the code

If you want to understand the system by tracing one request through it:

1. `start_console.py` → bootstraps Console Hub
2. `console/console_hub.py` → main loop, slash command dispatch
3. `agents/oracle_agent.py` → request → triage → delegation
4. `agents/operator_agent.py` or `agents/knowledge_agent.py` → execution / RAG
5. `rag/embedding/embedding_engine.py` → embedding backends
6. `rag/vector_stores/lancedb_store.py` → the actual vector store

If you want to add a new agent: subclass `core/base_agent.py:BaseAgent`,
define `_create_*_prompt()` and the methods Oracle expects, register it in
`console/console_hub.py`'s agent map.
