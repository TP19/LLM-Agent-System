# manage_rag.py — the ingestion + RAG store tool

`scripts/manage_rag.py` is the entry point for **everything that touches the
LanceDB store**. The Knowledge agent and the RAG side of the Summarization
agent both query whatever you put in here. If `/agent knowledge` returns
"no information in the knowledge base", it's because nothing has been indexed
yet — that's what this tool is for.

The store lives at `~/.llm_engine/lance_db/{private,public}/` and is reused
across console sessions.

## Quick reference

```bash
# Activate the venv (manage_rag uses the same deps as the console)
source .venv/bin/activate

# Index a single file (PDF, EPUB, TXT, MD, code, docx, ...)
python scripts/manage_rag.py index ~/notes/report.pdf

# Index a whole folder, recursively
python scripts/manage_rag.py index-folder ~/notes --recursive

# Limit to certain extensions
python scripts/manage_rag.py index-folder ~/code --extensions .py .md

# See what's in the store
python scripts/manage_rag.py stats         # high-level stats
python scripts/manage_rag.py list          # collections + doc counts
python scripts/manage_rag.py inspect documents --db private --limit 5
python scripts/manage_rag.py metadata-detailed

# Quick search without going through the console
python scripts/manage_rag.py query documents "what is RAG?" --top-k 5

# Test the full Knowledge-agent path (retrieval + answer)
python scripts/manage_rag.py test-query "who created Python?"

# Wipe (only if you really mean it)
python scripts/manage_rag.py reset --db private --confirm
```

`python scripts/manage_rag.py --help` lists every subcommand.

## What happens when you index a file

```
file → read_file()          (decodes PDF/EPUB/DOCX/text)
     → detect_document_type (text / code / markdown / book / …)
     → adaptive chunker     (chapter-aware, code-aware, paragraph-aware)
     → embedding engine     (GGUF via llama-server OR HuggingFace ST)
     → LanceDB private collection ("documents")
     → SQLite metadata store (titles, authors, chunk counts, doc IDs)
```

Each chunk gets an embedding vector and metadata (source path, chunk index,
detected language, etc.). The Knowledge agent retrieves by vector similarity
on the embedding store, optionally reranks (if you enable
`reranking.enable: true` in `config/rag_config.yaml`), and feeds the top
chunks to the LLM as context.

## Prompting the Knowledge agent

Once docs are indexed, in the console:

```
/agent knowledge
Knowledge>: <your question>
```

Things that work well:
- **Direct factual lookups** — "What model is recommended for embedding?"
- **Multi-doc synthesis** — "Compare the auth approaches across the API docs"
- **Definitions / explanations grounded in your corpus** — "Define the X
  parameter as described in the spec"
- **Source-citation style questions** — "Which section of the PDF covers
  database migration?"

Things to **avoid**:
- Off-corpus chat ("What's the weather?"). Knowledge will look in the store,
  find nothing relevant, and tell you so honestly.
- Multi-turn casual conversation. Each prompt is treated as an independent
  retrieval — there's no chained conversational memory across turns.

After each answer you'll see:
- a verification panel (confidence + risk + missing sources warnings),
- a Sources list with score-ranked chunks,
- `View source details? [y/n]` — press `y` to enter the rich chunk viewer
  (vim-style keys: `j` / `k` to scroll chunks, `+` / `-` to widen / narrow
  the surrounding context, `g<n>` to jump, `q` to quit).

If you see a fallback message instead of the rich answer panel, the RAG
retriever didn't initialize — usually because no documents have been
indexed yet, or the embedding model couldn't load (see Troubleshooting).

## Prompting the Summarizer agent

```
/agent summarizer
Summarizer>: Summarize /path/to/file.pdf
Summarizer>: Summarize ~/notes.md --style narrative
Summarizer>: Summarize chapters 3-5 from ~/books/book.epub --style technical
```

Styles: `narrative` (default), `academic`, `technical`, `explanatory`, `quick`.

Compression levels: `detailed` (~20% of original), `standard` (~10%),
`condensed` (~5%), `outline` (~2%). Add as `--compression <level>` after the
style.

The Summarizer reads files directly off disk — it does NOT pull from the
LanceDB store. So you don't have to index a doc before summarizing it.

## Typical workflow

```bash
# 1. Index your reference material
python scripts/manage_rag.py index-folder ~/work/docs --recursive

# 2. Confirm it landed
python scripts/manage_rag.py stats
python scripts/manage_rag.py inspect documents --db private --limit 3

# 3. Use it
python start_console.py --ephemeral
> /agent knowledge
Knowledge>: What were the deployment steps for the staging release?
```

The Knowledge agent shows source citations, a verification panel, and lets
you press `y` to open the **rich chunk viewer** for any source — vim-style
keys (`j`/`k`/`+`/`-`/`g<n>`/`q`) to scroll through chunks with surrounding
context.

## Two collections: private vs public

`rag_config.yaml` defines two LanceDB collections:

- `private` — your local indexed content. Read/write. Default for ingestion.
- `public` — read-only mount for shared/curated content (empty by default).

You normally only need `--collection private` (the default).

## Supported formats

`read_file()` handles:

- Text & markdown (`.txt`, `.md`, `.rst`)
- PDF (`.pdf`) — via `pypdf2` / `pdfplumber`
- EPUB (`.epub`)
- DOCX (`.docx`)
- Code files (Python, JS, TS, Go, Rust, C/C++, Java, etc. — auto language detection)
- Spreadsheets (`.xlsx`, `.csv`)
- Plain text fallback for anything else

If an optional decoder is missing, manage_rag prints a clear `pip install ...`
hint and skips that file.

## Re-indexing

By default, a file that's already in the metadata store is skipped. Use
`--force` to re-chunk and re-embed:

```bash
python scripts/manage_rag.py index ~/notes/report.pdf --force
```

This is useful after changing chunking settings in `rag_config.yaml` or
swapping the embedding model.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Collection is empty" on `inspect` | The collection name doesn't match — try `documents` not `private_documents` |
| "The lance library is required" | `pip install pylance` (also in `requirements.txt`) |
| Embedding download hangs on first index | First-run HF download of `Qwen3-Embedding-0.6B` (~1.2 GB). Either wait, or run `python install.py` to grab the local Nomic GGUF instead |
| `llama-server not found` | Run `python install.py` to install the binary; the embedding engine auto-detects `<project>/bin/llama-server` |
| Knowledge says "no relevant info" | Either the corpus is empty or the LLM is being conservative — check sources panel; if a relevant doc IS there but the LLM ignores it, try `accurate` mode |

## Tying it all together

```
manage_rag.py index ...    ──>  LanceDB (private/documents)
                                    │
                                    ▼
                              embedding engine
                                    │
                                    ▼
        /agent knowledge ──>  Knowledge agent  ──>  LLM
                                    │
                                    ▼
                            answer + source panel
                                    │
                              (press y) ──>  rich chunk viewer
                                                  (vim keys)
```

The Summarization agent uses the same chunker + embedding stack internally,
so the install step that wires Nomic also benefits `/agent summarizer`.
