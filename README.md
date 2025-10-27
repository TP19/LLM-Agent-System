# LLM Agent System

A modular multi-agent orchestration framework for executing complex tasks using local LLMs with advanced RAG capabilities.

## Overview

The LLM Agent System is a sophisticated framework that coordinates multiple specialized AI agents to handle complex user requests. It features intelligent task decomposition, security validation, code generation, knowledge retrieval, and hierarchical summarization—all running locally with efficient memory management.

### Key Features

- **Multi-Agent Architecture**: 7 specialized agents (Triage, Security, Executor, Coder, Knowledge, Summarization, Feedback) that collaborate to solve complex tasks
- **Interactive Checkpointing**: Review and approve agent actions before execution with rollback capability
- **Advanced RAG System**: Dual vector database (private/public) with semantic chunking, reranking, and quality modes
- **Lazy Model Loading**: Memory-efficient architecture loads only one 8B model at a time
- **Rich Terminal UI**: Beautiful, scannable output with progress tracking and syntax highlighting
- **Validation System**: Anti-hallucination checks to prevent invalid routing and commands
- **Flexible Operation Modes**: Interactive (with checkpoints) or autonomous (file-based monitoring)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Interactive Session Layer                   │
│            (Session Manager + Terminal UI)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Interactive Orchestrator                        │
│     (Checkpoint Manager + Workflow Validator)                │
└────┬────┬────┬────┬────┬────┬────┬───────────────────────┘
     │    │    │    │    │    │    │
┌────▼────▼────▼────▼────▼────▼────▼──────────────────────────┐
│  Triage  Security  Executor  Coder  Knowledge  Summ.  FB     │
│  Agent    Agent     Agent   Agent    Agent    Agent  Agent   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Core Services Layer                             │
│  (BaseAgent, LazyModelManager, AgentRegistry)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│      Infrastructure & Persistence Layer                      │
│  (Vector Stores, Embeddings, Reranker, Metadata)             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- 16GB+ RAM

### Setup

1. **Create Virtual Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install llama-cpp-python**

For GPU support (CUDA):
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

For CPU only:
```bash
pip install llama-cpp-python
```

For Metal (macOS):
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Model Configuration (`config/models.yaml`)

Create a `config/models.yaml` file with your model paths:

```yaml
models:
  executor:
    model_path: "/path/to/your/model.gguf"  # 8B instruction model
    n_ctx: 4096
    n_gpu_layers: 5
    verbose: false

  security:
    model_path: "/path/to/your/model.gguf"
    n_ctx: 16384
    n_gpu_layers: 5
    verbose: false

  triage:
    model_path: "/path/to/your/model.gguf"
    n_ctx: 16384
    n_gpu_layers: 5
    verbose: false

  coder:
    model_path: "/path/to/your/coder-model.gguf"  # Code-specialized model
    n_gpu_layers: 7
    n_ctx: 16384

  summarization:
    model_path: "/path/to/your/model.gguf"
    n_gpu_layers: 10
    n_ctx: 8192
    enable_rag: true

  knowledge:
    model_path: "/path/to/your/model.gguf"
    n_ctx: 8192
    n_gpu_layers: 7
    verbose: false
    temperature: 0.3
    top_p: 0.9
    repeat_penalty: 1.1

settings:
  command_timeout: 30
  enable_follow_up: true
  max_collaboration_cycles: 3
```

**Recommended Models:**
- General agents: IBM Granite 3.3 8B Instruct (Q8_0)
- Coder agent: Qwen3-Coder 30B (Q5_K_M) or similar code-specialized model

### 2. Retrieval Configuration (`config/retrieval_config.yaml`)

Create a `config/retrieval_config.yaml` file:

```yaml
# Quality modes configuration
quality_modes:
  fast:
    initial_k: 10
    final_k: 5
    use_context: false
    use_reranking: false
    timeout_seconds: 5

  balanced:
    initial_k: 20
    final_k: 10
    use_context: true
    use_reranking: false
    timeout_seconds: 10

  accurate:
    initial_k: 50
    final_k: 20
    use_context: true
    use_reranking: true
    timeout_seconds: 20

  thorough:
    initial_k: 100
    final_k: 50
    use_context: true
    use_reranking: true
    timeout_seconds: 60

# Embedding model configuration
embedding:
  default_model: "all-MiniLM-L6-v2"  # Fast, 384 dims
  accurate_model: "all-mpnet-base-v2"  # Better, 768 dims
  use_accurate_for: ["accurate", "thorough"]

# Retrieval parameters
retrieval:
  default_quality: "balanced"
  max_chunks_per_query: 50
  min_similarity_score: 0.5
  context_window: 1

# Answer generation
answer_generation:
  max_context_chunks: 10
  max_tokens: 800
  temperature: 0.3
  top_p: 0.9

# Source citation
sources:
  max_sources_display: 10
  show_scores: true
  show_metadata: true
  content_preview_length: 150
```

### 3. RAG Configuration

The `config/rag_config.yaml` is included in the repository and can be customized for your RAG setup.

## Usage

### Interactive Mode

The primary way to use the system is through interactive mode:

```bash
python start_interactive.py
```

**Example Session:**

```
╭─────────────────────────────────────────────────╮
│   🤖 LLM Agent System - Interactive Mode        │
╰─────────────────────────────────────────────────╯

You: Analyze the security implications of adding CORS to our API

[Agent Workflow]
1. Triage Agent analyzes your request
2. Security Agent evaluates security concerns
3. System proposes approach → Checkpoint (You review & approve)
4. Executor/Coder Agent implements solution
5. Summarization creates documentation

Enter 'help' for commands, 'quit' to exit
```

**Available Commands:**
- `help` - Show available commands
- `stats` - Display session statistics
- `history` - View command history
- `rollback` - Rollback to previous checkpoint
- `quality <mode>` - Set RAG quality mode (fast/balanced/accurate/thorough)
- `quit` - Exit session

### Autonomous Mode

For automated file-based processing:

```bash
python orchestration/file_monitor.py
```

This monitors the `requests/` directory for new task files and processes them automatically.

## Agent Roles

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **Triage** | Request analysis & routing | Classifies requests, determines complexity, routes to appropriate agents |
| **Security** | Security validation | Risk assessment, permission validation, security best practices |
| **Executor** | Command execution | Runs shell commands, file operations, system tasks |
| **Coder** | Code generation | Writes code, refactors, generates implementations |
| **Knowledge** | Information retrieval | RAG-powered Q&A, document search, semantic retrieval |
| **Summarization** | Document processing | Hierarchical summarization, chunk-based analysis |
| **Feedback** | User interaction | Collects feedback, refines responses |

## RAG System

The system includes a sophisticated RAG implementation:

- **Dual Vector Databases**: Private (user-specific) and public (shared knowledge)
- **Quality Modes**: Fast, Balanced, Accurate, Thorough
- **Semantic Chunking**: Intelligent text splitting with boundary detection
- **Reranking**: Optional cross-encoder reranking for improved relevance
- **Metadata-Aware**: Tracks document metadata, tokens, and chunk relationships

## RAG Document Management

The system includes a powerful `manage_rag.py` script for managing your knowledge base.

### Indexing Documents

**Index a single file:**
```bash
# Index to private database (user-specific knowledge)
python scripts/manage_rag.py index /path/to/document.txt --collection private

# Index to public database (shared knowledge)
python scripts/manage_rag.py index /path/to/document.pdf --collection public

# Force re-index existing file
python scripts/manage_rag.py index /path/to/document.txt --collection private --force
```

**Index entire folders:**
```bash
# Index all files in a folder (recursive by default)
python scripts/manage_rag.py index-folder /path/to/docs/ --collection private

# Index specific file types only
python scripts/manage_rag.py index-folder /path/to/docs/ --collection private --extensions .txt .md .pdf

# Non-recursive indexing
python scripts/manage_rag.py index-folder /path/to/docs/ --collection private --no-recursive
```

**Supported file formats:**
- Text files (`.txt`, `.md`)
- PDF files (`.pdf`) - requires `pip install PyPDF2`
- Source code files (`.py`, `.js`, `.java`, `.cpp`, etc.)

### Querying and Managing

**Query your knowledge base:**
```bash
# Query with default quality (balanced)
python scripts/manage_rag.py test-query "How does the authentication work?"

# Use different quality modes
python scripts/manage_rag.py test-query "Explain the architecture" --quality accurate
python scripts/manage_rag.py test-query "Quick info about API" --quality fast
python scripts/manage_rag.py test-query "Comprehensive analysis" --quality thorough
```

**Database management:**
```bash
# Show database statistics
python scripts/manage_rag.py stats

# List all collections
python scripts/manage_rag.py list --db all

# Inspect a specific collection
python scripts/manage_rag.py inspect semantic --db private --limit 10

# System health check
python scripts/manage_rag.py health

# Reset databases (use with caution!)
python scripts/manage_rag.py reset --db private --confirm
```

### Recommended Workflow

1. **Index your documentation:**
   ```bash
   python scripts/manage_rag.py index-folder ~/my-project/docs/ --collection private
   ```

2. **Test retrieval:**
   ```bash
   python scripts/manage_rag.py test-query "your question" --quality balanced
   ```

3. **Use in interactive mode:**
   ```bash
   python start_interactive.py
   # Ask questions and the Knowledge Agent will use your indexed documents
   ```

### Project Structure

```
.
├── agents/              # Specialized agent implementations
├── core/                # BaseAgent, model manager, registry
├── rag/                 # RAG system (vector stores, embeddings, retrieval)
├── interactive/         # Interactive mode (orchestrator, session, checkpoints)
├── utilities/           # Helpers (chunking, metadata, token counting)
├── orchestration/       # Autonomous file-based execution
├── config/              # Configuration files (YAML)
└── start_interactive.py # Main entry point
```

## Performance

### Typical Timings (8B Models on GPU)
- Triage: 1-2 seconds
- Security Analysis: 2-3 seconds
- Code Generation: 5-15 seconds (depending on complexity)
- RAG Retrieval (balanced): 1-3 seconds
- Document Summarization: 3-8 seconds per chunk

### Memory Optimization
- Lazy loading: Only one model in memory at a time
- Automatic cleanup after agent execution
- GPU memory monitoring and management

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

This means you are free to:
- ✅ Use this software for any purpose
- ✅ Study and modify the source code
- ✅ Share copies of the software
- ✅ Share modified versions

**Requirements:**
- If you distribute this software or modifications, you must:
  - Make the source code available
  - License your modifications under GPL-3.0
  - Include copyright and license notices

See the [LICENSE](LICENSE) file for full details, or visit https://www.gnu.org/licenses/gpl-3.0.html

## Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/TP19/LLM-Agent-System.git
   cd llm_agent_system
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style and patterns
   - Add tests if applicable

4. **Test your changes**
   ```bash
   # Ensure your changes work with the system
   python start_interactive.py
   # Test RAG functionality if modified
   python scripts/manage_rag.py health
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Contribution Areas

We especially welcome contributions in:

- **New Agents**: Create specialized agents for new domains (research, testing, deployment, etc.)
- **RAG Improvements**: Enhanced retrieval strategies, new embedding models, better chunking
- **Documentation**: Tutorials, examples, architecture explanations
- **Performance**: Optimization, caching strategies, faster inference
- **Integration**: Support for more LLM backends, APIs, or model formats
- **Testing**: Unit tests, integration tests, benchmarking
- **Bug Fixes**: Always appreciated!

### Code Guidelines

- Use descriptive variable and function names
- Add docstrings to classes and functions
- Keep functions focused and modular
- Handle errors gracefully with informative messages
- Update README if you change functionality

### Reporting Issues

Found a bug or have a feature request?

1. Check if the issue already exists
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU/CPU)
   - Relevant logs or error messages

## Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/TP19/LLM-Agent-System/issues)
- **Discussions**: Ask questions or share ideas in [GitHub Discussions](https://github.com/TP19/LLM-Agent-System/discussions)
- **Documentation**: Check this README and code documentation for guidance

## Acknowledgments

This project builds upon excellent open-source libraries:
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings for llama.cpp
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - Embeddings
- [Rich](https://github.com/Textualize/rich) - Terminal UI

---

**Built with ❤️ for the open-source community**
