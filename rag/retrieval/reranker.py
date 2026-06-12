from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging
import os
import shutil
import yaml
from ..vector_stores.base_store import MemoryEntry


def _default_llama_server_path() -> str:
    """Resolve llama-server binary path.

    Search order:
      1. $LLM_ENGINE_LLAMA_SERVER
      2. shutil.which("llama-server")
      3. <project_root>/bin/llama-server     (install.py default)
      4. ~/.local/bin/llama-server
    """
    project_local = Path(__file__).resolve().parents[2] / "bin" / "llama-server"
    return (
        os.environ.get("LLM_ENGINE_LLAMA_SERVER")
        or shutil.which("llama-server")
        or (str(project_local) if project_local.exists() else None)
        or str(Path.home() / ".local" / "bin" / "llama-server")
    )

class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision

    Supports:
    - HuggingFace CrossEncoder models
    - GGUF models (via llama-cpp-python)

    Reads configuration from config/rag_config.yaml
    """

    def __init__(self,
                model_name: str = None,
                lazy_load: bool = None,
                config_path: str = None,
                embedding_engine = None):
        """
        Args:
            model_name: Model name (HF) or path (GGUF .gguf file) - overrides config
            lazy_load: Load model only when needed - overrides config
            config_path: Path to config file (default: config/rag_config.yaml)
            embedding_engine: Optional EmbeddingEngine to reuse for GGUF models (avoids duplicate servers)
        """
        self.logger = logging.getLogger("Reranker")

        # Load configuration
        self.config = self._load_config(config_path)

        # Use provided values OR fallback to config OR use defaults
        self.model_name = model_name or self.config.get('model', 'ibm-granite/granite-embedding-reranker-english-r2')
        self.lazy_load = lazy_load if lazy_load is not None else self.config.get('lazy_load', True)
        self.embedding_engine = embedding_engine  # Reuse existing embedding engine

        self._model = None
        self.active_backend = None  # Track which backend is used (llama-cpp-python or llama-server)
        self.server_process = None  # For llama-server subprocess

        # Detect format (HuggingFace or GGUF)
        self.model_format = self._detect_format(self.model_name)

        if not self.lazy_load:
            self._load_model()

    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from rag_config.yaml"""
        if config_path is None:
            # Try multiple locations
            possible_paths = [
                Path('config/rag_config.yaml'),
                Path('../config/rag_config.yaml'),
                Path(__file__).parent.parent.parent / 'config' / 'rag_config.yaml',
            ]

            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
        else:
            config_path = Path(config_path)

        # Load config if found
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    reranking_config = config.get('reranking', {})
                    self.logger.info(f"✅ Loaded reranker config from {config_path}")
                    return reranking_config
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
        else:
            self.logger.warning("No config file found, using defaults")

        return {}

    def _detect_format(self, model_name: str) -> str:
        """Detect if model is GGUF or HuggingFace"""
        if model_name.endswith('.gguf'):
            return 'gguf'
        elif Path(model_name).exists() and Path(model_name).suffix == '.gguf':
            return 'gguf'
        else:
            return 'hf'

    def _load_model(self) -> None:
        """Load the model (HuggingFace or GGUF)"""
        if self._model is None:
            if self.model_format == 'gguf':
                self._load_gguf_model()
            else:
                self._load_hf_model()

    def _load_hf_model(self) -> None:
        """Load HuggingFace CrossEncoder model"""
        self.logger.info(f"Loading HF reranker: {self.model_name}")
        self._model = CrossEncoder(self.model_name)
        self.logger.info("✅ HF Reranker loaded")

    def _load_gguf_model(self) -> None:
        """Load GGUF reranker model with dual-backend support (llama-cpp-python or llama-server)"""
        model_path = Path(self.model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF reranker not found: {model_path}")

        # Try llama-cpp-python first, fallback to llama-server
        try:
            self._load_gguf_with_python(model_path)
        except Exception as e:
            error_msg = str(e)
            # Check if it's a compatibility error
            if "undefined symbol" in error_msg or "libcudart" in error_msg or "llama_kv_self" in error_msg:
                self.logger.warning(f"⚠️ llama-cpp-python failed, trying llama-server fallback...")
                self.logger.debug(f"Error details: {e}")
            else:
                self.logger.warning(f"⚠️ llama-cpp-python failed: {e}, trying llama-server fallback...")

            # Fallback to llama-server
            try:
                self._load_gguf_with_server(model_path)
            except Exception as e2:
                raise RuntimeError(
                    f"Both backends failed for {model_path.name}:\n"
                    f"llama-cpp-python: {str(e)[:100]}\n"
                    f"llama-server: {str(e2)[:100]}"
                )

    def _load_gguf_with_python(self, model_path: Path) -> None:
        """Load GGUF model using llama-cpp-python"""
        from llama_cpp import Llama
        import sys
        import os

        self.logger.info(f"Loading GGUF reranker with llama-cpp-python: {model_path}")

        # Get GGUF-specific settings from config
        gguf_config = self.config.get('gguf', {})
        n_ctx = gguf_config.get('n_ctx', 2048)
        n_threads = gguf_config.get('n_threads', 4)
        n_batch = gguf_config.get('n_batch', 2048)  # CRITICAL: Must match n_ctx for long texts
        n_gpu_layers = gguf_config.get('n_gpu_layers', 99)  # 99=all layers (GPU), 0=CPU only

        # Suppress llama.cpp C++ warnings by redirecting stderr at OS level
        stderr_fd = sys.stderr.fileno()
        old_stderr_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        try:
            # Redirect stderr to /dev/null at the file descriptor level
            os.dup2(devnull_fd, stderr_fd)

            # Load with llama.cpp
            self._model = Llama(
                model_path=str(model_path),
                embedding=True,  # Reranking uses embedding similarity
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,  # GPU acceleration
                verbose=False
            )

        finally:
            # Restore stderr
            os.dup2(old_stderr_fd, stderr_fd)
            os.close(old_stderr_fd)
            os.close(devnull_fd)

        self.active_backend = 'llama-cpp-python'
        gpu_status = f"{n_gpu_layers} layers" if n_gpu_layers > 0 else "CPU only"
        self.logger.info(f"✅ GGUF Reranker loaded with llama-cpp-python (GPU={gpu_status})")

    def _load_gguf_with_server(self, model_path: Path) -> None:
        """Load GGUF reranker using llama-server subprocess"""
        import subprocess
        import time
        import requests

        self.logger.info(f"Loading GGUF reranker with llama-server: {model_path}")

        # Get configuration
        gguf_config = self.config.get('gguf', {})
        llama_server_path = gguf_config.get('llama_server_path') or _default_llama_server_path()
        llama_server_path = os.path.expanduser(llama_server_path)
        n_ctx = gguf_config.get('n_ctx', 2048)
        n_batch = gguf_config.get('n_batch', 2048)  # CRITICAL: Read batch size from config
        n_gpu_layers = gguf_config.get('n_gpu_layers', 99)

        # Use ephemeral port (find an available port)
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        self.logger.info(f"Using ephemeral port: {port} (n_batch={n_batch})")

        # Check if server binary exists
        if not Path(llama_server_path).exists():
            raise FileNotFoundError(f"llama-server not found at: {llama_server_path}")

        # Start server process
        cmd = [
            llama_server_path,
            '-m', str(model_path),
            '--port', str(port),
            '--ctx-size', str(n_ctx),
            '--batch-size', str(n_batch),  # CRITICAL: Pass batch size to server!
            '--ubatch-size', str(n_batch),  # Physical batch size (same as batch-size for embeddings)
            '--n-gpu-layers', str(n_gpu_layers),
            '--embedding'
        ]

        self.logger.info(f"Starting llama-server on port {port}...")
        import tempfile
        self._server_stderr_file = tempfile.NamedTemporaryFile(
            mode='w+', suffix='.log', prefix='reranker_server_', delete=False
        )
        env = os.environ.copy()
        bin_dir = str(Path(llama_server_path).parent)
        env["LD_LIBRARY_PATH"] = bin_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._server_stderr_file,
            env=env,
        )

        # 127.0.0.1 (not 'localhost') to avoid IPv6 resolution issues.
        # Probe /embedding (not /health) — llama-server in embedding mode
        # returns 503 on /health even when the model is fully loaded.
        server_url = f"http://127.0.0.1:{port}"
        max_retries = 120
        ready = False
        last_err = None
        for i in range(max_retries):
            if self.server_process.poll() is not None:
                self._server_stderr_file.seek(0)
                err_tail = self._server_stderr_file.read()[-2000:]
                raise RuntimeError(
                    f"reranker llama-server exited early (rc={self.server_process.returncode}). "
                    f"Last stderr:\n{err_tail}"
                )
            try:
                response = requests.post(
                    f"{server_url}/embedding",
                    json={"content": "ready_check"},
                    timeout=2,
                )
                if response.status_code == 200:
                    ready = True
                    break
                last_err = f"HTTP {response.status_code}"
            except Exception as e:
                last_err = type(e).__name__
            time.sleep(1)
        if not ready:
            try:
                self.server_process.kill()
            except Exception:
                pass
            self._server_stderr_file.seek(0)
            err_tail = self._server_stderr_file.read()[-2000:]
            raise RuntimeError(
                f"reranker llama-server readiness probe failed on {server_url} "
                f"after {max_retries}s (last: {last_err}).\nLast stderr:\n{err_tail}"
            )

        self._model = server_url
        self.active_backend = 'llama-server'

        self.logger.info(f"✅ GGUF Reranker loaded with llama-server")
        self.logger.info(f"   Server URL: {server_url}, port: {port}, GPU: {n_gpu_layers}")

    def _check_server_health(self) -> bool:
        """Check if llama-server is healthy"""
        if self.active_backend != 'llama-server' or not self._model:
            return False

        import requests
        try:
            response = requests.get(f"{self._model}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _restart_server(self):
        """Restart llama-server if it crashed"""
        self.logger.warning("Reranker server appears dead, attempting restart...")

        # Kill old process if it exists
        if self.server_process:
            try:
                self.server_process.kill()
                self.server_process.wait(timeout=5)
            except:
                pass
            self.server_process = None

        # Reload the model (will start new server)
        model_path = Path(self.model_name)
        self._load_gguf_with_server(model_path)
        self.logger.info("✅ Reranker server restarted successfully")

    def _create_embedding_with_server(self, text: str, retry_on_error: bool = True) -> List[float]:
        """Create embedding using llama-server API with auto-restart on failure"""
        import requests

        try:
            response = requests.post(
                f"{self._model}/embedding",
                json={"content": text},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            # llama-server returns: [{"index": 0, "embedding": [[...]]}]
            return result[0]['embedding'][0]

        except Exception as e:
            # If server error and we haven't retried yet, try restarting server
            if retry_on_error and ('500' in str(e) or 'Connection' in str(e)):
                self.logger.warning(f"Reranker server error: {e}, checking health...")

                if not self._check_server_health():
                    self._restart_server()
                    # Retry once after restart
                    return self._create_embedding_with_server(text, retry_on_error=False)

            # If retry failed or different error, re-raise
            raise
    
    @property
    def model(self):
        """Lazy load model"""
        if self._model is None:
            self._load_model()
        return self._model

    def rerank(self,
               query: str,
               memories: List[MemoryEntry],
               top_k: int = 5) -> List[MemoryEntry]:
        """
        Rerank retrieved memories using cross-encoder

        Args:
            query: User query
            memories: Retrieved memories from vector search
            top_k: Number of top results to return

        Returns:
            Reranked list of memories
        """
        if not memories:
            return []

        if self.model_format == 'gguf':
            return self._rerank_gguf(query, memories, top_k)
        else:
            return self._rerank_hf(query, memories, top_k)

    def _rerank_hf(self, query: str, memories: List[MemoryEntry], top_k: int) -> List[MemoryEntry]:
        """Rerank using HuggingFace CrossEncoder"""
        # Create query-document pairs
        pairs = [[query, mem.content] for mem in memories]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort memories by score
        scored_memories = list(zip(memories, scores))
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Update importance scores based on reranking
        reranked = []
        for mem, score in scored_memories[:top_k]:
            mem.importance_score = float(score)
            reranked.append(mem)

        return reranked

    def _rerank_gguf(self, query: str, memories: List[MemoryEntry], top_k: int) -> List[MemoryEntry]:
        """Rerank using GGUF model (via embedding similarity)"""
        import numpy as np

        # If we have a shared embedding engine, use it instead of starting our own server
        if self.embedding_engine is not None:
            self.logger.debug("Using shared embedding engine for reranking")
            query_emb = np.array(self.embedding_engine.embed_text(query))
        else:
            # Ensure model is loaded (lazy loading)
            if self._model is None:
                self._load_model()

            # Get query embedding based on active backend
            if self.active_backend == 'llama-server':
                query_emb = np.array(self._create_embedding_with_server(query))
            else:
                # llama-cpp-python backend - self._model is a Llama object
                query_emb = self._model.create_embedding(query)['data'][0]['embedding']
                query_emb = np.array(query_emb)

        # Handle 2D embeddings (token-level) by taking mean pooling
        if query_emb.ndim == 2:
            query_emb = np.mean(query_emb, axis=0)

        # Ensure 1D
        query_emb = query_emb.flatten()

        # Calculate scores for each memory
        scores = []
        for mem in memories:
            # Get document embedding
            if self.embedding_engine is not None:
                # Use shared embedding engine
                doc_emb = np.array(self.embedding_engine.embed_text(mem.content))
            elif self.active_backend == 'llama-server':
                doc_emb = np.array(self._create_embedding_with_server(mem.content))
            else:
                # llama-cpp-python backend - self._model is a Llama object
                doc_emb = self._model.create_embedding(mem.content)['data'][0]['embedding']
                doc_emb = np.array(doc_emb)

            # Handle 2D embeddings (token-level) by taking mean pooling
            if doc_emb.ndim == 2:
                doc_emb = np.mean(doc_emb, axis=0)

            # Ensure 1D
            doc_emb = doc_emb.flatten()

            # Cosine similarity
            similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            scores.append(float(similarity))

        # Sort memories by score
        scored_memories = list(zip(memories, scores))
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Update importance scores based on reranking
        reranked = []
        for mem, score in scored_memories[:top_k]:
            mem.importance_score = score
            reranked.append(mem)

        return reranked
    
    def unload_model(self) -> None:
        """Unload model to free memory and cleanup resources"""
        if self._model is not None:
            del self._model
            self._model = None

        # Cleanup llama-server process if running
        if self.server_process is not None:
            self.logger.info("Shutting down llama-server process...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()
            self.server_process = None

        self.logger.info("Reranker unloaded")

    def __del__(self):
        """Cleanup on deletion"""
        self.unload_model()