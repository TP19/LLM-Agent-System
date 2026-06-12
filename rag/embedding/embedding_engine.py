#!/usr/bin/env python3
"""
Config-Aware Embedding Engine - Supports HuggingFace and GGUF models

Reads from config/rag_config.yaml
Supports:
- HuggingFace models (sentence-transformers)
- GGUF models (llama-cpp-python)

Auto-detects format based on .gguf extension
"""

import logging
import os
import shutil
import numpy as np
import yaml
from pathlib import Path
from typing import List, Union, Optional, Dict


def _default_llama_server_path() -> str:
    """Resolve llama-server binary path.

    Search order:
      1. $LLM_ENGINE_LLAMA_SERVER (explicit override)
      2. shutil.which("llama-server")             (system PATH)
      3. <project_root>/bin/llama-server          (install.py default)
      4. ~/.local/bin/llama-server                (user-managed install)
    """
    # rag/embedding/embedding_engine.py -> project root is parents[2]
    project_local = Path(__file__).resolve().parents[2] / "bin" / "llama-server"
    return (
        os.environ.get("LLM_ENGINE_LLAMA_SERVER")
        or shutil.which("llama-server")
        or (str(project_local) if project_local.exists() else None)
        or str(Path.home() / ".local" / "bin" / "llama-server")
    )


class EmbeddingEngine:
    """
    Generate embeddings for semantic search
    
    ALL CONFIGURATION FROM rag_config.yaml:
    - Model selection
    - Device (CPU/GPU)
    - Batch sizes
    - Normalization settings
    
    Supports models:
    - all-MiniLM-L6-v2: Fast, 384 dims (default)
    - all-mpnet-base-v2: Better accuracy, 768 dims
    - instructor-large: Best accuracy, 768 dims
    """
    
    def __init__(self, model_name: str = None, lazy_load: bool = None,
                 device: str = None, batch_size: int = None, config_path: str = None,
                 backend: str = None):
        """
        Initialize embedding engine

        Args:
            model_name: Model to use (overrides config) - HF name or GGUF path
            lazy_load: Load model on first use (overrides config)
            device: Device to use (overrides config)
            batch_size: Batch size (overrides config)
            config_path: Path to config file (default: config/rag_config.yaml)
            backend: Backend to use for GGUF models (overrides config)
                    'auto' = try llama-cpp-python first, fallback to llama-server
                    'llama-cpp-python' = use llama-cpp-python only
                    'llama-server' = use llama-server only
        """
        self.logger = logging.getLogger("EmbeddingEngine")

        # Load configuration
        self.config = self._load_config(config_path)

        # Use provided values OR fallback to config OR use final defaults
        self.model_name = model_name or self.config.get('model', 'all-MiniLM-L6-v2')
        self.lazy_load = lazy_load if lazy_load is not None else self.config.get('lazy_load', True)

        # Detect model format (HuggingFace or GGUF)
        self.model_format = self._detect_format(self.model_name)

        # Backend configuration for GGUF models
        self.backend = backend or self.config.get('backend', 'auto')

        # Device detection and configuration
        self.device = self._resolve_device(device)

        # Batch size configuration
        self.batch_size = self._resolve_batch_size(batch_size)

        # Get model dimension (will be updated after loading for GGUF)
        self.dimension = self._get_model_dimension(self.model_name)

        # Model placeholder
        self.model = None

        # Track which backend was actually used
        self.active_backend = None

        # Server process reference (for cleanup)
        self.server_process = None

        # Load immediately if not lazy
        if not self.lazy_load:
            self._load_model()

        format_str = "GGUF" if self.model_format == "gguf" else "HuggingFace"
        self.logger.info(
            f"✅ Embedding engine initialized "
            f"(format: {format_str}, model: {self.model_name}, dims: {self.dimension}, "
            f"device: {self.device}, batch_size: {self.batch_size}, lazy: {self.lazy_load}, "
            f"backend: {self.backend})"
        )
    
    def _load_config(self, config_path: str = None) -> Dict:
        """
        Load configuration from rag_config.yaml
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
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
                    embedding_config = config.get('embedding', {})
                    self.logger.info(f"✅ Loaded config from {config_path}")
                    return embedding_config
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
        else:
            self.logger.warning("No config file found, using defaults")
        
        # Return empty dict if no config
        return {}
    
    def _resolve_device(self, device_override: str = None) -> str:
        """
        Resolve device to use
        
        Priority:
        1. Explicit override parameter
        2. Config file setting
        3. Auto-detection
        
        Args:
            device_override: Explicit device setting
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        # 1. Check override
        if device_override:
            return device_override
        
        # 2. Check config
        config_device = self.config.get('device', 'auto')
        
        # 3. Auto-detect if set to 'auto'
        if config_device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    self.logger.info("✅ GPU detected, using CUDA")
                    return "cuda"
            except ImportError:
                pass
            
            self.logger.info("Using CPU")
            return "cpu"
        
        return config_device
    
    def _resolve_batch_size(self, batch_size_override: int = None) -> int:
        """
        Resolve batch size to use
        
        Priority:
        1. Explicit override parameter
        2. Config 'batch_size' (if set)
        3. Device-specific config (cpu_batch_size or gpu_batch_size)
        4. Auto-calculated default
        
        Args:
            batch_size_override: Explicit batch size
            
        Returns:
            Batch size integer
        """
        # 1. Check override
        if batch_size_override:
            return batch_size_override
        
        # 2. Check explicit batch_size in config
        config_batch_size = self.config.get('batch_size')
        if config_batch_size:
            return config_batch_size
        
        # 3. Check device-specific config
        if self.device == 'cuda':
            device_batch_size = self.config.get('gpu_batch_size')
        else:
            device_batch_size = self.config.get('cpu_batch_size')
        
        if device_batch_size:
            return device_batch_size
        
        # 4. Calculate default based on device and model
        return self._get_default_batch_size()
    
    def _get_default_batch_size(self) -> int:
        """
        Calculate default batch size based on device and model
        
        Returns:
            Batch size (higher for GPU, lower for CPU)
        """
        # Base batch sizes
        if self.device == "cuda":
            base_batch = 64
        else:
            base_batch = 16
        
        # Adjust based on model size
        if "mpnet" in self.model_name.lower() or "large" in self.model_name.lower():
            base_batch = base_batch // 2
        
        self.logger.info(f"Auto-calculated batch size: {base_batch}")
        return base_batch
    
    def _detect_format(self, model_name: str) -> str:
        """
        Detect model format based on file extension

        Args:
            model_name: Model name or path

        Returns:
            'gguf' or 'hf'
        """
        if model_name.endswith('.gguf'):
            return 'gguf'
        elif Path(model_name).exists() and Path(model_name).suffix == '.gguf':
            return 'gguf'
        else:
            return 'hf'

    def _get_model_dimension(self, model_name: str) -> int:
        """Get embedding dimension for model"""
        # For GGUF models, dimension will be detected after loading
        if self.model_format == 'gguf':
            return self.config.get('dimension', 768)  # Use config or default

        # HuggingFace model dimensions
        dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "instructor-large": 768,
            "instructor-xl": 768,
            "Qwen/Qwen3-Embedding-0.6B": 1024,
        }
        return dimensions.get(model_name, 384)
    
    def _load_model(self):
        """Load the embedding model (HuggingFace or GGUF)"""
        if self.model is not None:
            return  # Already loaded

        if self.model_format == 'gguf':
            self._load_gguf_model()
        else:
            self._load_hf_model()

    def _load_hf_model(self):
        """Load HuggingFace model using sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading HuggingFace model: {self.model_name}")
            self.logger.info(f"Device: {self.device}")

            # Load model with specified device
            self.model = SentenceTransformer(self.model_name, device=self.device)

            self.logger.info(f"✅ HF Model loaded: {self.model_name} ({self.dimension} dims)")
            self.logger.info(f"   Using device: {self.model.device}")

        except ImportError:
            self.logger.error("sentence-transformers not installed")
            raise ImportError(
                "Please install: pip install sentence-transformers"
            )

        except Exception as e:
            self.logger.error(f"Failed to load HF model {self.model_name}: {e}")
            raise

    def _load_gguf_model(self):
        """Load GGUF model with dual-backend support (llama-cpp-python or llama-server)"""
        model_path = Path(self.model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        errors = []

        # Strategy based on backend preference
        if self.backend == 'llama-server':
            # User explicitly wants llama-server
            self._load_gguf_with_server(model_path)

        elif self.backend == 'llama-cpp-python':
            # User explicitly wants llama-cpp-python
            self._load_gguf_with_python(model_path)

        else:  # 'auto' - try both with smart fallback
            # Try llama-cpp-python first (faster, less overhead)
            try:
                self.logger.info("Attempting llama-cpp-python backend...")
                self._load_gguf_with_python(model_path)
            except Exception as e:
                errors.append(f"llama-cpp-python failed: {str(e)[:100]}")
                self.logger.warning(f"⚠️ llama-cpp-python failed, trying llama-server fallback...")
                self.logger.debug(f"Error details: {e}")

                # Fallback to llama-server
                try:
                    self._load_gguf_with_server(model_path)
                except Exception as e2:
                    errors.append(f"llama-server failed: {str(e2)[:100]}")
                    raise RuntimeError(
                        f"Both backends failed for {model_path.name}:\n" +
                        "\n".join(errors)
                    )

    def _load_gguf_with_python(self, model_path: Path):
        """Load GGUF model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            import sys
            import os

            self.logger.info(f"Loading GGUF model with llama-cpp-python: {model_path}")

            # Get GGUF-specific settings from config
            gguf_config = self.config.get('gguf', {})
            n_ctx = gguf_config.get('n_ctx', 2048)
            n_threads = gguf_config.get('n_threads', 4)
            n_batch = gguf_config.get('n_batch', 2048)  # CRITICAL: Must match n_ctx for long texts
            n_gpu_layers = gguf_config.get('n_gpu_layers', 99)
            use_mlock = gguf_config.get('use_mlock', False)
            use_mmap = gguf_config.get('use_mmap', True)

            # Suppress llama.cpp C++ warnings
            stderr_fd = sys.stderr.fileno()
            old_stderr_fd = os.dup(stderr_fd)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)

            try:
                os.dup2(devnull_fd, stderr_fd)

                # Load model
                self.model = Llama(
                    model_path=str(model_path),
                    embedding=True,
                    n_ctx=n_ctx,
                    n_batch=n_batch,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    use_mlock=use_mlock,
                    use_mmap=use_mmap,
                    verbose=False
                )

                # Detect dimension
                test_emb = self.model.create_embedding("test")
                embedding_data = test_emb['data'][0]['embedding']
                if isinstance(embedding_data, list) and len(embedding_data) > 0:
                    if isinstance(embedding_data[0], list):
                        embedding_data = np.mean(embedding_data, axis=0).tolist()
                self.dimension = len(embedding_data)

            finally:
                os.dup2(old_stderr_fd, stderr_fd)
                os.close(old_stderr_fd)
                os.close(devnull_fd)

            self.active_backend = 'llama-cpp-python'
            self.logger.info(f"✅ GGUF Model loaded with llama-cpp-python ({self.dimension} dims)")
            self.logger.info(f"   Settings: ctx={n_ctx}, threads={n_threads}, batch={n_batch}, GPU={n_gpu_layers}")

        except ImportError as e:
            raise ImportError(f"llama-cpp-python not installed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load with llama-cpp-python: {e}")

    def _load_gguf_with_server(self, model_path: Path):
        """Load GGUF model using llama-server subprocess"""
        import subprocess
        import time
        import requests

        self.logger.info(f"Loading GGUF model with llama-server: {model_path}")

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
        # Capture stderr to a temp file so we can surface real errors on failure
        # instead of just timing out silently.
        import tempfile
        self._server_stderr_file = tempfile.NamedTemporaryFile(
            mode='w+', suffix='.log', prefix='llama_server_', delete=False
        )
        # Make sure llama-server can find its bundled shared libs (.so files
        # in the same dir) — install.py drops them next to the binary, but
        # they're not on the system LD path.
        env = os.environ.copy()
        bin_dir = str(Path(llama_server_path).parent)
        env["LD_LIBRARY_PATH"] = bin_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._server_stderr_file,
            env=env,
        )

        # Use 127.0.0.1 (not 'localhost') to avoid IPv6 resolution issues —
        # llama-server binds to 127.0.0.1 unless --host is passed.
        #
        # Readiness check: llama-server's /health returns 503 while in embedding
        # mode even after the model is loaded (model is ready, no slot has been
        # used yet — see llama.cpp issue tracker). The reliable signal is that
        # the /embedding endpoint successfully returns a vector for a tiny test
        # input. We probe that directly so we know the server is *actually*
        # usable, not just "process started".
        base_url = f"http://127.0.0.1:{port}"

        max_wait = 120
        ready = False
        last_err: Optional[str] = None
        for i in range(max_wait):
            if self.server_process.poll() is not None:
                self._server_stderr_file.seek(0)
                err_tail = self._server_stderr_file.read()[-2000:]
                raise RuntimeError(
                    f"llama-server exited early (rc={self.server_process.returncode}). "
                    f"Last stderr:\n{err_tail}"
                )
            try:
                response = requests.post(
                    f"{base_url}/embedding",
                    json={"content": "ready_check"},
                    timeout=3,
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
                f"llama-server readiness probe failed after {max_wait}s on "
                f"{base_url}/embedding (last: {last_err}).\nLast stderr:\n{err_tail}"
            )

        # Store server URL
        self.model = f"http://localhost:{port}"
        self.active_backend = 'llama-server'

        # Detect dimension with test embedding (no retry to avoid infinite loop during restart)
        test_emb = self._create_embedding_with_server("test", retry_on_error=False)
        self.dimension = len(test_emb)

        self.logger.info(f"✅ GGUF Model loaded with llama-server ({self.dimension} dims)")
        self.logger.info(f"   Server URL: {self.model}, port: {port}")
        self.logger.info(f"   Config: n_ctx={n_ctx}, n_batch={n_batch}, n_gpu_layers={n_gpu_layers}")

    def _check_server_health(self, test_embedding: bool = True) -> bool:
        """Check if llama-server is healthy and can process embeddings"""
        if self.active_backend != 'llama-server' or not self.model:
            return False

        import requests
        try:
            # First check basic health
            response = requests.get(f"{self.model}/health", timeout=2)
            if response.status_code != 200:
                return False

            # If requested, test actual embedding generation
            if test_embedding:
                try:
                    test_response = requests.post(
                        f"{self.model}/embedding",
                        json={"content": "test"},
                        timeout=5
                    )
                    return test_response.status_code == 200
                except:
                    return False

            return True
        except:
            return False

    def _restart_server(self):
        """Restart llama-server if it crashed"""
        self.logger.warning("🔄 Server appears dead, attempting restart...")

        # Kill old process if it exists
        old_url = self.model
        if self.server_process:
            try:
                self.logger.info(f"   Killing old server process (PID: {self.server_process.pid})...")
                self.server_process.kill()
                self.server_process.wait(timeout=5)
            except Exception as e:
                self.logger.debug(f"   Error killing process: {e}")
            self.server_process = None

        # Reload the model (will start new server with new ephemeral port)
        self.logger.info("   Starting new server...")
        model_path = Path(self.model_name)
        self._load_gguf_with_server(model_path)
        self.logger.info(f"✅ Server restarted: {old_url} → {self.model}")

    def _create_embedding_with_server(self, text: str, retry_on_error: bool = True) -> List[float]:
        """Create embedding using llama-server API with auto-restart on failure"""
        import requests

        try:
            response = requests.post(
                f"{self.model}/embedding",
                json={"content": text},
                timeout=30
            )

            # If 500 error, try to get server error message
            if response.status_code == 500:
                try:
                    error_detail = response.json()
                    self.logger.error(f"Server returned 500: {error_detail}")
                except:
                    self.logger.error(f"Server returned 500 (no details): {response.text[:200]}")

            response.raise_for_status()
            result = response.json()
            # llama-server returns: [{"index": 0, "embedding": [[...]]}]
            return result[0]['embedding'][0]

        except Exception as e:
            # If server error and we haven't retried yet, try restarting server
            if retry_on_error and ('500' in str(e) or 'Connection' in str(e)):
                self.logger.warning(f"Server error: {e}, checking health...")

                if not self._check_server_health():
                    try:
                        self._restart_server()
                        # Retry once after restart
                        self.logger.info("   Retrying request with new server...")
                        return self._create_embedding_with_server(text, retry_on_error=False)
                    except Exception as restart_error:
                        self.logger.error(f"❌ Restart failed: {restart_error}")
                        raise
                else:
                    self.logger.warning("   Server is healthy but request failed - not restarting")

            # If retry failed or different error, re-raise
            raise
    
    def embed_text(self, text: str, normalize: bool = None) -> List[float]:
        """
        Generate embedding for single text

        Args:
            text: Text to embed
            normalize: Normalize to unit length (uses config if None)

        Returns:
            Embedding vector as list
        """
        # Lazy load model if needed
        if self.model is None:
            self._load_model()

        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension

        # Use config setting if not specified
        if normalize is None:
            normalize = self.config.get('normalize_embeddings', True)

        try:
            if self.model_format == 'gguf':
                return self._embed_text_gguf(text, normalize)
            else:
                return self._embed_text_hf(text, normalize)

        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension

    def _embed_text_hf(self, text: str, normalize: bool) -> List[float]:
        """Generate embedding using HuggingFace model"""
        # Generate embedding
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        # Convert to list
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        elif isinstance(embedding, np.ndarray):
            return embedding.tolist()
        else:
            return list(embedding)

    def _embed_text_gguf(self, text: str, normalize: bool) -> List[float]:
        """Generate embedding using GGUF model (handles both backends)"""
        if self.active_backend == 'llama-server':
            # Use server API
            embedding = self._create_embedding_with_server(text)

            # Normalize if requested
            if normalize:
                embedding = np.array(embedding)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (embedding / norm).tolist()
                else:
                    embedding = embedding.tolist()
            return embedding

        else:
            # Use llama-cpp-python
            import sys
            import os

            stderr_fd = sys.stderr.fileno()
            old_stderr_fd = os.dup(stderr_fd)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)

            try:
                os.dup2(devnull_fd, stderr_fd)

                result = self.model.create_embedding(text)
                embedding = result['data'][0]['embedding']

                # Handle 2D embeddings
                if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                    embedding = np.mean(embedding, axis=0).tolist()

                # Normalize if requested
                if normalize:
                    embedding = np.array(embedding)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    embedding = embedding.tolist()

            finally:
                os.dup2(old_stderr_fd, stderr_fd)
                os.close(old_stderr_fd)
                os.close(devnull_fd)

            return embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = None,
                    show_progress: bool = None, normalize: bool = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses config if None)
            show_progress: Show progress bar (uses config if None)
            normalize: Normalize embeddings (uses config if None)

        Returns:
            List of embedding vectors
        """
        # Lazy load model if needed
        if self.model is None:
            self._load_model()

        if not texts:
            return []

        # Use config settings if not specified
        if batch_size is None:
            batch_size = self.batch_size

        if show_progress is None:
            show_progress = self.config.get('show_progress', False)

        if normalize is None:
            normalize = self.config.get('normalize_embeddings', True)

        # Log batch processing info
        format_str = "GGUF" if self.model_format == "gguf" else f"HF on {self.device}"
        self.logger.info(
            f"Embedding {len(texts)} texts with batch_size={batch_size} "
            f"using {format_str}"
        )

        # Filter out empty texts, replace with space
        valid_texts = [t if t and t.strip() else " " for t in texts]

        try:
            if self.model_format == 'gguf':
                return self._embed_texts_gguf(valid_texts, batch_size, show_progress, normalize)
            else:
                return self._embed_texts_hf(valid_texts, batch_size, show_progress, normalize)

        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            # Return zero vectors
            return [[0.0] * self.dimension for _ in texts]

    def _embed_texts_hf(self, texts: List[str], batch_size: int, show_progress: bool, normalize: bool) -> List[List[float]]:
        """Generate batch embeddings using HuggingFace model"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                device=self.device
            )

            # Convert to list of lists
            return [
                emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                for emb in embeddings
            ]

        except RuntimeError as e:
            # If CUDA out of memory, retry with smaller batch
            if "out of memory" in str(e).lower() and batch_size > 8:
                self.logger.warning(
                    f"GPU OOM with batch_size={batch_size}, "
                    f"retrying with batch_size=8"
                )
                return self._embed_texts_hf(texts, 8, show_progress, normalize)
            raise

    def _embed_texts_gguf(self, texts: List[str], batch_size: int, show_progress: bool, normalize: bool) -> List[List[float]]:
        """Generate batch embeddings using GGUF model (handles both backends)"""
        embeddings = []

        if self.active_backend == 'llama-server':
            # Use server API for batch (one request at a time to avoid overwhelming server)
            import time
            for i, text in enumerate(texts):
                try:
                    embedding = self._create_embedding_with_server(text)

                    # Normalize if requested
                    if normalize:
                        embedding = np.array(embedding)
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = (embedding / norm).tolist()

                    embeddings.append(embedding)

                except Exception as e:
                    # If individual text fails, log and use zero vector
                    self.logger.warning(f"Failed to embed text {i+1}/{len(texts)}: {str(e)[:100]}")
                    self.logger.debug(f"Problematic text: {text[:100]}...")
                    embeddings.append([0.0] * self.dimension)

                # Add tiny delay every N requests to prevent server overload
                if (i + 1) % 50 == 0:
                    time.sleep(0.1)  # 100ms pause every 50 requests

        else:
            # Use llama-cpp-python
            import sys
            import os

            stderr_fd = sys.stderr.fileno()
            old_stderr_fd = os.dup(stderr_fd)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)

            try:
                os.dup2(devnull_fd, stderr_fd)

                for text in texts:
                    result = self.model.create_embedding(text)
                    embedding = result['data'][0]['embedding']

                    # Handle 2D embeddings
                    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                        embedding = np.mean(embedding, axis=0).tolist()

                    # Normalize if requested
                    if normalize:
                        embedding = np.array(embedding)
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = (embedding / norm).tolist()

                    embeddings.append(embedding)

            finally:
                os.dup2(old_stderr_fd, stderr_fd)
                os.close(old_stderr_fd)
                os.close(devnull_fd)

        return embeddings
    
    def switch_model(self, model_name: str):
        """
        Switch to a different embedding model
        
        Args:
            model_name: New model to use
        """
        self.logger.info(f"Switching from {self.model_name} to {model_name}")
        
        self.model_name = model_name
        self.dimension = self._get_model_dimension(model_name)
        self.model = None  # Clear old model
        
        # Recalculate default batch size for new model
        self.batch_size = self._resolve_batch_size(None)
        
        if not self.lazy_load:
            self._load_model()
        
        self.logger.info(
            f"✅ Switched to {model_name} "
            f"(dims: {self.dimension}, batch_size: {self.batch_size})"
        )
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            self.logger.error("sklearn not installed")
            raise ImportError("Please install: pip install scikit-learn")
        
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        return float(similarity)
    
    def set_batch_size(self, batch_size: int):
        """
        Update the default batch size
        
        Args:
            batch_size: New batch size
        """
        self.logger.info(f"Changing batch size from {self.batch_size} to {batch_size}")
        self.batch_size = batch_size
    
    def reload_config(self, config_path: str = None):
        """
        Reload configuration from file

        Args:
            config_path: Path to config file (optional)
        """
        self.logger.info("Reloading configuration...")
        self.config = self._load_config(config_path)

        # Reapply settings
        self.device = self._resolve_device(None)
        self.batch_size = self._resolve_batch_size(None)

        self.logger.info(
            f"✅ Config reloaded: device={self.device}, batch_size={self.batch_size}"
        )

    def cleanup(self):
        """
        Cleanup resources (especially llama-server process)
        """
        if self.server_process is not None:
            self.logger.info("Shutting down llama-server process...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()
            self.server_process = None
            self.logger.info("✅ Server process terminated")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass