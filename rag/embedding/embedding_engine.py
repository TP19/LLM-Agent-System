#!/usr/bin/env python3
"""
Config-Aware Embedding Engine - Reads from rag_config.yaml

All settings are configurable via config/rag_config.yaml
No hardcoded defaults except as fallbacks if config missing
"""

import logging
import numpy as np
import yaml
from pathlib import Path
from typing import List, Union, Optional, Dict


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
                 device: str = None, batch_size: int = None, config_path: str = None):
        """
        Initialize embedding engine
        
        Args:
            model_name: Model to use (overrides config)
            lazy_load: Load model on first use (overrides config)
            device: Device to use (overrides config)
            batch_size: Batch size (overrides config)
            config_path: Path to config file (default: config/rag_config.yaml)
        """
        self.logger = logging.getLogger("EmbeddingEngine")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Use provided values OR fallback to config OR use final defaults
        self.model_name = model_name or self.config.get('model', 'all-MiniLM-L6-v2')
        self.lazy_load = lazy_load if lazy_load is not None else self.config.get('lazy_load', True)
        
        # Device detection and configuration
        self.device = self._resolve_device(device)
        
        # Batch size configuration
        self.batch_size = self._resolve_batch_size(batch_size)
        
        # Get model dimension
        self.dimension = self._get_model_dimension(self.model_name)
        
        # Model placeholder
        self.model = None
        
        # Load immediately if not lazy
        if not self.lazy_load:
            self._load_model()
        
        self.logger.info(
            f"✅ Embedding engine initialized "
            f"(model: {self.model_name}, dims: {self.dimension}, "
            f"device: {self.device}, batch_size: {self.batch_size}, lazy: {self.lazy_load})"
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
    
    def _get_model_dimension(self, model_name: str) -> int:
        """Get embedding dimension for model"""
        dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "instructor-large": 768,
            "instructor-xl": 768,
        }
        return dimensions.get(model_name, 384)
    
    def _load_model(self):
        """Load the embedding model"""
        if self.model is not None:
            return  # Already loaded
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.logger.info(f"Device: {self.device}")
            
            # Load model with specified device
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            self.logger.info(f"✅ Model loaded: {self.model_name} ({self.dimension} dims)")
            self.logger.info(f"   Using device: {self.model.device}")
            
        except ImportError:
            self.logger.error("sentence-transformers not installed")
            raise ImportError(
                "Please install: pip install sentence-transformers"
            )
        
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
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
        
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    
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
        self.logger.info(
            f"Embedding {len(texts)} texts with batch_size={batch_size} "
            f"on {self.device}"
        )
        
        # Filter out empty texts, replace with space
        valid_texts = [t if t and t.strip() else " " for t in texts]
        
        try:
            embeddings = self.model.encode(
                valid_texts,
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
                return self.embed_texts(
                    texts,
                    batch_size=8,
                    show_progress=show_progress,
                    normalize=normalize
                )
            raise
        
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            # Return zero vectors
            return [[0.0] * self.dimension for _ in texts]
    
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