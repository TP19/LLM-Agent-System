#!/usr/bin/env python3
"""
Embedding Engine for RAG

Generates text embeddings using sentence-transformers.
"""

import logging
from typing import List


class EmbeddingEngine:
    """Generate embeddings for text"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding engine
        
        Args:
            model_name: Sentence transformer model name
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"✓ Embedding engine initialized: {model_name}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embedding[0].tolist()