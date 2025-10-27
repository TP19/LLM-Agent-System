#!/usr/bin/env python3
"""
RAG Retriever

Combines vector search with reranking and context building.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from interactive.rag.vector_store import VectorStore
    from interactive.rag.embeddings import EmbeddingEngine


class RAGRetriever:
    """
    RAG retrieval system
    
    Combines vector search with reranking and context building
    """
    
    def __init__(self, vector_store: 'VectorStore',
                 embedding_engine: Optional['EmbeddingEngine'] = None):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: Vector store instance
            embedding_engine: Embedding engine (optional)
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.logger = logging.getLogger(__name__)
    
    def retrieve(self, query: str, k: int = 5,
                rerank: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query
            k: Number of results
            rerank: Whether to rerank results
            
        Returns:
            List of relevant documents
        """
        # Search vector store
        results = self.vector_store.search(query, k=k*2 if rerank else k)
        
        # Optional reranking
        if rerank and len(results) > k:
            results = self._rerank_results(query, results)
            results = results[:k]
        
        self.logger.debug(f"Retrieved {len(results)} documents for query")
        
        return results
    
    def _rerank_results(self, query: str,
                       results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results by relevance
        
        Args:
            query: Search query
            results: Initial results
            
        Returns:
            Reranked results
        """
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        
        for result in results:
            text_words = set(result['text'].lower().split())
            overlap = len(query_words & text_words)
            result['relevance_score'] = overlap / len(query_words) if query_words else 0
        
        # Sort by relevance
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return results
    
    def build_context(self, query: str,
                     k: int = 3,
                     max_tokens: int = 2000) -> str:
        """
        Build context string from retrieved documents
        
        Args:
            query: Search query
            k: Number of documents
            max_tokens: Max context size
            
        Returns:
            Context string
        """
        results = self.retrieve(query, k=k)
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results, 1):
            text = result['text']
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            tokens = len(text) // 4
            
            if total_length + tokens > max_tokens:
                # Truncate to fit
                remaining = (max_tokens - total_length) * 4
                if remaining > 100:  # Only add if meaningful
                    text = text[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[Document {i}]\n{text}\n")
            total_length += tokens
        
        return "\n".join(context_parts)