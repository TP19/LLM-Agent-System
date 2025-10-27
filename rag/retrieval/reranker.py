from sentence_transformers import CrossEncoder
from typing import List, Tuple
from ..vector_stores.base_store import MemoryEntry

class Reranker:
    """Cross-encoder reranker for improving retrieval precision"""
    
    def __init__(self,
                model_name: str = "ibm-granite/granite-embedding-reranker-english-r2",
                lazy_load: bool = True):
        """
        Args:
            model_name: Cross-encoder model name
            lazy_load: Load model only when needed
        """
        self.model_name = model_name
        self._model = None
        
        if not lazy_load:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the cross-encoder model"""
        if self._model is None:
            print(f"Loading reranker: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print("Reranker loaded")
    
    @property
    def model(self) -> CrossEncoder:
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
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self._model is not None:
            del self._model
            self._model = None
            print("Reranker unloaded")