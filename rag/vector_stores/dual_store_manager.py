from typing import List, Dict, Optional
from .chromadb_store import ChromaDBStore
from .base_store import MemoryEntry
import threading

class DualStoreManager:
    """Manages private and public vector databases"""
    
    def __init__(self, 
                 private_db_path: str,
                 public_db_path: str,
                 enable_parallel: bool = True):
        """
        Args:
            private_db_path: Path for user-specific database
            public_db_path: Path for shared knowledge database
            enable_parallel: Enable parallel search across both DBs
        """
        self.private_store = ChromaDBStore(
            db_path=private_db_path,
            collection_prefix="private"
        )
        self.public_store = ChromaDBStore(
            db_path=public_db_path,
            collection_prefix="public"
        )
        self.enable_parallel = enable_parallel
    
    def add_to_private(self, memories: List[MemoryEntry], 
                       collection_name: str = "default") -> None:
        """Add memories to private database"""
        self.private_store.add_memories(memories, collection_name)
    
    def add_to_public(self, memories: List[MemoryEntry],
                      collection_name: str = "default") -> None:
        """Add memories to public database (admin only)"""
        self.public_store.add_memories(memories, collection_name)
    
    def search_both(self, query_embedding: List[float], 
                    top_k: int = 10,
                    collection_name: str = "default",
                    private_weight: float = 0.6,
                    public_weight: float = 0.4) -> List[MemoryEntry]:
        """
        Search both databases and merge results
        
        Args:
            query_embedding: Query vector
            top_k: Total results to return
            private_weight: Weight for private results (0-1)
            public_weight: Weight for public results (0-1)
        """
        if self.enable_parallel:
            return self._parallel_search(
                query_embedding, top_k, collection_name,
                private_weight, public_weight
            )
        else:
            return self._sequential_search(
                query_embedding, top_k, collection_name,
                private_weight, public_weight
            )
    
    def _parallel_search(self, query_embedding, top_k, collection_name,
                        private_weight, public_weight) -> List[MemoryEntry]:
        """Search both databases in parallel using threads"""
        private_results = []
        public_results = []
        
        def search_private():
            nonlocal private_results
            private_results = self.private_store.search(
                query_embedding, top_k, collection_name
            )
        
        def search_public():
            nonlocal public_results
            public_results = self.public_store.search(
                query_embedding, top_k, collection_name
            )
        
        # Create and start threads
        private_thread = threading.Thread(target=search_private)
        public_thread = threading.Thread(target=search_public)
        
        private_thread.start()
        public_thread.start()
        
        # Wait for both to complete
        private_thread.join()
        public_thread.join()
        
        # Merge and rerank results
        return self._merge_results(
            private_results, public_results,
            private_weight, public_weight, top_k
        )
    
    def _merge_results(self, private_results, public_results,
                       private_weight, public_weight, top_k) -> List[MemoryEntry]:
        """Merge results from both databases with weights"""
        # Adjust importance scores based on weights
        for mem in private_results:
            mem.importance_score *= private_weight
        for mem in public_results:
            mem.importance_score *= public_weight
        
        # Combine and sort by importance
        all_results = private_results + public_results
        all_results.sort(key=lambda x: x.importance_score, reverse=True)
        
        return all_results[:top_k]