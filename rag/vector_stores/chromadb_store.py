import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from .base_store import BaseVectorStore, MemoryEntry
import json

class ChromaDBStore(BaseVectorStore):
    """ChromaDB implementation with in-memory and persistent modes"""
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 collection_prefix: str = "llm_engine"):
        """
        Args:
            db_path: Path for persistent storage (None = in-memory)
            collection_prefix: Prefix for collection names
        """
        if db_path:
            self.client = chromadb.PersistentClient(path=db_path)
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        self.collection_prefix = collection_prefix
    
    def add_memories(self, memories: List[MemoryEntry], 
                     collection_name: str = "default") -> None:
        """Store memories with metadata"""
        collection = self.client.get_or_create_collection(
            name=f"{self.collection_prefix}_{collection_name}"
        )
        
        collection.add(
            ids=[m.id for m in memories],
            embeddings=[m.embedding for m in memories],
            documents=[m.content for m in memories],
            metadatas=[self._prepare_metadata(m) for m in memories]
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5,
            collection_name: str = "default",
            filters: Optional[Dict] = None) -> List[MemoryEntry]:
        """Search with optional metadata filters"""
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            collection = self.client.get_collection(name=full_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )
            
            return self._parse_results(results)
        except Exception as e:
            # Collection doesn't exist or is empty
            print(f"⚠️ Could not query collection '{collection_name}': {e}")
            return []  # Return empty list instead of crashing
    
    def _prepare_metadata(self, memory: MemoryEntry) -> Dict:
        """Convert MemoryEntry to ChromaDB metadata format"""
        return {
            "memory_type": memory.memory_type,
            "timestamp": memory.timestamp,
            "importance_score": memory.importance_score,
            "access_count": memory.access_count,
            **memory.metadata
        }
    
    def _parse_results(self, results: Dict) -> List[MemoryEntry]:
        """Parse ChromaDB results into MemoryEntry objects"""
        memories = []
        for i, doc_id in enumerate(results['ids'][0]):
            memories.append(MemoryEntry(
                id=doc_id,
                content=results['documents'][0][i],
                embedding=results['embeddings'][0][i] if results.get('embeddings') else [],
                metadata=results['metadatas'][0][i],
                memory_type=results['metadatas'][0][i].get('memory_type', 'unknown'),
                timestamp=results['metadatas'][0][i].get('timestamp', ''),
                importance_score=results['metadatas'][0][i].get('importance_score', 0.5),
                access_count=results['metadatas'][0][i].get('access_count', 0)
            ))
        return memories
    
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection"""
        try:
            full_name = f"{self.collection_prefix}_{collection_name}"
            self.client.delete_collection(name=full_name)
        except Exception as e:
            print(f"⚠️ Could not delete collection {collection_name}: {e}")

    def update_memory(self, memory_id: str, updates: Dict) -> None:
        """Update memory metadata - MVP version"""
        pass  # Skip for Phase 4 MVP