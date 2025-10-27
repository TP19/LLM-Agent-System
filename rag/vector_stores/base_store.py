from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    """Represents a stored memory with metadata"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    memory_type: str  # 'episodic', 'semantic', 'procedural'
    timestamp: str
    importance_score: float
    access_count: int

class BaseVectorStore(ABC):
    """Abstract interface for vector storage"""
    
    @abstractmethod
    def add_memories(self, memories: List[MemoryEntry]) -> None:
        """Store multiple memories"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, 
               filters: Optional[Dict] = None) -> List[MemoryEntry]:
        """Search for similar memories"""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection"""
        pass
    
    @abstractmethod
    def update_memory(self, memory_id: str, updates: Dict) -> None:
        """Update memory metadata (e.g., access_count, importance)"""
        pass