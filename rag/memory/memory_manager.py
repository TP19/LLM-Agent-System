from typing import List, Dict, Optional
from datetime import datetime
from ..vector_stores.dual_store_manager import DualStoreManager
from ..vector_stores.base_store import MemoryEntry
from ..embedding.embedding_engine import EmbeddingEngine
from ..retrieval.reranker import Reranker
import uuid

class MemoryManager:
    """Unified interface for episodic and semantic memory"""
    
    def __init__(self,
                 store_manager: DualStoreManager,
                 embedding_engine: EmbeddingEngine,
                 reranker: Optional[Reranker] = None,
                 enable_reranking: bool = True):
        """
        Args:
            store_manager: Dual database manager
            embedding_engine: Embedding generator
            reranker: Reranking model (optional)
            enable_reranking: Use reranking for retrieval
        """
        self.store = store_manager
        self.embedder = embedding_engine
        self.reranker = reranker if reranker and enable_reranking else None
    
    def store_episodic_memory(self,
                             content: str,
                             metadata: Dict,
                             is_private: bool = True,
                             importance: float = 0.5) -> str:
        """
        Store an episodic memory (specific interaction)
        
        Args:
            content: The interaction text
            metadata: Additional context (user, task, outcome, etc.)
            is_private: Store in private DB
            importance: Initial importance score (0-1)
            
        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())
        embedding = self.embedder.embed_text(content)
        
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            memory_type="episodic",
            timestamp=datetime.now().isoformat(),
            importance_score=importance,
            access_count=0
        )
        
        if is_private:
            self.store.add_to_private([memory], "episodic")
        else:
            self.store.add_to_public([memory], "episodic")
        
        return memory_id
    
    def store_semantic_memory(self,
                             knowledge: str,
                             metadata: Dict,
                             is_private: bool = False,
                             importance: float = 0.7) -> str:
        """
        Store semantic memory (extracted knowledge)
        
        Args:
            knowledge: The extracted knowledge/pattern
            metadata: Context about the knowledge
            is_private: Store in private DB
            importance: Initial importance score
        """
        memory_id = str(uuid.uuid4())
        embedding = self.embedder.embed_text(knowledge)
        
        memory = MemoryEntry(
            id=memory_id,
            content=knowledge,
            embedding=embedding,
            metadata=metadata,
            memory_type="semantic",
            timestamp=datetime.now().isoformat(),
            importance_score=importance,
            access_count=0
        )
        
        if is_private:
            self.store.add_to_private([memory], "semantic")
        else:
            self.store.add_to_public([memory], "semantic")
        
        return memory_id
    
    def retrieve_relevant_memories(self,
                                query: str,
                                top_k: int = 10,
                                memory_types: Optional[List[str]] = None,
                                use_reranking: bool = True) -> List[MemoryEntry]:
        """
        Retrieve relevant memories with optional reranking
        
        Args:
            query: User query
            top_k: Number of results
            memory_types: Filter by type ['episodic', 'semantic', 'documents']
            use_reranking: Apply reranking step
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Retrieve more initially for reranking
        initial_k = top_k * 3 if use_reranking and self.reranker else top_k
        
        # Search across relevant collections
        all_memories = []
        
        # Determine which collections to search
        if memory_types:
            collections_to_search = memory_types
        else:
            # Search all common collections
            collections_to_search = ["episodic", "semantic", "documents"]
        
        for collection_name in collections_to_search:
            try:
                memories = self.store.search_both(
                    query_embedding,
                    top_k=initial_k,
                    collection_name=collection_name
                )
                all_memories.extend(memories)
            except Exception as e:
                # Collection might not exist yet - that's OK
                print(f"⚠️ Collection '{collection_name}' not found or empty")
                continue
        
        if not all_memories:
            print(f"⚠️ No memories found for query: '{query}'")
            return []
        
        # Sort by importance score and take top results
        all_memories.sort(key=lambda x: x.importance_score, reverse=True)
        memories = all_memories[:initial_k]
        
        # Rerank if enabled
        if use_reranking and self.reranker and memories:
            memories = self.reranker.rerank(query, memories, top_k)
        else:
            memories = memories[:top_k]
        
        # Update access counts
        for mem in memories:
            mem.access_count += 1
            # In production, update DB here
        
        return memories
    
    def get_debugging_context(self,
                             error_description: str,
                             top_k: int = 5) -> str:
        """
        Get relevant debugging context from past interactions
        
        Args:
            error_description: Description of the error/issue
            top_k: Number of past examples to retrieve
            
        Returns:
            Formatted context string
        """
        memories = self.retrieve_relevant_memories(
            query=error_description,
            top_k=top_k,
            memory_types=["episodic", "procedural"]
        )
        
        if not memories:
            return "No relevant past experiences found."
        
        context_parts = ["=== Relevant Past Experiences ===\n"]
        
        for i, mem in enumerate(memories, 1):
            outcome = mem.metadata.get('outcome', 'unknown')
            solution = mem.metadata.get('solution', 'N/A')
            
            context_parts.append(f"\n--- Experience {i} (Score: {mem.importance_score:.2f}) ---")
            context_parts.append(f"Situation: {mem.content[:200]}...")
            context_parts.append(f"Outcome: {outcome}")
            context_parts.append(f"Solution: {solution}")
        
        return "\n".join(context_parts)
    
    def store_semantic_memory(self,
                            knowledge: str,
                            metadata: dict,
                            is_private: bool = False,
                            importance: float = 0.7) -> str:
        """
        Store semantic memory (extracted knowledge)
        """
        memory_id = str(uuid.uuid4())
        embedding = self.embedder.embed_text(knowledge)
        
        memory = MemoryEntry(
            id=memory_id,
            content=knowledge,
            embedding=embedding,
            metadata=metadata,
            memory_type="semantic",
            timestamp=datetime.now().isoformat(),
            importance_score=importance,
            access_count=0
        )
        
        # Store in "semantic" collection
        if is_private:
            self.store.add_to_private([memory], "semantic")  # Use "semantic" not "documents"
        else:
            self.store.add_to_public([memory], "semantic")
        
        return memory_id