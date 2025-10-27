#!/usr/bin/env python3
"""
RAG Infrastructure - Vector Store

Unified RAG system supporting ChromaDB and LanceDB.
Can be used by any agent for context retrieval.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to store"""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_collection(self):
        """Delete the collection"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass


class ChromaDBStore(VectorStore):
    """ChromaDB implementation - Fast, in-memory"""
    
    def __init__(self, collection_name: str = "llm_engine",
                 persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB store
        
        Args:
            collection_name: Name of collection
            persist_directory: Where to persist data
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        # Setup client
        if persist_directory:
            persist_dir = Path(persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.logger.info(f"✓ ChromaDB store initialized: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to ChromaDB
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata'
        """
        ids = [str(doc['id']) for doc in documents]
        texts = [doc['text'] for doc in documents]
        # ChromaDB requires metadata to be None or non-empty dict
        metadatas = [doc.get('metadata') or None for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        self.logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of matching documents
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        documents = []
        
        if results['documents']:
            for i in range(len(results['documents'][0])):
                documents.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return documents
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection.name)
        self.logger.info(f"Deleted collection: {self.collection.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        count = self.collection.count()
        return {
            'type': 'chromadb',
            'collection': self.collection.name,
            'document_count': count
        }


class LanceDBStore(VectorStore):
    """LanceDB implementation - Better for large datasets"""
    
    def __init__(self, collection_name: str = "llm_engine",
                 persist_directory: Optional[str] = None):
        """
        Initialize LanceDB store
        
        Args:
            collection_name: Name of table
            persist_directory: Database directory
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "LanceDB not installed. Install with: pip install lancedb"
            )
        
        # Setup database
        if persist_directory:
            db_path = Path(persist_directory)
            db_path.mkdir(parents=True, exist_ok=True)
        else:
            db_path = Path.home() / ".llm_engine" / "lancedb"
            db_path.mkdir(parents=True, exist_ok=True)
        
        self.db = lancedb.connect(str(db_path))
        self.collection_name = collection_name
        self.table = None
        
        self.logger.info(f"✓ LanceDB store initialized: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to LanceDB
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata', 'vector'
        """
        # Convert to LanceDB format
        data = []
        for doc in documents:
            data.append({
                'id': str(doc['id']),
                'text': doc['text'],
                'metadata': json.dumps(doc.get('metadata', {})),
                'vector': doc.get('vector', [0.0] * 384)  # Default dimension
            })
        
        # Create or open table
        if self.table is None:
            self.table = self.db.create_table(
                self.collection_name,
                data=data,
                mode="overwrite"
            )
        else:
            self.table.add(data)
        
        self.logger.info(f"Added {len(documents)} documents to LanceDB")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query (or query vector)
            k: Number of results
            
        Returns:
            List of matching documents
        """
        if self.table is None:
            return []
        
        # For now, return empty - would need embedding function
        # In production, integrate with embedding model
        self.logger.warning("LanceDB search requires query vector")
        return []
    
    def delete_collection(self):
        """Delete the table"""
        if self.table:
            self.db.drop_table(self.collection_name)
            self.logger.info(f"Deleted table: {self.collection_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get table statistics"""
        count = len(self.table) if self.table else 0
        return {
            'type': 'lancedb',
            'table': self.collection_name,
            'document_count': count
        }


def create_rag_system(store_type: str = "chromadb",
                     collection_name: str = "llm_engine",
                     persist_directory: Optional[str] = None) -> Tuple[VectorStore, 'RAGRetriever']:
    """
    Factory function to create RAG system
    
    Args:
        store_type: Type of vector store ('chromadb' or 'lancedb')
        collection_name: Name of collection/table
        persist_directory: Where to persist data
        
    Returns:
        Tuple of (VectorStore, RAGRetriever)
    """
    from interactive.rag.retriever import RAGRetriever
    
    # Create vector store
    if store_type.lower() == "chromadb":
        store = ChromaDBStore(collection_name, persist_directory)
    elif store_type.lower() == "lancedb":
        store = LanceDBStore(collection_name, persist_directory)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
    
    # Create retriever
    retriever = RAGRetriever(store)
    
    return store, retriever