#!/usr/bin/env python3
"""
Chunk Viewer - Core Logic

Loads and navigates document chunks from vector DB
Provides context around specific chunks
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkData:
    """Data for a single chunk"""
    chunk_number: int
    chunk_id: str
    content: str
    doc_id: str
    file_name: str
    token_count: int
    has_code: bool
    chapter_num: Optional[str] = None
    chapter_title: Optional[str] = None
    relevance_score: Optional[float] = None
    is_target: bool = False  # Highlight this chunk

    # New: Adaptive chunking metadata
    content_type: Optional[str] = None  # 'code' | 'text' | 'mixed'
    language: Optional[str] = None  # 'python' | 'c' | 'cpp' | 'text' | etc.
    is_complete: Optional[bool] = None  # Function/paragraph not split
    functions: Optional[list] = None  # List of function names in chunk
    chunk_strategy: Optional[str] = None  # How it was chunked

    # Enhanced metadata
    classes: Optional[list] = None  # Class names
    imports: Optional[list] = None  # Import statements
    keywords: Optional[list] = None  # Key terms
    domains: Optional[list] = None  # High-level domains (security, science, etc.)
    topics: Optional[list] = None  # Specific topics (buffer_overflow, etc.)
    complexity: Optional[str] = None  # simple, medium, complex
    has_examples: Optional[bool] = None
    has_comments: Optional[bool] = None
    has_documentation: Optional[bool] = None
    code_patterns: Optional[list] = None
    language_confidence: Optional[float] = None


class ChunkViewer:
    """
    Core chunk viewing logic
    
    Loads chunks from the vector store and provides navigation/context
    """
    
    def __init__(self, store_manager, metadata_store):
        """
        Initialize chunk viewer
        
        Args:
            store_manager: DualStoreManager instance
            metadata_store: MetadataStore instance
        """
        self.store_manager = store_manager
        self.metadata_store = metadata_store
        self.logger = logging.getLogger("ChunkViewer")
        
        # Cache for loaded chunks
        self._chunk_cache = {}
    
    def load_document_chunks(self, doc_id: str, collection: str = "private") -> Tuple[List[ChunkData], Dict]:
        """
        Load all chunks for a document
        
        Args:
            doc_id: Document ID
            collection: Collection name (private/public)
            
        Returns:
            (chunks, doc_info) tuple
        """
        # Check cache
        cache_key = f"{collection}:{doc_id}"
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]
        
        self.logger.info(f"Loading chunks for doc_id: {doc_id}")
        
        # Get document info from metadata store
        doc_info = self.metadata_store.get_document(doc_id=doc_id)
        
        if not doc_info:
            self.logger.error(f"Document not found: {doc_id}")
            return [], {}
        
        # Get store
        store = (self.store_manager.private_store if collection == "private" 
                else self.store_manager.public_store)
        
        # Get all chunks for this document using BaseVectorStore interface
        collection_name = "documents"

        try:
            # Use LanceDB-compatible method to get documents by filter
            results = store.get_documents_by_filter(
                filter_dict={"doc_id": doc_id},
                collection_name=collection_name,
                limit=10000  # Large limit to get all chunks
            )
        except Exception as e:
            self.logger.error(f"Error loading chunks: {e}")
            return [], doc_info
        
        # Parse results into ChunkData objects
        chunks = []
        
        if results and results['ids']:
            for i, (chunk_id, content, metadata) in enumerate(zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            )):
                chunk = ChunkData(
                    chunk_number=metadata.get('chunk_number', i),
                    chunk_id=chunk_id,
                    content=content,
                    doc_id=doc_id,
                    file_name=metadata.get('file_name', 'Unknown'),
                    token_count=metadata.get('token_count', 0),
                    has_code=metadata.get('has_code', False),
                    chapter_num=metadata.get('chapter_num'),
                    chapter_title=metadata.get('chapter_title'),
                    # New: Adaptive chunking metadata
                    content_type=metadata.get('content_type'),
                    language=metadata.get('language'),
                    is_complete=metadata.get('is_complete'),
                    functions=metadata.get('functions', []),
                    chunk_strategy=metadata.get('chunk_strategy'),
                    # Enhanced metadata
                    classes=metadata.get('classes', []),
                    imports=metadata.get('imports', []),
                    keywords=metadata.get('keywords', []),
                    domains=metadata.get('domains', []),
                    topics=metadata.get('topics', []),
                    complexity=metadata.get('complexity'),
                    has_examples=metadata.get('has_examples'),
                    has_comments=metadata.get('has_comments'),
                    has_documentation=metadata.get('has_documentation'),
                    code_patterns=metadata.get('code_patterns', []),
                    language_confidence=metadata.get('language_confidence')
                )
                chunks.append(chunk)
        
        # Sort by chunk number
        chunks.sort(key=lambda x: x.chunk_number)
        
        # Cache results
        self._chunk_cache[cache_key] = (chunks, doc_info)
        
        self.logger.info(f"Loaded {len(chunks)} chunks")
        return chunks, doc_info
    
    def get_chunk_with_context(self, doc_id: str, chunk_number: int, 
                               context_size: int = 5, 
                               collection: str = "private",
                               relevance_scores: Dict[int, float] = None) -> Dict:
        """
        Get a chunk with surrounding context
        
        Args:
            doc_id: Document ID
            chunk_number: Target chunk number
            context_size: Number of chunks before/after to include
            collection: Collection name
            relevance_scores: Optional dict of chunk_number -> relevance score
            
        Returns:
            Dict with chunks, metadata, and navigation info
        """
        # Load all chunks for document
        chunks, doc_info = self.load_document_chunks(doc_id, collection)
        
        if not chunks:
            return {
                'error': 'No chunks found for document',
                'doc_info': doc_info
            }
        
        # Find target chunk
        target_idx = None
        for i, chunk in enumerate(chunks):
            if chunk.chunk_number == chunk_number:
                target_idx = i
                break
        
        if target_idx is None:
            return {
                'error': f'Chunk {chunk_number} not found',
                'doc_info': doc_info,
                'available_range': (chunks[0].chunk_number, chunks[-1].chunk_number)
            }
        
        # Calculate context range
        start_idx = max(0, target_idx - context_size)
        end_idx = min(len(chunks), target_idx + context_size + 1)
        
        # Extract context chunks
        context_chunks = chunks[start_idx:end_idx]
        
        # Mark target chunk and add relevance scores
        for chunk in context_chunks:
            if chunk.chunk_number == chunk_number:
                chunk.is_target = True
            
            if relevance_scores and chunk.chunk_number in relevance_scores:
                chunk.relevance_score = relevance_scores[chunk.chunk_number]
        
        # Build result
        return {
            'chunks': context_chunks,
            'target_chunk_number': chunk_number,
            'target_index': target_idx,
            'doc_info': doc_info,
            'total_chunks': len(chunks),
            'context_range': (context_chunks[0].chunk_number, context_chunks[-1].chunk_number),
            'can_go_prev': target_idx > 0,
            'can_go_next': target_idx < len(chunks) - 1,
            'progress': (target_idx + 1) / len(chunks)
        }
    
    def search_chunks(self, doc_id: str, query: str, 
                     collection: str = "private") -> List[Tuple[int, str]]:
        """
        Search chunks for a text query
        
        Args:
            doc_id: Document ID
            query: Search query
            collection: Collection name
            
        Returns:
            List of (chunk_number, preview) tuples
        """
        chunks, _ = self.load_document_chunks(doc_id, collection)
        
        query_lower = query.lower()
        results = []
        
        for chunk in chunks:
            if query_lower in chunk.content.lower():
                # Find position of match
                pos = chunk.content.lower().index(query_lower)
                
                # Extract preview around match (50 chars before/after)
                start = max(0, pos - 50)
                end = min(len(chunk.content), pos + len(query) + 50)
                preview = chunk.content[start:end]
                
                if start > 0:
                    preview = "..." + preview
                if end < len(chunk.content):
                    preview = preview + "..."
                
                results.append((chunk.chunk_number, preview))
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str, collection: str = "private") -> Optional[ChunkData]:
        """
        Get a single chunk by its vector store ID
        
        Args:
            chunk_id: vector store chunk ID
            collection: Collection name
            
        Returns:
            ChunkData or None if not found
        """
        # Extract doc_id from chunk_id (format: doc_xxx_chunk_N)
        parts = chunk_id.split('_')
        if len(parts) < 3:
            return None
        
        # Reconstruct doc_id (everything before "_chunk_")
        chunk_part_idx = chunk_id.rfind('_chunk_')
        if chunk_part_idx == -1:
            return None
        
        doc_id = chunk_id[:chunk_part_idx]
        
        # Load document chunks
        chunks, _ = self.load_document_chunks(doc_id, collection)
        
        # Find matching chunk
        for chunk in chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        
        return None
    
    def clear_cache(self):
        """Clear the chunk cache"""
        self._chunk_cache = {}
        self.logger.info("Chunk cache cleared")