#!/usr/bin/env python3
"""
Intelligent Retriever - Multi-Stage Retrieval Pipeline

Implements quality-aware retrieval with metadata filtering and context expansion.

Quality Modes:
- Fast: 5 chunks, ~2 seconds
- Balanced: 10 chunks, ~5 seconds (default)
- Accurate: 20 chunks, ~10 seconds
- Thorough: 50+ chunks, ~30 seconds
"""

import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from rag.query_parser import QueryIntent
from utilities.metadata_store import MetadataStore


class QualityMode(Enum):
    """Retrieval quality modes"""
    FAST = "fast"          # Quick results, good enough
    BALANCED = "balanced"  # Balance speed and accuracy (default)
    ACCURATE = "accurate"  # Thorough search with verification
    THOROUGH = "thorough"  # Comprehensive multi-document analysis


@dataclass
class RetrievedChunk:
    """Retrieved chunk with metadata and scoring"""
    
    # Chunk identification
    chunk_id: str
    doc_id: str
    chunk_number: int
    
    # Content
    content: str
    summary: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    # Scoring
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    
    # Context
    has_context: bool = False
    context_chunks: List[int] = None
    
    # Source info
    file_path: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    chapter: Optional[int] = None


@dataclass
class RetrievalResult:
    """Complete retrieval result"""
    
    query: str
    chunks: List[RetrievedChunk]
    quality_mode: QualityMode
    
    # Stats
    total_candidates: int = 0
    metadata_filtered: int = 0
    semantic_filtered: int = 0
    final_count: int = 0
    
    # Timing
    retrieval_time: float = 0.0
    
    # Metadata
    documents_searched: List[str] = None


class IntelligentRetriever:
    """Multi-stage intelligent retrieval system"""
    
    def __init__(self, metadata_store: MetadataStore,
                 store_manager,
                 embedding_engine,
                 reranker=None):
        """Initialize retriever"""
        self.metadata_store = metadata_store
        self.store_manager = store_manager
        self.embedder = embedding_engine
        self.reranker = reranker
        self.logger = logging.getLogger("IntelligentRetriever")
        
        # Quality mode configurations
        self.quality_configs = {
            QualityMode.FAST: {
                'initial_k': 10,
                'final_k': 5,
                'use_context': False,
                'use_reranking': False,
                'metadata_filter': True,
            },
            QualityMode.BALANCED: {
                'initial_k': 20,
                'final_k': 10,
                'use_context': True,
                'use_reranking': False,
                'metadata_filter': True,
            },
            QualityMode.ACCURATE: {
                'initial_k': 50,
                'final_k': 20,
                'use_context': True,
                'use_reranking': True,
                'metadata_filter': True,
            },
            QualityMode.THOROUGH: {
                'initial_k': 100,
                'final_k': 50,
                'use_context': True,
                'use_reranking': True,
                'metadata_filter': True,
            },
        }
    
    def retrieve(self, query: str, intent: QueryIntent,
                quality: QualityMode = QualityMode.BALANCED,
                is_private: bool = True) -> RetrievalResult:
        """Main retrieval pipeline"""
        
        start_time = time.time()
        config = self.quality_configs[quality]
        
        self.logger.info(f"Starting {quality.value} retrieval for: {query}")
        
        # Stage 1: Metadata filtering
        candidate_docs = self._filter_by_metadata(intent) if config['metadata_filter'] else None
        
        # Stage 2: Semantic search
        chunks = self._semantic_search(
            query=query,
            doc_ids=candidate_docs,
            intent=intent,
            top_k=config['initial_k'],
            is_private=is_private
        )
        
        if not chunks:
            self.logger.warning("No chunks found in semantic search")
            return RetrievalResult(
                query=query,
                chunks=[],
                quality_mode=quality,
                retrieval_time=time.time() - start_time
            )
        
        semantic_count = len(chunks)
        
        # Stage 3: Context expansion - NOW ENABLED!
        if config['use_context']:
            chunks = self._expand_context(chunks, is_private)
        
        # Stage 4: Reranking
        if config['use_reranking'] and self.reranker:
            chunks = self._rerank_chunks(query, chunks, config['final_k'])
        else:
            chunks = chunks[:config['final_k']]
        
        # Calculate final scores
        self._calculate_final_scores(chunks, intent)
        
        # Sort by final score
        chunks.sort(key=lambda c: c.final_score, reverse=True)
        
        retrieval_time = time.time() - start_time
        
        # Get document info
        doc_ids = list(set(c.doc_id for c in chunks))
        documents = [self.metadata_store.get_document(doc_id=doc_id) 
                    for doc_id in doc_ids]
        doc_titles = [d['title'] if d else 'Unknown' for d in documents]
        
        result = RetrievalResult(
            query=query,
            chunks=chunks,
            quality_mode=quality,
            metadata_filtered=len(candidate_docs) if candidate_docs else 0,
            semantic_filtered=semantic_count,
            final_count=len(chunks),
            retrieval_time=retrieval_time,
            documents_searched=doc_titles
        )
        
        self.logger.info(
            f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s "
            f"(mode: {quality.value})"
        )
        
        return result
    
    def _filter_by_metadata(self, intent: QueryIntent) -> Optional[List[str]]:
        """Stage 1: Fast metadata filtering using SQLite"""
        filters = {}
        
        if intent.author:
            filters['author'] = intent.author
        if intent.title:
            filters['title'] = intent.title
        if intent.doc_type:
            filters['doc_type'] = intent.doc_type
        if intent.language:
            filters['language'] = intent.language
        
        if not filters:
            self.logger.debug("No metadata filters to apply")
            return None
        
        self.logger.debug(f"Applying metadata filters: {filters}")
        doc_ids = self.metadata_store.find_documents(**filters)
        
        self.logger.info(f"Metadata filter found {len(doc_ids)} matching documents")
        return doc_ids if doc_ids else None
    
    def _semantic_search(self, query: str, doc_ids: Optional[List[str]],
                        intent: QueryIntent, top_k: int,
                        is_private: bool) -> List[RetrievedChunk]:
        """Stage 2: Semantic search with optional filtering"""
        
        store = (self.store_manager.private_store if is_private 
                else self.store_manager.public_store)
        
        # Build where filter for ChromaDB
        where_filter = {}
        
        if doc_ids:
            where_filter['doc_id'] = {'$in': doc_ids}
        
        if intent.chapter is not None:
            where_filter['chapter_num'] = intent.chapter
        
        if intent.doc_type:
            where_filter['source_type'] = intent.doc_type
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Determine which collections to search
        collections_to_search = []
        collection_names = ['documents', 'episodic', 'semantic', 'private_semantic', 'private_episodic']
        
        for coll_name in collection_names:
            try:
                full_name = f"{store.collection_prefix}_{coll_name}"
                collection = store.client.get_collection(full_name)
                collections_to_search.append((full_name, collection))
                self.logger.info(f"✓ Will search collection: {full_name}")
            except:
                pass
        
        if not collections_to_search:
            self.logger.error("No collections found to search!")
            return []
        
        # Search all collections and combine results
        all_results = []
        
        for coll_name, collection in collections_to_search:
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, 100),
                    where=where_filter if where_filter else None,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['ids'] and results['ids'][0]:
                    self.logger.info(f"Found {len(results['ids'][0])} results in {coll_name}")
                    all_results.append(results)
            
            except Exception as e:
                self.logger.warning(f"Error searching {coll_name}: {e}")
        
        if not all_results:
            self.logger.warning("No results from any collection")
            return []
        
        # Parse results into RetrievedChunk objects
        chunks = self._parse_search_results(all_results, top_k)
        return chunks
    
    def _parse_search_results(self, results_list: List, top_k: int) -> List[RetrievedChunk]:
        """Parse ChromaDB results into RetrievedChunk objects"""
        all_chunks = []
        
        for results in results_list:
            for i, chunk_id in enumerate(results['ids'][0]):
                try:
                    content = results['documents'][0][i] if results.get('documents') else ""
                    distance = results['distances'][0][i] if results.get('distances') else 0.0
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    
                    similarity = 1.0 - distance if distance <= 1.0 else 0.0
                    
                    doc_id = metadata.get('doc_id', metadata.get('document_id', 'unknown'))
                    chunk_number = metadata.get('chunk_number', i)
                    
                    chunk = RetrievedChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        chunk_number=chunk_number,
                        content=content,
                        summary=metadata.get('chunk_summary'),
                        metadata=metadata,
                        similarity_score=similarity,
                        file_path=metadata.get('file_path'),
                        author=metadata.get('author'),
                        title=metadata.get('title', metadata.get('doc_title')),
                        chapter=metadata.get('chapter_num', metadata.get('chapter'))
                    )
                    
                    all_chunks.append(chunk)
                
                except Exception as e:
                    self.logger.warning(f"Error parsing chunk {i}: {e}")
        
        # Sort by similarity and return top_k
        all_chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        return all_chunks[:top_k]
    
    def _expand_context(self, chunks: List[RetrievedChunk], is_private: bool) -> List[RetrievedChunk]:
        """
        Stage 3: Expand chunks with surrounding context
        
        IMPROVED VERSION:
        - Fetches actual surrounding chunks from ChromaDB
        - Handles missing chunks gracefully
        - Preserves original chunk ordering
        """
        
        self.logger.debug("Expanding context for chunks")
        
        store = (self.store_manager.private_store if is_private 
                else self.store_manager.public_store)
        
        expanded = []
        seen_ids = set()
        
        for chunk in chunks:
            # Calculate surrounding chunk numbers
            prev_chunk = chunk.chunk_number - 1
            next_chunk = chunk.chunk_number + 1
            
            # Try to fetch surrounding chunks from same document
            for chunk_num in [prev_chunk, chunk.chunk_number, next_chunk]:
                if chunk_num < 0:
                    continue
                
                chunk_key = f"{chunk.doc_id}_{chunk_num}"
                
                if chunk_key in seen_ids:
                    continue
                
                # If this is the original chunk, just add it
                if chunk_num == chunk.chunk_number:
                    chunk.has_context = True
                    chunk.context_chunks = [prev_chunk, chunk.chunk_number, next_chunk]
                    expanded.append(chunk)
                    seen_ids.add(chunk_key)
                    continue
                
                # Try to fetch the context chunk from ChromaDB
                try:
                    context_chunk = self._fetch_chunk_by_number(
                        store, chunk.doc_id, chunk_num, chunk.file_path
                    )
                    
                    if context_chunk:
                        # Mark it as context (lower score so it doesn't dominate)
                        context_chunk.similarity_score = chunk.similarity_score * 0.7
                        expanded.append(context_chunk)
                        seen_ids.add(chunk_key)
                
                except Exception as e:
                    self.logger.debug(f"Could not fetch context chunk {chunk_num}: {e}")
        
        self.logger.debug(f"Expanded from {len(chunks)} to {len(expanded)} chunks")
        return expanded
    
    def _fetch_chunk_by_number(self, store, doc_id: str, chunk_number: int, 
                               file_path: str) -> Optional[RetrievedChunk]:
        """Fetch a specific chunk by its number from ChromaDB"""
        
        # Search all possible collections
        collection_names = ['documents', 'semantic', 'episodic']
        
        for coll_name in collection_names:
            try:
                full_name = f"{store.collection_prefix}_{coll_name}"
                collection = store.client.get_collection(full_name)
                
                # Query with metadata filter
                results = collection.get(
                    where={
                        '$and': [
                            {'doc_id': doc_id},
                            {'chunk_number': chunk_number}
                        ]
                    },
                    include=['documents', 'metadatas']
                )
                
                if results['ids'] and len(results['ids']) > 0:
                    chunk_id = results['ids'][0]
                    content = results['documents'][0] if results.get('documents') else ""
                    metadata = results['metadatas'][0] if results.get('metadatas') else {}
                    
                    return RetrievedChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        chunk_number=chunk_number,
                        content=content,
                        summary=metadata.get('chunk_summary'),
                        metadata=metadata,
                        similarity_score=0.0,  # Will be set by caller
                        file_path=file_path,
                        author=metadata.get('author'),
                        title=metadata.get('title'),
                        chapter=metadata.get('chapter_num')
                    )
            
            except Exception as e:
                continue
        
        return None
    
    def _rerank_chunks(self, query: str, chunks: List[RetrievedChunk],
                      final_k: int) -> List[RetrievedChunk]:
        """Stage 4: Rerank using reranker model (if available)"""
        
        if not self.reranker:
            self.logger.debug("No reranker available, skipping reranking")
            return chunks[:final_k]
        
        self.logger.debug(f"Reranking {len(chunks)} chunks")
        
        try:
            reranked = self.reranker.rerank(query, chunks, final_k)
            return reranked
        except Exception as e:
            self.logger.warning(f"Reranking failed: {e}, using original order")
            return chunks[:final_k]
    
    def _calculate_final_scores(self, chunks: List[RetrievedChunk],
                               intent: QueryIntent):
        """Calculate final scores based on multiple factors"""
        
        for chunk in chunks:
            # Start with similarity score
            score = chunk.similarity_score
            
            # Boost if reranked
            if chunk.rerank_score > 0:
                score = (score * 0.4) + (chunk.rerank_score * 0.6)
            
            # Boost if matches author filter
            if intent.author and chunk.author:
                if intent.author.lower() in chunk.author.lower():
                    score *= 1.2
            
            # Boost if matches chapter filter
            if intent.chapter is not None and chunk.chapter == intent.chapter:
                score *= 1.15
            
            # Boost if has summary
            if chunk.summary:
                score *= 1.05
            
            chunk.final_score = min(score, 1.0)  # Cap at 1.0
    
    def format_results(self, result: RetrievalResult, 
                      show_metadata: bool = True) -> str:
        """Format results for display"""
        
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Query: {result.query}")
        lines.append(f"Quality: {result.quality_mode.value}")
        lines.append(f"Time: {result.retrieval_time:.2f}s")
        lines.append(f"{'='*60}\n")
        
        if result.documents_searched:
            lines.append(f"📚 Searched {len(result.documents_searched)} document(s):")
            for doc in result.documents_searched:
                lines.append(f"  • {doc}")
            lines.append("")
        
        lines.append(f"📊 Pipeline Stats:")
        lines.append(f"  Metadata filtered: {result.metadata_filtered} docs")
        lines.append(f"  Semantic search: {result.semantic_filtered} chunks")
        lines.append(f"  Final results: {result.final_count} chunks")
        lines.append("")
        
        lines.append(f"📄 Results:\n")
        
        for i, chunk in enumerate(result.chunks[:10], 1):  # Show top 10
            lines.append(f"[{i}] Score: {chunk.final_score:.3f}")
            
            if show_metadata and chunk.metadata:
                if chunk.title:
                    lines.append(f"    Title: {chunk.title}")
                if chunk.author:
                    lines.append(f"    Author: {chunk.author}")
                if chunk.chapter:
                    lines.append(f"    Chapter: {chunk.chapter}")
            
            # Show content preview
            preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            lines.append(f"    {preview}")
            lines.append("")
        
        return "\n".join(lines)