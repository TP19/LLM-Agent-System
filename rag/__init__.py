#!/usr/bin/env python3
"""
RAG Package - Retrieval Augmented Generation

Components:
- query_parser: Natural language query understanding
- intelligent_retriever: Multi-stage retrieval pipeline
- embedding: Text embedding generation
- vector_stores: Vector database management
- memory: Memory management system
"""

__version__ = "0.1.0"

# Make key classes easily importable
from rag.query_parser import QueryParser, QueryIntent, IntentType
from rag.intelligent_retriever import IntelligentRetriever, QualityMode, RetrievedChunk

__all__ = [
    'QueryParser',
    'QueryIntent',
    'IntentType',
    'IntelligentRetriever',
    'QualityMode',
    'RetrievedChunk',
]