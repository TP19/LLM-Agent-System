#!/usr/bin/env python3
"""
Semantic Chunker Utility

Intelligent text chunking that preserves semantic boundaries.
MVP: Simple token-based chunking with smart boundaries.
Future: Content-aware chunking (code/docs/books).
"""

import logging
from typing import List
from utilities.token_counter import TokenCounter

class SemanticChunker:
    """
    Intelligent text chunking
    
    MVP: Token-based chunking with boundary detection
    TODO: Add semantic chunking strategies for:
        - Code files (by function/class)
        - Documents (by section/heading)
        - Books (by chapter)
    """
    
    def __init__(self, token_counter: TokenCounter = None):
        self.logger = logging.getLogger("SemanticChunker")
        # Accept token_counter from parent, or create default
        self.token_counter = token_counter if token_counter else TokenCounter()
    
    def chunk_by_tokens(self, text: str, chunk_size: int = 3000,
                       overlap: int = 200) -> List[str]:
        """
        Chunk text by token count with smart boundaries
        
        This is the main MVP chunking method. It:
        1. Splits text into token-sized chunks
        2. Tries to break at semantic boundaries (paragraphs, sentences)
        3. Maintains overlap for context preservation
        
        Args:
            text: Text to chunk
            chunk_size: Target tokens per chunk (default: 3000)
            overlap: Tokens to overlap between chunks (default: 200)
            
        Returns:
            List of text chunks
        """
        
        # DON'T check fits_in_context here - let caller decide
        # The summarization agent already made that decision
        
        self.logger.info(f"✂️ Chunking text with size={chunk_size}, overlap={overlap}")
        
        # Always chunk - don't skip based on context check
        chunks = self.token_counter.chunk_by_tokens(text, chunk_size, overlap)
        
        self.logger.info(f"📦 Created {len(chunks)} chunks")
        
        # Log chunk sizes for debugging
        for i, chunk in enumerate(chunks, 1):
            token_count = self.token_counter.count_tokens(chunk)
            self.logger.debug(f"Chunk {i}: {token_count} tokens")
        
        return chunks
    
    # ========================================================================
    # TODO: Future semantic chunking strategies
    # ========================================================================
    
    def chunk_by_paragraphs(self, text: str, max_tokens: int = 3000) -> List[str]:
        """
        TODO: Chunk by paragraph boundaries (for documents)
        
        Will be useful for:
        - Essays
        - Documentation
        - Blog posts
        """
        raise NotImplementedError("Paragraph chunking not yet implemented")
    
    def chunk_by_sections(self, text: str, max_tokens: int = 3000) -> List[str]:
        """
        TODO: Chunk by section headings (for structured docs)
        
        Will detect:
        - Markdown headers (##, ###)
        - LaTeX sections
        - HTML headers
        """
        raise NotImplementedError("Section chunking not yet implemented")
    
    def chunk_code_by_functions(self, code: str, language: str = "python") -> List[str]:
        """
        TODO: Chunk code by function/class boundaries
        
        Will preserve:
        - Function definitions
        - Class definitions
        - Import statements
        """
        raise NotImplementedError("Code chunking not yet implemented")
    
    def chunk_book_by_chapters(self, text: str) -> List[str]:
        """
        TODO: Chunk books by chapter boundaries
        
        Will detect:
        - Chapter headers
        - Section breaks
        - Part divisions
        """
        raise NotImplementedError("Chapter chunking not yet implemented")
    
    # ========================================================================
    # Helper methods
    # ========================================================================
    
    def _find_best_break_point(self, text: str, target_pos: int,
                              search_range: int = 200) -> int:
        """
        Find the best place to break text near target position
        
        Priority:
        1. Paragraph break (\n\n)
        2. Sentence end (. ! ?)
        3. Word boundary
        4. Character position (fallback)
        
        Args:
            text: Text to search in
            target_pos: Ideal break position
            search_range: How far to search for better break
            
        Returns:
            Best break position
        """
        
        start = max(0, target_pos - search_range)
        end = min(len(text), target_pos + search_range)
        search_text = text[start:end]
        
        # Look for paragraph break
        para_break = search_text.rfind('\n\n')
        if para_break != -1:
            return start + para_break + 2
        
        # Look for sentence end
        sentence_ends = [
            search_text.rfind('. '),
            search_text.rfind('.\n'),
            search_text.rfind('! '),
            search_text.rfind('? ')
        ]
        best_sentence = max(sentence_ends)
        if best_sentence != -1:
            return start + best_sentence + 2
        
        # Look for word boundary
        space = search_text.rfind(' ')
        if space != -1:
            return start + space + 1
        
        # Fallback to target position
        return target_pos