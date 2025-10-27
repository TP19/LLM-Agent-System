#!/usr/bin/env python3
"""
Token Counter Utility

Accurate token counting for text using tiktoken (OpenAI's tokenizer)
Falls back to simple estimation if tiktoken not available.
"""

import logging
from typing import Optional, Dict, Any, List


class TokenCounter:
    """
    Token counting and management
    
    Uses tiktoken for accurate counting when available,
    falls back to character-based estimation otherwise.
    """
    
    def __init__(self, max_context: int = 16000):
        self.max_context = max_context
        self.logger = logging.getLogger("TokenCounter")
        
        # Try to import tiktoken
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
            self.logger.info("✅ Using tiktoken for accurate token counting")
        except ImportError:
            self.encoding = None
            self.use_tiktoken = False
            self.logger.warning("⚠️ tiktoken not available, using estimation")
        
    def _normalize_whitespace(self, text: str, preserve_paragraphs: bool = True) -> str:
        """
        Normalize whitespace in text while optionally preserving structure
        
        Args:
            text: Input text
            preserve_paragraphs: If True, keep paragraph breaks (double newline)
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        if preserve_paragraphs:
            # Split by paragraphs
            paragraphs = text.split('\n\n')
            
            # Clean each paragraph
            cleaned_paragraphs = []
            for para in paragraphs:
                # Collapse multiple spaces, tabs, newlines within paragraph
                cleaned = ' '.join(para.split())
                if cleaned:  # Only add non-empty paragraphs
                    cleaned_paragraphs.append(cleaned)
            
            # Rejoin with double newline
            return '\n\n'.join(cleaned_paragraphs)
        else:
            # Aggressive collapse - all whitespace becomes single space
            return ' '.join(text.split())


    def count_tokens(self, text: str, normalize: bool = False) -> int:
        """
        Count tokens in text with optional whitespace normalization
        
        Args:
            text: Text to count tokens for
            normalize: If True, normalize whitespace before counting
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        # Optional normalization
        if normalize:
            text = self._normalize_whitespace(text, preserve_paragraphs=False)
        
        if self.use_tiktoken:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                self.logger.warning(f"tiktoken encoding failed: {e}, falling back to estimation")
                return len(text) // 4
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4


    def chunk_by_tokens(self, text: str, chunk_size: int, 
                    overlap: int = 0, normalize: bool = True) -> List[str]:
        """
        Split text into chunks by token count with improved whitespace handling
        
        Args:
            text: Text to chunk
            chunk_size: Maximum tokens per chunk
            overlap: Tokens to overlap between chunks
            normalize: If True, normalize whitespace before chunking
            
        Returns:
            List of text chunks
        """
        
        # Normalize whitespace if requested (preserves paragraph structure)
        if normalize:
            text = self._normalize_whitespace(text, preserve_paragraphs=True)
            self.logger.debug(f"Normalized text: {len(text)} chars")
        
        if self.use_tiktoken:
            return self._chunk_with_tiktoken(text, chunk_size, overlap)
        else:
            return self._chunk_by_estimation(text, chunk_size, overlap)


    def _chunk_with_tiktoken(self, text: str, chunk_size: int, 
                            overlap: int) -> List[str]:
        """
        Chunk using exact token counting - IMPROVED VERSION
        
        This version has better safeguards against infinite loops.
        """
        
        # Encode entire text
        try:
            tokens = self.encoding.encode(text)
        except Exception as e:
            self.logger.error(f"Failed to encode text: {e}")
            return [text]  # Return whole text as fallback
        
        chunks = []
        start = 0
        iteration = 0
        max_iterations = len(tokens) + 100  # Safety limit
        
        self.logger.debug(f"Chunking {len(tokens)} tokens into chunks of {chunk_size}")
        
        while start < len(tokens):
            # Safety check for infinite loop
            iteration += 1
            if iteration > max_iterations:
                self.logger.error(
                    f"Chunking loop exceeded max iterations ({max_iterations}). "
                    f"Breaking to prevent infinite loop."
                )
                break
            
            # Calculate end position
            end = min(start + chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            try:
                chunk_text = self.encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                self.logger.debug(f"Created chunk {len(chunks)}: {len(chunk_tokens)} tokens")
            except Exception as e:
                self.logger.error(f"Failed to decode chunk: {e}")
                break
            
            # Check if we've reached the end
            if end >= len(tokens):
                self.logger.debug("Reached end of tokens")
                break
            
            # Calculate next start position with overlap
            next_start = end - overlap
            
            # CRITICAL: Ensure forward progress
            # If overlap is too large or causes us to go backwards, force forward
            if next_start <= start:
                self.logger.warning(
                    f"Overlap too large (overlap={overlap}, chunk_size={chunk_size}). "
                    f"Forcing forward progress."
                )
                next_start = start + max(1, chunk_size // 2)  # Move at least half chunk forward
            
            start = next_start
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(tokens)} tokens")
        return chunks


    def _chunk_by_estimation(self, text: str, chunk_size: int,
                            overlap: int) -> List[str]:
        """
        Chunk using character estimation - IMPROVED VERSION
        
        Better handling of boundary detection and whitespace.
        """
        
        # Estimate characters per chunk (4 chars ≈ 1 token)
        chars_per_chunk = chunk_size * 4
        overlap_chars = overlap * 4
        
        chunks = []
        start = 0
        iteration = 0
        max_iterations = (len(text) // chars_per_chunk) + 100  # Safety limit
        
        self.logger.debug(
            f"Chunking {len(text)} chars into chunks of ~{chars_per_chunk} chars "
            f"(target: {chunk_size} tokens)"
        )
        
        while start < len(text):
            # Safety check
            iteration += 1
            if iteration > max_iterations:
                self.logger.error("Chunking exceeded max iterations")
                break
            
            # Calculate initial end position
            end = min(start + chars_per_chunk, len(text))
            
            # Extract chunk
            chunk_text = text[start:end]
            
            # Try to find a good boundary if not at text end
            if end < len(text):
                # Priority 1: Try to break at paragraph boundary (double newline)
                last_para = chunk_text.rfind('\n\n')
                if last_para > chars_per_chunk * 0.6:  # At least 60% through
                    chunk_text = chunk_text[:last_para + 2]  # Include the newlines
                    end = start + last_para + 2
                
                # Priority 2: Try to break at sentence boundary
                elif last_para == -1:
                    sentence_ends = [
                        chunk_text.rfind('. '),
                        chunk_text.rfind('.\n'),
                        chunk_text.rfind('! '),
                        chunk_text.rfind('? ')
                    ]
                    last_sentence = max(sentence_ends)
                    
                    if last_sentence > chars_per_chunk * 0.7:  # At least 70% through
                        chunk_text = chunk_text[:last_sentence + 2]
                        end = start + last_sentence + 2
                    
                    # Priority 3: Try to break at word boundary
                    else:
                        last_space = chunk_text.rfind(' ')
                        if last_space > chars_per_chunk * 0.8:  # At least 80% through
                            chunk_text = chunk_text[:last_space + 1]
                            end = start + last_space + 1
            
            chunks.append(chunk_text.strip())  # Strip whitespace from chunk
            self.logger.debug(f"Created chunk {len(chunks)}: {len(chunk_text)} chars")
            
            # Check if we've reached the end
            if end >= len(text):
                self.logger.debug("Reached end of text")
                break
            
            # Calculate next start with overlap
            next_start = end - overlap_chars
            
            # CRITICAL: Ensure forward progress
            if next_start <= start + 10:  # Need at least 10 chars progress
                next_start = start + max(10, chars_per_chunk // 2)
                self.logger.warning("Forcing forward progress in chunking")
            
            start = next_start
        
        # Filter out any empty chunks
        chunks = [c for c in chunks if c.strip()]
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks


    def estimate_processing_time(self, token_count: int, 
                                tokens_per_second: float = 50) -> float:
        """
        Estimate processing time based on token count
        
        Args:
            token_count: Number of tokens to process
            tokens_per_second: Processing speed (default: 50 tokens/sec)
            
        Returns:
            Estimated time in seconds
        """
        base_time = token_count / tokens_per_second
        
        # Add overhead for chunk processing (context switching, etc.)
        if token_count > 8000:  # If requires chunking
            chunk_count = token_count // 3000  # Approximate chunks
            overhead = chunk_count * 2  # 2 seconds per chunk for processing
            return base_time + overhead
        
        return base_time


    def validate_chunk_integrity(self, chunks: List[str]) -> Dict[str, any]:
        """
        Validate that chunks are well-formed
        
        Returns diagnostic information about chunks.
        """
        if not chunks:
            return {'valid': False, 'error': 'No chunks provided'}
        
        stats = {
            'valid': True,
            'chunk_count': len(chunks),
            'empty_chunks': 0,
            'token_counts': [],
            'total_tokens': 0,
            'avg_tokens': 0,
            'warnings': []
        }
        
        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                stats['empty_chunks'] += 1
                stats['warnings'].append(f"Chunk {i} is empty")
                continue
            
            token_count = self.count_tokens(chunk)
            stats['token_counts'].append(token_count)
            stats['total_tokens'] += token_count
            
            # Check if chunk is suspiciously small
            if token_count < 50:
                stats['warnings'].append(f"Chunk {i} very small: {token_count} tokens")
        
        if stats['token_counts']:
            stats['avg_tokens'] = stats['total_tokens'] / len(stats['token_counts'])
        
        # Validation
        if stats['empty_chunks'] > 0:
            stats['valid'] = False
            stats['warnings'].append(f"{stats['empty_chunks']} empty chunks found")
        
        return stats
    
    def fits_in_context(self, text: str, reserve_tokens: int = 1000) -> bool:
        """
        Check if text fits in context window
        
        Args:
            text: Text to check
            reserve_tokens: Tokens to reserve for prompt/response
            
        Returns:
            True if fits, False otherwise
        """
        token_count = self.count_tokens(text)
        available = self.max_context - reserve_tokens
        
        fits = token_count <= available
        
        # Log for debugging
        if not fits:
            self.logger.debug(f"Text does not fit: {token_count} tokens > {available} available")
        
        return fits
