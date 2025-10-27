#!/usr/bin/env python3
"""
Modular Summarization Agent - MVP

Handles large files that exceed LLM context windows through:
- Intelligent chunking
- Hierarchical summarization  
- Persistent storage

MVP Focus: Text/Markdown files only, basic functionality
"""

import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from core.base_agent import BaseAgent

# Will need these utilities (to be created)
from utilities.token_counter import TokenCounter
from utilities.semantic_chunker import SemanticChunker
from utilities.context_store import ContextStore
from utilities.metadata_store import MetadataStore

class SummaryDepth(Enum):
    """Depth of analysis"""
    QUICK = "quick"          # 2-3 min - High-level only
    STANDARD = "standard"    # 5-10 min - Detailed summaries
    DEEP = "deep"           # 15+ min - Full analysis

@dataclass
class ChunkSummary:
    """Summary of a single chunk"""
    chunk_number: int
    content: str
    summary: str
    token_count: int
    key_points: List[str]

@dataclass
class FileSummary:
    """Complete file summary result"""
    file_path: str
    file_hash: str
    total_tokens: int
    chunk_count: int
    chunks_processed: int
    summary: str
    key_insights: List[str]
    processing_time: float
    depth: SummaryDepth
    timestamp: datetime
    success: bool
    errors: List[str]

class ModularSummarizationAgent(BaseAgent):
    """
    Intelligent summarization agent for large contexts
    
    MVP Capabilities:
    - Process text/markdown files exceeding context window
    - Hierarchical summarization (chunk → combined → insights)
    - SQLite storage for retrieval
    - Progress tracking
    
    Future Enhancements (TODO):
    - PDF/DOCX/EPUB support
    - Interactive Q&A mode
    - Codebase analysis
    - Semantic chunking strategies
    - Incremental updates
    """

    def __init__(self, model_manager, memory_manager=None):
        super().__init__("summarization", model_manager)
        # Get actual model context from config
        model_context = 8192  # Safe default
        self.memory_manager = memory_manager
        self.enable_rag = True  # Can be configured
        self.metadata_store = MetadataStore()
        self.logger.info("✅ Summarization agent initialized with metadata store")
        # Access the models_config from model_manager
        try:
            # The model_manager stores models_config during initialization
            if hasattr(model_manager, 'models_config'):
                models_config = model_manager.models_config
            elif hasattr(model_manager, 'config'):
                models_config = model_manager.config
            else:
                # Fallback: try to load from file
                import yaml
                with open('config/models.yaml', 'r') as f:
                    config_data = yaml.safe_load(f)
                    models_config = config_data.get('models', {})
            
            summarization_config = models_config.get('summarization', {})
            model_context = summarization_config.get('n_ctx', 8192)
            
            self.logger.info(f"📏 Using model context size: {model_context} tokens")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not read model context from config: {e}")
            self.logger.warning(f"⚠️ Using default context: {model_context} tokens")

            # Initialize utilities with actual model context
            self.token_counter = TokenCounter(max_context=model_context)
            self.chunker = SemanticChunker(token_counter=self.token_counter)
            self.context_store = ContextStore("summaries/summary_cache.db")
            
            # ✅ FIX: ADD THESE LINES - Initialize chunk_size and overlap
            # These are used in _summarize_hierarchically but were never set!
            self.chunk_size = 3000  # Target tokens per chunk
            self.overlap = 200      # Overlap between chunks

            self.logger.info(f"📦 Chunking config: size={self.chunk_size}, overlap={self.overlap}")
        
        # Initialize utilities with actual model context
        self.token_counter = TokenCounter(max_context=model_context)
        self.chunker = SemanticChunker(token_counter=self.token_counter)  # Pass token counter!
        self.context_store = ContextStore("summaries/summary_cache.db")

        self.chunk_size = 3000  # Target tokens per chunk
        self.overlap = 200      # Overlap between chunks
        self.logger.info(f"📦 Chunking config: size={self.chunk_size}, overlap={self.overlap}")

        # Prompts for different summarization stages
        self.prompts = {
            'chunk_summary': self._create_chunk_summary_prompt(),
            'combine_summaries': self._create_combine_prompt(),
            'extract_insights': self._create_insights_prompt()
        }
        
        # Enhanced stats
        self.stats.update({
            'files_summarized': 0,
            'chunks_processed': 0,
            'total_tokens_processed': 0,
            'successful_summaries': 0,
            'cache_hits': 0,
            'chunks_stored_in_vector_db': 0
        })
        
        self.logger.info("✅ Modular summarization agent initialized")

    def summarize_file(self, file_path: str, depth: SummaryDepth = SummaryDepth.STANDARD, 
                    force_refresh: bool = False, request_id: str = None) -> FileSummary:
        """
        Summarize a large file that may exceed context window
        
        Args:
            file_path: Path to file to summarize
            depth: Analysis depth (quick/standard/deep)
            force_refresh: Force re-processing even if cached
            request_id: Request tracking ID
            
        Returns:
            FileSummary with results
        """
        
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        self.logger.info(f"[{request_id}] 📄 Starting summarization: {file_path}")
        
        try:
            # 1. Load and validate file
            self.logger.info(f"[{request_id}] 📖 Step 1: Loading file")
            file_content, file_hash = self._load_file(file_path, request_id)
            
            if not file_content:
                return self._create_error_summary(
                    file_path, "Failed to load file", start_time
                )
            
            # 2. Check cache
            if not force_refresh:
                self.logger.info(f"[{request_id}] 🔍 Checking cache for {file_path}")
                cached_summary = self.context_store.get_summary(file_path, file_hash)
                if cached_summary:
                    self.logger.info(f"[{request_id}] ⚡ Cache HIT! Using cached summary")
                    self.stats['cache_hits'] += 1
                    return self._create_summary_from_cache(cached_summary, start_time)
                else:
                    self.logger.info(f"[{request_id}] ❌ Cache MISS - processing file")
            
            # 3. Count tokens and determine if chunking needed
            total_tokens = self.token_counter.count_tokens(file_content)
            self.logger.info(f"[{request_id}] 📊 File size: {total_tokens} tokens")
            
            # 4. Check if file fits in context
            safety_reserve = int(self.token_counter.max_context * 0.4)
            min_reserve = 2500
            reserve_tokens = max(safety_reserve, min_reserve)
            
            self.logger.info(f"[{request_id}] 📏 Context: {self.token_counter.max_context} tokens, "
                        f"Reserve: {reserve_tokens} tokens, "
                        f"Available: {self.token_counter.max_context - reserve_tokens} tokens")
            
            if self.token_counter.fits_in_context(file_content, reserve_tokens=reserve_tokens):
                # Small enough - summarize directly
                self.logger.info(f"[{request_id}] ⚡ File fits in context, direct summary")
                summary_result = self._summarize_directly(
                    file_content, file_path, depth, request_id
                )
            else:
                # Too large - use hierarchical summarization
                self.logger.info(f"[{request_id}] 🔄 File too large, using hierarchical approach")
                summary_result = self._summarize_hierarchically(
                    file_content, file_path, depth, request_id
                )
            
            # 5. Store in cache
            self.logger.info(f"[{request_id}] 💾 Storing summary in cache")
            summary_id = self.context_store.store_summary(
                file_path=file_path,
                file_hash=file_hash,
                summary=summary_result['summary'],
                key_insights=summary_result['key_insights'],
                chunk_count=summary_result.get('chunk_count', 1),
                total_tokens=total_tokens
            )
            self.logger.info(f"[{request_id}] ✅ Summary cached with ID: {summary_id}")
            
            # 6. Create result BEFORE vector storage (so we can return it properly)
            processing_time = time.time() - start_time
            result = FileSummary(
                file_path=file_path,
                file_hash=file_hash,
                total_tokens=total_tokens,
                chunk_count=summary_result.get('chunk_count', 1),
                chunks_processed=summary_result.get('chunks_processed', 1),
                summary=summary_result['summary'],
                key_insights=summary_result['key_insights'],
                processing_time=processing_time,
                depth=depth,
                timestamp=datetime.now(),
                success=True,
                errors=[]
            )
            
            # Update stats
            self._update_summary_stats(result)
            
            self.logger.info(f"[{request_id}] ✅ Summarization complete in {processing_time:.2f}s")
            return result
        
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Summarization failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_summary(file_path, str(e), start_time)
            

    def _load_file(self, file_path: str, request_id: str) -> Tuple[str, str]:
        """Load file and calculate hash - supports txt, md, pdf"""
        
        try:
            path = Path(file_path)
            
            if not path.exists():
                self.logger.error(f"[{request_id}] ❌ File not found: {file_path}")
                return None, None
            
            # Read based on file type
            suffix = path.suffix.lower()
            
            if suffix == '.pdf':
                content = self._extract_pdf_text(path)
            elif suffix in ['.txt', '.md', '.markdown']:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                self.logger.error(f"[{request_id}] ❌ Unsupported file type: {suffix}")
                return None, None
            
            # Calculate hash
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            self.logger.info(f"[{request_id}] ✅ Loaded {len(content)} characters")
            return content, file_hash
            
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Error loading file: {e}")
            return None, None

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            import PyPDF2
            
            text_parts = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            self.logger.error("PyPDF2 not installed. Install: pip install PyPDF2")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting PDF: {e}")
            return None

    def _summarize_directly(self, content: str, file_path: str, 
                        depth: SummaryDepth, request_id: str) -> Dict[str, Any]:
        """
        Summarize file directly without chunking
        
        FIXED: Now stores in vector DB even for small files!
        """
        
        prompt = self._create_direct_summary_prompt(depth)
        full_prompt = prompt + content
        
        # Final safety check - if prompt is too large, force chunking
        prompt_tokens = self.token_counter.count_tokens(full_prompt)
        if prompt_tokens > (self.token_counter.max_context - 1500):
            self.logger.warning(f"[{request_id}] ⚠️ Prompt too large ({prompt_tokens} tokens), forcing chunking")
            # Fall back to hierarchical summarization
            return self._summarize_hierarchically(content, file_path, depth, request_id)
        
        self.logger.debug(f"[{request_id}] 📝 Direct summary prompt: {prompt_tokens} tokens")
        
        try:
            response = self.generate_with_logging(
                full_prompt,
                request_id,
                max_tokens=1000 if depth == SummaryDepth.QUICK else 1500,
                temperature=0.3,
                top_p=0.9
            )
            
            # Parse response
            summary, insights = self._parse_summary_response(response)
            
            # CRITICAL: Store in vector DB even for small files!
            # Treat entire content as one "chunk"
            self.logger.info(f"[{request_id}] 🔍 RAG Storage Check (Direct):")
            self.logger.info(f"[{request_id}]    - memory_manager exists: {self.memory_manager is not None}")
            self.logger.info(f"[{request_id}]    - enable_rag: {self.enable_rag}")
            
            if self.memory_manager is None:
                self.logger.warning(f"[{request_id}] ⚠️  No memory manager - RAG disabled")
            elif not self.enable_rag:
                self.logger.warning(f"[{request_id}] ⚠️  enable_rag is False - RAG disabled")
            else:
                # Store entire document as single chunk
                self.logger.info(f"[{request_id}] 💾 Storing document in vector DB (direct mode)...")
                try:
                    # Wrap content in a list to treat as single chunk
                    chunks = [content]
                    chunk_summaries = [summary]
                    
                    self._store_chunks_in_memory(
                        file_path=file_path,
                        chunks=chunks,
                        final_summary=summary,
                        chunk_summaries=chunk_summaries
                    )
                    self.logger.info(f"[{request_id}] ✅ Document stored successfully")
                except Exception as e:
                    self.logger.error(f"[{request_id}] ❌ Failed to store document: {e}")
                    import traceback
                    traceback.print_exc()
            
            return {
                'summary': summary,
                'key_insights': insights,
                'chunk_count': 1,
                'chunks_processed': 1
            }
        except ValueError as e:
            if "exceed context window" in str(e):
                self.logger.warning(f"[{request_id}] ⚠️ Context exceeded despite checks, forcing chunking")
                # Fall back to chunking
                return self._summarize_hierarchically(content, file_path, depth, request_id)
            else:
                raise  # Re-raise other ValueError exceptions

    """
    Complete Fixed Methods for agents/summarization_agent.py

    INSTRUCTIONS:
    1. Backup your file: cp agents/summarization_agent.py agents/summarization_agent.py.backup
    2. Replace these TWO methods in your ModularSummarizationAgent class
    3. Test by summarizing a document
    """


    def _summarize_hierarchically(self, content: str, file_path: str, 
                                depth: SummaryDepth, request_id: str) -> Dict:
        """
        Hierarchical summarization for large files - FIXED VERSION
        """
        
        # 1. Chunk the content
        self.logger.info(f"[{request_id}] 📦 Chunking content...")
        chunks = self.chunker.chunk_by_tokens(
            text=content,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        self.logger.info(f"[{request_id}] Created {len(chunks)} chunks")
        
        # 2. Summarize each chunk
        self.logger.info(f"[{request_id}] 📝 Summarizing chunks...")
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"[{request_id}] Processing chunk {i+1}/{len(chunks)}")
            
            # Get chunk content
            if hasattr(chunk, 'content'):
                chunk_text = chunk.content
            elif hasattr(chunk, 'text'):
                chunk_text = chunk.text
            elif isinstance(chunk, str):
                chunk_text = chunk
            else:
                chunk_text = str(chunk)
            
            # Summarize chunk
            summary = self._summarize_chunk(chunk_text, i, request_id)
            chunk_summaries.append(summary)
        
        # 3. Roll up summaries into final summary
        self.logger.info(f"[{request_id}] 🔄 Rolling up summaries...")
        summary_result = self._roll_up_summaries(chunk_summaries, depth, request_id)
        
        # ✅ FIX: summary_result is a dict with 'summary' and 'key_insights'
        final_summary = summary_result['summary']
        key_insights = summary_result['key_insights']
        
        self.logger.info(f"[{request_id}] ✅ Final summary: {len(final_summary)} chars")
        self.logger.info(f"[{request_id}] ✅ Key insights: {len(key_insights)} items")
        
        # 4. Store in vector DB if RAG enabled
        if self.enable_rag and self.memory_manager:
            try:
                self.logger.info(f"[{request_id}] 💾 Storing chunks in vector DB...")
                self._store_chunks_in_memory(
                    chunks=chunks,
                    chunk_summaries=chunk_summaries,
                    file_path=file_path,
                    final_summary=final_summary,  # ✅ FIX: Pass string, not dict
                    request_id=request_id
                )
            except Exception as e:
                self.logger.error(f"[{request_id}] ❌ Failed to store in vector DB: {e}")
                # Don't fail the whole summarization if storage fails
        
        return {
            'summary': final_summary,
            'key_insights': key_insights,
            'chunk_count': len(chunks),
            'chunks_processed': len(chunk_summaries)
        }

    def _roll_up_summaries(self, chunk_summaries: List[ChunkSummary], 
                        depth: SummaryDepth, request_id: str) -> Dict:
        """
        Roll up chunk summaries into final summary - FIXED VERSION
        
        This creates a progressively refined summary using a rolling approach.
        Each chunk is integrated into the running summary.
        
        Args:
            chunk_summaries: List of individual chunk summaries
            depth: Summary depth level
            request_id: Request tracking ID
            
        Returns:
            Dict with 'summary' and 'key_insights'
        """
        
        if not chunk_summaries:
            self.logger.warning(f"[{request_id}] No chunk summaries to roll up")
            return {'summary': '', 'key_insights': []}
        
        self.logger.info(f"[{request_id}] 🔄 Rolling up {len(chunk_summaries)} chunk summaries")
        
        # ========================================================================
        # PHASE 1: Build Rolling Summary
        # ========================================================================
        
        # Start with the first chunk's summary as the base
        rolling_summary = chunk_summaries[0].summary
        self.logger.info(
            f"[{request_id}] 📝 Starting with chunk 1/{len(chunk_summaries)} "
            f"({len(rolling_summary)} chars)"
        )
        
        # Progressively integrate each subsequent chunk
        for i in range(1, len(chunk_summaries)):
            chunk = chunk_summaries[i]
            
            self.logger.info(
                f"[{request_id}] 🔄 Integrating chunk {i+1}/{len(chunk_summaries)} "
                f"(current rolling summary: {len(rolling_summary)} chars)"
            )
            
            # ✅ FIX: Use rolling_summary (not combined_summary) as previous_summary
            rolling_summary = self._combine_with_rolling_summary(
                previous_summary=rolling_summary,
                new_chunk_summary=chunk.summary,
                current_chunk=i+1,
                total_chunks=len(chunk_summaries),
                request_id=request_id
            )
            
            self.logger.debug(
                f"[{request_id}] ✅ Updated rolling summary: {len(rolling_summary)} chars"
            )
            
            # Optional: Check if rolling summary is getting too large
            summary_tokens = self.token_counter.count_tokens(rolling_summary)
            if summary_tokens > 1000:
                self.logger.warning(
                    f"[{request_id}] ⚠️ Rolling summary growing large "
                    f"({summary_tokens} tokens), may condense"
                )
        
        # ========================================================================
        # PHASE 2: Extract Key Insights from Final Summary
        # ========================================================================
        
        self.logger.info(
            f"[{request_id}] 💡 Extracting key insights from final summary "
            f"({len(rolling_summary)} chars)"
        )
        
        key_insights = self._extract_insights(rolling_summary, depth, request_id)
        
        self.logger.info(
            f"[{request_id}] ✅ Roll-up complete! "
            f"Final summary: {len(rolling_summary)} chars, "
            f"Insights: {len(key_insights)}"
        )
        
        return {
            'summary': rolling_summary,
            'key_insights': key_insights
        }


    def _extract_insights(self, summary: str, depth: SummaryDepth, request_id: str) -> List[str]:
        """
        Extract key insights from final summary
        
        Args:
            summary: The final rolled-up summary
            depth: Summary depth level (affects insight detail)
            request_id: Request tracking ID
            
        Returns:
            List of key insights
        """
        
        # Adjust insight extraction based on depth
        if depth == SummaryDepth.QUICK:
            insight_count = "3-5 key points"
            max_tokens = 300
        elif depth == SummaryDepth.STANDARD:
            insight_count = "5-7 key insights"
            max_tokens = 500
        else:  # DEEP
            insight_count = "7-10 detailed insights"
            max_tokens = 700
        
        prompt = f"""Extract the {insight_count} from this summary.

    Focus on:
    - Main themes and patterns
    - Critical findings
    - Important conclusions
    - Actionable insights

    Format as a numbered list.

    Summary:
    {summary}

    Key Insights:
    """
        
        response = self.generate_with_logging(
            prompt,
            request_id,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9
        )
        
        # Parse insights from response
        insights = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith(('•', '-', '*')) or 
                        any(line.startswith(f'{i}.') for i in range(1, 20))):
                # Clean up the line
                cleaned = line.lstrip('•-*0123456789. ')
                if len(cleaned) > 10:  # Must be substantial
                    insights.append(cleaned)
        
        self.logger.info(f"[{request_id}] Extracted {len(insights)} insights")
        return insights


    def _store_chunks_in_memory(self, chunks: List[str], 
                            chunk_summaries: List[ChunkSummary],
                            file_path: str,
                            final_summary: str,  # ✅ This is now a string
                            request_id: str):
        """Store chunks in memory manager - FIXED"""
        
        if not self.memory_manager:
            self.logger.warning(f"[{request_id}] No memory manager available")
            return
        
        stored_count = 0
        failed_count = 0
        
        for i, (chunk, chunk_summary) in enumerate(zip(chunks, chunk_summaries)):
            try:
                # Get chunk text
                if hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                elif isinstance(chunk, str):
                    chunk_text = chunk
                else:
                    chunk_text = str(chunk)
                
                # Prepare metadata
                metadata = {
                    'file_path': str(file_path),
                    'chunk_number': i,
                    'total_chunks': len(chunks),
                    'chunk_summary': chunk_summary.summary,
                    # ✅ FIX: Handle string slicing properly
                    'document_summary': final_summary[:500] if final_summary else "",
                    'source_type': 'document_chunk',
                    'request_id': request_id
                }
                
                # Store as semantic memory (knowledge)
                memory_id = self.memory_manager.store_semantic_memory(
                    knowledge=chunk_text,
                    metadata=metadata,
                    is_private=True,  # User's documents are private
                    importance=0.7
                )
                
                stored_count += 1
                self.logger.debug(f"[{request_id}] Stored chunk {i+1} as {memory_id}")
                
            except Exception as e:
                failed_count += 1
                self.logger.error(f"[{request_id}] Failed to store chunk {i}: {e}")
        
        if stored_count > 0:
            self.logger.info(f"✅ Successfully stored {stored_count}/{len(chunks)} chunks")
            self.stats['chunks_stored_in_vector_db'] = stored_count
        else:
            self.logger.error(f"❌ Failed to store any chunks ({failed_count} failures)")
        
        if failed_count > 0:
            self.logger.warning(f"⚠️  {failed_count} chunks failed to store")

    def _extract_document_metadata(self, file_path: str, content: str) -> Dict:
        """
        Extract metadata from document
        
        For now, simple extraction. Phase 5.2 will add LLM-based extraction.
        """
        
        file_path_obj = Path(file_path)
        
        # Basic metadata
        metadata = {
            'title': file_path_obj.stem,  # Filename without extension
            'author': 'Unknown',  # Will extract later
            'doc_type': 'document',  # Will classify later
            'language': 'en',  # Will detect later
            'file_size': len(content),
        }
        
        # Simple heuristics for doc_type
        extension = file_path_obj.suffix.lower()
        if extension in ['.py', '.js', '.java', '.cpp', '.c', '.go']:
            metadata['doc_type'] = 'code'
        elif extension in ['.log', '.txt'] and 'log' in file_path.lower():
            metadata['doc_type'] = 'log'
        elif extension in ['.md', '.txt']:
            metadata['doc_type'] = 'document'
        
        # Check for code indicators in content
        code_indicators = ['def ', 'class ', 'function ', 'import ', '#include']
        if any(indicator in content[:1000] for indicator in code_indicators):
            metadata['has_code'] = True
        else:
            metadata['has_code'] = False
        
        # Check for math indicators
        math_indicators = ['$$', '\\begin{', '\\frac', '\\sum']
        if any(indicator in content for indicator in math_indicators):
            metadata['has_math'] = True
        else:
            metadata['has_math'] = False
        
        return metadata

    def _summarize_chunk(self, chunk: str, chunk_number: int, 
                        request_id: str) -> ChunkSummary:
        """Summarize a single chunk with robust validation"""
        
        # Build prompt
        prompt = self.prompts['chunk_summary'] + chunk
        
        # Count tokens (might be estimation)
        chunk_tokens = self.token_counter.count_tokens(chunk)
        prompt_tokens = self.token_counter.count_tokens(prompt)
        
        self.logger.debug(f"[{request_id}] 🔍 Chunk {chunk_number}: {chunk_tokens} tokens (estimated)")
        self.logger.debug(f"[{request_id}] 📝 Prompt: {prompt_tokens} tokens (estimated)")
        
        # Adaptive max_tokens based on available space
        # Leave buffer for potential estimation errors
        max_response_tokens = 300
        estimated_total = prompt_tokens + max_response_tokens
        
        # If estimation suggests we're close to limit, be even more conservative
        safety_threshold = int(self.token_counter.max_context * 0.8)  # 80% of context
        
        if estimated_total > safety_threshold:
            self.logger.warning(f"[{request_id}] ⚠️ Chunk {chunk_number} approaching limit "
                              f"({estimated_total} est. tokens), truncating")
            
            # Aggressively truncate - cut chunk in half
            if self.token_counter.use_tiktoken:
                chunk_token_list = self.token_counter.encoding.encode(chunk)
                truncated_tokens = chunk_token_list[:len(chunk_token_list)//2]
                chunk = self.token_counter.encoding.decode(truncated_tokens)
            else:
                # Cut by characters
                chunk = chunk[:len(chunk)//2]
            
            # Rebuild prompt with truncated chunk
            prompt = self.prompts['chunk_summary'] + chunk
            prompt_tokens = self.token_counter.count_tokens(prompt)
            self.logger.info(f"[{request_id}] ✂️ Truncated to {prompt_tokens} tokens (estimated)")
        
        # Generate with conservative max_tokens
        try:
            response = self.generate_with_logging(
                prompt,
                request_id,
                max_tokens=max_response_tokens,
                temperature=0.3,
                top_p=0.9
            )
        except ValueError as e:
            if "exceed context window" in str(e):
                # Last resort - truncate even more aggressively
                self.logger.error(f"[{request_id}] ❌ Chunk {chunk_number} still too large, "
                                f"emergency truncation")
                
                # Cut to 1/4 of original
                if self.token_counter.use_tiktoken:
                    chunk_token_list = self.token_counter.encoding.encode(chunk)
                    truncated_tokens = chunk_token_list[:len(chunk_token_list)//4]
                    chunk = self.token_counter.encoding.decode(truncated_tokens)
                else:
                    chunk = chunk[:len(chunk)//4]
                
                prompt = self.prompts['chunk_summary'] + chunk
                
                # Try again with even smaller response
                response = self.generate_with_logging(
                    prompt,
                    request_id,
                    max_tokens=200,  # Emergency: only 200 tokens
                    temperature=0.3,
                    top_p=0.9
                )
            else:
                raise  # Re-raise if it's a different error
        
        # Parse response
        summary = response.strip()
        self.logger.debug(f"[{request_id}] ✅ Chunk {chunk_number} response: {len(summary)} chars")
        
        # Extract key points (simple parsing)
        key_points = []
        for line in summary.split('\n'):
            if line.strip().startswith(('•', '-', '*')):
                key_points.append(line.strip().lstrip('•-* '))
        
        return ChunkSummary(
            chunk_number=chunk_number,
            content=chunk,
            summary=summary,
            token_count=chunk_tokens,
            key_points=key_points
        )

    def _combine_chunk_summaries(self, combined_text: str, 
                                chunk_summaries: List[ChunkSummary],
                                request_id: str) -> str:
        """Combine multiple chunk summaries into coherent summary"""
        
        prompt = self.prompts['combine_summaries']
        prompt += f"\n\nNumber of chunks: {len(chunk_summaries)}\n\n"
        prompt += combined_text
        
        response = self.generate_with_logging(
            prompt,
            request_id,
            max_tokens=800,  # Reduced from 1200 for safety
            temperature=0.3,
            top_p=0.9
        )
        
        return response.strip()

    def _parse_summary_response(self, response: str) -> Tuple[str, List[str]]:
        """Parse LLM response into summary and insights"""
        
        # Simple parsing - look for insights section
        insights = []
        summary_text = response
        
        # Try to find insights section
        if 'key insights:' in response.lower() or 'key points:' in response.lower():
            parts = response.split('\n')
            summary_lines = []
            insight_mode = False
            
            for line in parts:
                if 'key insights:' in line.lower() or 'key points:' in line.lower():
                    insight_mode = True
                elif insight_mode and line.strip().startswith(('•', '-', '*')):
                    insights.append(line.strip().lstrip('•-* '))
                elif not insight_mode:
                    summary_lines.append(line)
            
            if summary_lines:
                summary_text = '\n'.join(summary_lines).strip()
        
        return summary_text, insights if insights else ["Summary analysis complete"]

    # ========================================================================
    # PROMPT TEMPLATES
    # ========================================================================

    def _create_chunk_summary_prompt(self) -> str:
        """Prompt for summarizing individual chunks"""
        return """You are summarizing a section of a larger document. Create a concise summary that captures the main points and key information.

Keep the summary focused and informative. This summary will be combined with other sections later.

FORMAT:
- 2-4 sentences covering main points
- Include any critical details or facts
- Be clear and concise

Text section to summarize:
"""

    def _create_combine_prompt(self) -> str:
        """Prompt for combining chunk summaries"""
        return """You are combining multiple section summaries into a coherent overall summary.

Create a well-structured summary that:
- Captures the main themes and ideas
- Flows logically from start to end
- Highlights the most important information
- Is readable and clear

IMPORTANT: Create ONE cohesive summary, not a list of sections.

Section summaries to combine:
"""

    def _create_insights_prompt(self) -> str:
        """Prompt for extracting key insights"""
        return """Based on this summary, extract the KEY INSIGHTS - the most important takeaways.

List 5-10 bullet points that capture:
- Main arguments or themes
- Critical information
- Important conclusions
- Notable patterns or trends

Format as bullet points (•) - be concise and actionable.

Summary to analyze:
"""

    def _create_direct_summary_prompt(self, depth: SummaryDepth) -> str:
        """Prompt for direct summarization (file fits in context)"""
        
        if depth == SummaryDepth.QUICK:
            return """Create a QUICK overview of this document:
- 2-3 paragraphs maximum
- Main topic and purpose
- Key takeaways

Document:
"""
        elif depth == SummaryDepth.DEEP:
            return """Create a COMPREHENSIVE analysis of this document:
- Detailed summary covering all major sections
- Key arguments and evidence
- Main conclusions and implications
- Important details worth noting

Document:
"""
        else:  # STANDARD
            return """Create a clear, balanced summary of this document:
- 3-5 paragraphs covering main content
- Key points and important details
- Main conclusions

Document:
"""

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _create_summary_from_cache(self, cached: Dict, start_time: float) -> FileSummary:
        """Create FileSummary from cached data"""
        
        return FileSummary(
            file_path=cached['file_path'],
            file_hash=cached['file_hash'],
            total_tokens=cached['total_tokens'],
            chunk_count=cached['chunk_count'],
            chunks_processed=cached['chunk_count'],
            summary=cached['summary'],
            key_insights=cached['key_insights'],
            processing_time=time.time() - start_time,
            depth=SummaryDepth.STANDARD,  # Cached, so assume standard
            timestamp=datetime.now(),
            success=True,
            errors=[]
        )

    def _create_error_summary(self, file_path: str, error: str, 
                             start_time: float) -> FileSummary:
        """Create error result"""
        
        return FileSummary(
            file_path=file_path,
            file_hash="",
            total_tokens=0,
            chunk_count=0,
            chunks_processed=0,
            summary="",
            key_insights=[],
            processing_time=time.time() - start_time,
            depth=SummaryDepth.STANDARD,
            timestamp=datetime.now(),
            success=False,
            errors=[error]
        )

    def _update_summary_stats(self, result: FileSummary):
        """Update statistics"""
        self.stats['files_summarized'] += 1
        self.stats['total_tokens_processed'] += result.total_tokens
        
        if result.success:
            self.stats['successful_summaries'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        base_stats = super().get_stats()
        return {
            **base_stats,
            'files_summarized': self.stats['files_summarized'],
            'chunks_processed': self.stats['chunks_processed'],
            'total_tokens_processed': self.stats['total_tokens_processed'],
            'successful_summaries': self.stats['successful_summaries'],
            'cache_hits': self.stats['cache_hits'],
            'success_rate': self.stats['successful_summaries'] / max(1, self.stats['files_summarized'])
        }

    def _combine_with_rolling_summary(self, previous_summary: str, new_chunk_summary: str,
                                    current_chunk: int, total_chunks: int,
                                    request_id: str) -> str:
        """
        Combine previous rolling summary with new chunk summary.
        
        FIXED: More aggressive condensation to prevent context overflow
        """
        
        
        # First, check if previous summary needs condensing BEFORE building prompt
        prev_tokens = self.token_counter.count_tokens(previous_summary)
        new_tokens = self.token_counter.count_tokens(new_chunk_summary)
        
        # Reserve space for prompt structure and output (estimate ~400 tokens)
        max_input_tokens = int(self.token_counter.max_context * 0.6)  # More conservative: 60%
        max_prev_summary_tokens = max_input_tokens // 2  # Half for previous summary
        max_new_chunk_tokens = max_input_tokens // 2  # Half for new chunk
        
        # Condense previous summary if too large
        if prev_tokens > max_prev_summary_tokens:
            self.logger.warning(
                f"[{request_id}] Previous summary too large ({prev_tokens} tokens), "
                f"condensing to {max_prev_summary_tokens}"
            )
            previous_summary = self._condense_summary(
                previous_summary, 
                request_id,
                target_tokens=max_prev_summary_tokens
            )
        
        # Condense new chunk summary if too large
        if new_tokens > max_new_chunk_tokens:
            self.logger.warning(
                f"[{request_id}] New chunk summary too large ({new_tokens} tokens), "
                f"condensing to {max_new_chunk_tokens}"
            )
            new_chunk_summary = self._condense_summary(
                new_chunk_summary,
                request_id,
                target_tokens=max_new_chunk_tokens
            )
        
        # Build prompt
        prompt = f"""Combine these summaries progressively.

    Previous summary (chunks 1-{current_chunk-1}):
    {previous_summary}

    New chunk summary (chunk {current_chunk}/{total_chunks}):
    {new_chunk_summary}

    Create a combined summary that:
    1. Integrates new information
    2. Maintains key points from previous chunks
    3. Flows naturally (max 300 words)

    Combined summary:
    """
        
        # Final safety check
        prompt_tokens = self.token_counter.count_tokens(prompt)
        max_safe = int(self.token_counter.max_context * 0.8)
        
        if prompt_tokens > max_safe:
            # Emergency: use ultra-short prompt
            self.logger.error(
                f"[{request_id}] Emergency: prompt still too large ({prompt_tokens} tokens), "
                "using ultra-short combination"
            )
            prompt = f"""Combine: {previous_summary[:500]} + {new_chunk_summary[:500]}"""
        
        response = self.generate_with_logging(
            prompt,
            request_id,
            max_tokens=500,  # Reduced from 600
            temperature=0.3,
            top_p=0.9
        )
        
        return response.strip()

    def _condense_summary(self, summary: str, request_id: str, target_tokens: int = 300) -> str:
        """
        Condense a summary to target token count.
        
        FIXED: Accept target_tokens parameter for more control
        """
        
        current_tokens = self.token_counter.count_tokens(summary)
        
        # If already small enough, return as-is
        if current_tokens <= target_tokens:
            return summary
        
        # Calculate how much to condense
        reduction_ratio = target_tokens / current_tokens
        target_words = int(reduction_ratio * 150)  # Rough estimate
        
        prompt = f"""Condense this summary to its core points (max {target_words} words):

    {summary}

    Condensed version:
"""