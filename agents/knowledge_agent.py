#!/usr/bin/env python3
"""
Knowledge Agent - Complete with Interactive Query Support

COMPLETE IMPLEMENTATION:
- Full query_interactive() method
- Rich formatted output
- Source viewing
- Chunk viewer integration hooks
- All missing imports added
"""

import logging
import re
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path

# Rich imports for interactive display
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.markup import escape

from core.base_agent import BaseAgent
from rag.query_parser import QueryParser, IntentType
from rag.intelligent_retriever import IntelligentRetriever, QualityMode
from rag.source_verification import SourceVerificationSystem, VerificationResult
from utilities.metadata_store import MetadataStore


class KnowledgeAgent(BaseAgent):
    """
    Knowledge retrieval agent with interactive Q&A capabilities
    
    Features:
    - Natural language query understanding
    - Multi-stage intelligent retrieval
    - Quality modes (fast/balanced/accurate/thorough)
    - Rich formatted output
    - Interactive source viewing
    - Chunk viewer integration
    """
    
    def __init__(self, model_manager, memory_manager=None):
        super().__init__("knowledge", model_manager)

        self.memory_manager = memory_manager

        # Initialize components
        self.metadata_store = MetadataStore()
        self.query_parser = QueryParser()

        # Initialize source verification system
        self.verifier = SourceVerificationSystem()

        # Initialize retriever if memory manager available
        if memory_manager:
            self.retriever = IntelligentRetriever(
                metadata_store=self.metadata_store,
                store_manager=memory_manager.store,
                embedding_engine=memory_manager.embedder,
                reranker=memory_manager.reranker
            )
        else:
            self.retriever = None
            self.logger.warning("No memory manager - retriever not available")

        # Test model loading
        try:
            test_result = self.generate_with_logging(
                "Test",
                "init",
                max_tokens=5,
                temperature=0.3
            )
            self.logger.info("✅ Knowledge agent LLM loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to load knowledge LLM: {e}")
            self.logger.error("Make sure 'knowledge' model is configured in config/models.yaml")

        # Cache for last sources (for chunk viewer)
        self._last_sources = []

        self.logger.info("✅ Knowledge agent initialized with source verification")

    def query(self, user_query: str, quality: str = "balanced",
            request_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for knowledge queries
        
        Args:
            user_query: Natural language query
            quality: Quality mode (fast/balanced/accurate/thorough)
            request_id: Request tracking ID
            
        Returns:
            Dict with answer, sources, confidence
        """
        
        if request_id is None:
            import uuid
            request_id = str(uuid.uuid4())[:8]
        
        start_time = datetime.now()
        
        self.logger.info(f"[{request_id}] Knowledge query: {user_query}")
        
        try:
            # Step 1: Parse query
            intent = self.query_parser.parse(user_query)
            
            self.logger.info(
                f"[{request_id}] Parsed intent: {intent.intent_type.value}, "
                f"confidence: {intent.confidence}"
            )
            
            # Step 2: Retrieve relevant chunks
            if not self.retriever:
                return {
                    'success': False,
                    'error': 'Retriever not available - check memory manager initialization'
                }
            
            try:
                quality_mode = QualityMode[quality.upper()]
            except KeyError:
                quality_mode = QualityMode.BALANCED
            
            retrieval_result = self.retriever.retrieve(
                query=user_query,
                intent=intent,
                quality=quality_mode,
                is_private=True
            )
                        
            if not retrieval_result.chunks:
                return {
                    'success': True,
                    'answer': "I don't have enough information to answer that question. The knowledge base might not contain relevant information on this topic.",
                    'sources': [],
                    'confidence': 0.0,
                    'request_id': request_id
                }
            
            # Step 3: Generate answer
            answer = self._generate_answer(user_query, retrieval_result.chunks, intent, request_id)
            
            # Step 4: Format sources
            sources = self._format_sources(retrieval_result.chunks)
            
            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(retrieval_result.chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                f"[{request_id}] Query complete: {len(retrieval_result.chunks)} chunks, "
                f"confidence: {confidence:.2f}, time: {processing_time:.2f}s"
            )
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'intent': self._intent_to_dict(intent),
                'retrieval_stats': {
                    'quality_mode': quality,
                    'time': retrieval_result.retrieval_time,
                    'chunks_found': len(retrieval_result.chunks),
                    'documents_searched': retrieval_result.documents_searched
                },
                'processing_time': processing_time,
                'request_id': request_id
            }
        
        except Exception as e:
            self.logger.error(f"[{request_id}] Error in knowledge query: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'request_id': request_id
            }
    
    def query_with_verification(self, user_query: str, quality: str = "balanced",
                                request_id: str = None) -> Dict[str, Any]:
        """
        Enhanced query with source verification

        Returns:
            {
                'answer': str,
                'sources': List[Dict],
                'confidence': float,
                'verification': VerificationResult,  # NEW
                'verified': bool,  # Quick check
                ... (all other query fields)
            }
        """
        # Get standard result
        result = self.query(user_query, quality, request_id)

        if not result.get('success'):
            # Don't verify if query failed
            result['verified'] = False
            result['verification'] = None
            return result

        # Verify the answer
        verification = self.verifier.verify(
            answer=result['answer'],
            sources=result['sources'],
            original_query=user_query
        )

        # Add verification to result
        result['verification'] = verification
        result['verified'] = verification.is_verified

        # Adjust confidence based on verification
        if not verification.is_verified:
            result['confidence'] = min(
                result['confidence'],
                verification.confidence
            )

        return result

    def query_interactive(self, user_query: str, quality: str = "balanced") -> Dict[str, Any]:
        """
        Interactive query with source viewing option

        Returns:
            Query result dict with 'continue_qa' flag
        """
        console = Console()

        # Get answer WITH verification
        result = self.query_with_verification(user_query, quality)

        if not result.get('success'):
            console.print(f"[red]❌ Error: {result.get('error', 'Unknown error')}[/red]")
            result['continue_qa'] = False  # ✅ Don't continue on error
            return result

        # Display answer with Rich formatting
        self._display_answer_rich(console, result)

        # Show verification results (NEW)
        if result.get('verification'):
            self._display_verification(console, result['verification'])

        # Show confidence BEFORE asking about source details (Fix 2)
        confidence = result.get('confidence', 0)
        if confidence > 0:
            self._display_confidence(console, confidence)

        # Display sources
        if result.get('sources'):
            self._display_sources_rich(console, result['sources'])

            # Offer to view sources in detail
            if Confirm.ask("\n🔍 View source details?", default=False):
                self._interactive_source_viewer(console, result['sources'])

        console.print()
        result['continue_qa'] = Confirm.ask(
            "💬 Ask another question?",
            default=True
        )

        return result

    def _display_answer_rich(self, console: Console, result: Dict):
        """Display answer with Rich formatting"""
        # Escape the answer to prevent Rich markup errors (Fix 1)
        safe_answer = escape(result['answer'])

        console.print("\n")
        console.print(Panel(
            safe_answer,
            title="💡 Answer",
            border_style="cyan",
            padding=(1, 2)
        ))

    def _display_sources_rich(self, console: Console, sources: List[Dict]):
        """Display sources with Rich formatting"""
        if not sources:
            return

        console.print("\n[bold cyan]📚 Sources:[/bold cyan]")

        for i, source in enumerate(sources[:5], 1):  # Show top 5
            # Extract source info
            score = source.get('score', source.get('similarity', 'N/A'))
            preview = source.get('content_preview', source.get('text_preview', ''))

            # Format source entry
            console.print(f"\n  [{i}] ", style="bold yellow", end="")

            # Title/file info (escaped to prevent Rich markup errors - Fix 1)
            if source.get('title'):
                safe_title = escape(str(source['title']))
                console.print(f"{safe_title}", style="cyan")
            elif source.get('file'):
                safe_filename = escape(str(Path(source['file']).name))
                console.print(f"{safe_filename}", style="cyan")
            else:
                console.print("Unknown source", style="dim cyan")

            # Chunk and score info
            chunk_num = source.get('chunk_number', source.get('number', '?'))
            total = source.get('total_chunks', '?')
            console.print(f"      Chunk {chunk_num}/{total} • ", style="dim", end="")

            # Parse score for color coding
            try:
                score_float = float(str(score).replace('%', ''))
            except:
                score_float = 0

            score_color = "green" if score_float > 0.7 else "yellow"
            console.print(f"Score: {score}", style=score_color)

            # Chapter info if available (escaped - Fix 1)
            if source.get('chapter'):
                safe_chapter = escape(str(source['chapter']))
                console.print(f"      Chapter: {safe_chapter}", style="dim cyan")
            elif source.get('chapter_info'):
                ch = source['chapter_info']
                safe_chapter_title = escape(str(ch.get('title', '')))
                console.print(f"      Chapter {ch.get('number')}: {safe_chapter_title}", style="dim cyan")

            # Preview (first 100 chars, escaped - Fix 1)
            if preview:
                preview_text = preview[:100] + "..." if len(preview) > 100 else preview
                safe_preview = escape(str(preview_text))
                console.print(f"      {safe_preview}", style="dim white")

    def _display_verification(self, console: Console, verification: VerificationResult):
        """
        Display verification results to user

        Shows:
        - Verification status
        - Confidence level
        - Issues (if any)
        - Recommendation
        """
        # Choose style based on verification
        if verification.is_verified:
            style = "green"
            icon = "✅"
            title = "Verified"
        elif verification.hallucination_risk < 0.5:
            style = "yellow"
            icon = "⚠️"
            title = "Partially Verified"
        else:
            style = "red"
            icon = "❌"
            title = "Verification Failed"

        # Build content
        lines = []
        lines.append(f"Verification Confidence: {verification.confidence:.0%}")
        lines.append(f"Hallucination Risk: {verification.hallucination_risk:.0%}")

        if verification.issues:
            lines.append("\nIssues:")
            for issue in verification.issues:
                safe_issue = escape(str(issue))
                lines.append(f"  • {safe_issue}")

        lines.append(f"\n{verification.recommendation}")

        console.print("\n")
        console.print(Panel(
            "\n".join(lines),
            title=f"{icon} {title}",
            border_style=style
        ))

    def _display_confidence(self, console: Console, confidence: float):
        """Display confidence score with color coding"""
        color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"

        console.print(f"\n[{color}]📊 Confidence: {confidence:.0%}[/{color}]")

        # Add suggestion for low confidence
        if confidence < 0.5:
            console.print("[dim]💡 Tip: Try rephrasing your question or using 'accurate' mode for better results[/dim]")

    def _interactive_source_viewer(self, console: Console, sources: List[Dict]):
        """
        Interactive source viewing mode
        
        Allows user to select and view sources in detail using chunk viewer
        """
        while True:
            # Show source menu
            console.print("\n[bold cyan]Select source to view:[/bold cyan]")
            console.print("[dim]Enter number (1-{}), 'b' to go back, 'q' to quit[/dim]\n".format(len(sources[:10])))
            
            for i, source in enumerate(sources[:10], 1):
                title = source.get('title', source.get('file', 'Unknown'))
                score = source.get('score', source.get('similarity', 'N/A'))
                console.print(f"  [{i}] {title} (score: {score})")
            
            choice = Prompt.ask("\nYour choice", default="b").strip().lower()
            
            if choice in ['q', 'quit']:
                break
            
            if choice in ['b', 'back']:
                break
            
            # Try to parse as number
            try:
                source_idx = int(choice) - 1
                
                if 0 <= source_idx < len(sources):
                    source = sources[source_idx]
                    self._display_source_detail(console, source, source_idx + 1)
                    
                    # Check if chunk viewer is available
                    if hasattr(self, 'memory_manager') and self.memory_manager:
                        if Confirm.ask("\n🔍 View in chunk viewer with full context?", default=False):
                            self._launch_chunk_viewer(source)
                
                else:
                    console.print(f"[red]Invalid choice. Please enter 1-{len(sources[:10])}[/red]")
            
            except ValueError:
                console.print("[red]Invalid input. Please enter a number or 'b' to go back[/red]")

    def _display_source_detail(self, console: Console, source: Dict, source_num: int):
        """Display detailed information about a source"""
        console.print(f"\n[bold cyan]{'═' * 60}[/bold cyan]")
        console.print(f"[bold cyan]Source {source_num} Details[/bold cyan]")
        console.print(f"[bold cyan]{'═' * 60}[/bold cyan]\n")
        
        # Document info
        if source.get('title'):
            console.print(f"[bold]Document:[/bold] {source['title']}")
        elif source.get('file'):
            console.print(f"[bold]File:[/bold] {source['file']}")
        
        # Chunk info
        chunk_num = source.get('chunk_number', source.get('number', '?'))
        total_chunks = source.get('total_chunks', '?')
        console.print(f"[bold]Chunk:[/bold] {chunk_num}/{total_chunks}")
        
        # Score
        score = source.get('score', source.get('similarity', 'N/A'))
        console.print(f"[bold]Relevance:[/bold] {score}")
        
        # Chapter if available
        if source.get('chapter'):
            console.print(f"[bold]Chapter:[/bold] {source['chapter']}")
        elif source.get('chapter_info'):
            ch = source['chapter_info']
            console.print(f"[bold]Chapter:[/bold] {ch.get('number')} - {ch.get('title')}")
        
        # Full text
        console.print(f"\n[bold]Content:[/bold]")
        full_text = source.get('text_preview', source.get('content_preview', 'No content available'))
        console.print(full_text, style="white")
        console.print()

    def _launch_chunk_viewer(self, source: Dict):
        """
        Launch chunk viewer for detailed source exploration
        
        Integrates with utilities/chunk_viewer.py and utilities/rich_chunk_ui.py
        """
        try:
            from utilities.chunk_viewer import ChunkViewer
            from utilities.rich_chunk_ui import RichChunkUI
            
            # Extract doc_id and chunk_number
            doc_id = source.get('doc_id')
            chunk_num = source.get('chunk_number', source.get('number', 0))
            
            if not doc_id:
                # Try to extract from chunk_id
                chunk_id = source.get('chunk_id', '')
                if chunk_id and '_chunk_' in chunk_id:
                    doc_id = chunk_id[:chunk_id.rfind('_chunk_')]
            
            if not doc_id:
                print("[red]❌ Could not determine document ID from source[/red]")
                return
            
            # Initialize viewer
            viewer = ChunkViewer(
                store_manager=self.memory_manager.store,
                metadata_store=self.metadata_store
            )
            rich_ui = RichChunkUI(viewer)
            
            # Build relevance scores for highlighting
            relevance_scores = {}
            for s in self._last_sources:
                s_chunk_num = s.get('chunk_number', s.get('number', 0))
                s_score_str = str(s.get('score', s.get('similarity', '0'))).replace('%', '')
                try:
                    relevance_scores[s_chunk_num] = float(s_score_str) / 100.0
                except:
                    relevance_scores[s_chunk_num] = 0.0
            
            # Launch viewer
            rich_ui.show_interactive(
                doc_id=doc_id,
                initial_chunk=chunk_num,
                collection="private",
                relevance_scores=relevance_scores
            )
        
        except ImportError as e:
            print(f"[red]❌ Chunk viewer not available: {e}[/red]")
            print("[dim]Make sure chunk_viewer.py and rich_chunk_ui.py are in utilities/[/dim]")
        except Exception as e:
            print(f"[red]❌ Error launching chunk viewer: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    def _intent_to_dict(self, intent) -> Dict[str, Any]:
        """Convert QueryIntent object to JSON-serializable dict"""
        return {
            'raw_query': intent.raw_query,
            'intent_type': intent.intent_type.value if hasattr(intent.intent_type, 'value') else str(intent.intent_type),
            'confidence': intent.confidence,
            'author': intent.author,
            'title': intent.title,
            'chapter': intent.chapter,
            'section': intent.section,
            'time_range': intent.time_range,
            'doc_type': intent.doc_type,
            'topic': intent.topic,
            'language': intent.language,
            'entities_found': intent.entities_found,
            'keywords': intent.keywords
        }    

    def _build_metadata_filters(self, intent) -> Dict[str, Any]:
        """Build metadata filters from query intent"""
        filters = {}
        
        if intent.author:
            filters['author'] = intent.author
        if intent.title:
            filters['title'] = intent.title
        if intent.doc_type:
            filters['doc_type'] = intent.doc_type
        if intent.chapter:
            filters['chapter'] = intent.chapter
        
        return filters
    
    def _generate_answer(self, query: str, chunks, intent, request_id: str) -> str:
        """
        Generate answer using LLM with retrieved context
        """

        # Build context from chunks with SOURCE NUMBERS matching display (Fix 4)
        context_parts = []
        for i, chunk in enumerate(chunks[:10], 1):  # Use top 10, start from 1
            source_info = ""
            if chunk.title:
                source_info = f" (from: {chunk.title}"
                if chunk.chapter:
                    source_info += f", Chapter {chunk.chapter}"
                source_info += ")"

            # Clean up content - remove extra whitespace
            content = ' '.join(chunk.content.split())
            if len(content) > 500:  # Limit chunk size in prompt
                content = content[:500] + "..."

            # Use Source [{i}] format to match display numbering (Fix 4)
            context_parts.append(f"Source [{i}]{source_info}:\n{content}\n")

        context = "\n".join(context_parts)

        # Build prompt
        prompt = f"""Context from knowledge base:

{context}

User question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Be specific and cite which source number supports your answer (e.g., "According to source [1]...")
4. Keep the answer concise but complete
5. If multiple sources contradict each other, mention this
6. DO NOT repeat these instructions in your answer

Answer:"""

        try:
            # Generate answer with LLM (Fix 3: better stop tokens)
            response = self.generate_with_logging(
                prompt,
                f"query_{request_id}",
                max_tokens=500,
                temperature=0.1,  # Very low temperature for factual answers
                stop=["User question:", "Context from", "Instructions:"]  # Fix 3: prevent continuation
            )

            answer = response.strip()

            # Clean the response to remove any leaked instructions (Fix 3)
            answer = self._clean_llm_response(answer)

            # Fallback if answer is empty or too short
            if not answer or len(answer) < 10:
                self.logger.warning(f"[{request_id}] Generated answer too short, using fallback")
                answer = self._create_fallback_answer(chunks)

            return answer

        except Exception as e:
            self.logger.error(f"[{request_id}] Error generating answer: {e}", exc_info=True)
            return self._create_fallback_answer(chunks)
    
    def _create_fallback_answer(self, chunks) -> str:
        """Create fallback answer from chunks without LLM"""
        if not chunks:
            return "I couldn't find relevant information to answer your question."

        # Get top chunk
        top_chunk = chunks[0]

        # Create simple answer from top chunk
        content = top_chunk.content[:300]
        source_info = f" (from {top_chunk.title})" if top_chunk.title else ""

        return f"Based on the available information{source_info}, here's what I found:\n\n{content}...\n\n(Note: Full LLM-generated answer not available. Showing raw excerpts instead.)"

    def _clean_llm_response(self, answer: str) -> str:
        """
        Remove any leaked instructions or formatting from LLM response (Fix 3)

        This fixes the issue where the LLM includes the instruction prompt in its answer.
        """
        # Remove instruction sections that might leak
        patterns_to_remove = [
            r'Instructions?:\s*\n.*?(?=\n\n|\Z)',  # Instructions section
            r'Answer:\s*',  # "Answer:" prefix
            r'User question:.*?(?=\n\n|\Z)',  # User question repetition
            r'Context from knowledge base:.*?(?=\n\n|\Z)',  # Context header
        ]

        for pattern in patterns_to_remove:
            answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = answer.strip()

        return answer
    
    def _format_sources(self, chunks) -> list:
        """
        Format chunk sources for display
        
        Ensures all fields needed for interactive viewing are present
        """
        sources = []
        
        # Cache this for chunk viewer
        self._last_sources = []
        
        for i, chunk in enumerate(chunks, 1):
            source = {
                'number': i,
                'score': f"{chunk.final_score:.2f}",
                'similarity': chunk.final_score,  # Numeric version
                'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                'text_preview': chunk.content,  # Full text for detail view
                
                # IDs for chunk viewer
                'doc_id': chunk.doc_id if hasattr(chunk, 'doc_id') else None,
                'chunk_id': chunk.chunk_id if hasattr(chunk, 'chunk_id') else None,
                'chunk_number': chunk.chunk_number if hasattr(chunk, 'chunk_number') else i-1,
            }
            
            # Document metadata
            if chunk.title:
                source['title'] = chunk.title
            if chunk.author:
                source['author'] = chunk.author
            if chunk.file_path:
                source['file'] = chunk.file_path
            
            # Chapter info
            if hasattr(chunk, 'chapter') and chunk.chapter:
                source['chapter'] = chunk.chapter
            if hasattr(chunk, 'chapter_title') and chunk.chapter_title:
                source['chapter_info'] = {
                    'number': getattr(chunk, 'chapter_num', None),
                    'title': chunk.chapter_title
                }
            
            # Get total_chunks from metadata if available
            if hasattr(chunk, 'doc_id') and chunk.doc_id:
                try:
                    doc_info = self.metadata_store.get_document(doc_id=chunk.doc_id)
                    if doc_info:
                        source['total_chunks'] = doc_info.get('total_chunks', '?')
                        if not source.get('title'):
                            source['title'] = doc_info.get('title', 'Unknown')
                except:
                    source['total_chunks'] = '?'
            else:
                source['total_chunks'] = '?'
            
            sources.append(source)
            self._last_sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, chunks) -> float:
        """Calculate overall confidence score"""

        if not chunks:
            return 0.0

        # Average of top 3 chunk scores
        top_scores = [chunk.final_score for chunk in chunks[:3]]
        confidence = sum(top_scores) / len(top_scores)

        return min(confidence, 1.0)

    def chat(self, message: str, history: list = None) -> str:
        """
        Chat interface for Knowledge Agent - uses query_interactive().

        Leverages the existing RAG pipeline with IntelligentRetriever,
        rich formatting, source viewing, and chunk viewer integration.

        Args:
            message: User's question/query
            history: Optional conversation history

        Returns:
            Response string with answer and sources
        """
        # Check if retriever is available
        if not self.retriever:
            return (
                "Knowledge agent's RAG retriever isn't initialized — you'll get "
                "plain LLM answers without source grounding.\n\n"
                "Likely causes (check the console boot output for the real reason):\n\n"
                "  1) The RAG init crashed during console startup. Scroll up in this\n"
                "     pane — if you see a yellow 'RAG not available: …' line, that\n"
                "     IS the actual error. Common culprits: torch/CUDA mismatch when\n"
                "     running from the wrong venv, missing llama-server binary,\n"
                "     embedding model can't load.\n\n"
                "  2) The LanceDB store has no documents yet. Run:\n"
                "       python scripts/manage_rag.py index <file>\n"
                "       python scripts/manage_rag.py index-folder <dir> --recursive\n"
                "       python scripts/manage_rag.py stats   # verify\n\n"
                "After fixing the underlying issue, exit this agent ( /back ), restart\n"
                "the console, and try again."
            )

        # Use the interactive query method which has rich output
        result = self.query_interactive(message, quality="balanced")

        if not result.get('success'):
            return f"Error: {result.get('error', 'Unknown error')}"

        # The interactive method already displayed rich output
        # Return the answer for conversation history
        return result.get('answer', 'No answer found.')

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge agent statistics"""
        
        base_stats = super().get_stats()
        
        # Add knowledge-specific stats
        metadata_stats = self.metadata_store.get_stats()
        
        return {
            **base_stats,
            'documents_indexed': metadata_stats['documents'],
            'chunks_indexed': metadata_stats['chunks'],
            'total_tokens': metadata_stats['total_tokens']
        }