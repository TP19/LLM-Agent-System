#!/usr/bin/env python3
"""
Interactive Summarization Agent Wrapper

Adds interactive capabilities to SummarizationAgent for Phase 3.
This is critical for RAG/vector DB integration.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.summarization_agent import ModularSummarizationAgent
from interactive.base_interactive_agent import InteractiveAgentMixin
from interactive.modes import InteractiveSession


class InteractiveSummarizationAgent(InteractiveAgentMixin, ModularSummarizationAgent):
    """
    Interactive-capable Summarization Agent
    
    Extends SummarizationAgent with:
    - Progress reporting to UI
    - Checkpoint creation during summarization
    - Session history recording
    - RAG integration for Q&A
    - Vector storage support
    """
    
    def __init__(self, model_manager, enable_rag: bool = False):
        """
        Initialize interactive summarization agent
        
        Args:
            model_manager: Model manager instance (can be None for fallback mode)
            enable_rag: Whether to enable RAG/vector storage
        """
        super().__init__(model_manager)
        self.enable_rag = enable_rag
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"InteractiveSummarizationAgent initialized (RAG: {enable_rag})")
    
    def summarize_file_interactive(self, 
                                   file_path: str,
                                   max_summary_length: int = 500) -> Dict[str, Any]:
        """
        Summarize file with interactive features
        
        Args:
            file_path: Path to file to summarize
            max_summary_length: Maximum summary length
            
        Returns:
            Dictionary with summarization results
        """
        self.report_progress("Starting file summarization...", 0)
        
        try:
            # Read file
            self.report_progress("Reading file...", 10)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Summarize
            self.report_progress("Analyzing content...", 30)
            
            # Call base agent's summarize method
            if hasattr(self, 'summarize_text'):
                summary = self.summarize_text(content, max_summary_length)
            else:
                # Fallback if method doesn't exist
                summary = self._create_fallback_summary(content, max_summary_length)
            
            self.report_progress("Summarization complete", 100)
            
            # Create result dict
            result_dict = {
                'success': True,
                'file_path': file_path,
                'content_length': len(content),
                'summary': summary,
                'summary_length': len(summary),
                'timestamp': datetime.now().isoformat()
            }
            
            # If RAG enabled, store in vector DB
            if self.enable_rag and hasattr(self, 'store_in_vector_db'):
                self.report_progress("Storing in vector database...", 95)
                try:
                    self.store_in_vector_db(file_path, content, summary)
                    result_dict['rag_stored'] = True
                except Exception as e:
                    self.logger.warning(f"Failed to store in vector DB: {e}")
                    result_dict['rag_stored'] = False
            
            # Record in session
            if self.is_interactive():
                self._record_in_session(result_dict)
            
            # Request checkpoint
            self.request_checkpoint(
                checkpoint_type='summarization',
                accumulated_results=self.interactive_session.accumulated_results if self.is_interactive() else None
            )
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}", exc_info=True)
            return self._create_error_result(file_path, str(e))
    
    def query_document_interactive(self,
                                  query: str,
                                  document_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Query document using RAG (if enabled)
        
        Args:
            query: Question to ask
            document_path: Optional specific document to query
            
        Returns:
            Dictionary with query results
        """
        self.report_progress("Processing query...", 0)
        
        if not self.enable_rag:
            return {
                'success': False,
                'error': 'RAG not enabled for this agent',
                'query': query
            }
        
        try:
            # Query vector DB
            self.report_progress("Searching knowledge base...", 30)
            
            if hasattr(self, 'query_vector_db'):
                results = self.query_vector_db(query, document_path)
            else:
                results = self._fallback_query(query)
            
            self.report_progress("Query complete", 100)
            
            result_dict = {
                'success': True,
                'query': query,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Record in session
            if self.is_interactive():
                self._record_query_in_session(result_dict)
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    def _create_fallback_summary(self, content: str, max_length: int) -> str:
        """Create simple fallback summary"""
        if len(content) <= max_length:
            return content
        
        # Simple truncation with ellipsis
        return content[:max_length-3] + "..."
    
    def _fallback_query(self, query: str) -> Dict:
        """Fallback query when RAG unavailable"""
        return {
            'answer': 'RAG functionality not fully implemented',
            'sources': [],
            'confidence': 0.0
        }
    
    def _create_error_result(self, file_path: str, error: str) -> Dict:
        """Create error result"""
        return {
            'success': False,
            'file_path': file_path,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
    
    def _record_in_session(self, result_dict: Dict):
        """Record execution in interactive session"""
        self.interactive_session.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent': 'SummarizationAgent',
            'action': 'summarize_file',
            'result': {
                'success': result_dict['success'],
                'file_path': result_dict.get('file_path', 'N/A'),
                'content_length': result_dict.get('content_length', 0),
                'summary_length': result_dict.get('summary_length', 0)
            }
        })
        
        self.interactive_session.accumulated_results['summarization'] = result_dict
        self.logger.debug(f"Recorded summarization result: {result_dict['success']}")
    
    def _record_query_in_session(self, result_dict: Dict):
        """Record query in interactive session"""
        self.interactive_session.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent': 'SummarizationAgent',
            'action': 'query_document',
            'result': {
                'query': result_dict['query'],
                'success': result_dict['success']
            }
        })


def create_interactive_summarization_agent(model_manager, 
                                          session: InteractiveSession,
                                          enable_rag: bool = False):
    """
    Factory function to create interactive summarization agent
    
    Args:
        model_manager: Model manager instance
        session: Interactive session
        enable_rag: Whether to enable RAG/vector storage
        
    Returns:
        InteractiveSummarizationAgent instance attached to session
    """
    agent = InteractiveSummarizationAgent(model_manager, enable_rag)
    
    # Attach to session if provided
    if session:
        agent.attach_session(
            session=session,
            checkpoint_callback=None,  # Set by orchestrator
            progress_callback=None      # Set by orchestrator
        )
    
    return agent