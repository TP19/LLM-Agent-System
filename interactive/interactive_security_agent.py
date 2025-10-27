#!/usr/bin/env python3
"""
Interactive Security Agent

Wraps SecurityAgent with interactive capabilities and RAG support.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# Fix imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.security_agent import ModularSecurityAgent, SecuritySuggestion
from interactive.base_interactive_agent import InteractiveAgentMixin
from interactive.modes import InteractiveSession


class InteractiveSecurityAgent(InteractiveAgentMixin, ModularSecurityAgent):
    """
    Interactive-capable Security Agent
    
    Extends ModularSecurityAgent with:
    - Checkpoint support
    - Progress reporting
    - User approval workflow
    - RAG-enhanced command history (optional)
    """
    
    def __init__(self, model_manager):
        """
        Initialize interactive security agent
        
        Args:
            model_manager: Model manager instance
        """
        super().__init__(model_manager)
        self.logger = logging.getLogger(__name__)
        
        # RAG for command history (optional)
        self.rag_store = None
        self.command_history = []
    
    def suggest_approach_interactive(self, user_request: str,
                                    triage_result: Optional[Dict] = None,
                                    context: Optional[Dict] = None) -> Dict:
        """
        Suggest security approach with interactive features - WITH TIME TRACKING
        """
        import time

        start_time = time.time()
        
        # Report progress
        self.report_progress("Starting security analysis...", 0)
        
        # Call base agent's suggest_approach method
        self.report_progress("Analyzing security implications...", 30)
        
        try:
            result = self.suggest_approach(user_request, triage_result)
            
        except Exception as e:
            self.logger.error(f"Security analysis failed: {e}")
            result = self._create_fallback_suggestion(user_request)
        
        self.report_progress("Security analysis complete", 100)
        
        # CALCULATE PROCESSING TIME - ADD THIS BLOCK
        processing_time = time.time() - start_time
        if self.is_interactive():
            self.interactive_session.total_processing_time += processing_time
        
        # Convert SecuritySuggestion to dict
        if hasattr(result, 'commands'):
            # It's a SecuritySuggestion object
            result_dict = {
                'success': True,
                'commands': result.commands if hasattr(result, 'commands') else [],
                'reasoning': result.reasoning if hasattr(result, 'reasoning') else '',
                'approach': result.approach if hasattr(result, 'approach') else '',
                'next_steps': result.next_steps if hasattr(result, 'next_steps') else [],
                'confidence': result.confidence if hasattr(result, 'confidence') else 0.7,
                'risk_level': 'medium',
                'requires_approval': True,
                'security_notes': [],
                'suggested_commands': result.commands if hasattr(result, 'commands') else [],
                'processing_time': processing_time
            }
        else:
            # Already a dict
            result_dict = result
            result_dict.setdefault('success', True)
            result_dict.setdefault('risk_level', 'medium')
            result_dict.setdefault('requires_approval', True)
            result_dict.setdefault('security_notes', [])
            result_dict.setdefault('suggested_commands', [])
            result_dict['processing_time'] = processing_time
        
        # Record in session
        if self.is_interactive():
            self.interactive_session.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'agent': 'SecurityAgent',
                'action': 'suggest_approach',
                'result': {
                    'risk_level': result_dict['risk_level'],
                    'reasoning': result_dict['reasoning'],
                    'suggested_commands': result_dict['suggested_commands'],
                    'requires_approval': result_dict['requires_approval'],
                    'processing_time': processing_time
                }
            })
            
            # Add to accumulated results
            self.interactive_session.accumulated_results['security'] = result_dict
        
        # Request checkpoint
        self.request_checkpoint(
            checkpoint_type='security',
            accumulated_results=self.interactive_session.accumulated_results if self.is_interactive() else None
        )
        
        return result_dict
    
    def _store_commands_in_history(self, commands: List[str], context: str):
        """
        Store commands in history for RAG retrieval
        
        Args:
            commands: List of commands
            context: Context/request that generated commands
        """
        for cmd in commands:
            self.command_history.append({
                'command': cmd,
                'context': context,
                'timestamp': datetime.now().isoformat()
            })
        
        # If RAG is enabled, store in vector DB
        if self.rag_store:
            try:
                documents = []
                for i, entry in enumerate(self.command_history[-len(commands):]):
                    documents.append({
                        'id': f"cmd_{len(self.command_history)-len(commands)+i}",
                        'text': f"Command: {entry['command']}\nContext: {entry['context']}",
                        'metadata': {
                            'command': entry['command'],
                            'timestamp': entry['timestamp']
                        }
                    })
                
                self.rag_store.add_documents(documents)
                self.logger.debug(f"Stored {len(commands)} commands in RAG")
            except Exception as e:
                self.logger.warning(f"Failed to store in RAG: {e}")
    
    def enable_rag(self, rag_store):
        """
        Enable RAG for command history
        
        Args:
            rag_store: Vector store instance
        """
        self.rag_store = rag_store
        self.logger.info("✓ RAG enabled for security agent")
    
    def find_similar_commands(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar commands from history using RAG
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of similar command entries
        """
        if not self.rag_store:
            return []
        
        try:
            results = self.rag_store.search(query, k=k)
            return results
        except Exception as e:
            self.logger.warning(f"RAG search failed: {e}")
            return []
    
    def explain_current_state(self) -> str:
        """Explain what security agent is doing"""
        return "Analyzing security implications and suggesting safe commands..."
    
    def suggest_user_actions(self) -> List[str]:
        """Suggest actions for user"""
        return [
            "Approve all suggested commands",
            "Select specific commands to execute",
            "Modify commands before execution",
            "Request alternative approach",
            "Stop and review security concerns"
        ]


def create_interactive_security_agent(model_manager,
                                     session: Optional[InteractiveSession] = None,
                                     enable_rag: bool = False):
    """
    Factory function for interactive security agent
    
    Args:
        model_manager: Model manager instance
        session: Optional interactive session
        enable_rag: Whether to enable RAG
        
    Returns:
        Configured InteractiveSecurityAgent
    """
    agent = InteractiveSecurityAgent(model_manager)
    
    if session:
        agent.attach_session(session)
    
    if enable_rag:
        try:
            from interactive.rag.vector_store import create_rag_system
            
            vector_store, _ = create_rag_system(
                store_type="chromadb",
                collection_name="security_commands",
                persist_directory=str(Path.home() / ".llm_engine" / "rag")
            )
            
            agent.enable_rag(vector_store)
        except Exception as e:
            logging.warning(f"Failed to enable RAG: {e}")
    
    return agent