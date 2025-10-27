#!/usr/bin/env python3
"""
Interactive Coder Agent Wrapper

Adds interactive capabilities to ModularCoderAgent for Phase 3.
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

from agents.coder_agent import ModularCoderAgent, CoderResult
from interactive.base_interactive_agent import InteractiveAgentMixin
from interactive.modes import InteractiveSession


class InteractiveCoderAgent(InteractiveAgentMixin, ModularCoderAgent):
    """
    Interactive-capable Coder Agent
    
    Extends ModularCoderAgent with:
    - Progress reporting to UI
    - Checkpoint creation during code generation
    - Session history recording
    - User interaction support
    """
    
    def __init__(self, model_manager):
        """
        Initialize interactive coder agent
        
        Args:
            model_manager: Model manager instance (can be None for fallback mode)
        """
        super().__init__(model_manager)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("InteractiveCoderAgent initialized")
    
    def process_specification_interactive(self, 
                                         spec_content: str,
                                         base_path: str = ".") -> Dict[str, Any]:
        """
        Process specification with interactive features
        
        Args:
            spec_content: Specification content to process
            base_path: Base path for project creation
            
        Returns:
            Dictionary with generation results
        """
        self.report_progress("Starting code generation...", 0)
        
        try:
            # Call base agent's process_specification method
            self.report_progress("Parsing specification...", 20)
            result = self.process_specification(spec_content, base_path)
            
            self.report_progress("Code generation complete", 100)
            
            # Convert CoderResult to dict
            result_dict = self._convert_result_to_dict(result)
            
            # Record in session
            if self.is_interactive():
                self._record_in_session(result_dict)
            
            # Request checkpoint
            self.request_checkpoint(
                checkpoint_type='code_generation',
                accumulated_results=self.interactive_session.accumulated_results if self.is_interactive() else None
            )
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}", exc_info=True)
            return self._create_fallback_result(spec_content)
    
    def _convert_result_to_dict(self, result: CoderResult) -> Dict:
        """Convert CoderResult to dictionary"""
        return {
            'success': result.success,
            'files_created': result.files_created,
            'directories_created': result.directories_created,
            'errors': result.errors,
            'warnings': result.warnings,
            'processing_time': result.processing_time,
            'timestamp': result.timestamp.isoformat()
        }
    
    def _create_fallback_result(self, spec_content: str) -> Dict:
        """Create fallback result when generation fails"""
        return {
            'success': False,
            'files_created': [],
            'directories_created': [],
            'errors': ['Code generation failed'],
            'warnings': ['Using fallback result'],
            'processing_time': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _record_in_session(self, result_dict: Dict):
        """Record execution in interactive session"""
        self.interactive_session.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent': 'CoderAgent',
            'action': 'process_specification',
            'result': {
                'success': result_dict['success'],
                'files_created': len(result_dict['files_created']),
                'directories_created': len(result_dict['directories_created'])
            }
        })
        
        self.interactive_session.accumulated_results['code_generation'] = result_dict
        self.logger.debug(f"Recorded coder result: {result_dict['success']}")


def create_interactive_coder_agent(model_manager, session: InteractiveSession):
    """
    Factory function to create interactive coder agent
    
    Args:
        model_manager: Model manager instance
        session: Interactive session
        
    Returns:
        InteractiveCoderAgent instance attached to session
    """
    agent = InteractiveCoderAgent(model_manager)
    
    # Attach to session if provided
    if session:
        # These callbacks will be set by the orchestrator
        agent.attach_session(
            session=session,
            checkpoint_callback=None,  # Set by orchestrator
            progress_callback=None      # Set by orchestrator
        )
    
    return agent