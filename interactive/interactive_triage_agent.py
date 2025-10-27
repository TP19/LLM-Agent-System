#!/usr/bin/env python3
"""
Interactive Triage Agent Wrapper

Wraps ModularTriageAgent with interactive capabilities.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

# Import the base triage agent and its types
from agents.triage_agent import ModularTriageAgent, TriageResult, TaskCategory, TaskDifficulty

# Import interactive components
from interactive.base_interactive_agent import InteractiveAgentMixin
from interactive.modes import InteractiveSession

import yaml
from pathlib import Path
from core.model_manager import LazyModelManager

# Load config
config_path = Path.home() / "repos" / "LLM-Engine" / "config" / "models.yaml"
with open(config_path) as f:
    models_config = yaml.safe_load(f)['models']

model_manager = LazyModelManager(models_config)


class InteractiveTriageAgent(InteractiveAgentMixin, ModularTriageAgent):
    """
    Interactive wrapper for Triage Agent
    
    Adds:
    - Progress reporting
    - Checkpoint creation
    - Session history recording
    - User interaction support
    """
    
    def __init__(self, model_manager):
        """Initialize interactive triage agent"""
        # Use super() for cooperative multiple inheritance
        # This will properly initialize both parent classes in MRO order
        super().__init__(model_manager)
        
        self.logger = logging.getLogger(__name__)
        self.logger.debug("✅ InteractiveTriageAgent initialized")
    
    def classify_request_interactive(self, user_request: str,
                                    available_agents: Optional[List[str]] = None,
                                    context: Optional[Dict] = None) -> Dict:
        """
        Classify request with interactive features
        
        Args:
            user_request: User's request text
            available_agents: List of available agent names
            context: Optional context dict
            
        Returns:
            Dictionary with classification results
        """

        self._original_request = user_request

        print(f"\n🔍 DEBUG: classify_request_interactive called")
        print(f"🔍 DEBUG: user_request = '{user_request}'")
        print(f"🔍 DEBUG: self._original_request = '{self._original_request}'")
        
        start_time = time.time()
        
        # Report progress
        self.report_progress("Starting triage analysis...", 0)
        
        try:
            # Call the analyze method from ModularTriageAgent
            self.report_progress("Analyzing request type...", 30)
            result = self.analyze(user_request)
            
            self.report_progress("Triage complete", 100)
            
        except Exception as e:
            self.logger.error(f"Triage analysis failed: {e}")
            # Create fallback result
            result = self._create_fallback_result(user_request)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        if self.is_interactive():
            self.interactive_session.total_processing_time += processing_time
        
        # Convert TriageResult to dict
        result_dict = self._convert_result_to_dict(result)
        result_dict['processing_time'] = processing_time
        
        # Add terminal category flag
        result_dict['is_terminal'] = result_dict['classification'] in [
            'summarization', 
            'coding', 
            'knowledge_query'
        ]
        
        # Record in session history
        if self.is_interactive():
            self._record_in_session(result_dict)
        
        # Request checkpoint
        if self.is_interactive():
            self.request_checkpoint(
                checkpoint_type='triage',
                accumulated_results=self.interactive_session.accumulated_results
            )
        
        return result_dict
    
    def _convert_result_to_dict(self, result: TriageResult) -> Dict:
        """Convert TriageResult dataclass to dictionary"""
        
        result_dict = {
            'classification': result.category.value,
            'confidence': result.confidence,
            'recommended_agent': self._map_category_to_agent(result.category),
            'analysis': result.reasoning,
            'needs_clarification': result.needs_clarification,
            'difficulty': result.difficulty.value,
            'timestamp': result.timestamp.isoformat(),
            'original_request': getattr(self, '_original_request', '')
        }
        
        # ✅ ADD THESE DEBUG LINES
        print(f"\n🔍 DEBUG: _convert_result_to_dict called")
        print(f"🔍 DEBUG: result_dict keys = {list(result_dict.keys())}")
        print(f"🔍 DEBUG: original_request in dict = '{result_dict.get('original_request')}'")
        
        return result_dict

    
    def _map_category_to_agent(self, category: TaskCategory) -> str:
        """Map task category to recommended agent"""
        agent_map = {
            TaskCategory.SYSADMIN: 'executor',
            TaskCategory.FILEOPS: 'executor',
            TaskCategory.NETWORK: 'executor',
            TaskCategory.SECURITY: 'security',
            TaskCategory.CODING: 'coder',
            TaskCategory.SUMMARIZATION: 'summarization',
            TaskCategory.KNOWLEDGE_QUERY: 'knowledge',
            TaskCategory.DEVELOPMENT: 'executor',
            TaskCategory.CONTENT: 'summarization',
            TaskCategory.UNKNOWN: 'executor'
        }
        return agent_map.get(category, 'executor')
    
    def _create_fallback_result(self, request: str) -> TriageResult:
        """Create fallback result when analysis fails"""
        return TriageResult(
            category=TaskCategory.UNKNOWN,
            difficulty=TaskDifficulty.MODERATE,
            confidence=0.3,
            reasoning="Analysis failed, using fallback classification",
            needs_clarification=True,
            processing_time=0.0,
            timestamp=datetime.now()
        )
    
    def _record_in_session(self, result_dict: Dict):
        """Record execution in interactive session"""
        if not self.is_interactive():
            return
        
        # Add to execution history
        self.interactive_session.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent': 'TriageAgent',
            'action': 'classify_request',
            'result': {
                'classification': result_dict['classification'],
                'confidence': result_dict['confidence'],
                'recommended_agent': result_dict['recommended_agent']
            }
        })
        
        # Add to accumulated results
        self.interactive_session.accumulated_results['triage'] = result_dict
        
        self.logger.debug(f"Recorded triage result: {result_dict['classification']}")
    
    def explain_current_state(self) -> str:
        """Explain what the agent is currently doing"""
        return "Analyzing your request to determine the best approach and which specialized agent should handle it..."
    
    def suggest_user_actions(self) -> List[str]:
        """Suggest possible actions for the user"""
        return [
            "Continue with the recommended agent",
            "Override the agent selection",
            "Provide more context about your request",
            "Stop and refine your request"
        ]


def create_interactive_triage_agent(model_manager, 
                                   session: Optional[InteractiveSession] = None):
    """
    Factory function to create interactive triage agent
    
    Args:
        model_manager: Model manager instance
        session: Optional interactive session to attach
        
    Returns:
        Configured InteractiveTriageAgent
    """
    agent = InteractiveTriageAgent(model_manager)
    
    if session:
        agent.attach_session(session)
        agent.logger.debug(f"Attached to session {session.session_id}")
    
    return agent