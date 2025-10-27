#!/usr/bin/env python3
"""
Base Interactive Agent

Base class for agents that support interactive mode with checkpoints.
All agents should inherit from this to enable checkpoint support.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from interactive.modes import InteractiveSession, CheckpointFrequency


class BaseInteractiveAgent(ABC):
    """
    Base class for interactive-capable agents
    
    Provides:
    - Session attachment
    - Progress callbacks
    - State serialization
    - Interactive prompts
    """
    
    def __init__(self, model_manager=None):
        """
        Initialize base interactive agent
        
        Args:
            model_manager: Model manager instance (optional for testing)
        """
        self.model_manager = model_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_progress_time = None
        
        # Interactive session
        self.interactive_session: Optional[InteractiveSession] = None
        self.checkpoint_callback = None
        self.progress_callback = None
        
        # State
        self._state = {}
    
    def attach_session(self, session: InteractiveSession,
                      checkpoint_callback=None,
                      progress_callback=None):
        """
        Attach to interactive session
        
        Args:
            session: Interactive session to attach to
            checkpoint_callback: Function to call when checkpoint should be created
            progress_callback: Function to call to report progress
        """
        self.interactive_session = session
        self.checkpoint_callback = checkpoint_callback
        self.progress_callback = progress_callback
        
        self.logger.info(
            f"Attached to session {session.session_id} "
            f"(mode: {session.mode.value}, freq: {session.checkpoint_frequency.value})"
        )
    
    def detach_session(self):
        """Detach from interactive session"""
        if self.interactive_session:
            self.logger.info(f"Detached from session {self.interactive_session.session_id}")
        
        self.interactive_session = None
        self.checkpoint_callback = None
        self.progress_callback = None
    
    def is_interactive(self) -> bool:
        """Check if agent is in interactive mode"""
        return self.interactive_session is not None
    
    def should_checkpoint(self, checkpoint_type: str) -> bool:
        """
        Check if checkpoint should be created
        
        Args:
            checkpoint_type: Type of checkpoint
            
        Returns:
            True if checkpoint should be created
        """
        if not self.is_interactive():
            return False
        
        frequency = self.interactive_session.checkpoint_frequency
        
        if frequency == CheckpointFrequency.DISABLED:
            return False
        
        if frequency == CheckpointFrequency.MINIMAL:
            return checkpoint_type == "completion_review"
        
        if frequency == CheckpointFrequency.STANDARD:
            return checkpoint_type in [
                "triage", "security", "execution", "completion_review"
            ]
        
        if frequency == CheckpointFrequency.DETAILED:
            return True
        
        # SMART mode - let agent decide
        return self._smart_checkpoint_decision(checkpoint_type)
    
    def _smart_checkpoint_decision(self, checkpoint_type: str) -> bool:
        """
        Smart checkpoint decision (to be overridden by subclasses)
        
        Args:
            checkpoint_type: Type of checkpoint
            
        Returns:
            True if checkpoint should be created
        """
        # Default: same as STANDARD
        return checkpoint_type in [
            "triage", "security", "execution", "completion_review"
        ]
    
    def report_progress(self, 
                    message: str,
                    percentage: Optional[float] = None,
                    details: Optional[Dict] = None):
        """
        Report progress - works in BOTH modes
        Also tracks time for session
        """
        if self.is_interactive() and self.progress_callback:
            try:
                self.progress_callback(
                    agent=self.agent_name,
                    message=message,
                    percentage=percentage,
                    details=details
                )
                
                # Track time in session
                if hasattr(self, 'interactive_session') and self.interactive_session:
                    current_time = time.time()
                    if hasattr(self, '_last_progress_time'):
                        elapsed = current_time - self._last_progress_time
                        self.interactive_session.total_processing_time += elapsed
                    self._last_progress_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")
        else:
            log_msg = f"{self.agent_name}: {message}"
            if percentage is not None:
                log_msg += f" [{percentage:.0f}%]"
            self.logger.info(log_msg)
    
    def request_checkpoint(self, checkpoint_type: str,
                          accumulated_results: Optional[Dict[str, Any]] = None):
        """
        Request checkpoint creation
        
        Args:
            checkpoint_type: Type of checkpoint
            accumulated_results: Results to include in checkpoint
        """
        if self.checkpoint_callback and self.should_checkpoint(checkpoint_type):
            self.checkpoint_callback(
                checkpoint_type=checkpoint_type,
                current_agent=self.__class__.__name__,
                accumulated_results=accumulated_results,
                agent_states=self.get_state()
            )
    
    def get_state(self) -> Optional[bytes]:
        """
        Get serialized agent state for checkpoint
        
        Returns:
            Serialized state or None
        """
        # Subclasses can override to provide state serialization
        return None
    
    def set_state(self, state: bytes):
        """
        Restore agent state from checkpoint
        
        Args:
            state: Serialized state
        """
        # Subclasses can override to restore state
        pass
    
    def record_execution(self, action: str, result: Any,
                        details: Optional[Dict[str, Any]] = None):
        """
        Record execution in session history
        
        Args:
            action: Action performed
            result: Result of action
            details: Optional additional details
        """
        if self.interactive_session:
            self.interactive_session.execution_history.append({
                'timestamp': datetime.now().isoformat(),
                'agent': self.__class__.__name__,
                'action': action,
                'result': result,
                'details': details or {}
            })
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute agent logic
        
        Must be implemented by subclasses
        """
        pass
    
    def explain_current_state(self) -> str:
        """
        Explain what agent is currently doing
        
        Returns:
            Human-readable explanation
        """
        return f"{self.__class__.__name__} is processing..."
    
    def suggest_user_actions(self) -> List[str]:
        """
        Suggest actions user could take
        
        Returns:
            List of suggested actions
        """
        return [
            "Continue with current approach",
            "Modify the request",
            "Stop and review results"
        ]
    
    def check_user_input_needed(self, context: Dict[str, Any]) -> bool:
        """
        Determine if user input would be helpful
        
        Args:
            context: Current execution context
            
        Returns:
            True if user input recommended
        """
        # Subclasses can override for smart interruption
        return False
    
    def __repr__(self) -> str:
        interactive = "interactive" if self.is_interactive() else "autonomous"
        return f"<{self.__class__.__name__} mode={interactive}>"


class InteractiveAgentMixin:
    """
    Mixin to add interactive capabilities to existing agents
    
    Usage:
        class MyAgent(InteractiveAgentMixin, ExistingAgent):
            pass
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactive_session = None
        self.checkpoint_callback = None
        self.progress_callback = None
        
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(self.__class__.__name__)
    
    def attach_session(self, session: InteractiveSession,
                      checkpoint_callback=None,
                      progress_callback=None):
        """Attach to interactive session"""
        self.interactive_session = session
        self.checkpoint_callback = checkpoint_callback
        self.progress_callback = progress_callback
        
        self.logger.info(f"Attached to session {session.session_id}")
    
    def is_interactive(self) -> bool:
        """Check if in interactive mode"""
        return self.interactive_session is not None
    
    def report_progress(self, message: str, percentage: Optional[float] = None,
                       details: Optional[Dict[str, Any]] = None):
        """Report progress"""
        if self.progress_callback:
            self.progress_callback(
                agent=self.__class__.__name__,
                message=message,
                percentage=percentage,
                details=details
            )
    
    def request_checkpoint(self, checkpoint_type: str,
                          accumulated_results: Optional[Dict[str, Any]] = None):
        """Request checkpoint"""
        if self.checkpoint_callback:
            self.checkpoint_callback(
                checkpoint_type=checkpoint_type,
                current_agent=self.__class__.__name__,
                accumulated_results=accumulated_results
            )