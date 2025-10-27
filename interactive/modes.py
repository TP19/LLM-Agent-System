#!/usr/bin/env python3
"""
Interactive Mode Data Structures

Core enums and dataclasses for interactive mode functionality.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


class CheckpointFrequency(Enum):
    """How often to create checkpoints"""
    DISABLED = "disabled"        # No checkpoints (autonomous mode)
    MINIMAL = "minimal"          # Only at completion review
    STANDARD = "standard"        # After each agent (default)
    DETAILED = "detailed"        # After each significant operation
    SMART = "smart"              # AI decides when checkpoint needed
    NONE = "none"


class UserAction(Enum):
    """Actions user can take at a checkpoint"""
    CONTINUE = "continue"        # Proceed with current plan
    STOP = "stop"                # End task here
    MODIFY = "modify"            # Change approach
    QUERY = "query"              # Ask questions
    ROLLBACK = "rollback"        # Go to earlier checkpoint
    SETTINGS = "settings"        # Adjust checkpoint frequency


class ReviewAction(Enum):
    """Actions user can take at review session"""
    APPROVE = "approve"          # Task complete, finalize
    CONTINUE = "continue"        # Do more work
    QUERY = "query"              # Ask questions about results
    MODIFY = "modify"            # Change approach and restart
    ROLLBACK = "rollback"        # Go to earlier checkpoint
    SHOW_DETAILS = "details"     # View full execution log


class InteractionMode(Enum):
    """Interaction mode for the session"""
    AUTONOMOUS = "autonomous"    # No interaction (file-based)
    INTERACTIVE = "interactive"  # Full interactive with checkpoints
    REVIEW_ONLY = "review_only"  # Only review at completion


@dataclass
class Checkpoint:
    """
    Represents a checkpoint in the interactive session
    
    This captures the complete state at a point in time,
    allowing rollback and resume functionality.
    """
    # Identification
    checkpoint_id: str
    session_id: str
    checkpoint_number: int
    timestamp: datetime
    checkpoint_type: str            # 'triage', 'security', 'execution', etc.
    
    # Current State
    current_agent: str
    current_stage: str
    cycle_number: int = 0
    
    # Results Accumulated So Far
    accumulated_results: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict] = field(default_factory=list)
    
    # Rollback Support
    agent_states: Optional[bytes] = None  # Serialized agent states
    can_rollback: bool = True
    parent_checkpoint_id: Optional[str] = None
    
    # User Context
    user_actions: List[str] = field(default_factory=list)
    user_guidance: Optional[str] = None
    
    # Performance Tracking
    processing_time_so_far: float = 0.0
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'checkpoint_id': self.checkpoint_id,
            'session_id': self.session_id,
            'checkpoint_number': self.checkpoint_number,
            'timestamp': self.timestamp.isoformat(),
            'checkpoint_type': self.checkpoint_type,
            'current_agent': self.current_agent,
            'current_stage': self.current_stage,
            'cycle_number': self.cycle_number,
            'accumulated_results': self.accumulated_results,
            'execution_history': self.execution_history,
            'can_rollback': self.can_rollback,
            'parent_checkpoint_id': self.parent_checkpoint_id,
            'user_actions': self.user_actions,
            'user_guidance': self.user_guidance,
            'processing_time_so_far': self.processing_time_so_far,
            'tokens_used': self.tokens_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        # agent_states is stored separately in DB, not in dict
        data['agent_states'] = None
        return cls(**data)


@dataclass
class InteractiveSession:
    """
    Represents an interactive session
    
    Manages the lifecycle of a user's interactive request,
    including checkpoints, state, and user interactions.
    """
    session_id: str
    created_at: datetime
    mode: InteractionMode
    checkpoint_frequency: CheckpointFrequency
    
    # Request
    user_request: str
    original_request: str  # Keep original for reference
    
    # Current State
    current_stage: str = "initialization"
    current_agent: Optional[str] = None
    current_cycle: int = 0
    is_complete: bool = False
    
    # Session Data
    accumulated_results: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict] = field(default_factory=list)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    user_interactions: List[Dict] = field(default_factory=list)
    
    # Performance
    total_processing_time: float = 0.0
    total_tokens_used: int = 0
    
    # Status
    completion_reason: Optional[str] = None
    error_message: Optional[str] = None
    
    def add_checkpoint(self, checkpoint: Checkpoint):
        """Add checkpoint to session"""
        self.checkpoints.append(checkpoint)
    
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get most recent checkpoint"""
        return self.checkpoints[-1] if self.checkpoints else None
    
    def add_user_interaction(self, interaction_type: str, user_input: str, 
                            agent_response: str = ""):
        """Record user interaction"""
        self.user_interactions.append({
            'timestamp': datetime.now().isoformat(),
            'type': interaction_type,
            'user_input': user_input,
            'agent_response': agent_response
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'mode': self.mode.value,
            'checkpoint_frequency': self.checkpoint_frequency.value,
            'user_request': self.user_request,
            'original_request': self.original_request,
            'current_stage': self.current_stage,
            'current_agent': self.current_agent,
            'current_cycle': self.current_cycle,
            'is_complete': self.is_complete,
            'accumulated_results': self.accumulated_results,
            'execution_history': self.execution_history,
            'total_processing_time': self.total_processing_time,
            'total_tokens_used': self.total_tokens_used,
            'completion_reason': self.completion_reason,
            'error_message': self.error_message
        }


class UserStoppedException(Exception):
    """Raised when user stops execution at a checkpoint"""
    pass


class CheckpointRollbackException(Exception):
    """Raised when rollback to checkpoint is requested"""
    def __init__(self, checkpoint_id: str):
        self.checkpoint_id = checkpoint_id
        super().__init__(f"Rollback to checkpoint: {checkpoint_id}")