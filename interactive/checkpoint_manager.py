#!/usr/bin/env python3
"""
Checkpoint Manager

Handles checkpoint creation, validation, and rollback operations.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from interactive.modes import (
    Checkpoint,
    InteractiveSession,
    CheckpointFrequency,
    CheckpointRollbackException,
    UserAction
)
from interactive.state_store import StateStore
from interactive.terminal_ui import TerminalUI


class CheckpointManager:
    """
    Manages checkpoint lifecycle
    
    Responsibilities:
    - Create checkpoints at appropriate times
    - Save checkpoint state
    - Handle checkpoint rollback
    - Determine when checkpoints are needed
    """
    
    def __init__(self, session: InteractiveSession, state_store: StateStore, 
                    ui: TerminalUI):
            """
            Initialize checkpoint manager
            
            Args:
                session: Current interactive session
                state_store: Database for persistence
                ui: Terminal UI for user interaction
            """
            self.session = session
            self.state_store = state_store
            self.ui = ui
            self.logger = logging.getLogger(__name__)

            # FIX: Handle both enum and string values properly
            if hasattr(session.checkpoint_frequency, 'value'):
                # It's an enum
                self.checkpoint_frequency = session.checkpoint_frequency.value
            else:
                # It's already a string
                self.checkpoint_frequency = str(session.checkpoint_frequency).lower()
            
            # Validate frequency
            valid_frequencies = ['standard', 'minimal', 'detailed']
            if self.checkpoint_frequency not in valid_frequencies:
                self.logger.warning(f"Invalid frequency '{self.checkpoint_frequency}', using 'standard'")
                self.checkpoint_frequency = 'standard'
            
            self.logger.info(f"✅ Checkpoint manager initialized (frequency: {self.checkpoint_frequency})")

    def should_create_checkpoint(self, stage: str, action_type: str = None) -> bool:
        """
        Determine if checkpoint should be created based on frequency setting
        
        Args:
            stage: Current workflow stage
            action_type: Type of action (optional)
            
        Returns:
            True if checkpoint should be created
        """
        frequency = getattr(self, 'checkpoint_frequency', 'standard')
        
        self.logger.debug(f"Checking checkpoint for stage={stage}, frequency={frequency}")
        
        if frequency == 'none':
            # NEW: Never create checkpoints except at final completion
            should_create = stage == 'final_completion'
            self.logger.debug(f"None mode: {should_create}")
            return should_create
        
        elif frequency == 'minimal':
            # Only create at critical points
            critical_stages = ['triage', 'execution_complete', 'completion_review', 'error']
            should_create = stage in critical_stages
            self.logger.debug(f"Minimal mode: {should_create}")
            return should_create
        
        elif frequency == 'detailed':
            # Create after every action
            self.logger.debug("Detailed mode: always create checkpoint")
            return True
        
        else:  # standard (default)
            # After each major stage
            major_stages = ['triage', 'security', 'execution', 'completion_review', 'knowledge_query']
            should_create = stage in major_stages
            self.logger.debug(f"Standard mode: {should_create}")
            return should_create
    
    def create_checkpoint(self,
                        checkpoint_type: str,
                        current_agent: str,
                        current_stage: str,
                        accumulated_results: Optional[Dict] = None,
                        execution_history: Optional[List] = None,
                        agent_states: Optional[bytes] = None) -> Checkpoint:
        """
        Create new checkpoint - USES SESSION TIME
        """
        # Generate IDs
        checkpoint_id = f"cp_{str(uuid.uuid4())[:8]}"
        checkpoint_number = len(self.session.checkpoints) + 1
        
        # Determine parent checkpoint
        parent_id = None
        if self.session.checkpoints:
            parent_id = self.session.checkpoints[-1].checkpoint_id
        
        # Use session data if not provided
        if accumulated_results is None:
            accumulated_results = self.session.accumulated_results.copy()
        if execution_history is None:
            execution_history = self.session.execution_history.copy()
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=self.session.session_id,
            checkpoint_number=checkpoint_number,
            timestamp=datetime.now(),
            checkpoint_type=checkpoint_type,
            current_agent=current_agent,
            current_stage=current_stage,
            cycle_number=self.session.current_cycle,
            accumulated_results=accumulated_results,
            execution_history=execution_history,
            agent_states=agent_states,
            can_rollback=(agent_states is not None),
            parent_checkpoint_id=parent_id,
            user_actions=[],
            user_guidance=None,
            processing_time_so_far=self.session.total_processing_time,  # FIXED: Use session time
            tokens_used=self.session.total_tokens_used
        )
        
        # Save to database
        self.state_store.save_checkpoint(checkpoint)
        
        # Add to session
        self.session.add_checkpoint(checkpoint)
        
        self.logger.info(
            f"Created checkpoint #{checkpoint_number}: {checkpoint_type} "
            f"at stage {current_stage} (time: {self.session.total_processing_time:.2f}s)"
        )
        
        return checkpoint
    
    def display_checkpoint(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        """
        Display checkpoint and get user action
        
        Args:
            checkpoint: Checkpoint to display
            
        Returns:
            Dict with user action and optional guidance
        """
        self.ui.display_checkpoint(checkpoint)
        action = self.ui.prompt_user_action()
        
        # Record user action
        checkpoint.user_actions.append(action.value)
        
        # Update checkpoint in database
        self.state_store.save_checkpoint(checkpoint)
        
        result = {
            'action': action,
            'guidance': None
        }
        
        # Handle different actions
        if action == UserAction.STOP:
            self.logger.info("User stopped execution at checkpoint")
            result['stop_reason'] = "User requested stop"
            
        elif action == UserAction.MODIFY:
            guidance = self.ui.get_user_guidance()
            checkpoint.user_guidance = guidance
            self.state_store.save_checkpoint(checkpoint)
            result['guidance'] = guidance
            self.logger.info(f"User provided guidance: {guidance}")
            
        elif action == UserAction.ROLLBACK:
            # Show available checkpoints
            checkpoints = self.state_store.load_checkpoints(self.session.session_id)
            target_cp = self.ui.select_checkpoint_for_rollback(checkpoints)
            
            if target_cp:
                self.logger.info(f"User requested rollback to checkpoint {target_cp.checkpoint_id}")
                result['rollback_to'] = target_cp.checkpoint_id
        
        return result
    
    def rollback_to(self, checkpoint_id: str) -> Checkpoint:
        """
        Rollback session to earlier checkpoint
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            Target checkpoint
            
        Raises:
            CheckpointRollbackException: Signals rollback to orchestrator
        """
        # Load target checkpoint
        target_cp = self.state_store.load_checkpoint(checkpoint_id)
        
        if not target_cp:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        if not target_cp.can_rollback:
            raise ValueError("Cannot rollback to this checkpoint (no state saved)")
        
        # Invalidate later checkpoints
        self._invalidate_later_checkpoints(target_cp.checkpoint_number)
        
        # Restore session state
        self.session.accumulated_results = target_cp.accumulated_results.copy()
        self.session.execution_history = target_cp.execution_history.copy()
        self.session.current_cycle = target_cp.cycle_number
        self.session.current_stage = target_cp.current_stage
        self.session.current_agent = target_cp.current_agent
        
        # Save updated session
        self.state_store.save_session(self.session)
        
        # Display rollback success
        self.ui.display_rollback_success(target_cp)
        
        self.logger.info(f"Rolled back to checkpoint #{target_cp.checkpoint_number}")
        
        # Raise exception to signal rollback to orchestrator
        raise CheckpointRollbackException(checkpoint_id)
    
    def _invalidate_later_checkpoints(self, checkpoint_number: int):
        """
        Invalidate checkpoints after the target
        
        Args:
            checkpoint_number: Number of checkpoint to keep
        """
        # Remove from session
        self.session.checkpoints = [
            cp for cp in self.session.checkpoints 
            if cp.checkpoint_number <= checkpoint_number
        ]
        
        # Note: We don't delete from database, just mark as invalidated
        # This allows for debugging and audit trail
        self.logger.debug(
            f"Invalidated checkpoints after #{checkpoint_number}"
        )
    
    def get_checkpoint_summary(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        """
        Get summary of checkpoint for display
        
        Args:
            checkpoint: Checkpoint to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            'number': checkpoint.checkpoint_number,
            'type': checkpoint.checkpoint_type,
            'agent': checkpoint.current_agent,
            'stage': checkpoint.current_stage,
            'cycle': checkpoint.cycle_number,
            'timestamp': checkpoint.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'can_rollback': checkpoint.can_rollback,
            'results_count': len(checkpoint.accumulated_results),
            'execution_count': len(checkpoint.execution_history)
        }