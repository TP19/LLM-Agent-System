#!/usr/bin/env python3
"""
Interactive Executor Agent

Wraps ExecutorAgent with interactive capabilities.
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

from agents.executor_agent import ModularExecutorAgent
from interactive.base_interactive_agent import InteractiveAgentMixin
from interactive.modes import InteractiveSession


class InteractiveExecutorAgent(InteractiveAgentMixin, ModularExecutorAgent):
    """
    Interactive-capable Executor Agent
    
    Extends ModularExecutorAgent with:
    - Checkpoint support after each cycle
    - Real-time progress updates
    - User guidance integration
    - Command validation with user
    """
    
    def __init__(self, model_manager):
        """
        Initialize interactive executor agent
        
        Args:
            model_manager: Model manager instance
        """
        super().__init__(model_manager)
        self.logger = logging.getLogger(__name__)
        
        # Execution state
        self.current_cycle = 0
        self.max_cycles = 5
    
    def execute_task_interactive(self, user_request: str,
                                security_suggestion,
                                request_id: Optional[str] = None,
                                max_cycles: int = 5) -> Dict[str, Any]:
        """
        Execute task with interactive checkpoints - WITH TIME TRACKING
        
        Args:
            user_request: Original request
            security_suggestion: Security suggestions (dict or object)
            request_id: Request ID
            max_cycles: Maximum execution cycles
            
        Returns:
            Execution result dict
        """
        import time
        
        overall_start_time = time.time()
        
        self.max_cycles = max_cycles
        all_results = []
        
        # Report start
        self.report_progress("Starting task execution...", 0)
        
        for cycle in range(1, max_cycles + 1):
            cycle_start_time = time.time()
            
            self.current_cycle = cycle
            
            # Progress update
            progress = (cycle / max_cycles) * 100
            self.report_progress(
                f"Executing cycle {cycle}/{max_cycles}",
                progress,
                details={'cycle': cycle, 'max_cycles': max_cycles}
            )
            
            # Execute one cycle
            cycle_result = self._execute_single_cycle(
                user_request,
                security_suggestion,
                request_id,
                cycle
            )
            
            all_results.append(cycle_result)
            
            # Track cycle time
            cycle_time = time.time() - cycle_start_time
            if self.is_interactive():
                self.interactive_session.total_processing_time += cycle_time
            
            # Record in session
            if self.is_interactive():
                self.interactive_session.execution_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'agent': 'ExecutorAgent',
                    'action': 'execute_cycle',
                    'cycle': cycle,
                    'result': {
                        'commands_executed': cycle_result.get('commands_executed', 0),
                        'success': cycle_result.get('success', False),
                        'is_complete': cycle_result.get('is_complete', False)
                    }
                })
                
                # Update accumulated results
                self.interactive_session.accumulated_results['execution'] = {
                    'cycles_completed': cycle,
                    'total_commands': sum(r.get('commands_executed', 0) for r in all_results),
                    'all_cycles': all_results
                }
            
            # Checkpoint after each cycle
            self.request_checkpoint(
                checkpoint_type='execution',
                accumulated_results=self.interactive_session.accumulated_results if self.is_interactive() else None
            )
            
            # Check if complete
            if cycle_result.get('is_complete', False):
                self.report_progress("Task completed successfully", 100)
                break
        
        # Calculate total processing time
        total_processing_time = time.time() - overall_start_time
        
        # Compile final result
        final_result = {
            'success': True,
            'cycles_completed': cycle,
            'total_commands': sum(r.get('commands_executed', 0) for r in all_results),
            'all_cycles': all_results,
            'is_complete': all_results[-1].get('is_complete', False) if all_results else False,
            'processing_time': total_processing_time
        }
        
        # Final update to session
        if self.is_interactive():
            self.interactive_session.accumulated_results['execution'] = final_result
        
        return final_result
    
    def _execute_single_cycle(self, user_request: str,
                             security_suggestion,
                             request_id: Optional[str],
                             cycle: int) -> Dict[str, Any]:
        """
        Execute a single cycle
        
        Args:
            user_request: Original request
            security_suggestion: Security suggestions
            request_id: Request ID
            cycle: Current cycle number
            
        Returns:
            Cycle result dict
        """
        self.logger.info(f"[{request_id}] 🔄 Executing cycle {cycle}")
        
        # Use original execute_task method
        # This calls the real executor logic
        result = self.execute_task(
            user_request=user_request,
            security_suggestion=security_suggestion,
            request_id=request_id
        )
        
        # Add cycle info
        result['cycle'] = cycle
        
        return result
    
    def explain_current_state(self) -> str:
        """Explain what executor is doing"""
        return f"Executing task (cycle {self.current_cycle}/{self.max_cycles})..."
    
    def suggest_user_actions(self) -> List[str]:
        """Suggest actions for user"""
        actions = [
            "Continue execution",
            "Stop here (task may be complete)",
            "Modify approach for next cycle"
        ]
        
        if self.current_cycle > 1:
            actions.append("Review previous cycle results")
        
        return actions
    
    def check_user_input_needed(self, context: Dict[str, Any]) -> bool:
        """
        Check if user input would be helpful
        
        Args:
            context: Execution context
            
        Returns:
            True if user input recommended
        """
        # Suggest user input if:
        # - Multiple failed commands
        # - Unclear if task is complete
        # - Approaching max cycles
        
        failed_count = context.get('failed_commands', 0)
        is_complete = context.get('is_complete', False)
        
        if failed_count > 2:
            return True
        
        if self.current_cycle >= self.max_cycles - 1 and not is_complete:
            return True
        
        return False


def create_interactive_executor_agent(model_manager,
                                     session: Optional[InteractiveSession] = None):
    """
    Factory function for interactive executor agent
    
    Args:
        model_manager: Model manager instance
        session: Optional interactive session
        
    Returns:
        Configured InteractiveExecutorAgent
    """
    agent = InteractiveExecutorAgent(model_manager)
    
    if session:
        agent.attach_session(session)
    
    return agent