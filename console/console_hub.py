"""
Console Hub - Unified LLM-Agent-System Interface

The Console provides a single entry point for all LLM-Agent-System interactions,
combining Oracle chat, agent invocation, and session management.
"""

import logging
import os
import queue
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from .console_ui import ConsoleUI
from .command_router import CommandRouter
from .session_controller import SessionController, ConsoleSession
from .tmux_manager import TmuxManager, BackgroundTask
from .task_duration_estimator import TaskDurationEstimator, ExecutionMode
from .agent_context import AgentContextManager
from .task_registry import TaskRegistry, TrackedTask, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)

# Try to import ProjectManager
try:
    from projects.project_manager import ProjectManager, Project
    PROJECT_MANAGER_AVAILABLE = True
except ImportError:
    PROJECT_MANAGER_AVAILABLE = False
    ProjectManager = None
    Project = None

# Try to import ModularContextProvider for doc-manager integration
try:
    from core.modular_context import ModularContextProvider
    MODULAR_CONTEXT_AVAILABLE = True
except ImportError:
    MODULAR_CONTEXT_AVAILABLE = False
    ModularContextProvider = None

# Resource monitoring for task acceptance
try:
    from orchestration.resource_monitor import ResourceMonitor
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError:
    RESOURCE_MONITOR_AVAILABLE = False
    ResourceMonitor = None


@dataclass
class ConsoleConfig:
    """Console configuration"""
    default_ephemeral: bool = False
    auto_save_interval: int = 30  # seconds
    max_history: int = 100
    enable_tmux: bool = True
    project_base_path: str = "~/.llm_engine/projects"
    fast_mode: bool = True  # Auto-triage without asking (standard mode)


class Console:
    """
    Unified Console Hub

    Features:
    - Oracle as primary chat interface
    - Project and session management
    - Agent invocation
    - Background task management (tmux)
    - Chunk viewer integration

    Usage:
        from console import Console

        console = Console(model_manager)
        console.run()
    """

    def __init__(self, model_manager=None, config: Optional[ConsoleConfig] = None):
        """
        Initialize Console

        Args:
            model_manager: LLM model manager instance
            config: Console configuration
        """
        self.config = config or ConsoleConfig()
        self.model_manager = model_manager

        # Initialize components
        self.ui = ConsoleUI()
        self.command_router = CommandRouter(self)
        self.session_controller = SessionController()

        # Initialize ProjectManager if available
        self._project_manager: Optional[ProjectManager] = None
        if PROJECT_MANAGER_AVAILABLE:
            try:
                self._project_manager = ProjectManager(self.config.project_base_path)
                logger.info("ProjectManager initialized")
            except Exception as e:
                logger.warning(f"Could not initialize ProjectManager: {e}")

        # State
        self.active_project: Optional[str] = None
        self._active_project_id: Optional[str] = None
        self.active_session: Optional[ConsoleSession] = None
        self.active_session_name: Optional[str] = None
        self.ephemeral_mode: bool = self.config.default_ephemeral
        self.running: bool = False

        # Agent instances (lazy loaded)
        self._oracle = None
        self._agents: Dict[str, Any] = {}

        # Background tasks tracking
        self._background_tasks: Dict[str, BackgroundTask] = {}

        # Initialize tmux manager and task estimator
        self.tmux_manager = TmuxManager()
        self.task_estimator = TaskDurationEstimator(threshold=30)

        # Task registry for background task tracking
        self.task_registry = TaskRegistry()

        # Agent context manager (initialized when session is created)
        self._agent_context: Optional[AgentContextManager] = None

        # Track active agent panes
        self._active_agent_panes: Dict[str, str] = {}  # agent_name -> pane_id

        # Conversation history for Oracle
        self._conversation_history: List[Dict[str, str]] = []

        # Modular context provider for doc-manager integration (lazy loaded)
        self._modular_context: Optional[ModularContextProvider] = None

        # Async processing state for non-blocking Oracle
        self._processing: bool = False
        self._processing_thread: Optional[threading.Thread] = None
        self._abort_requested: bool = False
        self._response_queue: queue.Queue = queue.Queue()
        self._current_request: Optional[str] = None

        # Resource monitor for task acceptance
        self._resource_monitor = None
        if RESOURCE_MONITOR_AVAILABLE:
            try:
                self._resource_monitor = ResourceMonitor()
                logger.info("ResourceMonitor initialized")
            except Exception as e:
                logger.warning(f"Could not initialize ResourceMonitor: {e}")

        # Resource thresholds for task acceptance
        self._resource_thresholds = {
            'cpu_busy': 90.0,        # CPU % above which system is "busy"
            'memory_busy': 90.0,     # Memory % above which system is "busy"
            'gpu_memory_min': 2000,  # Minimum free GPU memory (MB) for LLM tasks
        }

        logger.info("Console initialized")

    @property
    def oracle(self):
        """Lazy load Oracle agent"""
        if self._oracle is None and self.model_manager:
            try:
                from agents.oracle_agent import OracleAgent
                self._oracle = OracleAgent(self.model_manager)
                logger.info("Oracle agent loaded")
            except ImportError as e:
                logger.warning(f"Could not load Oracle agent: {e}")
        return self._oracle

    @property
    def modular_context(self) -> Optional[ModularContextProvider]:
        """Lazy load ModularContextProvider for doc-manager integration"""
        if self._modular_context is None and MODULAR_CONTEXT_AVAILABLE:
            project_name = self.active_project or "LLM-Agent-System"
            try:
                self._modular_context = ModularContextProvider(project_name)
                if self._modular_context.is_available():
                    logger.info(f"ModularContextProvider initialized for {project_name}")
                else:
                    logger.info("ModularContextProvider not available (doc-manager not configured)")
                    self._modular_context = None
            except Exception as e:
                logger.warning(f"Could not initialize ModularContextProvider: {e}")
        return self._modular_context

    def check_system_resources(self) -> Dict[str, Any]:
        """
        Check system resources before accepting a task.

        Returns:
            Dict with:
            - 'ready': bool - True if system can accept the task
            - 'warnings': List[str] - Any resource warnings
            - 'resources': Dict - Current resource snapshot
        """
        result = {
            'ready': True,
            'warnings': [],
            'resources': {}
        }

        if not self._resource_monitor:
            return result  # No monitoring available, proceed

        try:
            resources = self._resource_monitor.get_system_resources()
            result['resources'] = resources.to_dict()

            # Check CPU
            if resources.cpu_percent > self._resource_thresholds['cpu_busy']:
                result['warnings'].append(f"CPU busy: {resources.cpu_percent:.1f}%")

            # Check Memory
            if resources.memory_percent > self._resource_thresholds['memory_busy']:
                result['warnings'].append(f"Memory pressure: {resources.memory_percent:.1f}%")
                if resources.memory_percent > 95:
                    result['ready'] = False  # Critical memory pressure

            # Check GPU (if using LLM)
            if resources.gpu_available:
                if resources.gpu_memory_free_mb and resources.gpu_memory_free_mb < self._resource_thresholds['gpu_memory_min']:
                    result['warnings'].append(f"Low GPU memory: {resources.gpu_memory_free_mb:.0f}MB free")
                if resources.gpu_utilization and resources.gpu_utilization > 95:
                    result['warnings'].append(f"GPU busy: {resources.gpu_utilization:.1f}%")

        except Exception as e:
            logger.warning(f"Resource check failed: {e}")

        return result

    def get_resource_summary(self) -> str:
        """Get human-readable resource summary for /status command"""
        if not self._resource_monitor:
            return "Resource monitoring not available"
        return self._resource_monitor.get_resource_summary()

    def _get_module_context_for_request(self, message: str) -> Optional[str]:
        """
        Get relevant module context from doc-manager for a request.

        Uses semantic search to find modules relevant to the user's request,
        then formats the context for inclusion in the Oracle prompt.

        Args:
            message: User's request message

        Returns:
            Formatted context string, or None if not available
        """
        if not self.modular_context or not self.modular_context.is_available():
            return None

        try:
            # Use semantic search to find relevant modules
            task_context = self.modular_context.get_task_context(message, top_k=3)

            if not task_context.get('suggested_modules'):
                return None

            # Format context for LLM consumption
            lines = ["[Module Context from doc-manager]"]

            for module_name in task_context['suggested_modules'][:2]:
                module_ctx = self.modular_context.get_module_context(module_name)
                if module_ctx:
                    formatted = self.modular_context.format_context_for_llm(
                        module_ctx, verbosity="brief"
                    )
                    lines.append(formatted)
                    lines.append("")

            if len(lines) > 1:
                context_str = "\n".join(lines)
                logger.debug(f"Retrieved module context: {context_str[:200]}...")
                return context_str

        except Exception as e:
            logger.warning(f"Error getting module context: {e}")

        return None

    def run(self):
        """Main Console loop"""
        self.running = True

        # Display welcome
        self.ui.display_welcome()

        # Create initial session if no session active
        if not self.active_session:
            if self.ephemeral_mode:
                self.active_session = self.session_controller.create_session(
                    name="default",
                    ephemeral=True
                )
            else:
                # Reuse recent empty session to avoid session bloat
                self.active_session = self.session_controller.get_or_create_default_session()
            self.active_session_name = self.active_session.name

        # Update UI context
        self._update_ui_context()

        # Start background task polling
        self.task_registry.start_polling()

        # Main loop
        try:
            while self.running:
                try:
                    # Check for notifications before prompt
                    self._check_notifications()

                    # Check for pending Oracle response
                    self._check_oracle_response()

                    # Modify prompt to show processing state
                    prompt_name = "Console"
                    if self._processing:
                        prompt_name = "Console [processing...]"

                    user_input = self.ui.prompt(prompt_name)

                    if not user_input:
                        continue

                    # Check for command
                    if user_input.startswith("/"):
                        self.command_router.handle(user_input)
                    elif self._processing:
                        # Block new chat while processing
                        self.ui.print_warning("Oracle is still processing. Use /abort to cancel or wait.")
                    else:
                        # Chat with Oracle
                        self._handle_chat(user_input)

                except KeyboardInterrupt:
                    if self._processing:
                        self.ui.print_system("Aborting current request...")
                        self._abort_oracle_request()
                    else:
                        self.ui.print_system("Use /exit to quit")
                    continue

        except SystemExit:
            pass
        finally:
            # Stop polling
            self.task_registry.stop_polling()
            # Abort any pending Oracle request
            if self._processing:
                self._abort_oracle_request()
            self.ui.goodbye()
            self.running = False

    def _handle_chat(self, message: str):
        """
        Handle chat message (delegate to Oracle)

        Now runs Oracle in background thread for non-blocking console.

        Supports two modes:
        - Fast mode (default): Auto-triage actionable tasks without asking
        - Interactive mode: Ask user for triage approval

        Args:
            message: User message
        """
        # Update session name if this is first message
        if self.active_session and self.active_session.message_count == 0:
            new_name = self.session_controller.generate_name_from_message(
                message, self.active_project
            )
            self.active_session.name = new_name
            self.active_session_name = new_name
            self._update_ui_context()

        # Add user message to session
        if self.active_session:
            self.active_session.add_message("user", message)
            self._conversation_history.append({"role": "user", "content": message})

        # Get Oracle response
        if self.oracle:
            # Check system resources before accepting task
            resource_check = self.check_system_resources()
            if resource_check['warnings']:
                for warning in resource_check['warnings']:
                    self.ui.print_warning(f"⚠ {warning}")

            if not resource_check['ready']:
                self.ui.print_error("System resources critical - task may be slow or fail")
                self.ui.print_info("Consider waiting or using /bg for background processing")
                # Still proceed but warn user

            # Spawn Oracle call in background thread
            self._processing = True
            self._current_request = message
            self._abort_requested = False

            # Get module context from doc-manager (if available)
            module_context = self._get_module_context_for_request(message)
            history_with_context = self._conversation_history.copy()
            if module_context:
                # Inject module context as a system message before user's message
                history_with_context.insert(-1, {
                    "role": "system",
                    "content": module_context
                })
                logger.info("Injected module context into Oracle request")

            # Create and start background thread
            self._processing_thread = threading.Thread(
                target=self._oracle_worker,
                args=(message, history_with_context),
                daemon=True
            )
            self._processing_thread.start()

            self.ui.print_info("Processing... (use /status for info, /abort to cancel)")
        else:
            self.ui.print_warning("Oracle agent not available (no model manager)")
            self.ui.print_info("Use /agent <name> to chat with a specific agent")

    def _oracle_worker(self, message: str, history: List[Dict[str, str]]):
        """
        Background worker thread for Oracle calls.

        Now displays response immediately instead of waiting for main loop to poll queue.

        Args:
            message: User message
            history: Conversation history with context
        """
        try:
            # Check for abort before starting
            if self._abort_requested:
                self._processing = False
                self.ui.print_warning("\nRequest aborted")
                return

            # Call Oracle
            response = self.oracle.chat(
                message,
                history,
                auto_triage=self.config.fast_mode
            )

            # Check for abort after call
            if self._abort_requested:
                self._processing = False
                self.ui.print_warning("\nRequest aborted")
                return

            # Immediately display response (don't wait for main loop)
            self._display_oracle_response(response, message)

        except Exception as e:
            logger.error(f"Oracle worker error: {e}", exc_info=True)
            self._processing = False
            self.ui.print_error(f"\nOracle error: {e}")

    def _display_oracle_response(self, response: str, message: str):
        """
        Display Oracle response immediately from worker thread.

        This bypasses the queue polling to show results as soon as they're ready.

        Args:
            response: Oracle response text
            message: Original user message
        """
        import json

        # Check if this is a plan approval or triage response (JSON with special keys)
        # These need interactive prompts, so route to main thread via queue
        try:
            data = json.loads(response)
            if isinstance(data, dict) and (data.get('plan_approval') or data.get('triage_trigger')):
                # Route to main thread for interactive handling
                self._response_queue.put({
                    'response': response,
                    'message': message,
                })
                return
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Clear processing flag
        self._processing = False
        self._current_request = None

        # Add response to session
        if self.active_session:
            self.active_session.add_message("oracle", response)
            self._conversation_history.append({"role": "assistant", "content": response})
            self.session_controller.update_session(self.active_session)

        # Display response
        self.ui.print_oracle(response)

    def _check_oracle_response(self):
        """
        Check for pending Oracle response in the queue.

        Called in main loop to process async Oracle responses.
        """
        try:
            # Non-blocking check for response
            result = self._response_queue.get_nowait()

            # Response received - process it
            self._processing = False
            self._current_request = None

            if result.get('error'):
                self.ui.print_error(f"Oracle error: {result['error']}")
                return

            if result.get('aborted'):
                self.ui.print_warning("Request aborted")
                return

            response = result.get('response', '')
            message = result.get('message', '')

            # Handle triage or plan approval if needed
            from rich.console import Console as RichConsole
            console = RichConsole()

            # Check for triage trigger
            import json
            triage_data = None
            try:
                triage_data = json.loads(response)
                if not isinstance(triage_data, dict) or not triage_data.get('triage_trigger'):
                    triage_data = None
            except (json.JSONDecodeError, TypeError):
                pass

            if triage_data:
                # Handle triage approval interactively
                response = self._handle_triage_approval(triage_data, message, console)

            # Check for plan approval
            if not triage_data:
                response = self._handle_plan_approval(response, message, console)

            # Add response to session
            if self.active_session:
                self.active_session.add_message("oracle", response)
                self._conversation_history.append({"role": "assistant", "content": response})
                self.session_controller.update_session(self.active_session)

            # Display response
            self.ui.print_oracle(response)

        except queue.Empty:
            # No response yet, continue
            pass

    def _handle_triage_approval(self, triage_data: dict, message: str, console) -> str:
        """
        Handle triage approval flow interactively.

        Args:
            triage_data: Parsed triage trigger data
            message: Original user message
            console: Rich console instance

        Returns:
            Final response after triage handling
        """
        from rich.prompt import Confirm
        from rich.status import Status

        self.ui.print_info(f"\n{triage_data.get('prompt', 'Triage this task?')}")

        try:
            approved = Confirm.ask("Proceed with triage?", default=True)
        except KeyboardInterrupt:
            approved = False
            self.ui.print_warning("Cancelled")

        if approved:
            # Re-call Oracle with triage_approved=True
            with console.status("[bold cyan]Triaging and executing...[/bold cyan]", spinner="dots"):
                response = self.oracle.chat(
                    message,
                    self._conversation_history,
                    triage_approved=True
                )

            # Check if response is a plan approval request
            response = self._handle_plan_approval(response, message, console)
        else:
            # User declined - get conversational response
            with console.status("[bold cyan]Generating response...[/bold cyan]", spinner="dots"):
                response = self.oracle.chat(
                    f"Instead of executing, please explain how I would: {message}",
                    self._conversation_history
                )

        return response

    def _abort_oracle_request(self):
        """
        Abort the current Oracle request.

        Sets abort flag and clears processing state.
        """
        if not self._processing:
            return

        self._abort_requested = True
        self.ui.print_info("Abort requested, waiting for Oracle to stop...")

        # Wait briefly for thread to notice abort
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        # Clear state
        self._processing = False
        self._abort_requested = False
        self._current_request = None

        # Clear queue
        try:
            while True:
                self._response_queue.get_nowait()
        except queue.Empty:
            pass

        self.ui.print_info("Request aborted")

    def abort_current_request(self):
        """Public method to abort current request (called by /abort command)"""
        if self._processing:
            self._abort_oracle_request()
        else:
            self.ui.print_info("No request in progress")

    def _handle_plan_approval(self, response: str, original_message: str, console) -> str:
        """
        Handle plan approval flow for complex tasks.

        Phase D Enhanced Features:
        - Skip specific steps: "skip 2" or "skip step 2"
        - Remove agents: "remove security"
        - Reorder steps: "move 3 before 1"
        - General modification: free text

        Args:
            response: Oracle response (may be JSON plan approval trigger)
            original_message: Original user message
            console: Rich console for status

        Returns:
            Final response (execution result or original response)
        """
        import json
        import re
        from rich.prompt import Prompt
        from rich.panel import Panel

        # Try to parse as plan approval
        plan_data = None
        try:
            data = json.loads(response)
            if isinstance(data, dict) and data.get('plan_approval'):
                plan_data = data
        except (json.JSONDecodeError, TypeError):
            pass

        if not plan_data:
            return response

        # Display the plan
        plan_display = plan_data.get('plan_display', 'No plan details')
        triage_info = plan_data.get('triage', {})
        analysis_info = plan_data.get('analysis', {})

        # Show complexity info from TaskAnalyzer
        complexity = analysis_info.get('complexity', {})
        if complexity:
            complexity_level = complexity.get('complexity', 'unknown')
            risk_level = complexity.get('risk_level', 'unknown')
            self.ui.print_info(f"\n🔍 Complex task detected (Complexity: {complexity_level}, Risk: {risk_level})")
        else:
            self.ui.print_info(f"\n🔍 Complex task detected ({triage_info.get('difficulty', 'unknown')})")

        console.print(Panel(
            plan_display,
            title="[bold cyan]Execution Plan[/bold cyan]",
            border_style="cyan"
        ))

        # Show modification hints
        self.ui.print_info("Options: y (approve), n (cancel), or modify")
        self.ui.print_info("  Modify examples: 'skip 2', 'remove security', 'skip step 3', or describe changes")

        try:
            choice = Prompt.ask(
                "Your choice",
                default="y"
            ).strip().lower()
        except KeyboardInterrupt:
            choice = "n"
            self.ui.print_warning("Cancelled")

        if choice == "y":
            # Execute the approved plan
            with console.status("[bold green]Executing approved plan...[/bold green]", spinner="dots"):
                if hasattr(self.oracle, '_execute_approved_plan'):
                    response = self.oracle._execute_approved_plan(plan_data.get('plan_data', {}))
                else:
                    response = "❌ Plan execution not available"

        elif choice == "n":
            response = "Plan cancelled. Let me know if you'd like to try a different approach."

        else:
            # Handle modification - could be:
            # - "skip 2" or "skip step 2" -> Remove step 2
            # - "remove security" -> Remove all security steps
            # - "move 3 before 1" -> Reorder
            # - General text -> Re-generate with modification

            modified_plan = self._apply_plan_modification(
                choice, plan_data.get('plan_data', {}), console
            )

            if modified_plan:
                # Execute modified plan directly
                with console.status("[bold green]Executing modified plan...[/bold green]", spinner="dots"):
                    if hasattr(self.oracle, '_execute_approved_plan'):
                        response = self.oracle._execute_approved_plan(modified_plan)
                    else:
                        response = "❌ Plan execution not available"
            else:
                # Re-generate plan with modification text
                modified_request = f"{original_message}. Modification: {choice}"
                with console.status("[bold cyan]Regenerating plan...[/bold cyan]", spinner="dots"):
                    response = self.oracle.chat(
                        modified_request,
                        self._conversation_history,
                        triage_approved=True
                    )
                # Recursively handle if this is also a plan approval
                response = self._handle_plan_approval(response, modified_request, console)

        return response

    def _apply_plan_modification(self, modification: str, plan_data: dict, console) -> Optional[dict]:
        """
        Apply direct modifications to a plan without regenerating.

        Supports:
        - "skip N" or "skip step N" - Remove step N
        - "remove AGENT" - Remove all steps for AGENT
        - "only N,M,P" - Keep only steps N, M, P

        Args:
            modification: User's modification request
            plan_data: Current plan data
            console: Rich console for output

        Returns:
            Modified plan if direct modification was applied, None otherwise
        """
        import re
        from rich.panel import Panel

        modification = modification.lower().strip()
        subtasks = plan_data.get('subtasks', [])

        if not subtasks:
            return None

        # Pattern: "skip N" or "skip step N"
        skip_match = re.match(r'skip\s*(?:step\s*)?(\d+)', modification)
        if skip_match:
            step_num = int(skip_match.group(1))
            if 1 <= step_num <= len(subtasks):
                removed = subtasks.pop(step_num - 1)
                self.ui.print_info(f"✂️  Removed step {step_num}: {removed.get('description', 'N/A')}")
                # Renumber remaining tasks
                for i, task in enumerate(subtasks):
                    task['task_id'] = f'task_{i+1}'
                plan_data['subtasks'] = subtasks
                self._display_modified_plan(plan_data, console)
                return plan_data

        # Pattern: "remove AGENT"
        remove_match = re.match(r'remove\s+(\w+)', modification)
        if remove_match:
            agent_to_remove = remove_match.group(1)
            original_count = len(subtasks)
            subtasks = [t for t in subtasks if t.get('assigned_agent', '').lower() != agent_to_remove]
            removed_count = original_count - len(subtasks)
            if removed_count > 0:
                self.ui.print_info(f"✂️  Removed {removed_count} {agent_to_remove} step(s)")
                for i, task in enumerate(subtasks):
                    task['task_id'] = f'task_{i+1}'
                plan_data['subtasks'] = subtasks
                self._display_modified_plan(plan_data, console)
                return plan_data

        # Pattern: "only N,M,P" - keep only specified steps
        only_match = re.match(r'only\s+([\d,\s]+)', modification)
        if only_match:
            steps_str = only_match.group(1)
            keep_steps = [int(s.strip()) for s in steps_str.split(',') if s.strip().isdigit()]
            new_subtasks = []
            for step_num in keep_steps:
                if 1 <= step_num <= len(subtasks):
                    new_subtasks.append(subtasks[step_num - 1])
            if new_subtasks:
                self.ui.print_info(f"✂️  Keeping only steps: {', '.join(map(str, keep_steps))}")
                for i, task in enumerate(new_subtasks):
                    task['task_id'] = f'task_{i+1}'
                    task['dependencies'] = []  # Clear deps for simplicity
                plan_data['subtasks'] = new_subtasks
                self._display_modified_plan(plan_data, console)
                return plan_data

        # No direct modification pattern matched
        return None

    def _display_modified_plan(self, plan_data: dict, console):
        """Display the modified plan for confirmation"""
        from rich.panel import Panel

        subtasks = plan_data.get('subtasks', [])
        display_lines = [f"📋 Modified Plan: {plan_data.get('goal', 'Task')}"]
        display_lines.append(f"📝 {len(subtasks)} subtasks remaining:\n")

        for i, task in enumerate(subtasks, 1):
            display_lines.append(f"  {i}. {task.get('description', 'N/A')}")
            display_lines.append(f"     → Agent: {task.get('assigned_agent', 'unknown')}")

        console.print(Panel(
            "\n".join(display_lines),
            title="[bold yellow]Modified Plan[/bold yellow]",
            border_style="yellow"
        ))

    def _update_ui_context(self):
        """Update UI prompt context"""
        self.ui.set_context(
            project=self.active_project,
            session=self.active_session_name,
            ephemeral=self.ephemeral_mode
        )

    # =========================================================================
    # Project Management
    # =========================================================================

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects"""
        if not self._project_manager:
            return []

        try:
            projects = self._project_manager.list_projects()
            result = []
            for p in projects:
                # Count sessions for this project
                sessions = self.session_controller.list_sessions(
                    project_id=p.project_id,
                    include_ephemeral=False
                )
                result.append({
                    "name": p.name,
                    "project_id": p.project_id,
                    "template": p.template,
                    "session_count": len(sessions),
                    "last_accessed": p.last_accessed
                })
            return result
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return []

    def switch_project(self, name: str):
        """Switch to or create project"""
        if not self._project_manager:
            self.ui.print_error("ProjectManager not available")
            return

        try:
            # Try to find existing project by name
            projects = self._project_manager.list_projects()
            project = next((p for p in projects if p.name == name), None)

            if project:
                self._project_manager.switch_project(project.project_id)
                self.active_project = name
                self._active_project_id = project.project_id
                self.ui.print_success(f"Switched to project: {name}")

                # Create or switch to project's default session
                sessions = self.session_controller.list_sessions(
                    project_id=project.project_id
                )
                if sessions:
                    # Switch to most recent session
                    self.switch_session(sessions[0]['session_id'])
                else:
                    # Create new session for this project
                    self.create_session()
            else:
                # Create new project
                project_id = self._project_manager.create_project(
                    name=name,
                    description=f"Project {name}"
                )
                self._project_manager.switch_project(project_id)
                self.active_project = name
                self._active_project_id = project_id
                self.ui.print_success(f"Created and switched to project: {name}")

                # Create initial session
                self.create_session()

            self._update_ui_context()

        except Exception as e:
            logger.error(f"Error switching project: {e}")
            self.ui.print_error(f"Failed to switch project: {e}")

    def create_project(self, name: str, template: str = "default"):
        """Create new project"""
        if not self._project_manager:
            self.ui.print_error("ProjectManager not available")
            return

        try:
            project_id = self._project_manager.create_project(
                name=name,
                description=f"Project {name}",
                template=template
            )
            self.ui.print_success(f"Created project: {name}")

            if self.ui.confirm("Switch to this project?"):
                self._project_manager.switch_project(project_id)
                self.active_project = name
                self._active_project_id = project_id
                self.create_session()
                self._update_ui_context()

        except Exception as e:
            logger.error(f"Error creating project: {e}")
            self.ui.print_error(f"Failed to create project: {e}")

    # =========================================================================
    # Session Management
    # =========================================================================

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List sessions in current project"""
        return self.session_controller.list_sessions(
            project_id=self._active_project_id
        )

    def switch_session(self, session_id: str):
        """Switch to session"""
        # Handle partial session ID matching
        sessions = self.session_controller.list_sessions(
            project_id=self._active_project_id
        )

        # Try exact match first, then prefix match
        session = self.session_controller.get_session(session_id)
        if not session:
            # Try prefix match
            for s in sessions:
                if s['session_id'].startswith(session_id):
                    session = self.session_controller.get_session(s['session_id'])
                    break

        if not session:
            self.ui.print_error(f"Session not found: {session_id}")
            return

        self.active_session = session
        self.active_session_name = session.name

        # Restore conversation history
        self._conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in session.messages
        ]

        self._update_ui_context()
        self.ui.print_success(f"Switched to session: {session.name}")

    def create_session(self, name: Optional[str] = None):
        """Create new session"""
        session = self.session_controller.create_session(
            name=name,
            project_id=self._active_project_id,
            ephemeral=self.ephemeral_mode
        )

        self.active_session = session
        self.active_session_name = session.name
        self._conversation_history = []

        self._update_ui_context()
        self.ui.print_success(f"Created session: {session.name}")

    # =========================================================================
    # Mode Management
    # =========================================================================

    def toggle_ephemeral_mode(self):
        """Toggle fast/ephemeral mode

        Fast mode:
        - Ephemeral sessions (not persisted)
        - Auto-triage (no confirmation prompts for task delegation)
        """
        self.ephemeral_mode = not self.ephemeral_mode
        self.config.fast_mode = self.ephemeral_mode  # Link fast mode to ephemeral

        if self.ephemeral_mode:
            self.ui.print_info("⚡ FAST MODE enabled:")
            self.ui.print_info("   - Sessions are ephemeral (not persisted)")
            self.ui.print_info("   - Auto-triage enabled (no confirmation prompts)")
        else:
            self.ui.print_info("🔒 STANDARD MODE enabled:")
            self.ui.print_info("   - Sessions are persistent")
            self.ui.print_info("   - Triage requires confirmation")

        self._update_ui_context()

    # =========================================================================
    # Agent Management
    # =========================================================================

    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return [
            "oracle",
            "security",
            "operator",
            "coder",
            "knowledge",
            "triage",
            "summarizer",
        ]

    def invoke_agent(self, agent_name: str, use_tmux: bool = True):
        """
        Start direct chat with agent

        Args:
            agent_name: Name of the agent
            use_tmux: If True, spawn in tmux pane; if False, inline chat
        """
        if agent_name not in self.get_available_agents():
            self.ui.print_error(f"Unknown agent: {agent_name}")
            self.ui.print_info(f"Available: {', '.join(self.get_available_agents())}")
            return

        # Check if agent already has an active pane
        if agent_name in self._active_agent_panes:
            pane_id = self._active_agent_panes[agent_name]
            self.ui.print_info(f"{agent_name.title()} is already running in pane {pane_id}")
            self.ui.print_info("Use tmux to switch panes (Ctrl+B then arrow keys)")
            return

        if use_tmux and self._can_use_tmux():
            self._spawn_agent_pane(agent_name)
        else:
            self._agent_chat_loop(agent_name)

    def _can_use_tmux(self) -> bool:
        """Check if we can use tmux (running inside tmux)"""
        return 'TMUX' in os.environ

    def _spawn_agent_pane(self, agent_name: str):
        """Spawn agent chat in a tmux pane"""
        session_id = self.active_session.session_id if self.active_session else "default"

        # Build the command to run agent_chat.py
        cmd = (
            f"python -m console.agent_chat "
            f"--agent {agent_name} "
            f"--session {session_id}"
        )

        try:
            # Get current tmux session
            result = subprocess.run(
                ['tmux', 'display-message', '-p', '#{session_name}'],
                capture_output=True,
                text=True
            )
            current_session = result.stdout.strip()

            # Split the current window to create a new pane
            result = subprocess.run(
                ['tmux', 'split-window', '-h', '-P', '-F', '#{pane_id}', cmd],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                pane_id = result.stdout.strip()
                self._active_agent_panes[agent_name] = pane_id

                # Track in agent context
                if self._agent_context:
                    self._agent_context.set_pane(agent_name, pane_id)

                # Register with TaskRegistry so /tasks shows agent sessions
                if self.task_registry:
                    task = self.task_registry.register_existing_task(
                        task_id=f"agent-{agent_name}-{pane_id}",
                        name=f"{agent_name.title()} Chat",
                        description=f"Interactive {agent_name} agent session",
                        agent=agent_name,
                        pane_id=pane_id
                    )
                    logger.info(f"Registered agent task: {task.task_id}")

                self.ui.print_success(f"Started {agent_name.title()} in pane {pane_id}")
                self.ui.print_info("Use Ctrl+B then arrow keys to switch panes")
                self.ui.print_info("Type /back in agent pane to close it")
            else:
                self.ui.print_error(f"Failed to spawn pane: {result.stderr}")
                # Fallback to inline
                self._agent_chat_loop(agent_name)

        except Exception as e:
            logger.error(f"Failed to spawn agent pane: {e}")
            self.ui.print_warning(f"Could not spawn tmux pane: {e}")
            self.ui.print_info("Falling back to inline chat...")
            self._agent_chat_loop(agent_name)

    def _agent_chat_loop(self, agent_name: str):
        """Inline agent chat loop (fallback when tmux not available)"""
        self.ui.print_info(f"Chatting with {agent_name.title()}. Type /back to return to Console.")

        agent = self._get_agent(agent_name)

        # Initialize agent context if needed
        if self._agent_context is None and self.active_session:
            self._agent_context = AgentContextManager(self.active_session.session_id)

        while True:
            try:
                user_input = self.ui.prompt(agent_name.title())

                if not user_input:
                    continue

                if user_input.lower() in ["/back", "/exit", "/quit"]:
                    self.ui.print_system("Returning to Console...")
                    break

                # Track message in context
                if self._agent_context:
                    self._agent_context.add_message(agent_name, "user", user_input)

                # Get agent response
                history = []
                if self._agent_context:
                    history = self._agent_context.get_history(agent_name, limit=10)

                if agent and hasattr(agent, 'chat'):
                    response = agent.chat(user_input, history)
                    self.ui.print_agent(agent_name, response)
                elif agent and hasattr(agent, 'process'):
                    response = str(agent.process(user_input))
                    self.ui.print_agent(agent_name, response)
                else:
                    response = f"[{agent_name} agent not available]"
                    self.ui.print_agent(agent_name, response)

                # Track response in context
                if self._agent_context:
                    self._agent_context.add_message(agent_name, "assistant", response)

            except KeyboardInterrupt:
                self.ui.print_system("Returning to Console...")
                break

    def close_agent_pane(self, agent_name: str):
        """Close an agent's tmux pane"""
        if agent_name not in self._active_agent_panes:
            self.ui.print_error(f"No active pane for {agent_name}")
            return

        pane_id = self._active_agent_panes[agent_name]

        try:
            subprocess.run(['tmux', 'kill-pane', '-t', pane_id], check=True)
            del self._active_agent_panes[agent_name]

            if self._agent_context:
                self._agent_context.clear_pane(agent_name)

            self.ui.print_success(f"Closed {agent_name.title()} pane")

        except Exception as e:
            logger.error(f"Failed to close pane: {e}")
            self.ui.print_error(f"Failed to close pane: {e}")

    def _get_agent(self, agent_name: str):
        """Get or create agent instance"""
        if agent_name in self._agents:
            return self._agents[agent_name]

        if not self.model_manager:
            return None

        try:
            if agent_name == "security":
                from agents.security_agent import ModularSecurityAgent
                self._agents[agent_name] = ModularSecurityAgent(self.model_manager)
            elif agent_name == "operator":
                from agents.operator_agent import ModularOperatorAgent
                self._agents[agent_name] = ModularOperatorAgent(self.model_manager)
            elif agent_name == "oracle":
                self._agents[agent_name] = self.oracle
            elif agent_name == "coder":
                from agents.coder_agent import ModularCoderAgent
                self._agents[agent_name] = ModularCoderAgent(self.model_manager)
            elif agent_name == "knowledge":
                from agents.knowledge_agent import KnowledgeAgent
                self._agents[agent_name] = KnowledgeAgent(self.model_manager)
            elif agent_name == "triage":
                from agents.triage_agent import ModularTriageAgent
                self._agents[agent_name] = ModularTriageAgent(self.model_manager)
            elif agent_name == "summarizer":
                from agents.enhanced_summarization import EnhancedSummarizationAgent
                self._agents[agent_name] = EnhancedSummarizationAgent(self.model_manager)

            return self._agents.get(agent_name)

        except ImportError as e:
            logger.warning(f"Could not load {agent_name} agent: {e}")
            return None

    # =========================================================================
    # Task Management
    # =========================================================================

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get active tasks across all sessions"""
        tasks = []

        # Get tasks from internal tracking
        for task_id, task in self._background_tasks.items():
            tasks.append({
                'task_id': task.task_id,
                'session': task.session_name,
                'agent': task.agent,
                'status': task.status,
                'started': task.started_at
            })

        return tasks

    def get_background_tasks(self) -> List[Dict[str, Any]]:
        """Get background tmux tasks"""
        return self.tmux_manager.list_task_sessions()

    def attach_to_task(self, task_name: str):
        """Attach to background task"""
        tasks = self.get_background_tasks()

        # Find matching task
        matching = None
        for task in tasks:
            if task_name in task['name'] or task_name == task.get('task_id', ''):
                matching = task
                break

        if not matching:
            self.ui.print_error(f"Task not found: {task_name}")
            self.ui.print_info("Use /tasks to list available tasks")
            return

        session_name = matching['session_name']
        self.ui.print_system(f"Attaching to {session_name}...")
        self.ui.print_info("Press Ctrl+B then D to detach")

        # Attach to the tmux session
        self.tmux_manager.attach_session(session_name)

        self.ui.print_system("Detached from task session")

    def spawn_background_task(
        self,
        name: str,
        command: str,
        agent: str = "oracle"
    ) -> Optional[BackgroundTask]:
        """
        Spawn a task in tmux background

        Args:
            name: Task name
            command: Command to execute
            agent: Agent running the task

        Returns:
            BackgroundTask if successful
        """
        try:
            task = self.tmux_manager.spawn_task_session(
                name=name,
                command=command,
                agent=agent
            )

            # Track the task
            self._background_tasks[task.task_id] = task

            # Add to session's active tasks
            if self.active_session:
                self.session_controller.add_task(
                    self.active_session.session_id,
                    task.task_id
                )

            self.ui.print_success(f"Started background task: {name}")
            self.ui.print_info(f"Use /attach {task.task_id} to view output")

            return task

        except Exception as e:
            logger.error(f"Failed to spawn task: {e}")
            self.ui.print_error(f"Failed to start task: {e}")
            return None

    def _should_use_background(self, message: str) -> bool:
        """Check if message should trigger background execution"""
        estimate = self.task_estimator.estimate(message)
        return estimate.execution_mode == ExecutionMode.TMUX

    def kill_background_task(self, task_id: str):
        """Kill a background task"""
        tasks = self.get_background_tasks()

        # Find matching task
        matching = None
        for task in tasks:
            if task_id in task['name'] or task_id == task.get('task_id', ''):
                matching = task
                break

        if not matching:
            self.ui.print_error(f"Task not found: {task_id}")
            self.ui.print_info("Use /tasks to list available tasks")
            return

        session_name = matching['session_name']

        try:
            self.tmux_manager.kill_session(session_name)
            self.ui.print_success(f"Killed task: {matching['name']}")

            # Remove from internal tracking
            if task_id in self._background_tasks:
                del self._background_tasks[task_id]

        except Exception as e:
            logger.error(f"Failed to kill task: {e}")
            self.ui.print_error(f"Failed to kill task: {e}")

    # =========================================================================
    # Enhanced Background Task Management
    # =========================================================================

    def _check_notifications(self):
        """Check and display pending task notifications"""
        if not self.task_registry.has_notifications():
            return

        notifications = self.task_registry.pop_notifications()
        for notif in notifications:
            notif_type = notif.get('type', 'info')
            message = notif.get('message', '')
            task_id = notif.get('task_id', '')

            if notif_type == 'started':
                self.ui.print_notification(f"[Background] {message}", style="info")
            elif notif_type == 'completed':
                self.ui.print_notification(f"[Done] {message}", style="success")
            elif notif_type == 'failed':
                self.ui.print_notification(f"[Failed] {message}", style="error")
            elif notif_type == 'progress':
                self.ui.print_notification(f"[Progress] {message}", style="dim")
            elif notif_type == 'cancelled':
                self.ui.print_notification(f"[Cancelled] {message}", style="warning")

    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific task

        Args:
            task_id: Task ID

        Returns:
            Task details dict or None
        """
        return self.task_registry.get_task_status(task_id)

    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get all currently running tasks"""
        tasks = self.task_registry.get_running_tasks()
        return [
            {
                'task_id': t.task_id,
                'name': t.name,
                'description': t.description,
                'agent': t.agent,
                'progress': t.progress,
                'runtime': self.task_registry._calculate_runtime(t)
            }
            for t in tasks
        ]

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running background task

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancelled successfully
        """
        return self.task_registry.cancel_task(task_id)

    def spawn_background_request(
        self,
        request: str,
        description: str = ""
    ) -> Optional[TrackedTask]:
        """
        Spawn a request to run in background via Oracle

        Uses temp file for request to avoid shell escaping issues.

        Args:
            request: Natural language request
            description: Human-readable description

        Returns:
            TrackedTask if spawned successfully
        """
        if not self.oracle:
            self.ui.print_error("Oracle not available for background tasks")
            return None

        # Write request to temp file to avoid shell escaping issues
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        request_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            prefix='oracle_request_',
            delete=False
        )
        try:
            request_file.write(request)
            request_file.close()
            request_path = shlex.quote(request_file.name)
        except Exception as e:
            logger.error(f"Failed to write request file: {e}")
            self.ui.print_error(f"Failed to create background request: {e}")
            return None

        # Create wrapper script that reads request from file
        wrapper_script = f'''
import sys
import os
import yaml
from pathlib import Path

sys.path.insert(0, {repr(project_dir)})
from agents.oracle_agent import OracleAgent
from core.model_manager import LazyModelManager

# Read request from file
with open({repr(request_file.name)}, 'r') as f:
    request = f.read()

# Clean up request file
try:
    os.unlink({repr(request_file.name)})
except:
    pass

# Load models config (same as start_console.py)
config_path = Path({repr(project_dir)}) / 'config' / 'models.yaml'
models_config = {{}}
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        models_config = config.get('models', {{}})

# Process request (auto_triage=True for background - no interactive prompts)
mm = LazyModelManager(models_config=models_config)
oracle = OracleAgent(mm)
result = oracle.chat(request, auto_triage=True)
print('=== RESULT ===')
print(result)
print('=== COMPLETE ===')
'''
        # Write script to temp file
        script_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            prefix='oracle_bg_',
            delete=False
        )
        try:
            script_file.write(wrapper_script)
            script_file.close()
        except Exception as e:
            logger.error(f"Failed to write script file: {e}")
            self.ui.print_error(f"Failed to create background request: {e}")
            return None

        # Build command using the script file
        wrapper_cmd = (
            f"cd {shlex.quote(project_dir)} && "
            f"source ~/envs/base/bin/activate && "
            f"python {shlex.quote(script_file.name)} && "
            f"rm -f {shlex.quote(script_file.name)}"
        )

        # Spawn via task registry
        task = self.task_registry.spawn_task(
            name=request[:30].replace(" ", "-"),
            description=description or request,
            agent="oracle",
            command=wrapper_cmd,
            working_dir=project_dir
        )

        return task

    # =========================================================================
    # Viewer Integration
    # =========================================================================

    def open_chunk_viewer(self, doc_id: Optional[str] = None):
        """
        Open chunk viewer in tmux pane

        Args:
            doc_id: Optional document ID or search query
        """
        # Determine project for the viewer
        project = self.active_project or "default"

        # Build command
        cmd = f"python -m console.chunk_viewer --project {project}"
        if doc_id:
            cmd += f" --query '{doc_id}'"

        if self._can_use_tmux():
            self._spawn_viewer_pane(cmd)
        else:
            self._run_viewer_inline(project, doc_id)

    def _spawn_viewer_pane(self, cmd: str):
        """Spawn chunk viewer in tmux pane"""
        try:
            # Split window horizontally (side by side)
            result = subprocess.run(
                ['tmux', 'split-window', '-h', '-P', '-F', '#{pane_id}', cmd],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                pane_id = result.stdout.strip()
                self.ui.print_success(f"Opened chunk viewer in pane {pane_id}")
                self.ui.print_info("Use Ctrl+B then arrow keys to switch panes")
                self.ui.print_info("Type /back in viewer to close it")
            else:
                self.ui.print_error(f"Failed to spawn pane: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to spawn viewer pane: {e}")
            self.ui.print_error(f"Failed to open viewer: {e}")

    def _run_viewer_inline(self, project: str, query: Optional[str] = None):
        """Run chunk viewer inline (fallback when not in tmux)"""
        self.ui.print_info("Running chunk viewer inline...")
        self.ui.print_info("Type /back to return to Console")

        try:
            from .chunk_viewer import ChunkViewer

            viewer = ChunkViewer(project=project)

            if query:
                viewer.cmd_search(query)

            viewer.run()

        except ImportError as e:
            self.ui.print_error(f"Could not import chunk viewer: {e}")
        except Exception as e:
            logger.error(f"Viewer error: {e}")
            self.ui.print_error(f"Viewer error: {e}")
