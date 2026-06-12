"""
Command Router - Parses and routes /commands

Handles all slash commands for the Console.
"""

import logging
import re
from typing import Optional, Callable, Dict, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .console_hub import Console

logger = logging.getLogger(__name__)


class CommandError(Exception):
    """Raised when a command fails"""
    pass


@dataclass
class CommandDefinition:
    """Definition of a slash command"""
    name: str
    handler: Callable
    description: str = ""
    args: str = ""
    aliases: List[str] = field(default_factory=list)
    requires_project: bool = False
    requires_session: bool = False


class CommandRouter:
    """
    Routes /commands to appropriate handlers

    Usage:
        router = CommandRouter(console)
        router.handle("/help")
        router.handle("/project my-project")
    """

    def __init__(self, console: 'Console'):
        self.console = console
        self.commands: Dict[str, CommandDefinition] = {}
        self._alias_map: Dict[str, str] = {}
        self._register_builtin_commands()

    def register(self, command: CommandDefinition):
        """Register a command"""
        self.commands[command.name] = command

        # Register aliases
        for alias in command.aliases:
            self._alias_map[alias] = command.name

    def _register_builtin_commands(self):
        """Register all built-in commands"""

        # Help and system
        self.register(CommandDefinition(
            name="help",
            handler=self._cmd_help,
            description="Show this help message",
            aliases=["h", "?"]
        ))

        self.register(CommandDefinition(
            name="exit",
            handler=self._cmd_exit,
            description="Exit Console",
            aliases=["quit", "q"]
        ))

        self.register(CommandDefinition(
            name="status",
            handler=self._cmd_status,
            description="Show current status",
            aliases=["s"]
        ))

        self.register(CommandDefinition(
            name="resources",
            handler=self._cmd_resources,
            description="Show system resources (CPU, GPU, memory)",
            aliases=["res"]
        ))

        # Project commands
        self.register(CommandDefinition(
            name="projects",
            handler=self._cmd_projects,
            description="List all projects"
        ))

        self.register(CommandDefinition(
            name="project",
            handler=self._cmd_project,
            args="<name>",
            description="Switch to or create project"
        ))

        self.register(CommandDefinition(
            name="new-project",
            handler=self._cmd_new_project,
            args="<name> [template]",
            description="Create new project with template"
        ))

        # Session commands
        self.register(CommandDefinition(
            name="sessions",
            handler=self._cmd_sessions,
            description="List sessions in current project"
        ))

        self.register(CommandDefinition(
            name="session",
            handler=self._cmd_session,
            args="<id>",
            description="Switch to session"
        ))

        self.register(CommandDefinition(
            name="new",
            handler=self._cmd_new_session,
            args="<name>",
            description="Create new session"
        ))

        self.register(CommandDefinition(
            name="cleanup",
            handler=self._cmd_cleanup,
            description="Remove empty sessions (keep most recent)"
        ))

        # Mode commands
        self.register(CommandDefinition(
            name="fast",
            handler=self._cmd_fast,
            description="Toggle ephemeral/fast mode"
        ))

        # Agent commands
        self.register(CommandDefinition(
            name="agent",
            handler=self._cmd_agent,
            args="<name>",
            description="Chat directly with an agent",
            aliases=["a"]
        ))

        self.register(CommandDefinition(
            name="agents",
            handler=self._cmd_agents,
            description="List available agents"
        ))

        # Viewer commands
        self.register(CommandDefinition(
            name="chunks",
            handler=self._cmd_chunks,
            args="[doc_id]",
            description="Open chunk viewer"
        ))

        # The following commands depend on Oracle's task-runner / background
        # workflow system, which is not part of the v0.2 release. The handler
        # bodies are kept for v0.3 reactivation but the commands are not
        # registered so /help only advertises what actually works:
        #   tasks, attach, run, kill, what, cancel, bg, modules, abort

        # Diagnostics
        self.register(CommandDefinition(
            name="diag",
            handler=self._cmd_diag,
            args="[export]",
            description="Show diagnostics or export to file for sharing",
            aliases=["debug", "logs"]
        ))

    def handle(self, input_str: str) -> bool:
        """
        Parse and execute a command

        Args:
            input_str: User input starting with /

        Returns:
            True if command was handled successfully
        """
        if not input_str.startswith("/"):
            return False

        # Parse command and args
        parts = input_str[1:].split(maxsplit=1)
        if not parts:
            return False

        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Resolve alias
        if cmd_name in self._alias_map:
            cmd_name = self._alias_map[cmd_name]

        # Find command
        command = self.commands.get(cmd_name)

        if not command:
            self.console.ui.print_error(f"Unknown command: /{cmd_name}")
            self.console.ui.print_info("Type /help for available commands")
            return False

        # Check requirements
        if command.requires_project and not self.console.active_project:
            self.console.ui.print_error("This command requires an active project")
            self.console.ui.print_info("Use /project <name> to switch to a project")
            return False

        if command.requires_session and not self.console.active_session:
            self.console.ui.print_error("This command requires an active session")
            self.console.ui.print_info("Use /new <name> to create a session")
            return False

        # Execute command
        try:
            command.handler(args)
            return True
        except CommandError as e:
            self.console.ui.print_error(str(e))
            return False
        except Exception as e:
            logger.error(f"Command error: {e}", exc_info=True)
            self.console.ui.print_error(f"Command failed: {e}")
            return False

    def get_commands_for_help(self) -> Dict[str, Dict[str, str]]:
        """Get command info for help display"""
        return {
            name: {
                "args": cmd.args,
                "description": cmd.description
            }
            for name, cmd in self.commands.items()
        }

    # =========================================================================
    # Command Handlers
    # =========================================================================

    def _cmd_help(self, args: str):
        """Show help"""
        self.console.ui.display_help(self.get_commands_for_help())

    def _cmd_exit(self, args: str):
        """Exit Console"""
        raise SystemExit(0)

    def _cmd_status(self, args: str):
        """Show current status"""
        active_tasks = self.console.get_active_tasks()

        mode = "ephemeral" if self.console.ephemeral_mode else "persistent"

        self.console.ui.display_status(
            project=self.console.active_project,
            session=self.console.active_session_name,
            mode=mode,
            active_tasks=active_tasks
        )

        # Show system resources
        resource_summary = self.console.get_resource_summary()
        if resource_summary:
            self.console.ui.print_info("")
            self.console.ui.print_info(resource_summary)

    def _cmd_resources(self, args: str):
        """Show detailed system resources"""
        resource_check = self.console.check_system_resources()

        self.console.ui.print_info("System Resources:")
        self.console.ui.print_info("")

        # Show resource summary
        summary = self.console.get_resource_summary()
        if summary and summary != "Resource monitoring not available":
            for line in summary.split('\n'):
                self.console.ui.print_info(line)
        else:
            self.console.ui.print_warning("Resource monitoring not available")
            self.console.ui.print_info("Install psutil: pip install psutil")
            return

        # Show warnings
        if resource_check['warnings']:
            self.console.ui.print_info("")
            self.console.ui.print_warning("Warnings:")
            for warning in resource_check['warnings']:
                self.console.ui.print_warning(f"  - {warning}")

        # Show readiness
        self.console.ui.print_info("")
        if resource_check['ready']:
            self.console.ui.print_success("System ready for tasks")
        else:
            self.console.ui.print_error("System resources critical - consider waiting")

    def _cmd_projects(self, args: str):
        """List projects"""
        projects = self.console.list_projects()
        self.console.ui.display_projects(projects, self.console.active_project)

    def _cmd_project(self, args: str):
        """Switch to or create project"""
        if not args.strip():
            raise CommandError("Usage: /project <name>")

        name = args.strip()
        self.console.switch_project(name)

    def _cmd_new_project(self, args: str):
        """Create new project"""
        if not args.strip():
            raise CommandError("Usage: /new-project <name> [template]")

        parts = args.strip().split(maxsplit=1)
        name = parts[0]
        template = parts[1] if len(parts) > 1 else "default"

        self.console.create_project(name, template)

    def _cmd_sessions(self, args: str):
        """
        List sessions

        Auto-filter by active project unless 'all' is specified

        Usage:
            /sessions       - List sessions for current project
            /sessions all   - List all sessions across projects
        """
        sessions = self.console.list_sessions()
        active_id = self.console.active_session.session_id if self.console.active_session else None

        # Filter by active project unless 'all' specified
        show_all = args.strip().lower() == 'all'

        if not show_all and self.console.active_project:
            # Filter sessions by active project
            project_sessions = [
                s for s in sessions
                if s.get('project') == self.console.active_project or s.get('project') is None
            ]
            if project_sessions:
                self.console.ui.print_info(f"Sessions for project '{self.console.active_project}':")
                self.console.ui.print_info("(Use /sessions all to see all projects)")
                self.console.ui.print_info("")
            sessions = project_sessions

        self.console.ui.display_sessions(sessions, active_id)

    def _cmd_session(self, args: str):
        """Switch to session"""
        if not args.strip():
            raise CommandError("Usage: /session <id>")

        session_id = args.strip()
        self.console.switch_session(session_id)

    def _cmd_new_session(self, args: str):
        """Create new session"""
        name = args.strip() if args.strip() else None
        self.console.create_session(name)

    def _cmd_cleanup(self, args: str):
        """Clean up empty sessions"""
        deleted = self.console.session_controller.cleanup_empty_sessions(keep_recent=1)
        if deleted > 0:
            self.console.ui.print_info(f"Cleaned up {deleted} empty session(s)")
        else:
            self.console.ui.print_info("No empty sessions to clean up")

    def _cmd_fast(self, args: str):
        """Toggle fast/ephemeral mode"""
        self.console.toggle_ephemeral_mode()

    def _cmd_agent(self, args: str):
        """Chat with specific agent"""
        if not args.strip():
            raise CommandError("Usage: /agent <name>")

        agent_name = args.strip().lower()
        self.console.invoke_agent(agent_name)

    def _cmd_agents(self, args: str):
        """List available agents"""
        agents = self.console.get_available_agents()
        self.console.ui.display_agents(agents)

    def _cmd_tasks(self, args: str):
        """List background tasks"""
        tasks = self.console.get_background_tasks()
        self.console.ui.display_tasks(tasks)

    def _cmd_attach(self, args: str):
        """Attach to background task"""
        if not args.strip():
            raise CommandError("Usage: /attach <task>")

        task_name = args.strip()
        self.console.attach_to_task(task_name)

    def _cmd_chunks(self, args: str):
        """Open chunk viewer"""
        doc_id = args.strip() if args.strip() else None
        self.console.open_chunk_viewer(doc_id)

    def _cmd_run(self, args: str):
        """Run command in background"""
        if not args.strip():
            raise CommandError("Usage: /run <command>")

        command = args.strip()

        # Generate a task name from the command
        task_name = command.split()[0][:20] if command else "task"

        self.console.spawn_background_task(
            name=task_name,
            command=command,
            agent="console"
        )

    def _cmd_kill(self, args: str):
        """Kill a background task"""
        if not args.strip():
            raise CommandError("Usage: /kill <task>")

        task_id = args.strip()
        self.console.kill_background_task(task_id)

    # =========================================================================
    # Introspection Commands
    # =========================================================================

    def _cmd_what(self, args: str):
        """
        Ask Oracle about task progress or general status

        Usage:
            /what          - What's Oracle/system doing?
            /what <task>   - Details about specific task
        """
        task_id = args.strip() if args.strip() else None

        if task_id:
            # Get specific task info
            task_info = self.console.get_task_details(task_id)

            if not task_info:
                raise CommandError(f"Task not found: {task_id}")

            # Display task details
            self.console.ui.display_task_details(task_info)

            # If task is running, ask Oracle for interpretation
            if task_info.get('status') == 'running':
                output = task_info.get('recent_output', '')
                if output and self.console.oracle:
                    self.console.ui.print_info("Asking Oracle for interpretation...")
                    interpretation = self.console.oracle.interpret_task_output(
                        task_name=task_info.get('name', 'unknown'),
                        output=output[-1000:]  # Last 1000 chars
                    )
                    if interpretation:
                        self.console.ui.print_oracle_response(interpretation)
        else:
            # General status - what's happening?
            running_tasks = self.console.get_running_tasks()

            if not running_tasks:
                self.console.ui.print_info("No tasks currently running")
                self.console.ui.print_info("Use /bg <request> to run something in background")
            else:
                self.console.ui.print_info(f"Running tasks: {len(running_tasks)}")
                for task in running_tasks:
                    runtime = task.get('runtime', 'unknown')
                    self.console.ui.print_task_brief(
                        task['task_id'],
                        task['name'],
                        task.get('progress', 0),
                        runtime
                    )

    def _cmd_cancel(self, args: str):
        """
        Cancel a running background task

        Usage:
            /cancel <task_id>
        """
        if not args.strip():
            # Show running tasks for user to choose
            running = self.console.get_running_tasks()
            if not running:
                self.console.ui.print_info("No tasks currently running")
                return

            self.console.ui.print_info("Running tasks:")
            for task in running:
                self.console.ui.print_task_brief(
                    task['task_id'],
                    task['name'],
                    task.get('progress', 0),
                    task.get('runtime', '')
                )
            raise CommandError("Usage: /cancel <task_id>")

        task_id = args.strip()
        success = self.console.cancel_task(task_id)

        if success:
            self.console.ui.print_success(f"Cancelled task: {task_id}")
        else:
            raise CommandError(f"Could not cancel task: {task_id}")

    def _cmd_background(self, args: str):
        """
        Run a request in the background (non-blocking)

        Usage:
            /bg check system resources on all hosts
            /bg analyze this binary and generate report
        """
        if not args.strip():
            raise CommandError("Usage: /bg <request>")

        request = args.strip()

        # Spawn as background Oracle request
        task = self.console.spawn_background_request(
            request=request,
            description=request[:50] + "..." if len(request) > 50 else request
        )

        if task:
            self.console.ui.print_success(
                f"Background task started: {task.task_id}"
            )
            self.console.ui.print_info(
                f"Use /what {task.task_id} to check progress"
            )
            self.console.ui.print_info(
                f"Use /cancel {task.task_id} to stop"
            )

    def _cmd_modules(self, args: str):
        """
        Show module context from doc-manager

        Usage:
            /modules                    - List all available modules
            /modules <query>            - Find modules relevant to query
        """
        if not hasattr(self.console, 'modular_context') or not self.console.modular_context:
            self.console.ui.print_warning("Module context not available (doc-manager not configured)")
            return

        mc = self.console.modular_context
        if not mc.is_available():
            self.console.ui.print_warning("Module context not available")
            return

        if not args.strip():
            # List all modules
            modules = mc.list_modules()
            if modules:
                self.console.ui.print_info(f"Available modules ({len(modules)}):")
                for mod in modules:
                    ctx = mc.get_module_context(mod)
                    if ctx:
                        self.console.ui.print_info(f"  - {mod}: {ctx.description[:60]}...")
            else:
                self.console.ui.print_info("No modules registered in doc-manager")
            return

        # Find modules relevant to query
        query = args.strip()
        task_context = mc.get_task_context(query, top_k=3)
        suggested = task_context.get('suggested_modules', [])

        if suggested:
            self.console.ui.print_info(f"Modules relevant to '{query}':")
            for mod_name in suggested[:3]:
                ctx = mc.get_module_context(mod_name)
                if ctx:
                    self.console.ui.print_info(f"\n  [{mod_name}]")
                    self.console.ui.print_info(f"  Type: {ctx.type}")
                    self.console.ui.print_info(f"  Location: {ctx.code_location}")
                    self.console.ui.print_info(f"  {ctx.description}")
        else:
            self.console.ui.print_info(f"No modules found for: {query}")

    def _cmd_abort(self, args: str):
        """
        Abort current Oracle request

        Usage:
            /abort
        """
        self.console.abort_current_request()


    # =========================================================================
    # Diagnostics
    # =========================================================================

    def _cmd_diag(self, args: str):
        """Show diagnostics or export to shareable file.

        /diag         - Show diagnostics summary
        /diag export  - Export full diagnostics to file for sharing
        """
        import platform
        import os
        import psutil
        from pathlib import Path
        from datetime import datetime

        project_root = Path(__file__).parent.parent

        if args.strip().lower() == 'export':
            self._diag_export(project_root)
            return

        self.console.ui.print_info("=== LLM-Agent-System Diagnostics ===")
        self.console.ui.print_info("")

        # System info
        self.console.ui.print_info("System:")
        self.console.ui.print_info(f"  OS: {platform.system()} {platform.release()}")
        self.console.ui.print_info(f"  Python: {platform.python_version()}")
        self.console.ui.print_info(f"  CPU: {psutil.cpu_count()} cores, {psutil.cpu_percent()}% used")
        mem = psutil.virtual_memory()
        self.console.ui.print_info(f"  RAM: {mem.used // (1024**3)}GB / {mem.total // (1024**3)}GB ({mem.percent}%)")

        # GPU info
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.console.ui.print_info(f"  GPU: {gpu.name} ({gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB)")
        except Exception:
            self.console.ui.print_info("  GPU: Not detected")

        # Model config
        self.console.ui.print_info("")
        self.console.ui.print_info("Model Config:")
        config_file = project_root / "config" / "models.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                models = config.get('models', {})
                for agent_name, agent_conf in models.items():
                    model = agent_conf.get('model_path', 'not set')
                    layers = agent_conf.get('n_gpu_layers', '?')
                    chat = agent_conf.get('use_chat_api', False)
                    self.console.ui.print_info(f"  {agent_name}: layers={layers}, chat_api={chat}")
            except Exception as e:
                self.console.ui.print_warning(f"  Could not read config: {e}")
        else:
            self.console.ui.print_warning("  config/models.yaml not found")

        # Log files
        self.console.ui.print_info("")
        self.console.ui.print_info("Log Files:")
        log_files = [
            project_root / "console.log",
            project_root / "logs" / "oracle_detailed.log",
            project_root / "logs" / "oracle_full_responses.log",
        ]
        for lf in log_files:
            if lf.exists():
                size_kb = lf.stat().st_size / 1024
                self.console.ui.print_info(f"  {lf.name}: {size_kb:.1f} KB")
            else:
                self.console.ui.print_info(f"  {lf.name}: not found")

        # Recent errors
        self.console.ui.print_info("")
        self.console.ui.print_info("Recent Errors (last 5):")
        console_log = project_root / "console.log"
        if console_log.exists():
            try:
                errors = []
                with open(console_log, 'r') as f:
                    for line in f:
                        if 'ERROR' in line:
                            errors.append(line.strip())
                for err in errors[-5:]:
                    self.console.ui.print_error(f"  {err[:120]}")
                if not errors:
                    self.console.ui.print_success("  No errors found")
            except Exception:
                self.console.ui.print_warning("  Could not read console.log")
        else:
            self.console.ui.print_info("  No console.log found")

        self.console.ui.print_info("")
        self.console.ui.print_info("Run '/diag export' to save full diagnostics to a shareable file")

    def _diag_export(self, project_root):
        """Export full diagnostics to a shareable file."""
        import platform
        import os
        import psutil
        from pathlib import Path
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = project_root / f"diagnostics_{timestamp}.txt"

        lines = []
        lines.append("=" * 60)
        lines.append("LLM-Agent-System Diagnostics Report")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 60)

        # System
        lines.append("\n## System")
        lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
        lines.append(f"Python: {platform.python_version()}")
        lines.append(f"Hostname: {platform.node()}")
        lines.append(f"CPU: {psutil.cpu_count()} cores")
        mem = psutil.virtual_memory()
        lines.append(f"RAM: {mem.total // (1024**3)} GB total, {mem.available // (1024**3)} GB available")
        disk = psutil.disk_usage('/')
        lines.append(f"Disk: {disk.total // (1024**3)} GB total, {disk.free // (1024**3)} GB free")

        # GPU
        lines.append("\n## GPU")
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                lines.append(f"  {gpu.name}: {gpu.memoryTotal:.0f} MB total, {gpu.memoryUsed:.0f} MB used")
        except Exception as e:
            lines.append(f"  Not detected: {e}")

        # Model config
        lines.append("\n## Model Configuration (config/models.yaml)")
        config_file = project_root / "config" / "models.yaml"
        if config_file.exists():
            lines.append(config_file.read_text())
        else:
            lines.append("  NOT FOUND")

        # llama-server
        lines.append("\n## llama-server")
        import shutil
        llama_path = shutil.which('llama-server')
        lines.append(f"  PATH: {llama_path or 'not in PATH'}")
        home_build = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
        lines.append(f"  ~/llama.cpp: {'found' if home_build.exists() else 'not found'}")

        # Running processes
        lines.append("\n## Running LLM Processes")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'llama' in proc.info['name'].lower():
                    cmd = ' '.join(proc.info['cmdline'][:5]) if proc.info['cmdline'] else ''
                    lines.append(f"  PID {proc.info['pid']}: {cmd}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Recent log entries
        lines.append("\n## Recent Console Log (last 50 lines)")
        console_log = project_root / "console.log"
        if console_log.exists():
            try:
                with open(console_log, 'r') as f:
                    all_lines = f.readlines()
                for line in all_lines[-50:]:
                    lines.append(line.rstrip())
            except Exception as e:
                lines.append(f"  Could not read: {e}")
        else:
            lines.append("  No console.log found")

        # Recent errors only
        lines.append("\n## All Errors")
        if console_log.exists():
            try:
                with open(console_log, 'r') as f:
                    for line in f:
                        if 'ERROR' in line:
                            lines.append(line.rstrip())
            except Exception:
                pass

        # Oracle detailed log (last 30 lines)
        oracle_log = project_root / "logs" / "oracle_detailed.log"
        lines.append("\n## Oracle Detailed Log (last 30 lines)")
        if oracle_log.exists():
            try:
                with open(oracle_log, 'r') as f:
                    all_lines = f.readlines()
                for line in all_lines[-30:]:
                    lines.append(line.rstrip())
            except Exception as e:
                lines.append(f"  Could not read: {e}")
        else:
            lines.append("  No oracle_detailed.log found")

        # Write report
        report = '\n'.join(lines)
        export_path.write_text(report)

        size_kb = export_path.stat().st_size / 1024
        self.console.ui.print_success(f"Diagnostics exported to: {export_path}")
        self.console.ui.print_info(f"Size: {size_kb:.1f} KB")
        self.console.ui.print_info("Share this file when reporting issues")
