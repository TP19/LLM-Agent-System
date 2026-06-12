"""
Console UI - Rich-based terminal interface

Provides beautiful terminal output for the Console hub.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

logger = logging.getLogger(__name__)


class ConsoleUI:
    """
    Rich-based terminal UI for Console

    Provides consistent styling and formatting for:
    - Welcome/status banners
    - Command output
    - Progress indicators
    - Error/warning messages
    - Tables and panels
    """

    def __init__(self):
        # Force terminal settings for tmux compatibility
        # Set force_terminal=True to ensure proper output even in screen/tmux
        self.console = RichConsole(force_terminal=True, legacy_windows=False)
        self.current_project: Optional[str] = None
        self.current_session: Optional[str] = None
        self.ephemeral_mode: bool = False

    def display_welcome(self):
        """Display welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              🚀 LLM-Agent-System Console                            ║
║                                                               ║
║   Unified interface for AI-powered workflows                  ║
║   Type /help for commands or just chat with Oracle            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold cyan")
        self.console.print()

    def prompt(self, prefix: str = "Console") -> str:
        """
        Get user input with styled prompt

        Args:
            prefix: Prompt prefix (e.g., "Console", "Security")

        Returns:
            User input string
        """
        # Build prompt with context
        prompt_parts = []

        if self.current_project:
            prompt_parts.append(f"[dim]{self.current_project}[/dim]")

        if self.current_session:
            prompt_parts.append(f"[dim cyan]{self.current_session}[/dim cyan]")

        if self.ephemeral_mode:
            prompt_parts.append("[yellow]⚡[/yellow]")

        context = " ".join(prompt_parts)
        if context:
            full_prompt = f"{context} [bold cyan]{prefix}>[/bold cyan] "
        else:
            full_prompt = f"[bold cyan]{prefix}>[/bold cyan] "

        try:
            return Prompt.ask(full_prompt)
        except EOFError:
            return "/exit"
        except KeyboardInterrupt:
            return ""

    def confirm(self, message: str, default: bool = False) -> bool:
        """Get yes/no confirmation"""
        try:
            return Confirm.ask(message, default=default)
        except (EOFError, KeyboardInterrupt):
            return default

    def print_oracle(self, message: str):
        """Print Oracle's response"""
        self.console.print()
        self.console.print(f"[bold magenta]Oracle:[/bold magenta] {message}")
        self.console.print()

    def print_agent(self, agent_name: str, message: str):
        """Print agent response"""
        color = self._get_agent_color(agent_name)
        self.console.print()
        self.console.print(f"[bold {color}]{agent_name.title()}:[/bold {color}] {message}")
        self.console.print()

    def print_system(self, message: str):
        """Print system message"""
        self.console.print(f"[dim]ℹ {message}[/dim]")

    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"[green]✓ {message}[/green]")

    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"[red]✗ {message}[/red]")

    def print_warning(self, message: str):
        """Print warning message"""
        self.console.print(f"[yellow]⚠ {message}[/yellow]")

    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"[blue]ℹ {message}[/blue]")

    def print_progress(self, agent_name: str, status: str, message: str):
        """Print progress update from agent"""
        emoji = {
            "started": "🚀",
            "running": "⏳",
            "completed": "✅",
            "failed": "❌"
        }.get(status, "📊")

        color = self._get_agent_color(agent_name)
        self.console.print(f"   {emoji} [[{color}]{agent_name}[/{color}]] {message}")

    def display_help(self, commands: Dict[str, Dict[str, str]]):
        """Display help for all commands"""
        table = Table(title="Console Commands", box=None)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Args", style="dim")
        table.add_column("Description")

        for cmd_name, cmd_info in sorted(commands.items()):
            table.add_row(
                f"/{cmd_name}",
                cmd_info.get("args", ""),
                cmd_info.get("description", "")
            )

        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Or just type naturally to chat with Oracle[/dim]")
        self.console.print()

    def display_status(
        self,
        project: Optional[str],
        session: Optional[str],
        mode: str,
        active_tasks: List[Dict[str, Any]]
    ):
        """Display current status"""
        # Status panel
        status_lines = []

        if project:
            status_lines.append(f"Project: [cyan]{project}[/cyan]")
        else:
            status_lines.append("Project: [dim]none[/dim]")

        if session:
            status_lines.append(f"Session: [cyan]{session}[/cyan]")
        else:
            status_lines.append("Session: [dim]ephemeral[/dim]")

        status_lines.append(f"Mode: [yellow]{mode}[/yellow]")

        panel = Panel(
            "\n".join(status_lines),
            title="📊 Status",
            border_style="blue"
        )
        self.console.print(panel)

        # Active tasks
        if active_tasks:
            task_table = Table(title="Active Tasks", box=None)
            task_table.add_column("Session", style="cyan")
            task_table.add_column("Agent", style="magenta")
            task_table.add_column("Status")
            task_table.add_column("Started")

            for task in active_tasks:
                task_table.add_row(
                    task.get("session", "?"),
                    task.get("agent", "?"),
                    task.get("status", "?"),
                    task.get("started", "?")
                )

            self.console.print(task_table)
        else:
            self.console.print("[dim]No active tasks[/dim]")

        self.console.print()

    def display_projects(self, projects: List[Dict[str, Any]], active: Optional[str]):
        """Display project list"""
        if not projects:
            self.console.print("[dim]No projects found. Create one with /new-project <name>[/dim]")
            return

        table = Table(title="Projects", box=None)
        table.add_column("Name", style="cyan")
        table.add_column("Template")
        table.add_column("Sessions")
        table.add_column("Last Accessed")
        table.add_column("")

        for proj in projects:
            is_active = proj.get("name") == active
            marker = "[green]◆[/green]" if is_active else ""

            table.add_row(
                proj.get("name", "?"),
                proj.get("template", "default"),
                str(proj.get("session_count", 0)),
                proj.get("last_accessed", "?"),
                marker
            )

        self.console.print(table)
        self.console.print()

    def display_sessions(self, sessions: List[Dict[str, Any]], active: Optional[str]):
        """Display session list"""
        if not sessions:
            self.console.print("[dim]No sessions found. Create one with /new <name>[/dim]")
            return

        table = Table(title="Sessions", box=None)
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Mode")
        table.add_column("Messages")
        table.add_column("Last Accessed")
        table.add_column("")

        for sess in sessions:
            is_active = sess.get("session_id") == active
            marker = "[green]◆[/green]" if is_active else ""
            mode_style = "yellow" if sess.get("mode") == "ephemeral" else "green"

            table.add_row(
                sess.get("session_id", "?")[:8],
                sess.get("name", "?"),
                f"[{mode_style}]{sess.get('mode', '?')}[/{mode_style}]",
                str(sess.get("message_count", 0)),
                sess.get("last_accessed", "?"),
                marker
            )

        self.console.print(table)
        self.console.print()

    def display_tasks(self, tasks: List[Dict[str, Any]]):
        """Display tmux task sessions"""
        if not tasks:
            self.console.print("[dim]No background tasks running[/dim]")
            return

        table = Table(title="Background Tasks (tmux)", box=None)
        table.add_column("Name", style="cyan")
        table.add_column("Agent")
        table.add_column("Status")
        table.add_column("Started")

        for task in tasks:
            table.add_row(
                task.get("name", "?"),
                task.get("agent", "?"),
                task.get("status", "running"),
                task.get("started", "?")
            )

        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Use /attach <name> to view task output[/dim]")
        self.console.print()

    def display_agents(self, agents: List[str]):
        """Display available agents"""
        self.console.print("\n[bold]Available Agents:[/bold]")
        for agent in agents:
            color = self._get_agent_color(agent)
            self.console.print(f"  • [{color}]{agent}[/{color}]")
        self.console.print()
        self.console.print("[dim]Use /agent <name> to chat directly with an agent[/dim]")
        self.console.print()

    def set_context(
        self,
        project: Optional[str] = None,
        session: Optional[str] = None,
        ephemeral: bool = False
    ):
        """Update prompt context"""
        self.current_project = project
        self.current_session = session
        self.ephemeral_mode = ephemeral

    def _get_agent_color(self, agent_name: str) -> str:
        """Get color for agent name"""
        colors = {
            "oracle": "magenta",
            "security": "red",
            "operator": "green",
            "coder": "blue",
            "knowledge": "yellow",
            "summarizer": "cyan",
            "triage": "white",
        }
        return colors.get(agent_name.lower(), "white")

    def create_spinner(self, message: str = "Processing..."):
        """Create a spinner context manager"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        )

    def goodbye(self):
        """Display goodbye message"""
        self.console.print("\n👋 [cyan]Goodbye![/cyan]\n")

    # =========================================================================
    # Background Task UI
    # =========================================================================

    def print_notification(self, message: str, style: str = "info"):
        """
        Print a non-intrusive notification

        Args:
            message: Notification message
            style: Style (info, success, error, warning, dim)
        """
        style_map = {
            "info": "blue",
            "success": "green",
            "error": "red",
            "warning": "yellow",
            "dim": "dim"
        }
        color = style_map.get(style, "white")
        self.console.print(f"[{color}]{message}[/{color}]")

    def print_task_brief(
        self,
        task_id: str,
        name: str,
        progress: int,
        runtime: str
    ):
        """
        Print brief task info line

        Args:
            task_id: Task ID
            name: Task name
            progress: Progress percentage
            runtime: Runtime string
        """
        progress_bar = self._create_progress_bar(progress)
        self.console.print(
            f"  [{task_id}] {name[:30]:30} {progress_bar} {runtime}"
        )

    def display_task_details(self, task_info: Dict[str, Any]):
        """
        Display detailed task information

        Args:
            task_info: Task details dictionary
        """
        self.console.print()
        self.console.print(f"[bold]Task: {task_info.get('name', 'Unknown')}[/bold]")
        self.console.print(f"  ID: {task_info.get('task_id', 'N/A')}")
        self.console.print(f"  Agent: {task_info.get('agent', 'N/A')}")
        self.console.print(f"  Status: {task_info.get('status', 'unknown')}")
        self.console.print(f"  Runtime: {task_info.get('runtime', 'N/A')}")

        if task_info.get('progress', 0) > 0:
            progress_bar = self._create_progress_bar(task_info['progress'])
            self.console.print(f"  Progress: {progress_bar} {task_info['progress']}%")

        if task_info.get('progress_message'):
            self.console.print(f"  Message: {task_info['progress_message']}")

        if task_info.get('recent_output'):
            self.console.print()
            self.console.print("[dim]Recent output:[/dim]")
            output_lines = task_info['recent_output'].strip().split('\n')[-10:]
            for line in output_lines:
                self.console.print(f"  [dim]{line}[/dim]")

        self.console.print()

    def print_oracle_response(self, response: str):
        """Print Oracle's interpretation/response"""
        self.console.print()
        self.console.print("[magenta]Oracle:[/magenta]")
        self.console.print(response)
        self.console.print()

    def _create_progress_bar(self, progress: int, width: int = 10) -> str:
        """
        Create a simple progress bar

        Args:
            progress: Progress percentage (0-100)
            width: Bar width in characters

        Returns:
            Progress bar string
        """
        filled = int(width * progress / 100)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"
