#!/usr/bin/env python3
"""
Interactive Terminal UI

Rich-based terminal interface for interactive mode.
Beautiful, clean, and user-friendly.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box
from typing import Dict, List, Optional, Any

from interactive.modes import Checkpoint, UserAction, ReviewAction, InteractiveSession


class TerminalUI:
    """
    Rich-based terminal interface for interactive mode
    
    Provides beautiful, scannable displays for:
    - Checkpoints
    - Execution results
    - Progress tracking
    - User prompts
    """
    
    def __init__(self):
        self.console = Console()
    
    def clear_screen(self):
        """Clear terminal screen"""
        self.console.clear()
    
    def print_header(self, title: str):
        """Print section header"""
        self.console.print()
        self.console.print(f"[bold cyan]{title}[/bold cyan]")
        self.console.print("─" * 70)
    
    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"[green]✓[/green] {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"[red]✗[/red] {message}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        self.console.print(f"[yellow]⚠[/yellow] {message}")
    
    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"[blue]ℹ[/blue] {message}")
    
    def display_welcome(self):
        """Display welcome screen"""
        welcome_text = """
# 🚀 LLM-Engine Interactive Mode

Welcome to interactive mode! This allows you to:
- Guide agents through checkpoints
- Review work before completion
- Ask questions about results
- Rollback if needed
"""
        self.console.print(Panel(
            Markdown(welcome_text),
            title="Welcome",
            border_style="cyan",
            box=box.ROUNDED
        ))
    
    def display_session_start(self, session: InteractiveSession):
        """
        Display session start information
        
        Args:
            session: Interactive session
        """
        session_info = f"""
[bold]Session ID:[/bold] [cyan]{session.session_id}[/cyan]
[bold]Mode:[/bold] {session.mode.value}
[bold]Checkpoints:[/bold] {session.checkpoint_frequency.value}
[bold]Request:[/bold] {session.user_request}
"""
        
        self.console.print(Panel(
            session_info.strip(),
            title="📋 Session Details",
            border_style="blue",
            box=box.ROUNDED
        ))
        self.console.print()

    def prompt_user_action(self) -> UserAction:
        """Prompt for user action at checkpoint"""
        # This is the same as display_checkpoint_actions()
        return self.display_checkpoint_actions()
    
    def display_session_complete(self, session: InteractiveSession):
        """
        Display session completion
        
        Args:
            session: Completed session
        """
        completion_info = f"""
[bold green]✨ Session Completed Successfully![/bold green]

[bold]Session ID:[/bold] {session.session_id}
[bold]Total Time:[/bold] {session.total_processing_time:.2f}s
[bold]Checkpoints:[/bold] {len(session.checkpoints)}
[bold]Executions:[/bold] {len(session.execution_history)}
"""
        
        self.console.print()
        self.console.print(Panel(
            completion_info.strip(),
            title="🎉 Complete",
            border_style="green",
            box=box.DOUBLE
        ))
    
    def display_final_summary(self, session: InteractiveSession):
        """
        Display final summary before completion
        
        Args:
            session: Current session
        """
        summary_lines = []
        
        # Header
        summary_lines.append("[bold cyan]📊 Final Summary[/bold cyan]")
        summary_lines.append("")
        
        # Session info
        summary_lines.append(f"[bold]Session:[/bold] {session.session_id}")
        summary_lines.append(f"[bold]Request:[/bold] {session.user_request}")
        summary_lines.append("")
        
        # Results by stage
        if session.accumulated_results:
            summary_lines.append("[bold yellow]Results by Stage:[/bold yellow]")
            
            for stage, result in session.accumulated_results.items():
                summary_lines.append(f"  • [cyan]{stage}[/cyan]:")
                if isinstance(result, dict):
                    for key, value in list(result.items())[:5]:  # Show first 5 items
                        summary_lines.append(f"    - {key}: {value}")
                else:
                    summary_lines.append(f"    {result}")
            summary_lines.append("")
        
        # Execution history
        if session.execution_history:
            exec_count = len(session.execution_history)
            summary_lines.append(f"[bold green]Executions:[/bold green] {exec_count} operations")
            summary_lines.append("")
        
        # Checkpoints
        checkpoint_count = len(session.checkpoints)
        summary_lines.append(f"[bold blue]Checkpoints:[/bold blue] {checkpoint_count} saved")
        
        # Display in panel
        self.console.print()
        self.console.print(Panel(
            "\n".join(summary_lines),
            title="🎯 Final Review",
            border_style="yellow",
            box=box.ROUNDED
        ))
        self.console.print()

    def prompt_input(self, message: str, default: str = None) -> str:
        """Prompt for user input"""
        from rich.prompt import Prompt
        return Prompt.ask(message, default=default)
    
    def display_checkpoint(self, checkpoint: Checkpoint):
        """Display checkpoint information"""
        
        # Create checkpoint panel
        title = f"🎯 CHECKPOINT #{checkpoint.checkpoint_number}: {checkpoint.checkpoint_type.upper()}"
        
        # Build checkpoint content
        content_lines = []
        
        # Session info
        content_lines.append(f"[bold]Session:[/bold] {checkpoint.session_id}")
        content_lines.append(f"[bold]Time:[/bold] {checkpoint.timestamp.strftime('%H:%M:%S')}")
        content_lines.append(f"[bold]Agent:[/bold] {checkpoint.current_agent}")
        content_lines.append(f"[bold]Stage:[/bold] {checkpoint.current_stage}")
        
        if checkpoint.cycle_number > 0:
            content_lines.append(f"[bold]Cycle:[/bold] {checkpoint.cycle_number}")
        
        content_lines.append("")
        
        # Results summary - FIXED: Better handling of nested dicts
        if checkpoint.accumulated_results:
            content_lines.append("[bold cyan]📊 Current Results:[/bold cyan]")
            content_lines.append("")
            
            for stage, result in checkpoint.accumulated_results.items():
                content_lines.append(f"[yellow]▸ {stage.upper()}:[/yellow]")
                
                if isinstance(result, dict):
                    # Show relevant fields based on stage
                    if stage == 'triage':
                        content_lines.append(f"  • Classification: [green]{result.get('classification', 'N/A')}[/green]")
                        content_lines.append(f"  • Confidence: [cyan]{result.get('confidence', 0):.2f}[/cyan]")
                        content_lines.append(f"  • Agent: {result.get('recommended_agent', 'N/A')}")
                        if result.get('needs_clarification'):
                            content_lines.append(f"  • [yellow]⚠ Needs clarification[/yellow]")
                    
                    elif stage == 'security':
                        content_lines.append(f"  • Risk Level: [yellow]{result.get('risk_level', 'N/A')}[/yellow]")
                        commands = result.get('suggested_commands', [])
                        if commands:
                            content_lines.append(f"  • Commands: {len(commands)} suggested")
                            # Show first 2 commands
                            for cmd in commands[:2]:
                                content_lines.append(f"    - {cmd}")
                            if len(commands) > 2:
                                content_lines.append(f"    ... and {len(commands) - 2} more")
                        reasoning = result.get('reasoning', '')
                        if reasoning:
                            # Truncate long reasoning
                            reasoning_preview = reasoning[:100] + '...' if len(reasoning) > 100 else reasoning
                            content_lines.append(f"  • Reasoning: {reasoning_preview}")
                    
                    elif stage == 'execution':
                        content_lines.append(f"  • Status: [green]{result.get('status', 'N/A')}[/green]")
                        if 'cycles_completed' in result:
                            content_lines.append(f"  • Cycles: {result.get('cycles_completed', 0)}")
                        if 'total_commands' in result:
                            content_lines.append(f"  • Commands: {result.get('total_commands', 0)} executed")
                        if result.get('is_complete'):
                            content_lines.append(f"  • [green]✓ Complete[/green]")
                    
                    else:
                        # Generic dict display for other stages
                        for key, value in list(result.items())[:5]:
                            if isinstance(value, (str, int, float, bool)):
                                content_lines.append(f"  • {key}: {value}")
                            elif isinstance(value, list):
                                content_lines.append(f"  • {key}: {len(value)} items")
                            else:
                                content_lines.append(f"  • {key}: {type(value).__name__}")
                
                elif isinstance(result, (str, int, float, bool)):
                    content_lines.append(f"  {result}")
                else:
                    content_lines.append(f"  {type(result).__name__}")
                
                content_lines.append("")
        else:
            content_lines.append("[dim]No results accumulated yet[/dim]")
            content_lines.append("")
        
        # Performance
        content_lines.append(f"[dim]⏱️  Processing time: {checkpoint.processing_time_so_far:.2f}s[/dim]")
        if checkpoint.can_rollback:
            content_lines.append(f"[dim]💾 Rollback available[/dim]")
        
        # Create panel
        panel = Panel(
            "\n".join(content_lines),
            title=title,
            border_style="cyan",
            box=box.ROUNDED
        )
        
        self.console.print()
        self.console.print(panel)
    
    def display_checkpoint_actions(self) -> UserAction:
        """Display action menu and get user choice"""
        
        # Create actions table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Action", style="bold")
        table.add_column("Description", style="dim")
        
        table.add_row("1", "🚀 CONTINUE", "Let agents proceed")
        table.add_row("2", "✋ STOP", "End task here")
        table.add_row("3", "✏️  MODIFY", "Change approach")
        table.add_row("4", "💬 QUERY", "Ask questions")
        table.add_row("5", "🔙 ROLLBACK", "Go to earlier checkpoint")
        
        self.console.print()
        self.console.print(table)
        self.console.print()
        
        # Get user choice
        choice = Prompt.ask(
            "[bold]Your choice[/bold]",
            choices=["1", "2", "3", "4", "5"],
            default="1"
        )
        
        action_map = {
            "1": UserAction.CONTINUE,
            "2": UserAction.STOP,
            "3": UserAction.MODIFY,
            "4": UserAction.QUERY,
            "5": UserAction.ROLLBACK
        }
        
        return action_map[choice]
    
    def get_user_modification(self) -> str:
        """Get modified request or guidance from user"""
        self.console.print()
        self.console.print("[bold cyan]Provide new guidance or modified approach:[/bold cyan]")
        
        modification = Prompt.ask("[dim]>[/dim]")
        return modification
    
    def display_execution_results(self, results: List[Dict[str, Any]]):
        """Display execution results in a table"""
        
        if not results:
            self.console.print("[dim]No execution results yet[/dim]")
            return
        
        table = Table(
            title="Execution Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("#", style="dim")
        table.add_column("Command", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Output", style="dim", max_width=50)
        
        for i, result in enumerate(results[-10:], 1):  # Last 10 results
            status_emoji = "✓" if result.get('success', False) else "✗"
            status_color = "green" if result.get('success', False) else "red"
            
            table.add_row(
                str(i),
                result.get('command', 'N/A')[:50],
                f"[{status_color}]{status_emoji}[/{status_color}]",
                result.get('output', '')[:100]
            )
        
        self.console.print()
        self.console.print(table)
    
    def display_qa_prompt(self) -> str:
        """Prompt user for Q&A question"""
        self.console.print()
        self.console.print("[bold cyan]💬 Ask a question about the execution[/bold cyan]")
        self.console.print("[dim]Type 'done' to exit.[/dim]")
        self.console.print()
        
        question = Prompt.ask("[bold cyan]You[/bold cyan]")
        return question
    
    def display_qa_answer(self, question: str, answer: str):
        """Display Q&A answer"""
        self.console.print()
        self.console.print(f"[bold cyan]Q:[/bold cyan] {question}")
        self.console.print()
        self.console.print(Panel(
            answer,
            title="Answer",
            border_style="blue",
            box=box.ROUNDED
        ))
    
    def display_progress(self, agent: str, message: str, 
                        percentage: Optional[float] = None,
                        details: Optional[Dict] = None):
        """
        Display agent progress
        
        Args:
            agent: Agent name
            message: Progress message
            percentage: Optional progress percentage (0-100)
            details: Optional additional details
        """
        # Format progress message
        if percentage is not None:
            progress_str = f"[cyan]{agent}[/cyan]: {message} [{percentage:.0f}%]"
        else:
            progress_str = f"[cyan]{agent}[/cyan]: {message}"
        
        # Add details if provided
        if details:
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            progress_str += f" [dim]({detail_str})[/dim]"
        
        self.console.print(progress_str)
    
    def confirm(self, message: str, default: bool = True) -> bool:
        """Get yes/no confirmation"""
        return Confirm.ask(message, default=default)
    
    def display_completion(self, session: InteractiveSession):
        """Display final completion message"""
        
        completion_panel = Panel(
            f"[bold green]✅ Task Completed Successfully![/bold green]\n\n"
            f"Session ID: {session.session_id}\n"
            f"Total Time: {session.total_processing_time:.2f}s\n"
            f"Commands Executed: {len(session.execution_history)}",
            title="🎉 Completion",
            border_style="green",
            box=box.DOUBLE
        )
        
        self.console.print()
        self.console.print(completion_panel)
    
    def display_error(self, error_message: str):
        """Display error message"""
        error_panel = Panel(
            f"[bold red]Error:[/bold red]\n\n{error_message}",
            title="❌ Error",
            border_style="red",
            box=box.ROUNDED
        )
        
        self.console.print()
        self.console.print(error_panel)