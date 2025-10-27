#!/usr/bin/env python3
"""
Interactive Mode Entry Point

Main entry point for running LLM-Engine in interactive mode.

Usage:
    python start_interactive.py

Features:
    - Interactive checkpoint system
    - User guidance and control
    - Session persistence
    - Beautiful terminal UI
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from interactive.session_manager import SessionManager
from interactive.modes import CheckpointFrequency, UserStoppedException
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from interactive.modes import CheckpointFrequency
from interactive.session_manager import SessionManager
from interactive.state_store import StateStore
from interactive.terminal_ui import TerminalUI
from core.model_manager import LazyModelManager
from agents.triage_agent import ModularTriageAgent
from agents.security_agent import ModularSecurityAgent
from agents.executor_agent import ModularExecutorAgent
from agents.coder_agent import ModularCoderAgent
from agents.summarization_agent import ModularSummarizationAgent


def setup_logging():
    """
    Setup logging configuration
    
    Creates interactive-specific logs without interfering with agent logging.
    Similar to file_monitor.py approach.
    """
    # Create logs directory
    log_dir = Path.home() / ".llm_engine" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with date
    log_file = log_dir / f"interactive_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Get or create interactive logger
    logger = logging.getLogger("InteractiveMode")
    logger.setLevel(logging.INFO)
    
    # Only add handlers if not already present
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler (only for warnings and errors)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    return logger, log_file


def display_banner():
    """Display welcome banner"""
    console = Console()
    
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              🚀 LLM-Engine - Interactive Mode                ║
║                                                               ║
║  Collaborative AI workflow with checkpoints and user control  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    
    console.print(banner, style="bold cyan")
    console.print()


def get_user_request() -> str:
    """
    Get user request
    
    Returns:
        User's request string
    """
    console = Console()
    
    console.print("[bold]📝 What would you like me to do?[/bold]")
    console.print()
    console.print("Examples:")
    console.print("  • Analyze disk space usage on my system")
    console.print("  • Check Docker container status")
    console.print("  • Review system logs for errors")
    console.print("  • Create a backup script")
    console.print()
    
    request = Prompt.ask("[cyan]Your request[/cyan]")
    
    if not request.strip():
        console.print("[red]Error: Request cannot be empty[/red]")
        sys.exit(1)
    
    return request.strip()


def setup_checkpoint_frequency() -> 'CheckpointFrequency':
    """Let user choose checkpoint frequency"""
    from rich.prompt import Prompt
    from interactive.modes import CheckpointFrequency
    
    print("\n🎯 Checkpoint Frequency:")
    print("  [1] Standard - After each stage (Recommended)")
    print("  [2] Minimal - Only at critical points")
    print("  [3] Detailed - After every action")
    print("  [4] None - Skip all checkpoints (fastest)")
    
    choice = Prompt.ask(
        "Choose frequency",
        choices=["1", "2", "3", "4"],
        default="1"
    )
    
    frequency_map = {
        "1": CheckpointFrequency.STANDARD,
        "2": CheckpointFrequency.MINIMAL,
        "3": CheckpointFrequency.DETAILED,
        "4": CheckpointFrequency.NONE
    }
    
    frequency = frequency_map[choice]
    
    print(f"✅ Using {frequency.value} checkpoint frequency\n")
    
    # ✅ RETURN THE ENUM
    return frequency


def display_session_summary(session, log_file=None):
    """
    Display session completion summary
    
    Args:
        session: Completed InteractiveSession
        log_file: Path to log file
    """
    console = Console()
    
    console.print()
    console.print("[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]           ✨ Session Complete!                    [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print()
    
    # Session details
    details = f"""
Session ID: [cyan]{session.session_id}[/cyan]
Status: [green]{'Complete' if session.is_complete else 'In Progress'}[/green]
Checkpoints: [yellow]{len(session.checkpoints)}[/yellow]
Stages: [blue]{session.current_stage}[/blue]
    """
    
    console.print(Panel(details.strip(), title="📊 Session Summary", border_style="green"))
    
    # Database and log locations
    console.print()
    console.print(f"💾 Session saved to: [cyan]~/.llm_engine/interactive.db[/cyan]")
    if log_file:
        console.print(f"📋 Logs saved to: [cyan]{log_file}[/cyan]")
    else:
        console.print(f"📋 Logs saved to: [cyan]~/.llm_engine/logs/interactive_*.log[/cyan]")
    console.print()
    
    # Next steps
    console.print("[bold]Next steps:[/bold]")
    console.print("  • Run again: [cyan]python start_interactive.py[/cyan]")
    if log_file:
        console.print(f"  • View logs: [cyan]{log_file}[/cyan]")
    else:
        console.print("  • View logs: [cyan]~/.llm_engine/logs/interactive_*.log[/cyan]")
    console.print()


def display_user_stopped(session, log_file=None):
    """
    Display message when user stops workflow
    
    Args:
        session: Current session (may be None)
        log_file: Path to log file
    """
    console = Console()
    
    console.print()
    console.print("[bold yellow]═══════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold yellow]           🛑 Workflow Stopped by User            [/bold yellow]")
    console.print("[bold yellow]═══════════════════════════════════════════════════[/bold yellow]")
    console.print()
    
    console.print("The workflow has been stopped at your request.")
    console.print("Your session and progress have been saved.")
    console.print()
    
    if session:
        details = f"""
Session ID: [cyan]{session.session_id}[/cyan]
Checkpoints Saved: [yellow]{len(session.checkpoints)}[/yellow]
Last Stage: [magenta]{session.current_stage}[/magenta]
        """
        
        console.print(Panel(details.strip(), title="📊 Session Info", border_style="yellow"))
        console.print()
    
    console.print("💾 Session saved to: [cyan]~/.llm_engine/interactive.db[/cyan]")
    if log_file:
        console.print(f"📋 Logs saved to: [cyan]{log_file}[/cyan]")
    console.print()


def main():
    """
    Main entry point for interactive mode - Phase 4 with RAG
    """
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting LLM-Engine Interactive Mode (Phase 4)")
    
    try:
        # =================================================================
        # 1. Initialize Model Manager
        # =================================================================
        
        print("\n🔧 Initializing model manager...")
        
        # Load config
        with open('config/models.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Model Manager (lazy loading)
        model_manager = LazyModelManager(
            models_config=config.get('models', {})
        )
        print("   ✅ Model manager ready")
        
        # =================================================================
        # 2. Setup Checkpoint Frequency
        # =================================================================
        
        frequency = setup_checkpoint_frequency()
        
        # =================================================================
        # 3. Create Session Manager (it initializes everything else)
        # =================================================================
        
        print("\n📋 Initializing session manager...")
        
        session_manager = SessionManager(
            frequency=frequency,
            model_manager=model_manager
        )
        
        print("   ✅ Session manager ready")
        
        # =================================================================
        # 4. Main Loop
        # =================================================================
        
        print("\n" + "="*60)
        print("✨ Interactive Mode Ready!")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user request
                print("\n📝 What would you like me to do?")
                user_request = input("Your request (or 'quit' to exit): ").strip()
                
                # Check for exit
                if user_request.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_request:
                    continue
                
                # Run interactive session
                session = session_manager.run_interactive(user_request)
                
                # Display completion
                print("\n" + "="*60)
                print("✅ Session Complete")
                print(f"Session ID: {session.session_id}")
                if hasattr(session, 'total_processing_time'):
                    print(f"Duration: {session.total_processing_time:.1f}s")
                elif hasattr(session, 'created_at'):
                    # Calculate duration from created_at to now
                    from datetime import datetime
                    duration = (datetime.now() - session.created_at).total_seconds()
                    print(f"Duration: {duration:.1f}s")
                print("="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                print(f"\n❌ Error: {e}")
                print("\nYou can try another request or type 'quit' to exit.\n")
        
        # =================================================================
        # 5. Cleanup
        # =================================================================
        
        logger.info("Shutting down interactive mode")
        print("\n✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
