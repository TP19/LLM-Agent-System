#!/usr/bin/env python3
"""
Console Entry Point

Main entry point for running LLM-Agent-System Console - the unified interface.

Usage:
    python start_console.py
    python start_console.py --ephemeral     # Start in ephemeral mode
    python start_console.py --project NAME  # Start with specific project

Features:
    - Oracle as primary chat interface
    - Project and session management
    - Agent invocation (/agent security)
    - Background task management
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def cleanup_rogue_llama_servers():
    """
    Kill only orphaned llama-server processes that were spawned by LLM-Agent-System.

    Identifies LLM-Agent-System's own processes by checking if the parent process chain
    includes LLM-Agent-System paths, or if the process is a true orphan (parent is init/systemd).
    This avoids killing llama-server instances from text-generation-webui or other tools.
    """
    import subprocess
    import os
    from rich.console import Console

    # Use force_terminal for tmux compatibility
    console = Console(force_terminal=True)

    def _is_llm_engine_process(pid: int) -> bool:
        """Check if a llama-server process was spawned by LLM-Agent-System."""
        try:
            import psutil
            proc = psutil.Process(pid)

            # Primary check: LLM_ENGINE_PROCESS environment variable
            # Set by llama_server_backend.py when spawning processes
            try:
                proc_env = proc.environ()
                if proc_env.get('LLM_ENGINE_PROCESS') == '1':
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            # Check if it's a zombie - check parent chain
            if proc.status() == psutil.STATUS_ZOMBIE:
                try:
                    parent = proc.parent()
                    if parent:
                        parent_cmd = ' '.join(parent.cmdline())
                        if 'LLM-Agent-System' in parent_cmd or 'start_console' in parent_cmd:
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                return False

            # Fallback: check working directory
            try:
                cwd = proc.cwd()
                if 'LLM-Agent-System' in cwd:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            # Fallback: walk up the process tree
            try:
                parent = proc.parent()
                while parent and parent.pid > 1:
                    parent_cmd = ' '.join(parent.cmdline())
                    if 'LLM-Agent-System' in parent_cmd or 'start_console' in parent_cmd:
                        return True
                    if any(tool in parent_cmd for tool in [
                        'text-generation-webui', 'one_click.py', 'server.py --listen',
                        'SillyTavern', 'koboldcpp', 'ollama'
                    ]):
                        return False
                    parent = parent.parent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            # True orphan (parent is init) - likely from crashed LLM-Agent-System
            try:
                if proc.ppid() == 1:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    try:
        # Find llama-server processes
        result = subprocess.run(
            ['pgrep', '-f', 'llama-server'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            killed = 0

            for pid in pids:
                try:
                    pid_int = int(pid.strip())
                    if _is_llm_engine_process(pid_int):
                        os.kill(pid_int, 9)  # SIGKILL
                        console.print(f"   [green]Killed orphaned LLM-Agent-System llama-server (PID {pid_int})[/green]")
                        killed += 1
                except (ValueError, ProcessLookupError, PermissionError):
                    pass

            if killed > 0:
                import time
                time.sleep(1)
                console.print(f"[green]Cleaned up {killed} orphaned LLM-Agent-System process(es)[/green]")

    except (FileNotFoundError, ImportError):
        # psutil not available or pgrep not found - skip cleanup silently
        pass
    except Exception as e:
        console.print(f"[dim]Note: Could not check for orphaned processes: {e}[/dim]")


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('console.log'),
            logging.StreamHandler() if verbose else logging.NullHandler()
        ]
    )

    # Quiet some noisy loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLM-Agent-System Console - Unified Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_console.py                     # Start normally
    python start_console.py --ephemeral         # Don't persist sessions
    python start_console.py --project my-project  # Start with project
    python start_console.py -v                  # Verbose logging
        """
    )

    parser.add_argument(
        '--ephemeral', '-e',
        action='store_true',
        help='Start in ephemeral mode (sessions not persisted)'
    )

    parser.add_argument(
        '--project', '-p',
        type=str,
        default=None,
        help='Project to start with'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Skip orphaned process cleanup'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Cleanup orphaned processes
    if not args.no_cleanup:
        cleanup_rogue_llama_servers()

    # Import after path setup
    from console import Console, ConsoleConfig

    # Create config
    config = ConsoleConfig(
        default_ephemeral=args.ephemeral
    )

    # Initialize model manager
    model_manager = None
    try:
        import yaml
        from core.model_manager import LazyModelManager

        # Load models config
        models_config_path = project_root / 'config' / 'models.yaml'
        if models_config_path.exists():
            with open(models_config_path, 'r') as f:
                models_config = yaml.safe_load(f)

            print("Initializing model manager...")
            model_manager = LazyModelManager(
                models_config=models_config.get('models', {})
            )
            print("Model manager ready")
        else:
            print(f"Note: Config not found at {models_config_path}")
            print("Running without LLM support")

    except ImportError as e:
        print(f"Note: Model manager not available ({e})")
        print("Running in UI-only mode - agents will not have LLM support")
    except Exception as e:
        print(f"Warning: Could not initialize model manager: {e}")
        print("Running without LLM support - agents will not be available")

    # Flush stdout and add blank line before Rich console starts
    # This prevents output corruption when mixing print() with Rich
    sys.stdout.flush()
    print()

    # Create and run Console
    console = Console(
        model_manager=model_manager,
        config=config
    )

    # Switch to project if specified
    if args.project:
        console.switch_project(args.project)

    # Run main loop
    try:
        console.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logging.error(f"Console error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
