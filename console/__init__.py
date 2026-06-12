"""
Console - Unified LLM-Agent-System Interface

The Console provides a single entry point for all LLM-Agent-System interactions,
combining the functionality of Oracle Hub, Collaborative Mode, and Standard Mode.

Features:
- Oracle as the primary chat interface
- Project and session management
- Smart tmux integration for background tasks
- Direct agent invocation with context switching
- Chunk viewer integration

Usage:
    from console import Console

    console = Console(model_manager)
    console.run()

    # Or run directly:
    python start_console.py
"""

from .console_hub import Console, ConsoleConfig
from .command_router import CommandRouter, CommandDefinition, CommandError
from .console_ui import ConsoleUI
from .session_controller import SessionController, ConsoleSession
from .tmux_manager import TmuxManager, TmuxSession, BackgroundTask
from .task_duration_estimator import TaskDurationEstimator, TaskComplexity, ExecutionMode
from .agent_context import AgentContextManager, AgentConversation

__all__ = [
    'Console',
    'ConsoleConfig',
    'CommandRouter',
    'CommandDefinition',
    'CommandError',
    'ConsoleUI',
    'SessionController',
    'ConsoleSession',
    'TmuxManager',
    'TmuxSession',
    'BackgroundTask',
    'TaskDurationEstimator',
    'TaskComplexity',
    'ExecutionMode',
    'AgentContextManager',
    'AgentConversation',
]

__version__ = '1.0.0'
