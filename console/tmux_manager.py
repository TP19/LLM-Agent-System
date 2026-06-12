"""
Tmux Manager - Background Task Session Management

Manages tmux sessions for background tasks, agent panes, and viewers.
"""

import subprocess
import logging
import os
import time
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TmuxSession:
    """Represents a tmux session"""
    name: str
    session_id: str
    created_at: str
    windows: int
    attached: bool


@dataclass
class TmuxPane:
    """Represents a tmux pane"""
    pane_id: str
    session_name: str
    window_index: int
    pane_index: int
    active: bool
    command: str


@dataclass
class BackgroundTask:
    """Represents a background task running in tmux"""
    task_id: str
    name: str
    session_name: str
    agent: str
    command: str
    started_at: str
    status: str  # 'running', 'completed', 'failed'


class TmuxManager:
    """
    Manages tmux sessions for Console background tasks

    Features:
    - Spawn background task sessions
    - Create agent chat panes
    - Create viewer panes
    - Capture output from sessions
    - Monitor task status

    Usage:
        tmux = TmuxManager()

        # Spawn a background task
        task_id = tmux.spawn_task_session(
            name="analyze-binary",
            command="python run_security.py analyze target"
        )

        # List running tasks
        tasks = tmux.list_task_sessions()

        # Capture output
        output = tmux.capture_pane_output(task_id)
    """

    # Prefix for all Console tmux sessions
    SESSION_PREFIX = "llm-console"

    def __init__(self):
        self.logger = logging.getLogger("TmuxManager")
        self._check_tmux_available()

    def _check_tmux_available(self) -> bool:
        """Check if tmux is available"""
        try:
            result = subprocess.run(
                ['tmux', '-V'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.logger.info(f"tmux available: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            self.logger.warning("tmux not found in PATH")
        return False

    def _run_tmux(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a tmux command"""
        cmd = ['tmux'] + args
        self.logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if check and result.returncode != 0:
                self.logger.error(f"tmux error: {result.stderr}")
            return result
        except subprocess.TimeoutExpired:
            self.logger.error("tmux command timed out")
            raise
        except Exception as e:
            self.logger.error(f"tmux command failed: {e}")
            raise

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(self, name: str, command: Optional[str] = None) -> str:
        """
        Create a new tmux session

        Args:
            name: Session name
            command: Optional command to run in the session

        Returns:
            Full session name with prefix
        """
        session_name = f"{self.SESSION_PREFIX}-{name}"

        args = ['new-session', '-d', '-s', session_name]

        if command:
            args.extend([command])

        result = self._run_tmux(args, check=False)

        if result.returncode == 0:
            self.logger.info(f"Created tmux session: {session_name}")
            return session_name
        else:
            # Session might already exist
            if 'duplicate session' in result.stderr:
                self.logger.warning(f"Session already exists: {session_name}")
                return session_name
            raise RuntimeError(f"Failed to create session: {result.stderr}")

    def kill_session(self, session_name: str):
        """Kill a tmux session"""
        result = self._run_tmux(['kill-session', '-t', session_name], check=False)

        if result.returncode == 0:
            self.logger.info(f"Killed session: {session_name}")
        else:
            self.logger.warning(f"Could not kill session: {result.stderr}")

    def session_exists(self, session_name: str) -> bool:
        """Check if a session exists"""
        result = self._run_tmux(['has-session', '-t', session_name], check=False)
        return result.returncode == 0

    def list_sessions(self) -> List[TmuxSession]:
        """List all Console tmux sessions"""
        result = self._run_tmux([
            'list-sessions',
            '-F', '#{session_name}|#{session_id}|#{session_created}|#{session_windows}|#{session_attached}'
        ], check=False)

        if result.returncode != 0:
            return []

        sessions = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split('|')
            if len(parts) >= 5 and parts[0].startswith(self.SESSION_PREFIX):
                sessions.append(TmuxSession(
                    name=parts[0],
                    session_id=parts[1],
                    created_at=parts[2],
                    windows=int(parts[3]),
                    attached=parts[4] == '1'
                ))

        return sessions

    # =========================================================================
    # Task Session Management
    # =========================================================================

    def spawn_task_session(
        self,
        name: str,
        command: str,
        agent: str = "unknown",
        working_dir: Optional[str] = None
    ) -> BackgroundTask:
        """
        Spawn a background task in a new tmux session

        Args:
            name: Task name
            command: Command to execute
            agent: Agent running the task
            working_dir: Working directory for the command

        Returns:
            BackgroundTask object
        """
        import uuid
        task_id = str(uuid.uuid4())[:8]
        session_name = f"{self.SESSION_PREFIX}-task-{task_id}"

        # Build the command with working directory
        if working_dir:
            full_command = f"cd {working_dir} && {command}"
        else:
            full_command = command

        # Create session with command
        args = ['new-session', '-d', '-s', session_name, '-c', working_dir or os.getcwd()]

        result = self._run_tmux(args, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create task session: {result.stderr}")

        # Send the command to the session
        self._run_tmux(['send-keys', '-t', session_name, full_command, 'Enter'])

        task = BackgroundTask(
            task_id=task_id,
            name=name,
            session_name=session_name,
            agent=agent,
            command=command,
            started_at=datetime.now().isoformat(),
            status='running'
        )

        self.logger.info(f"Spawned task: {name} ({task_id}) in {session_name}")
        return task

    def list_task_sessions(self) -> List[Dict[str, Any]]:
        """List all background task sessions"""
        sessions = self.list_sessions()

        tasks = []
        for session in sessions:
            if '-task-' in session.name:
                # Extract task ID from session name
                parts = session.name.split('-task-')
                task_id = parts[1] if len(parts) > 1 else session.name

                # Get last line of output to check status
                output = self.capture_pane_output(session.name, lines=5)
                status = 'running'
                if output:
                    last_lines = output.strip().split('\n')
                    if last_lines:
                        last = last_lines[-1].lower()
                        if 'error' in last or 'failed' in last:
                            status = 'failed'
                        elif 'complete' in last or 'done' in last or 'finished' in last:
                            status = 'completed'

                tasks.append({
                    'task_id': task_id,
                    'name': session.name,
                    'session_name': session.name,
                    'started': session.created_at,
                    'status': status,
                    'agent': 'unknown'  # Could parse from session name
                })

        return tasks

    # =========================================================================
    # Pane Management
    # =========================================================================

    def spawn_agent_pane(
        self,
        agent_name: str,
        command: str,
        session_name: Optional[str] = None
    ) -> str:
        """
        Spawn a pane for agent interaction

        Args:
            agent_name: Name of the agent
            command: Command to run the agent
            session_name: Session to create pane in (uses current if None)

        Returns:
            Pane ID
        """
        if session_name is None:
            session_name = f"{self.SESSION_PREFIX}-agents"

        # Create session if it doesn't exist
        if not self.session_exists(session_name):
            self.create_session(session_name.replace(f"{self.SESSION_PREFIX}-", ""))

        # Split window to create new pane
        result = self._run_tmux([
            'split-window', '-h',
            '-t', session_name,
            '-P', '-F', '#{pane_id}'
        ], check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create pane: {result.stderr}")

        pane_id = result.stdout.strip()

        # Send command to the new pane
        self._run_tmux(['send-keys', '-t', pane_id, command, 'Enter'])

        self.logger.info(f"Spawned agent pane for {agent_name}: {pane_id}")
        return pane_id

    def spawn_viewer_pane(
        self,
        doc_id: Optional[str] = None,
        session_name: Optional[str] = None
    ) -> str:
        """
        Spawn a pane for the chunk viewer

        Args:
            doc_id: Document ID to view
            session_name: Session to create pane in

        Returns:
            Pane ID
        """
        if session_name is None:
            session_name = f"{self.SESSION_PREFIX}-viewer"

        # Build viewer command
        viewer_cmd = "python -m utilities.chunk_viewer"
        if doc_id:
            viewer_cmd += f" --doc {doc_id}"

        return self.spawn_agent_pane("viewer", viewer_cmd, session_name)

    def close_pane(self, pane_id: str):
        """Close a tmux pane"""
        result = self._run_tmux(['kill-pane', '-t', pane_id], check=False)

        if result.returncode == 0:
            self.logger.info(f"Closed pane: {pane_id}")
        else:
            self.logger.warning(f"Could not close pane: {result.stderr}")

    # =========================================================================
    # Output Capture
    # =========================================================================

    def capture_pane_output(
        self,
        target: str,
        lines: int = 100,
        start_line: Optional[int] = None
    ) -> str:
        """
        Capture output from a tmux pane

        Args:
            target: Session/pane target
            lines: Number of lines to capture
            start_line: Starting line (negative for history)

        Returns:
            Captured output text
        """
        args = ['capture-pane', '-t', target, '-p']

        if start_line is not None:
            args.extend(['-S', str(start_line)])

        args.extend(['-E', str(lines)])

        result = self._run_tmux(args, check=False)

        if result.returncode == 0:
            return result.stdout
        else:
            self.logger.warning(f"Could not capture output: {result.stderr}")
            return ""

    def send_keys(self, target: str, keys: str, enter: bool = True):
        """
        Send keys to a tmux pane

        Args:
            target: Session/pane target
            keys: Keys to send
            enter: Whether to send Enter after keys
        """
        args = ['send-keys', '-t', target, keys]

        if enter:
            args.append('Enter')

        self._run_tmux(args, check=False)

    # =========================================================================
    # Attach/Detach
    # =========================================================================

    def attach_session(self, session_name: str):
        """
        Attach to a tmux session (blocks until detached)

        Args:
            session_name: Session to attach to
        """
        # Use os.system for interactive attachment
        os.system(f"tmux attach-session -t {session_name}")

    def get_attach_command(self, session_name: str) -> str:
        """
        Get the command to attach to a session

        Args:
            session_name: Session name

        Returns:
            Command string
        """
        return f"tmux attach-session -t {session_name}"

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old Console tmux sessions"""
        sessions = self.list_sessions()

        for session in sessions:
            try:
                # Parse creation time
                created = datetime.fromtimestamp(int(session.created_at))
                age_hours = (datetime.now() - created).total_seconds() / 3600

                if age_hours > max_age_hours and not session.attached:
                    self.logger.info(f"Cleaning up old session: {session.name}")
                    self.kill_session(session.name)

            except (ValueError, TypeError):
                continue

    def cleanup_all_sessions(self):
        """Kill all Console tmux sessions"""
        sessions = self.list_sessions()

        for session in sessions:
            self.kill_session(session.name)

        self.logger.info(f"Cleaned up {len(sessions)} sessions")
