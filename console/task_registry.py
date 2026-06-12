"""
Task Registry - Background Task Tracking and Notifications

Implementation

Tracks background tasks spawned via TmuxManager and provides
status updates, notifications, and introspection capabilities.
"""

import json
import logging
import os
import time
import threading
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from console.tmux_manager import TmuxManager, BackgroundTask

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class TrackedTask:
    """A tracked background task with full metadata"""
    task_id: str
    name: str
    description: str
    agent: str
    command: str
    session_name: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0  # 0-100
    progress_message: str = ""
    result: Optional[str] = None
    error: Optional[str] = None
    notifications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackedTask':
        """Create from dictionary"""
        data['status'] = TaskStatus(data.get('status', 'pending'))
        data['priority'] = TaskPriority(data.get('priority', 'normal'))
        return cls(**data)


class TaskRegistry:
    """
    Registry for tracking background tasks

    Features:
    - Task lifecycle management (create, track, complete)
    - Status polling from tmux sessions
    - Notification queue for Console Hub
    - Persistence across restarts

    Usage:
        registry = TaskRegistry()

        # Spawn a background task
        task = registry.spawn_task(
            name="analyze-binary",
            description="Analyzing malware sample",
            agent="coder",
            command="python analyze.py sample.exe"
        )

        # Check status
        status = registry.get_task_status(task.task_id)

        # Get notifications
        notifications = registry.pop_notifications()
    """

    REGISTRY_FILE = ".llm-console-tasks.json"
    POLL_INTERVAL = 5  # seconds

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize task registry

        Args:
            data_dir: Directory for persistence file
        """
        self.data_dir = data_dir or os.path.expanduser("~/.llm-agent-system")
        os.makedirs(self.data_dir, exist_ok=True)

        self.registry_path = os.path.join(self.data_dir, self.REGISTRY_FILE)
        self.tmux = TmuxManager()

        self._tasks: Dict[str, TrackedTask] = {}
        self._notification_queue: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[TrackedTask], None]] = []
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None

        self._load_registry()

    # =========================================================================
    # Task Creation
    # =========================================================================

    def spawn_task(
        self,
        name: str,
        description: str,
        agent: str,
        command: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        working_dir: Optional[str] = None
    ) -> TrackedTask:
        """
        Spawn a new background task

        Args:
            name: Short task name
            description: Human-readable description
            agent: Agent running the task (oracle, operator, coder)
            command: Command to execute
            priority: Task priority
            working_dir: Working directory

        Returns:
            TrackedTask object
        """
        # Spawn via TmuxManager
        bg_task = self.tmux.spawn_task_session(
            name=name,
            command=command,
            agent=agent,
            working_dir=working_dir
        )

        # Create tracked task
        task = TrackedTask(
            task_id=bg_task.task_id,
            name=name,
            description=description,
            agent=agent,
            command=command,
            session_name=bg_task.session_name,
            status=TaskStatus.RUNNING,
            priority=priority,
            started_at=datetime.now().isoformat()
        )

        self._tasks[task.task_id] = task
        self._save_registry()

        # Queue notification
        self._queue_notification(
            task_id=task.task_id,
            type="started",
            message=f"Started: {description}"
        )

        logger.info(f"Spawned task: {name} ({task.task_id})")
        return task

    def register_existing_task(
        self,
        task_id: str,
        name: str,
        description: str,
        agent: str,
        session_name: str = "",
        pane_id: str = "",
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> TrackedTask:
        """
        Register an already-running task (e.g., agent pane)

        This allows tracking tasks that were spawned outside of spawn_task(),
        such as agent chat sessions spawned via tmux split-window.

        Args:
            task_id: Unique task identifier
            name: Short task name
            description: Human-readable description
            agent: Agent running the task
            session_name: tmux session name (optional)
            pane_id: tmux pane ID (for pane-based tasks)
            priority: Task priority

        Returns:
            TrackedTask object
        """
        task = TrackedTask(
            task_id=task_id,
            name=name,
            description=description,
            agent=agent,
            command="",  # No command for externally spawned tasks
            session_name=session_name or pane_id,  # Use pane_id as fallback
            status=TaskStatus.RUNNING,
            priority=priority,
            started_at=datetime.now().isoformat()
        )

        self._tasks[task.task_id] = task
        self._save_registry()

        # Queue notification
        self._queue_notification(
            task_id=task.task_id,
            type="started",
            message=f"Started: {description}"
        )

        logger.info(f"Registered existing task: {name} ({task.task_id})")
        return task

    # =========================================================================
    # Task Status
    # =========================================================================

    def get_task(self, task_id: str) -> Optional[TrackedTask]:
        """Get task by ID"""
        return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed task status

        Args:
            task_id: Task ID

        Returns:
            Status dict with task info and recent output
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        # Get recent output from tmux
        output = ""
        if task.status == TaskStatus.RUNNING:
            output = self.tmux.capture_pane_output(task.session_name, lines=20)

        return {
            **task.to_dict(),
            'recent_output': output,
            'runtime': self._calculate_runtime(task)
        }

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        agent: Optional[str] = None
    ) -> List[TrackedTask]:
        """
        List tasks with optional filtering

        Args:
            status: Filter by status
            agent: Filter by agent

        Returns:
            List of matching tasks
        """
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        if agent:
            tasks = [t for t in tasks if t.agent == agent]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def get_running_tasks(self) -> List[TrackedTask]:
        """Get all currently running tasks"""
        return self.list_tasks(status=TaskStatus.RUNNING)

    # =========================================================================
    # Task Control
    # =========================================================================

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancelled successfully
        """
        task = self._tasks.get(task_id)
        if not task:
            logger.warning(f"Task not found: {task_id}")
            return False

        if task.status != TaskStatus.RUNNING:
            logger.warning(f"Task not running: {task_id} ({task.status})")
            return False

        # Kill the tmux session
        self.tmux.kill_session(task.session_name)

        # Update task status
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now().isoformat()
        self._save_registry()

        # Queue notification
        self._queue_notification(
            task_id=task_id,
            type="cancelled",
            message=f"Cancelled: {task.name}"
        )

        logger.info(f"Cancelled task: {task_id}")
        return True

    def mark_completed(
        self,
        task_id: str,
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Mark a task as completed

        Args:
            task_id: Task ID
            result: Result summary
            error: Error message if failed
        """
        task = self._tasks.get(task_id)
        if not task:
            return

        if error:
            task.status = TaskStatus.FAILED
            task.error = error
        else:
            task.status = TaskStatus.COMPLETED
            task.result = result

        task.completed_at = datetime.now().isoformat()
        task.progress = 100
        self._save_registry()

        # Queue notification
        self._queue_notification(
            task_id=task_id,
            type="completed" if not error else "failed",
            message=f"{'Completed' if not error else 'Failed'}: {task.name}"
        )

    def update_progress(
        self,
        task_id: str,
        progress: int,
        message: str = ""
    ):
        """
        Update task progress

        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            message: Progress message
        """
        task = self._tasks.get(task_id)
        if not task:
            return

        task.progress = min(100, max(0, progress))
        task.progress_message = message
        self._save_registry()

        # Queue progress notification (throttled)
        if progress % 25 == 0:  # Only notify at 25%, 50%, 75%
            self._queue_notification(
                task_id=task_id,
                type="progress",
                message=f"{task.name}: {progress}% - {message}"
            )

    # =========================================================================
    # Notifications
    # =========================================================================

    def pop_notifications(self) -> List[Dict[str, Any]]:
        """
        Get and clear pending notifications

        Returns:
            List of notification dicts
        """
        notifications = self._notification_queue.copy()
        self._notification_queue.clear()
        return notifications

    def peek_notifications(self) -> List[Dict[str, Any]]:
        """Get pending notifications without clearing"""
        return self._notification_queue.copy()

    def has_notifications(self) -> bool:
        """Check if there are pending notifications"""
        return len(self._notification_queue) > 0

    def add_callback(self, callback: Callable[[TrackedTask], None]):
        """
        Add a callback for task status changes

        Args:
            callback: Function called with updated task
        """
        self._callbacks.append(callback)

    def _queue_notification(
        self,
        task_id: str,
        type: str,
        message: str
    ):
        """Queue a notification"""
        self._notification_queue.append({
            'task_id': task_id,
            'type': type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    # =========================================================================
    # Status Polling
    # =========================================================================

    def start_polling(self):
        """Start background status polling"""
        if self._polling:
            return

        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info("Started task status polling")

    def stop_polling(self):
        """Stop background status polling"""
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2)
        logger.info("Stopped task status polling")

    def _poll_loop(self):
        """Background polling loop"""
        while self._polling:
            try:
                self._check_task_statuses()
            except Exception as e:
                logger.error(f"Polling error: {e}")

            time.sleep(self.POLL_INTERVAL)

    def _check_task_statuses(self):
        """Check status of all running tasks"""
        running_tasks = self.get_running_tasks()

        for task in running_tasks:
            try:
                # Check if session still exists
                if not self.tmux.session_exists(task.session_name):
                    # Session ended - capture final output
                    self.mark_completed(task.task_id, result="Task session ended")
                    continue

                # Capture recent output for status detection
                output = self.tmux.capture_pane_output(task.session_name, lines=10)

                if output:
                    output_lower = output.lower()

                    # Detect completion patterns
                    if any(x in output_lower for x in ['complete', 'done', 'finished', 'success']):
                        self.mark_completed(task.task_id, result=output[-500:])

                    # Detect error patterns
                    elif any(x in output_lower for x in ['error:', 'failed:', 'exception:', 'traceback']):
                        self.mark_completed(task.task_id, error=output[-500:])

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(task)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"Error checking task {task.task_id}: {e}")

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save_registry(self):
        """Save registry to disk"""
        try:
            data = {
                'tasks': {tid: task.to_dict() for tid, task in self._tasks.items()},
                'saved_at': datetime.now().isoformat()
            }

            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def _load_registry(self):
        """Load registry from disk"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)

                for tid, task_data in data.get('tasks', {}).items():
                    try:
                        self._tasks[tid] = TrackedTask.from_dict(task_data)
                    except Exception as e:
                        logger.warning(f"Failed to load task {tid}: {e}")

                logger.info(f"Loaded {len(self._tasks)} tasks from registry")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        Remove old completed/failed tasks

        Args:
            max_age_hours: Maximum age in hours
        """
        now = datetime.now()
        to_remove = []

        for tid, task in self._tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                try:
                    completed = datetime.fromisoformat(task.completed_at)
                    age_hours = (now - completed).total_seconds() / 3600

                    if age_hours > max_age_hours:
                        to_remove.append(tid)
                except (ValueError, TypeError):
                    pass

        for tid in to_remove:
            del self._tasks[tid]

        if to_remove:
            self._save_registry()
            logger.info(f"Cleaned up {len(to_remove)} old tasks")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _calculate_runtime(self, task: TrackedTask) -> str:
        """Calculate task runtime as human-readable string"""
        try:
            start = datetime.fromisoformat(task.started_at) if task.started_at else None
            end = datetime.fromisoformat(task.completed_at) if task.completed_at else datetime.now()

            if start:
                delta = end - start
                seconds = int(delta.total_seconds())

                if seconds < 60:
                    return f"{seconds}s"
                elif seconds < 3600:
                    return f"{seconds // 60}m {seconds % 60}s"
                else:
                    hours = seconds // 3600
                    minutes = (seconds % 3600) // 60
                    return f"{hours}h {minutes}m"
        except (ValueError, TypeError):
            pass

        return "unknown"

    def get_summary(self) -> Dict[str, Any]:
        """
        Get registry summary statistics

        Returns:
            Summary dict with counts and status
        """
        tasks = list(self._tasks.values())

        return {
            'total': len(tasks),
            'running': len([t for t in tasks if t.status == TaskStatus.RUNNING]),
            'completed': len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            'failed': len([t for t in tasks if t.status == TaskStatus.FAILED]),
            'cancelled': len([t for t in tasks if t.status == TaskStatus.CANCELLED]),
            'pending_notifications': len(self._notification_queue)
        }
