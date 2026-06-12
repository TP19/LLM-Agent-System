"""
Session Controller - Console Session Management

Manages persistent and ephemeral sessions for the Console hub.
"""

import sqlite3
import json
import logging
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


logger = logging.getLogger(__name__)


@dataclass
class ConsoleSession:
    """Console session state"""
    session_id: str
    name: str
    project_id: Optional[str]
    mode: str  # 'persistent' or 'ephemeral'
    created_at: str
    last_accessed: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    active_tasks: List[str] = field(default_factory=list)
    message_count: int = 0

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the session"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(msg)
        self.message_count = len(self.messages)
        self.last_accessed = datetime.now().isoformat()


class SessionController:
    """
    Manages Console sessions

    Features:
    - Persistent sessions (saved to SQLite)
    - Ephemeral sessions (memory only)
    - Smart session naming
    - Project association
    """

    def __init__(self, db_path: str = "~/.llm_engine/console/sessions.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory sessions (for ephemeral mode)
        self._ephemeral_sessions: Dict[str, ConsoleSession] = {}

        # Session name counters for smart naming
        self._name_counters: Dict[str, int] = {}

        # Initialize database
        self._init_database()

        logger.info(f"Session controller initialized: {self.db_path}")

    def _init_database(self):
        """Create database schema if not exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS console_sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    project_id TEXT,
                    mode TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    messages TEXT,
                    context TEXT,
                    active_tasks TEXT,
                    message_count INTEGER DEFAULT 0
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_project
                ON console_sessions(project_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_accessed
                ON console_sessions(last_accessed)
            """)

            conn.commit()

    def create_session(
        self,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        ephemeral: bool = False,
        initial_context: Optional[Dict] = None
    ) -> ConsoleSession:
        """
        Create a new session

        Args:
            name: Session name (auto-generated if None)
            project_id: Associated project ID
            ephemeral: If True, session is not persisted
            initial_context: Initial context data

        Returns:
            New ConsoleSession
        """
        session_id = str(uuid.uuid4())[:4]  # Shorter 4-char IDs
        mode = "ephemeral" if ephemeral else "persistent"

        # Generate smart name if not provided
        if not name:
            name = self._generate_session_name(project_id)

        session = ConsoleSession(
            session_id=session_id,
            name=name,
            project_id=project_id,
            mode=mode,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            messages=[],
            context=initial_context or {},
            active_tasks=[],
            message_count=0
        )

        if ephemeral:
            self._ephemeral_sessions[session_id] = session
        else:
            self._save_session(session)

        logger.info(f"Created session: {name} ({session_id}) [{mode}]")
        return session

    def _session_name_exists(self, name: str) -> bool:
        """Check if session name already exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM console_sessions WHERE name = ?", (name,))
            return cursor.fetchone() is not None

    def _generate_session_name(self, project_id: Optional[str] = None) -> str:
        """Generate a smart session name (only add counter if name exists)"""
        base = project_id[:6] if project_id else "session"

        # First, try without counter
        if not self._session_name_exists(base):
            return base

        # Name exists, add counter
        counter = self._name_counters.get(base, 0) + 1
        self._name_counters[base] = counter

        return f"{base}-{counter}"

    def generate_name_from_message(self, message: str, project_id: Optional[str] = None) -> str:
        """
        Generate session name from first user message

        Examples:
            "analyze the binary" -> "analyze-binary" (or "analyze-binary-1" if exists)
            "debug these errors" -> "debug-errors"
        """
        # Extract key words
        words = message.lower().split()[:3]

        # Filter out common words
        stop_words = {'the', 'a', 'an', 'to', 'for', 'with', 'this', 'these', 'those', 'and', 'or'}
        key_words = [w for w in words if w not in stop_words and len(w) > 2]

        if key_words:
            # Clean and join
            base = '-'.join(re.sub(r'[^a-z0-9]', '', w) for w in key_words[:2])
        else:
            base = project_id[:6] if project_id else "session"

        # Only add counter if name already exists
        if not self._session_name_exists(base):
            return base

        counter = self._name_counters.get(base, 0) + 1
        self._name_counters[base] = counter

        return f"{base}-{counter}"

    def get_session(self, session_id: str) -> Optional[ConsoleSession]:
        """Get session by ID"""
        # Check ephemeral first
        if session_id in self._ephemeral_sessions:
            return self._ephemeral_sessions[session_id]

        # Check database
        return self._load_session(session_id)

    def _save_session(self, session: ConsoleSession):
        """Save session to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO console_sessions (
                    session_id, name, project_id, mode,
                    created_at, last_accessed,
                    messages, context, active_tasks, message_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.name,
                session.project_id,
                session.mode,
                session.created_at,
                session.last_accessed,
                json.dumps(session.messages),
                json.dumps(session.context),
                json.dumps(session.active_tasks),
                session.message_count
            ))

            conn.commit()

    def _load_session(self, session_id: str) -> Optional[ConsoleSession]:
        """Load session from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM console_sessions WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()

            if not row:
                return None

            data = dict(row)

            return ConsoleSession(
                session_id=data['session_id'],
                name=data['name'],
                project_id=data['project_id'],
                mode=data['mode'],
                created_at=data['created_at'],
                last_accessed=data['last_accessed'],
                messages=json.loads(data['messages'] or '[]'),
                context=json.loads(data['context'] or '{}'),
                active_tasks=json.loads(data['active_tasks'] or '[]'),
                message_count=data['message_count']
            )

    def update_session(self, session: ConsoleSession):
        """Update session (save if persistent)"""
        session.last_accessed = datetime.now().isoformat()

        if session.mode == "ephemeral":
            self._ephemeral_sessions[session.session_id] = session
        else:
            self._save_session(session)

    def list_sessions(
        self,
        project_id: Optional[str] = None,
        limit: int = 20,
        include_ephemeral: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List sessions

        Args:
            project_id: Filter by project
            limit: Max sessions to return
            include_ephemeral: Include ephemeral sessions

        Returns:
            List of session info dicts
        """
        sessions = []

        # Add ephemeral sessions
        if include_ephemeral:
            for session in self._ephemeral_sessions.values():
                if project_id and session.project_id != project_id:
                    continue
                sessions.append({
                    "session_id": session.session_id,
                    "name": session.name,
                    "mode": session.mode,
                    "message_count": session.message_count,
                    "last_accessed": session.last_accessed
                })

        # Add persistent sessions
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if project_id:
                cursor.execute("""
                    SELECT session_id, name, mode, message_count, last_accessed
                    FROM console_sessions
                    WHERE project_id = ?
                    ORDER BY last_accessed DESC
                    LIMIT ?
                """, (project_id, limit))
            else:
                cursor.execute("""
                    SELECT session_id, name, mode, message_count, last_accessed
                    FROM console_sessions
                    ORDER BY last_accessed DESC
                    LIMIT ?
                """, (limit,))

            for row in cursor.fetchall():
                sessions.append(dict(row))

        # Sort by last_accessed
        sessions.sort(key=lambda s: s['last_accessed'], reverse=True)

        return sessions[:limit]

    def delete_session(self, session_id: str):
        """Delete session"""
        # Remove from ephemeral
        if session_id in self._ephemeral_sessions:
            del self._ephemeral_sessions[session_id]
            return

        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM console_sessions WHERE session_id = ?", (session_id,))
            conn.commit()

        logger.info(f"Deleted session: {session_id}")

    def rename_session(self, session_id: str, new_name: str):
        """Rename a session"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.name = new_name
        self.update_session(session)

        logger.info(f"Renamed session {session_id} to: {new_name}")

    def add_task(self, session_id: str, task_id: str):
        """Add active task to session"""
        session = self.get_session(session_id)
        if session:
            if task_id not in session.active_tasks:
                session.active_tasks.append(task_id)
            self.update_session(session)

    def remove_task(self, session_id: str, task_id: str):
        """Remove task from session"""
        session = self.get_session(session_id)
        if session and task_id in session.active_tasks:
            session.active_tasks.remove(task_id)
            self.update_session(session)

    def cleanup_empty_sessions(self, keep_recent: int = 1) -> int:
        """
        Remove empty sessions (0 messages) except the most recent one

        Args:
            keep_recent: Number of empty sessions to keep (default 1)

        Returns:
            Number of sessions deleted
        """
        deleted = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Find empty sessions ordered by last_accessed (oldest first)
            cursor.execute("""
                SELECT session_id, name, last_accessed
                FROM console_sessions
                WHERE message_count = 0
                ORDER BY last_accessed ASC
            """)

            empty_sessions = cursor.fetchall()

            # Keep the most recent 'keep_recent' empty sessions
            to_delete = empty_sessions[:-keep_recent] if len(empty_sessions) > keep_recent else []

            for session_id, name, _ in to_delete:
                cursor.execute("DELETE FROM console_sessions WHERE session_id = ?", (session_id,))
                deleted += 1
                logger.info(f"Cleaned up empty session: {name} ({session_id})")

            conn.commit()

        return deleted

    def get_or_create_default_session(self, project_id: Optional[str] = None) -> ConsoleSession:
        """
        Get recent empty default session or create new one

        Prevents creation of many empty 'default' sessions by reusing recent ones
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Look for a recent empty session (created in last hour)
            cursor.execute("""
                SELECT session_id
                FROM console_sessions
                WHERE message_count = 0
                  AND name LIKE 'default%'
                  AND datetime(last_accessed) > datetime('now', '-1 hour')
                ORDER BY last_accessed DESC
                LIMIT 1
            """)

            row = cursor.fetchone()

            if row:
                session = self._load_session(row['session_id'])
                if session:
                    logger.info(f"Reusing recent empty session: {session.name} ({session.session_id})")
                    return session

        # Create new session
        return self.create_session(name="default", project_id=project_id)

    def cleanup_old_sessions(self, days: int = 7, exclude_current: Optional[str] = None) -> int:
        """
        Delete sessions older than N days

        Args:
            days: Delete sessions older than this many days
            exclude_current: Session ID to exclude from deletion

        Returns:
            Number of sessions deleted
        """
        deleted = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Find old sessions
            cursor.execute("""
                SELECT session_id, name, last_accessed
                FROM console_sessions
                WHERE datetime(last_accessed) < datetime('now', ? || ' days')
            """, (f"-{days}",))

            old_sessions = cursor.fetchall()

            for session_id, name, last_accessed in old_sessions:
                if session_id == exclude_current:
                    continue
                cursor.execute("DELETE FROM console_sessions WHERE session_id = ?", (session_id,))
                deleted += 1
                logger.info(f"Deleted old session: {name} ({session_id}) - last accessed: {last_accessed}")

            conn.commit()

        return deleted

    def find_session_by_name(self, query: str) -> List[Dict[str, Any]]:
        """
        Find sessions by name (fuzzy match)

        Args:
            query: Name pattern to search for

        Returns:
            List of matching sessions
        """
        matches = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Use LIKE for fuzzy matching
            cursor.execute("""
                SELECT session_id, name, mode, message_count, last_accessed
                FROM console_sessions
                WHERE name LIKE ?
                ORDER BY last_accessed DESC
                LIMIT 10
            """, (f"%{query}%",))

            for row in cursor.fetchall():
                matches.append(dict(row))

        # Also check ephemeral sessions
        for session in self._ephemeral_sessions.values():
            if query.lower() in session.name.lower():
                matches.append({
                    "session_id": session.session_id,
                    "name": session.name,
                    "mode": session.mode,
                    "message_count": session.message_count,
                    "last_accessed": session.last_accessed
                })

        return matches

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics for review

        Returns:
            Dict with session counts and suggestions
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total sessions
            cursor.execute("SELECT COUNT(*) FROM console_sessions")
            total = cursor.fetchone()[0]

            # Empty sessions
            cursor.execute("SELECT COUNT(*) FROM console_sessions WHERE message_count = 0")
            empty = cursor.fetchone()[0]

            # Sessions older than 7 days
            cursor.execute("""
                SELECT COUNT(*) FROM console_sessions
                WHERE datetime(last_accessed) < datetime('now', '-7 days')
            """)
            old = cursor.fetchone()[0]

            # Sessions older than 30 days
            cursor.execute("""
                SELECT COUNT(*) FROM console_sessions
                WHERE datetime(last_accessed) < datetime('now', '-30 days')
            """)
            very_old = cursor.fetchone()[0]

        return {
            "total": total,
            "empty": empty,
            "older_than_7_days": old,
            "older_than_30_days": very_old,
            "ephemeral": len(self._ephemeral_sessions),
            "suggestions": self._generate_cleanup_suggestions(total, empty, old, very_old)
        }

    def _generate_cleanup_suggestions(self, total: int, empty: int, old: int, very_old: int) -> List[str]:
        """Generate cleanup suggestions based on stats"""
        suggestions = []

        if empty > 5:
            suggestions.append(f"Consider running cleanup_empty_sessions() - {empty} empty sessions")
        if very_old > 0:
            suggestions.append(f"Consider running cleanup_old_sessions(30) - {very_old} sessions older than 30 days")
        elif old > 10:
            suggestions.append(f"Consider running cleanup_old_sessions(7) - {old} sessions older than 7 days")
        if total > 50:
            suggestions.append(f"You have {total} sessions. Consider archiving or cleaning up.")

        if not suggestions:
            suggestions.append("Session storage looks healthy!")

        return suggestions
