#!/usr/bin/env python3
"""
Interactive State Store

SQLite-based persistence for interactive sessions and checkpoints.
Handles session state, checkpoint history, and rollback functionality.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from interactive.modes import (
    InteractiveSession, 
    Checkpoint, 
    InteractionMode, 
    CheckpointFrequency
)


class StateStore:
    """
    SQLite storage for interactive sessions
    
    Manages persistence of:
    - Session metadata and state
    - Checkpoints for rollback
    - Execution history
    - User interactions
    """
    
    def __init__(self, db_path: str = "interactive/sessions.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger("StateStore")
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"✅ State store initialized: {db_path}")
    
    def _init_database(self):
        """Create database schema if not exists"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Mode configuration
                    mode TEXT NOT NULL,
                    checkpoint_frequency TEXT NOT NULL,
                    
                    -- Request
                    user_request TEXT NOT NULL,
                    original_request TEXT NOT NULL,
                    
                    -- Current state
                    current_stage TEXT,
                    current_agent TEXT,
                    current_cycle INTEGER DEFAULT 0,
                    is_complete BOOLEAN DEFAULT FALSE,
                    
                    -- Results (JSON)
                    accumulated_results TEXT,
                    execution_history TEXT,
                    
                    -- Performance
                    total_processing_time REAL DEFAULT 0.0,
                    total_tokens_used INTEGER DEFAULT 0,
                    
                    -- Completion
                    completion_reason TEXT,
                    error_message TEXT
                )
            """)
            
            # Checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    checkpoint_number INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Checkpoint metadata
                    checkpoint_type TEXT NOT NULL,
                    current_agent TEXT,
                    current_stage TEXT,
                    cycle_number INTEGER DEFAULT 0,
                    
                    -- State snapshot (JSON)
                    accumulated_results TEXT,
                    execution_history TEXT,
                    
                    -- Rollback support
                    agent_states BLOB,
                    can_rollback BOOLEAN DEFAULT TRUE,
                    parent_checkpoint_id TEXT,
                    
                    -- User context (JSON)
                    user_actions TEXT,
                    user_guidance TEXT,
                    
                    -- Performance
                    processing_time_so_far REAL DEFAULT 0.0,
                    tokens_used INTEGER DEFAULT 0,
                    
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_checkpoint_id) REFERENCES checkpoints(checkpoint_id)
                )
            """)
            
            # Execution history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    checkpoint_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Execution details
                    agent TEXT NOT NULL,
                    cycle_number INTEGER,
                    command TEXT,
                    status TEXT,
                    output TEXT,
                    exit_code INTEGER,
                    execution_time REAL,
                    
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id)
                )
            """)
            
            # User interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    checkpoint_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Interaction details
                    interaction_type TEXT NOT NULL,
                    user_input TEXT,
                    agent_response TEXT,
                    
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_checkpoints 
                ON checkpoints(session_id, checkpoint_number)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_executions 
                ON execution_history(session_id, cycle_number)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_interactions 
                ON user_interactions(session_id, timestamp)
            """)
            
            conn.commit()
    
    def save_session(self, session: InteractiveSession):
        """Save or update session"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (
                    session_id, created_at, updated_at,
                    mode, checkpoint_frequency,
                    user_request, original_request,
                    current_stage, current_agent, current_cycle, is_complete,
                    accumulated_results, execution_history,
                    total_processing_time, total_tokens_used,
                    completion_reason, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.created_at,
                datetime.now(),
                session.mode.value,
                session.checkpoint_frequency.value,
                session.user_request,
                session.original_request,
                session.current_stage,
                session.current_agent,
                session.current_cycle,
                session.is_complete,
                json.dumps(session.accumulated_results),
                json.dumps(session.execution_history),
                session.total_processing_time,
                session.total_tokens_used,
                session.completion_reason,
                session.error_message
            ))
            
            conn.commit()
            self.logger.debug(f"Saved session {session.session_id}")
    
    def load_session(self, session_id: str) -> Optional[InteractiveSession]:
        """Load session from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to dict
            data = dict(row)
            
            # Parse JSON fields
            data['accumulated_results'] = json.loads(data['accumulated_results'] or '{}')
            data['execution_history'] = json.loads(data['execution_history'] or '[]')
            
            # Convert enums
            data['mode'] = InteractionMode(data['mode'])
            data['checkpoint_frequency'] = CheckpointFrequency(data['checkpoint_frequency'])
            
            # Create session object
            session = InteractiveSession(
                session_id=data['session_id'],
                created_at=datetime.fromisoformat(data['created_at']),
                mode=data['mode'],
                checkpoint_frequency=data['checkpoint_frequency'],
                user_request=data['user_request'],
                original_request=data['original_request'],
                current_stage=data['current_stage'],
                current_agent=data['current_agent'],
                current_cycle=data['current_cycle'],
                is_complete=data['is_complete'],
                accumulated_results=data['accumulated_results'],
                execution_history=data['execution_history'],
                total_processing_time=data['total_processing_time'],
                total_tokens_used=data['total_tokens_used'],
                completion_reason=data['completion_reason'],
                error_message=data['error_message']
            )
            
            # Load checkpoints
            session.checkpoints = self.load_checkpoints(session_id)
            
            self.logger.debug(f"Loaded session {session_id}")
            return session
    
    def save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO checkpoints (
                    checkpoint_id, session_id, checkpoint_number, created_at,
                    checkpoint_type, current_agent, current_stage, cycle_number,
                    accumulated_results, execution_history,
                    agent_states, can_rollback, parent_checkpoint_id,
                    user_actions, user_guidance,
                    processing_time_so_far, tokens_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.checkpoint_id,
                checkpoint.session_id,
                checkpoint.checkpoint_number,
                checkpoint.timestamp,
                checkpoint.checkpoint_type,
                checkpoint.current_agent,
                checkpoint.current_stage,
                checkpoint.cycle_number,
                json.dumps(checkpoint.accumulated_results),
                json.dumps(checkpoint.execution_history),
                checkpoint.agent_states,
                checkpoint.can_rollback,
                checkpoint.parent_checkpoint_id,
                json.dumps(checkpoint.user_actions),
                checkpoint.user_guidance,
                checkpoint.processing_time_so_far,
                checkpoint.tokens_used
            ))
            
            conn.commit()
            self.logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load specific checkpoint"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM checkpoints WHERE checkpoint_id = ?
            """, (checkpoint_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            data = dict(row)
            
            # Parse JSON fields
            data['accumulated_results'] = json.loads(data['accumulated_results'] or '{}')
            data['execution_history'] = json.loads(data['execution_history'] or '[]')
            data['user_actions'] = json.loads(data['user_actions'] or '[]')
            
            # Convert timestamp
            data['timestamp'] = datetime.fromisoformat(data['created_at'])
            del data['created_at']
            
            return Checkpoint(**data)
    
    def load_checkpoints(self, session_id: str) -> List[Checkpoint]:
        """Load all checkpoints for a session"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM checkpoints 
                WHERE session_id = ?
                ORDER BY checkpoint_number
            """, (session_id,))
            
            checkpoints = []
            for row in cursor.fetchall():
                data = dict(row)
                
                # Parse JSON fields
                data['accumulated_results'] = json.loads(data['accumulated_results'] or '{}')
                data['execution_history'] = json.loads(data['execution_history'] or '[]')
                data['user_actions'] = json.loads(data['user_actions'] or '[]')
                
                # Convert timestamp
                data['timestamp'] = datetime.fromisoformat(data['created_at'])
                del data['created_at']
                
                checkpoints.append(Checkpoint(**data))
            
            return checkpoints
    
    def save_user_interaction(self, session_id: str, checkpoint_id: Optional[str],
                            interaction_type: str, user_input: str, 
                            agent_response: str = ""):
        """Record user interaction"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO user_interactions (
                    session_id, checkpoint_id, interaction_type,
                    user_input, agent_response
                ) VALUES (?, ?, ?, ?, ?)
            """, (session_id, checkpoint_id, interaction_type, user_input, agent_response))
            
            conn.commit()
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent sessions"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    session_id, created_at, mode, 
                    user_request, is_complete, completion_reason
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_session(self, session_id: str):
        """Delete session and all related data"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            self.logger.info(f"Deleted session {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            # Count checkpoints
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            total_checkpoints = cursor.fetchone()[0]
            
            # Count interactions
            cursor.execute("SELECT COUNT(*) FROM user_interactions")
            total_interactions = cursor.fetchone()[0]
            
            # Database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'total_sessions': total_sessions,
                'total_checkpoints': total_checkpoints,
                'total_interactions': total_interactions,
                'db_size_mb': db_size / (1024 * 1024),
                'db_path': str(self.db_path)
            }