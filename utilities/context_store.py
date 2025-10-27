#!/usr/bin/env python3
"""
Context Store Utility

SQLite-based storage for summaries and chunks.
Enables caching and retrieval of processed summaries.
"""
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import sqlite3
import logging

class ContextStore:
    """
    SQLite storage for summaries
    
    Features:
    - Store complete file summaries
    - Cache chunk summaries
    - Hash-based change detection
    - Query existing summaries
    
    TODO (Phase 2):
    - Vector embeddings for semantic search
    - ChromaDB integration
    - Cross-reference between summaries
    """
    
    def __init__(self, db_path: str = "summaries/summary_cache.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("ContextStore")
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"✅ Context store initialized: {db_path}")
    
    def _init_database(self):
        """Create database schema if not exists"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_insights TEXT,
                    chunk_count INTEGER DEFAULT 1,
                    total_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Chunks table (for detailed chunk storage)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_id INTEGER NOT NULL,
                    chunk_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (summary_id) REFERENCES summaries(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path 
                ON summaries(file_path)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash 
                ON summaries(file_hash)
            """)
            
            conn.commit()
    
    def store_summary(self, file_path: str, file_hash: str, summary: str,
                    key_insights: Any = None,  # Changed from List[str] to Any
                    chunk_count: int = 1,
                    total_tokens: int = 0) -> int:
        """
        Store a file summary - FIXED to handle dict/list
        
        Args:
            file_path: Path to the file
            file_hash: MD5 hash of file content
            summary: Generated summary text
            key_insights: Key insights (can be list, dict, or string)
            chunk_count: Number of chunks processed
            total_tokens: Total tokens in file
            
        Returns:
            Summary ID
        """
        
        # ✅ FIX: Convert key_insights to JSON if needed
        if key_insights is not None:
            if isinstance(key_insights, (list, dict)):
                insights_json = json.dumps(key_insights)
            else:
                insights_json = str(key_insights)
        else:
            insights_json = None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert or replace
            cursor.execute("""
                INSERT OR REPLACE INTO summaries 
                (file_path, file_hash, summary, key_insights, chunk_count, total_tokens, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_path, file_hash, summary, insights_json, chunk_count, 
                total_tokens, datetime.now()))
            
            summary_id = cursor.lastrowid
            conn.commit()
            
            self.logger.info(f"💾 Stored summary for {file_path} (ID: {summary_id})")
            return summary_id
        
    def store_chunks(self, summary_id: int, chunks: List[Dict[str, Any]]):
        """
        Store chunk summaries
        
        Args:
            summary_id: Parent summary ID
            chunks: List of chunk dictionaries
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO chunks 
                    (summary_id, chunk_number, content, summary, token_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    summary_id,
                    chunk.get('chunk_number', 0),
                    chunk.get('content', ''),
                    chunk.get('summary', ''),
                    chunk.get('token_count', 0)
                ))
            
            conn.commit()
            self.logger.info(f"💾 Stored {len(chunks)} chunks for summary {summary_id}")
    
    def get_summary(self, file_path: str, file_hash: str) -> Optional[Dict]:
        """
        Retrieve cached summary - FIXED to parse JSON
        
        Args:
            file_path: Path to the file
            file_hash: MD5 hash of file content
            
        Returns:
            Dict with summary data or None if not found
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, summary, key_insights, chunk_count, total_tokens, 
                    created_at, updated_at
                FROM summaries
                WHERE file_path = ? AND file_hash = ?
            """, (file_path, file_hash))
            
            row = cursor.fetchone()
            
            if row:
                # ✅ FIX: Parse key_insights from JSON
                key_insights_raw = row[2]
                try:
                    if key_insights_raw:
                        key_insights = json.loads(key_insights_raw)
                    else:
                        key_insights = []
                except (json.JSONDecodeError, TypeError):
                    # If not valid JSON, keep as string/split by newlines
                    if key_insights_raw:
                        key_insights = [line.strip() for line in key_insights_raw.split('\n') if line.strip()]
                    else:
                        key_insights = []
                
                return {
                    'id': row[0],
                    'summary': row[1],
                    'key_insights': key_insights,
                    'chunk_count': row[3],
                    'total_tokens': row[4],
                    'created_at': row[5],
                    'updated_at': row[6]
                }
            
            return None
    
    def get_chunks(self, summary_id: int) -> List[Dict[str, Any]]:
        """
        Get all chunks for a summary
        
        Args:
            summary_id: Summary ID
            
        Returns:
            List of chunk dicts
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM chunks 
                WHERE summary_id = ?
                ORDER BY chunk_number
            """, (summary_id,))
            
            chunks = [dict(row) for row in cursor.fetchall()]
            
            self.logger.info(f"Retrieved {len(chunks)} chunks for summary {summary_id}")
            return chunks
    
    def delete_summary(self, file_path: str):
        """
        Delete a summary and its chunks
        
        Args:
            file_path: Path to file
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM summaries WHERE file_path = ?
            """, (file_path,))
            
            conn.commit()
            self.logger.info(f"🗑️ Deleted summary for {file_path}")
    
    def list_summaries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List recent summaries
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of summary dicts
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, chunk_count, total_tokens, created_at, updated_at
                FROM summaries
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))
            
            summaries = [dict(row) for row in cursor.fetchall()]
            return summaries
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Stats dict with counts and totals
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count summaries
            cursor.execute("SELECT COUNT(*) FROM summaries")
            total_summaries = cursor.fetchone()[0]
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            # Total tokens processed
            cursor.execute("SELECT SUM(total_tokens) FROM summaries")
            total_tokens = cursor.fetchone()[0] or 0
            
            # Database size
            db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            
            return {
                'total_summaries': total_summaries,
                'total_chunks': total_chunks,
                'total_tokens_processed': total_tokens,
                'db_size_mb': db_size / (1024 * 1024),
                'db_path': self.db_path
            }
    
    def cleanup_old_summaries(self, days: int = 30):
        """
        Delete summaries older than specified days
        
        Args:
            days: Age threshold in days
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM summaries 
                WHERE updated_at < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            deleted = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"🧹 Deleted {deleted} old summaries (>{days} days)")
            return deleted