#!/usr/bin/env python3
"""
Metadata Store - Document Hierarchy Management

Manages structured metadata for documents, chapters, and chunks.
Complements ChromaDB by providing fast metadata filtering.
"""

import sqlite3
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

class MetadataStore:
    """
    SQLite storage for document hierarchy and metadata
    
    Provides:
    - Document registry with metadata
    - Chapter/section hierarchy
    - Chunk metadata and references
    - Fast filtering capabilities
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / ".llm_engine" / "metadata.db")
        
        self.db_path = Path(db_path)
        self.logger = logging.getLogger("MetadataStore")
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"✅ Metadata store initialized: {db_path}")
    
    def _init_database(self):
        """Create database schema"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    author TEXT,
                    doc_type TEXT NOT NULL,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER,
                    total_chunks INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT
                )
            """)
            
            # Chapters table (for hierarchical documents)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chapter_num INTEGER,
                    chapter_title TEXT,
                    start_chunk_num INTEGER,
                    end_chunk_num INTEGER,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)
            
            # Chunks metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chapter_id TEXT,
                    chunk_number INTEGER NOT NULL,
                    position_in_chapter INTEGER,
                    vector_id TEXT,
                    token_count INTEGER,
                    has_code BOOLEAN DEFAULT 0,
                    has_math BOOLEAN DEFAULT 0,
                    has_tables BOOLEAN DEFAULT 0,
                    has_images BOOLEAN DEFAULT 0,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id) ON DELETE SET NULL
                )
            """)
            
            # Create indices for fast lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_author ON documents(author)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_title ON documents(title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(doc_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_path ON documents(file_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chapters_doc ON chapters(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks_metadata(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks_metadata(chapter_id)")
            
            conn.commit()
    
    # ================================================================
    # Document Operations
    # ================================================================
    
# Replace the add_document() method in utilities/metadata_store.py:

    def add_document(self, title: str, author: str, doc_type: str,
                    file_path: str, file_hash: str, doc_id: str = None, **kwargs) -> str:
        """
        Add a document to the registry
        
        Args:
            title: Document title
            author: Author name
            doc_type: Type (book, article, code, log, etc.)
            file_path: Path to source file
            file_hash: Content hash for change detection
            doc_id: Optional specific doc_id to use (if None, generates UUID)
            **kwargs: Additional metadata (file_size, language, etc.)
        
        Returns:
            doc_id: Unique document ID
        """
        
        # ✅ FIX: Use provided doc_id or generate new one
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Extract optional fields
        file_size = kwargs.get('file_size', 0)
        total_chunks = kwargs.get('total_chunks', 0)
        total_tokens = kwargs.get('total_tokens', 0)
        language = kwargs.get('language', None)
        
        # Store additional metadata as JSON
        extra_metadata = {k: v for k, v in kwargs.items() 
                        if k not in ['file_size', 'total_chunks', 'total_tokens', 'language']}
        metadata_json = json.dumps(extra_metadata) if extra_metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO documents 
                (doc_id, title, author, doc_type, file_path, file_hash,
                file_size, total_chunks, total_tokens, language, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, title, author, doc_type, file_path, file_hash,
                file_size, total_chunks, total_tokens, language, metadata_json))
            
            conn.commit()
        
        self.logger.info(f"Added document: {title} (ID: {doc_id})")
        return doc_id
    
    def get_document(self, doc_id: str = None, file_path: str = None) -> Optional[Dict]:
        """Get document by ID or file path"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if doc_id:
                cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            elif file_path:
                cursor.execute("SELECT * FROM documents WHERE file_path = ?", (file_path,))
            else:
                return None
            
            row = cursor.fetchone()
            
            if row:
                doc = dict(row)
                # Parse JSON metadata
                if doc.get('metadata_json'):
                    try:
                        doc['extra_metadata'] = json.loads(doc['metadata_json'])
                    except:
                        doc['extra_metadata'] = {}
                return doc
            
            return None
    
    def update_document(self, doc_id: str, **kwargs):
        """Update document metadata"""
        
        allowed_fields = ['title', 'author', 'total_chunks', 'total_tokens', 'language']
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return
        
        updates['updated_at'] = datetime.now()
        
        # Build UPDATE query
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [doc_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE documents SET {set_clause} WHERE doc_id = ?", values)
            conn.commit()
        
        self.logger.debug(f"Updated document {doc_id}")
    
    def list_documents(self, doc_type: str = None, author: str = None,
                      limit: int = 100) -> List[Dict]:
        """List documents with optional filters"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM documents WHERE 1=1"
            params = []
            
            if doc_type:
                query += " AND doc_type = ?"
                params.append(doc_type)
            
            if author:
                query += " AND author LIKE ?"
                params.append(f"%{author}%")
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # ================================================================
    # Chapter Operations
    # ================================================================
    
# Replace the add_chapter() method in utilities/metadata_store.py with this:

    def add_chapter(self, doc_id: str, chapter_num: int, chapter_title: str,
                    start_chunk_num: int = 0, end_chunk_num: int = 0,
                    summary: str = None) -> str:
        """
        Add a chapter to a document
        
        Args:
            doc_id: Parent document ID
            chapter_num: Chapter number (1, 2, 3, ...)
            chapter_title: Chapter title/name
            start_chunk_num: First chunk number in this chapter
            end_chunk_num: Last chunk number in this chapter
            summary: Optional chapter summary
        
        Returns:
            chapter_id: Unique chapter ID
        """
        import uuid
        
        chapter_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chapters 
                (chapter_id, doc_id, chapter_num, chapter_title, 
                start_chunk_num, end_chunk_num, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chapter_id, doc_id, chapter_num, chapter_title,
                start_chunk_num, end_chunk_num, summary))
            
            conn.commit()
        
        self.logger.debug(f"Added chapter {chapter_num}: {chapter_title}")
        return chapter_id
    
    def get_chapters(self, doc_id: str) -> list:
        """
        Get all chapters for a document
        
        Args:
            doc_id: Document ID
        
        Returns:
            List of chapter dicts with all fields
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM chapters
                WHERE doc_id = ?
                ORDER BY chapter_num
            """, (doc_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    # ================================================================
    # Chunk Metadata Operations
    # ================================================================
    
    def add_chunk_metadata(self, doc_id: str, chunk_number: int,
                        vector_id: str, token_count: int,
                        has_code: bool = False, has_math: bool = False,
                        language: str = 'en', chapter_id: str = None) -> str:
        """
        Add chunk metadata record
        
        This is CRITICAL for context expansion to work!
        
        Args:
            doc_id: Document ID from add_document()
            chunk_number: Sequential chunk number (0, 1, 2, ...)
            vector_id: The ID used in ChromaDB (chunk_id)
            token_count: Number of tokens in this chunk
            has_code: Whether chunk contains code blocks
            has_math: Whether chunk contains mathematical notation
            language: Language code (default 'en')
            chapter_id: Optional chapter ID if chunk belongs to a chapter
        
        Returns:
            chunk_id: Unique metadata ID for this chunk record
        """
        import uuid
        
        chunk_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chunks_metadata 
                (chunk_id, doc_id, chapter_id, chunk_number, vector_id,
                token_count, has_code, has_math, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (chunk_id, doc_id, chapter_id, chunk_number, vector_id,
                token_count, has_code, has_math, language))
            
            conn.commit()
        
        self.logger.debug(f"Added chunk metadata: {chunk_id} for chunk {chunk_number}")
        return chunk_id


    def get_chunk_metadata(self, doc_id: str, chunk_number: int = None) -> list:
        """
        Get chunk metadata for a document
        
        Args:
            doc_id: Document ID
            chunk_number: Optional specific chunk number
        
        Returns:
            List of chunk metadata dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if chunk_number is not None:
                cursor.execute("""
                    SELECT * FROM chunks_metadata
                    WHERE doc_id = ? AND chunk_number = ?
                """, (doc_id, chunk_number))
            else:
                cursor.execute("""
                    SELECT * FROM chunks_metadata
                    WHERE doc_id = ?
                    ORDER BY chunk_number
                """, (doc_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


    def get_surrounding_chunks(self, doc_id: str, chunk_number: int, window: int = 1) -> list:
        """
        Get surrounding chunks for context expansion
        
        Args:
            doc_id: Document ID
            chunk_number: Center chunk number
            window: How many chunks before/after (default 1 = ±1)
        
        Returns:
            List of chunk metadata dicts for chunks in range
        """
        start_num = max(0, chunk_number - window)
        end_num = chunk_number + window
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM chunks_metadata
                WHERE doc_id = ? 
                AND chunk_number BETWEEN ? AND ?
                ORDER BY chunk_number
            """, (doc_id, start_num, end_num))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


    def delete_document(self, doc_id: str):
        """
        Delete a document and all its metadata
        
        Use this to fix UNIQUE constraint errors when re-indexing
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete in order (foreign keys)
            cursor.execute("DELETE FROM chunks_metadata WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM chapters WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            
            conn.commit()
        
        self.logger.info(f"Deleted document: {doc_id}")


    def find_documents(self, file_path: str = None, **filters) -> list:
        """
        Find documents by file_path or other filters
        
        Args:
            file_path: Exact file path to match
            **filters: Other filters (author, title, doc_type, etc.)
        
        Returns:
            List of doc_ids matching the criteria
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if file_path:
                cursor.execute("""
                    SELECT doc_id FROM documents WHERE file_path = ?
                """, (file_path,))
            else:
                # Build dynamic query from filters
                conditions = []
                params = []
                
                for key, value in filters.items():
                    if key in ['author', 'title', 'doc_type', 'language']:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                
                if conditions:
                    query = f"SELECT doc_id FROM documents WHERE {' AND '.join(conditions)}"
                    cursor.execute(query, params)
                else:
                    return []
            
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    
    def add_chunk_metadata_to_store():
        """
        Helper to add the add_chunk_metadata method to MetadataStore
        
        Add this to utilities/metadata_store.py:
        """
        code = '''
        def add_chunk_metadata(self, doc_id: str, chunk_number: int,
                            vector_id: str, token_count: int,
                            has_code: bool = False, has_math: bool = False,
                            language: str = 'en', chapter_id: str = None) -> str:
            """
            Add chunk metadata record
            
            Args:
                doc_id: Document ID
                chunk_number: Chunk number in document
                vector_id: Reference to ChromaDB vector ID
                token_count: Number of tokens in chunk
                has_code: Whether chunk contains code
                has_math: Whether chunk contains math/LaTeX
                language: Language code
                chapter_id: Optional chapter reference
            
            Returns:
                chunk_id: Unique chunk ID
            """
            import uuid
            
            chunk_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO chunks_metadata 
                    (chunk_id, doc_id, chapter_id, chunk_number, vector_id,
                    token_count, has_code, has_math, language)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (chunk_id, doc_id, chapter_id, chunk_number, vector_id,
                    token_count, has_code, has_math, language))
                
                conn.commit()
            
            self.logger.debug(f"Added chunk metadata: {chunk_id}")
            return chunk_id
        '''
    
    # ================================================================
    # Query Operations
    # ================================================================
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chapters")
            chapter_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks_metadata")
            chunk_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(total_tokens) FROM documents")
            total_tokens = cursor.fetchone()[0] or 0
            
            return {
                'documents': doc_count,
                'chapters': chapter_count,
                'chunks': chunk_count,
                'total_tokens': total_tokens
            }
        