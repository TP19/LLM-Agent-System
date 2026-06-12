#!/usr/bin/env python3
"""
LanceDB storage implementation for LLM-Agent-System.

Provides persistent vector storage.
Implements BaseVectorStore interface for compatibility with existing code.

- Versioned datasets (time-travel and rollback capability)
- Zero-copy reads (columnar format, no serialization overhead)
- Resilient under memory pressure (no SQLite corruption)
- Better concurrent access performance
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

try:
    import lancedb
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    logging.warning("LanceDB not available. Install with: pip install lancedb pyarrow")

from .base_store import BaseVectorStore, MemoryEntry


class LanceDBStore(BaseVectorStore):
    """
    LanceDB-backed vector store implementation.

    Features:
    - Persistent storage with automatic recovery
    - Versioned datasets for rollback
    - Efficient vector similarity search
    - Zero-copy reads (Apache Arrow format)
    - Resilient under memory pressure

    Example:
        store = LanceDBStore(db_path="~/.llm_engine/lance_db/private")
        store.add_memories(memories, collection_name="semantic")
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, db_path: Optional[str] = None, collection_prefix: str = "llm_engine"):
        """
        Initialize LanceDB store.

        Args:
            db_path: Path for persistent storage (default: ~/.llm_engine/lance_db)
            collection_prefix: Prefix for collection/table names
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError(
                "LanceDB not installed. Install with: pip install lancedb pyarrow"
            )

        self.logger = logging.getLogger("LanceDBStore")

        # Set up persistence directory
        if db_path:
            persist_dir = Path(db_path).expanduser()
        else:
            persist_dir = Path.home() / ".llm_engine" / "lance_db"

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_directory = str(persist_dir)
        self.collection_prefix = collection_prefix

        # Initialize LanceDB connection
        try:
            self.db = lancedb.connect(self.persist_directory)
            self.logger.info(f"✅ LanceDB store initialized at {self.persist_directory}")
        except Exception as e:
            self.logger.error(f"Failed to connect to LanceDB: {e}")
            raise

        # Cache for opened tables
        self._tables = {}

    def _get_table_name(self, collection_name: str) -> str:
        """Get full table name with prefix."""
        return f"{self.collection_prefix}_{collection_name}"

    def _get_or_create_table(self, collection_name: str, embedding_dim: Optional[int] = None):
        """
        Get or create a LanceDB table for a collection.

        Args:
            collection_name: Collection name
            embedding_dim: Dimension of embeddings (required for new tables)

        Returns:
            LanceDB table object
        """
        table_name = self._get_table_name(collection_name)

        # Check cache - but verify dimension if provided
        if table_name in self._tables:
            if embedding_dim is not None:
                # Verify cached table has correct dimension
                cached_table = self._tables[table_name]
                try:
                    schema = cached_table.schema
                    vector_field = schema.field("vector")
                    if hasattr(vector_field.type, 'list_size'):
                        existing_dim = vector_field.type.list_size
                        if existing_dim != embedding_dim:
                            self.logger.warning(f"⚠️ Dimension mismatch in cached table {table_name}: {existing_dim} vs {embedding_dim}")
                            del self._tables[table_name]
                            # Fall through to recreate
                        else:
                            return cached_table
                except Exception:
                    pass  # If we can't check, just return cached
            return self._tables[table_name]

        # Check if table exists
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)

            # Check dimension compatibility if embedding_dim provided
            if embedding_dim is not None:
                try:
                    schema = table.schema
                    vector_field = schema.field("vector")
                    if hasattr(vector_field.type, 'list_size'):
                        existing_dim = vector_field.type.list_size
                        if existing_dim != embedding_dim:
                            self.logger.warning(
                                f"⚠️ Embedding dimension mismatch in {table_name}: "
                                f"table has {existing_dim}, new embeddings have {embedding_dim}. "
                                f"Recreating table..."
                            )
                            # Drop and recreate table
                            self.db.drop_table(table_name)
                            # Fall through to create new table
                        else:
                            self._tables[table_name] = table
                            return table
                    else:
                        self._tables[table_name] = table
                        return table
                except Exception as e:
                    self.logger.warning(f"Could not verify dimension for {table_name}: {e}")
                    self._tables[table_name] = table
                    return table
            else:
                self._tables[table_name] = table
                return table

        # Create new table
        if embedding_dim is None:
            raise ValueError(f"embedding_dim required to create new table: {table_name}")

        # Define schema
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
            pa.field("memory_type", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("importance_score", pa.float64()),
            pa.field("access_count", pa.int64()),
            pa.field("metadata", pa.string()),  # JSON-encoded
        ])

        # Create empty table
        table = self.db.create_table(
            table_name,
            schema=schema,
            mode="create"
        )

        self._tables[table_name] = table
        self.logger.info(f"✅ Created new table: {table_name} (dim: {embedding_dim})")
        return table

    def add_memories(self, memories: List[MemoryEntry],
                     collection_name: str = "default") -> None:
        """
        Store multiple memories in a collection.

        Args:
            memories: List of MemoryEntry objects
            collection_name: Target collection name
        """
        if not memories:
            return

        # Detect embedding dimension from first memory
        embedding_dim = len(memories[0].embedding)

        # Get or create table
        table = self._get_or_create_table(collection_name, embedding_dim)

        # Convert MemoryEntry objects to LanceDB records
        records = []
        for mem in memories:
            record = {
                "id": mem.id,
                "content": mem.content,
                "vector": mem.embedding,
                "memory_type": mem.memory_type,
                "timestamp": mem.timestamp,
                "importance_score": mem.importance_score,
                "access_count": mem.access_count,
                "metadata": json.dumps(mem.metadata) if mem.metadata else "{}"
            }
            records.append(record)

        # Add to table
        try:
            table.add(records)
            self.logger.info(f"✅ Added {len(memories)} memories to {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to add memories: {e}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5,
               collection_name: str = "default",
               filters: Optional[Dict] = None) -> List[MemoryEntry]:
        """
        Search for similar memories using vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            collection_name: Collection to search
            filters: Optional metadata filters (not fully implemented yet)

        Returns:
            List of MemoryEntry objects sorted by similarity
        """
        table_name = self._get_table_name(collection_name)

        # Check if table exists
        if table_name not in self.db.table_names():
            self.logger.warning(f"Table {table_name} not found")
            return []

        try:
            table = self.db.open_table(table_name)

            # Perform vector search
            results = table.search(query_embedding).limit(top_k).to_list()

            # Convert to MemoryEntry objects
            memories = []
            for result in results:
                # Parse metadata JSON
                metadata_str = result.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}

                memory = MemoryEntry(
                    id=result["id"],
                    content=result["content"],
                    embedding=result["vector"],
                    memory_type=result["memory_type"],
                    timestamp=result["timestamp"],
                    importance_score=result.get("importance_score", 0.5),
                    access_count=result.get("access_count", 0),
                    metadata=metadata
                )
                memories.append(memory)

            self.logger.info(f"✅ Found {len(memories)} results in {collection_name}")
            return memories

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and all its data.

        Args:
            collection_name: Collection to delete
        """
        table_name = self._get_table_name(collection_name)

        if table_name not in self.db.table_names():
            self.logger.warning(f"Table {table_name} not found")
            return

        try:
            self.db.drop_table(table_name)
            if table_name in self._tables:
                del self._tables[table_name]
            self.logger.info(f"✅ Deleted collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            raise

    def update_memory(self, memory_id: str, updates: Dict,
                      collection_name: str = "default") -> None:
        """
        Update memory metadata (e.g., access_count, importance).

        Note: This is a simplified implementation. For production use,
        consider implementing proper UPDATE operations.

        Args:
            memory_id: ID of memory to update
            updates: Dictionary of fields to update
            collection_name: Collection containing the memory
        """
        table_name = self._get_table_name(collection_name)

        if table_name not in self.db.table_names():
            self.logger.warning(f"Table {table_name} not found")
            return

        try:
            table = self.db.open_table(table_name)

            # LanceDB doesn't have direct UPDATE, so we need to:
            # 1. Read the record
            # 2. Delete it
            # 3. Add updated version

            # Search for the record by ID
            all_records = table.to_pandas()
            record_idx = all_records[all_records['id'] == memory_id].index

            if len(record_idx) == 0:
                self.logger.warning(f"Memory {memory_id} not found")
                return

            # Get the record
            record = all_records.loc[record_idx[0]].to_dict()

            # Apply updates
            for key, value in updates.items():
                if key in record and key not in ['id', 'content', 'vector']:
                    record[key] = value

            # Delete old and add new (simple approach)
            # For production, use LanceDB's merge operation
            table.delete(f"id = '{memory_id}'")
            table.add([record])

            self.logger.info(f"✅ Updated memory: {memory_id}")

        except Exception as e:
            self.logger.error(f"Failed to update memory: {e}")
            raise

    # ===================================================================
    # LanceDB-Specific Features
    # ===================================================================

    def create_checkpoint(self, collection_name: str, description: str = "") -> int:
        """
        Create a checkpoint (version) of a collection.

        This allows rollback to previous states if needed.

        Args:
            collection_name: Collection to checkpoint
            description: Optional description

        Returns:
            Version number
        """
        table_name = self._get_table_name(collection_name)

        if table_name not in self.db.table_names():
            self.logger.warning(f"Table {table_name} not found")
            return 0

        try:
            table = self.db.open_table(table_name)
            version = table.version

            self.logger.info(f"✅ Checkpoint created for {collection_name} at version {version}")
            return version
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            return 0

    def rollback_to_version(self, collection_name: str, version: int) -> None:
        """
        Rollback a collection to a previous version.

        Args:
            collection_name: Collection to rollback
            version: Target version number
        """
        table_name = self._get_table_name(collection_name)

        if table_name not in self.db.table_names():
            self.logger.warning(f"Table {table_name} not found")
            return

        try:
            table = self.db.open_table(table_name)
            table.checkout(version)

            self.logger.info(f"✅ Rolled back {collection_name} to version {version}")
        except Exception as e:
            self.logger.error(f"Failed to rollback: {e}")
            raise

    def list_collections(self) -> List[str]:
        """
        List all collections (tables) in this database.

        Returns:
            List of collection names (without prefix)
        """
        try:
            all_tables = self.db.table_names()
            # Filter tables that match our prefix and extract collection name
            collections = []
            for table_name in all_tables:
                if table_name.startswith(f"{self.collection_prefix}_"):
                    # Remove prefix to get collection name
                    collection_name = table_name[len(self.collection_prefix) + 1:]
                    collections.append(collection_name)
            return collections
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []

    def get_collection_count(self, collection_name: str) -> int:
        """
        Get count of items in a collection.

        Args:
            collection_name: Name of collection

        Returns:
            Number of items in collection
        """
        table_name = self._get_table_name(collection_name)
        if table_name not in self.db.table_names():
            return 0

        try:
            table = self.db.open_table(table_name)
            return table.count_rows()
        except Exception as e:
            self.logger.error(f"Failed to count rows: {e}")
            return 0

    def get_stats(self, collection_name: str = "default") -> Dict[str, Any]:
        """
        Get statistics about a collection.

        Args:
            collection_name: Collection to analyze

        Returns:
            Dictionary with statistics
        """
        table_name = self._get_table_name(collection_name)

        if table_name not in self.db.table_names():
            return {
                "exists": False,
                "count": 0
            }

        try:
            table = self.db.open_table(table_name)
            df = table.to_pandas()

            return {
                "exists": True,
                "count": len(df),
                "version": table.version,
                "schema": str(table.schema),
                "memory_types": df['memory_type'].value_counts().to_dict() if len(df) > 0 else {}
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"exists": True, "error": str(e)}

    def get_all_documents(self, collection_name: str, limit: int = 100):
        """
        Get all documents from a collection (for compatibility with manage_rag.py).
        
        Args:
            collection_name: Collection name
            limit: Maximum number of documents to return
            
        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'embeddings'
        """
        table_name = self._get_table_name(collection_name)
        if table_name not in self.db.table_names():
            return {'ids': [], 'documents': [], 'metadatas': [], 'embeddings': []}
        
        try:
            table = self.db.open_table(table_name)
            df = table.to_pandas().head(limit)
            
            return {
                'ids': df['id'].tolist() if 'id' in df.columns else [],
                'documents': df['content'].tolist() if 'content' in df.columns else [],
                'metadatas': df['metadata'].tolist() if 'metadata' in df.columns else [],
                'embeddings': df['vector'].tolist() if 'vector' in df.columns else []
            }
        except Exception as e:
            self.logger.error(f"Failed to get documents: {e}")
            return {'ids': [], 'documents': [], 'metadatas': [], 'embeddings': []}
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        table_name = self._get_table_name(collection_name)
        return table_name in self.db.table_names()

    def get_documents_by_filter(self, filter_dict: Dict, collection_name: str = "documents", limit: int = 1000):
        """
        Get documents matching a filter (for chunk_viewer.py compatibility).

        Args:
            filter_dict: Dictionary with filter criteria (e.g., {'doc_id': 'some_id'})
            collection_name: Collection name
            limit: Maximum number of results

        Returns:
            Dict with 'ids', 'documents', 'metadatas'
        """
        table_name = self._get_table_name(collection_name)
        if table_name not in self.db.table_names():
            return {'ids': [], 'documents': [], 'metadatas': []}

        try:
            table = self.db.open_table(table_name)
            df = table.to_pandas()

            # Apply filters
            for key, value in filter_dict.items():
                if key in df.columns:
                    # Direct column filter
                    df = df[df[key] == value]
                else:
                    # Check in metadata JSON
                    def matches_metadata(metadata_str):
                        try:
                            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                            return metadata.get(key) == value
                        except:
                            return False

                    if 'metadata' in df.columns:
                        df = df[df['metadata'].apply(matches_metadata)]

            # Limit results
            df = df.head(limit)

            # Parse metadata JSON strings back to dicts
            metadatas = []
            for metadata_str in df['metadata'].tolist() if 'metadata' in df.columns else []:
                try:
                    metadatas.append(json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str)
                except:
                    metadatas.append({})

            return {
                'ids': df['id'].tolist() if 'id' in df.columns else [],
                'documents': df['content'].tolist() if 'content' in df.columns else [],
                'metadatas': metadatas
            }
        except Exception as e:
            self.logger.error(f"Failed to get documents by filter: {e}")
            return {'ids': [], 'documents': [], 'metadatas': []}

    def update_document_metadata(self, doc_id: str, metadata_updates: Dict, collection_name: str = "documents") -> bool:
        """
        Update metadata for a specific document (for manage_rag.py compatibility).

        Args:
            doc_id: Document ID to update
            metadata_updates: Dictionary of metadata fields to update
            collection_name: Collection name

        Returns:
            True if successful, False otherwise
        """
        table_name = self._get_table_name(collection_name)
        if table_name not in self.db.table_names():
            self.logger.warning(f"Table {table_name} not found")
            return False

        try:
            table = self.db.open_table(table_name)

            # Read all records to find the one to update
            all_records = table.to_pandas()
            record_idx = all_records[all_records['id'] == doc_id].index

            if len(record_idx) == 0:
                self.logger.warning(f"Document {doc_id} not found")
                return False

            # Get the record
            record = all_records.loc[record_idx[0]].to_dict()

            # Parse existing metadata JSON
            try:
                existing_metadata = json.loads(record.get('metadata', '{}'))
            except:
                existing_metadata = {}

            # Update metadata
            existing_metadata.update(metadata_updates)

            # Convert back to JSON
            record['metadata'] = json.dumps(existing_metadata)

            # Delete old and add new
            table.delete(f"id = '{doc_id}'")
            table.add([record])

            return True

        except Exception as e:
            self.logger.error(f"Failed to update document metadata: {e}")
            return False

