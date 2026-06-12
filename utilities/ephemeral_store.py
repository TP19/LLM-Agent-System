#!/usr/bin/env python3
"""
Ephemeral Summary Store Enhanced Summarizer

Temporary storage for summarization jobs. Creates ephemeral collections
that are automatically cleaned up after the job completes.
"""

import uuid
import logging
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class SummaryJob:
    """Metadata for a summarization job"""
    job_id: str
    file_path: str
    collection_name: str
    created_at: datetime
    chapter_count: int = 0
    total_chunks: int = 0
    status: str = "created"  # created, processing, completed, failed, cleaned

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()


class EphemeralSummaryStore:
    """
    Temporary storage for summarization job data.

    Creates ephemeral LanceDB collections for each summarization job.
    Collections are automatically cleaned up after job completion.

    Usage:
        store = EphemeralSummaryStore()

        # Create job
        job = store.create_job("/path/to/book.pdf")

        # Store chapters
        for chapter in chapters:
            store.store_chapter(job.job_id, chapter)

        # Retrieve for processing
        chunks = store.get_chapter_chunks(job.job_id, chapter_num=1)

        # Store summary
        store.store_chapter_summary(job.job_id, chapter_num=1, summary="...")

        # Cleanup when done
        store.cleanup_job(job.job_id)
    """

    def __init__(
        self,
        db_path: str = "summaries/ephemeral",
        max_job_age_hours: int = 24,
        use_lancedb: bool = True
    ):
        """
        Initialize ephemeral store.

        Args:
            db_path: Base path for ephemeral databases
            max_job_age_hours: Auto-cleanup jobs older than this
            use_lancedb: Use LanceDB (True) or simple file storage (False)
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.max_job_age_hours = max_job_age_hours
        self.use_lancedb = use_lancedb

        # Track active jobs
        self.active_jobs: Dict[str, SummaryJob] = {}

        # LanceDB connection (lazy init)
        self._db = None

        logger.info(f"EphemeralSummaryStore initialized: {self.db_path}")

    @property
    def db(self):
        """Lazy LanceDB connection"""
        if self._db is None and self.use_lancedb:
            try:
                import lancedb
                self._db = lancedb.connect(str(self.db_path / "lancedb"))
                logger.info("LanceDB connection established")
            except ImportError:
                logger.warning("LanceDB not available, using file storage")
                self.use_lancedb = False
        return self._db

    def create_job(self, file_path: str) -> SummaryJob:
        """
        Create a new summarization job.

        Args:
            file_path: Path to the source file

        Returns:
            SummaryJob with job_id and collection info
        """
        job_id = f"sum_{uuid.uuid4().hex[:12]}"
        collection_name = f"ephemeral_{job_id}"

        job = SummaryJob(
            job_id=job_id,
            file_path=file_path,
            collection_name=collection_name,
            created_at=datetime.now(),
        )

        self.active_jobs[job_id] = job

        # Create job directory for file-based storage
        job_dir = self.db_path / job_id
        job_dir.mkdir(exist_ok=True)

        logger.info(f"Created summarization job: {job_id} for {file_path}")
        return job

    def store_chapter(
        self,
        job_id: str,
        chapter_num: int,
        chapter_title: str,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        Store chapter chunks for a job.

        Args:
            job_id: Job identifier
            chapter_num: Chapter number
            chapter_title: Chapter title
            chunks: List of text chunks
            embeddings: Optional pre-computed embeddings

        Returns:
            True if successful
        """
        if job_id not in self.active_jobs:
            logger.error(f"Job not found: {job_id}")
            return False

        job = self.active_jobs[job_id]

        if self.use_lancedb and self.db:
            return self._store_chapter_lancedb(job, chapter_num, chapter_title, chunks, embeddings)
        else:
            return self._store_chapter_file(job, chapter_num, chapter_title, chunks)

    def _store_chapter_lancedb(
        self,
        job: SummaryJob,
        chapter_num: int,
        chapter_title: str,
        chunks: List[str],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """Store chapter in LanceDB"""
        try:
            import pyarrow as pa

            # Prepare data
            data = []
            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"{job.job_id}_ch{chapter_num}_c{i}",
                    "chapter_num": chapter_num,
                    "chapter_title": chapter_title,
                    "chunk_index": i,
                    "content": chunk,
                    "job_id": job.job_id,
                }
                if embeddings and i < len(embeddings):
                    record["embedding"] = embeddings[i]
                data.append(record)

            # Get or create table
            table_name = job.collection_name
            if table_name in self.db.table_names():
                table = self.db.open_table(table_name)
                table.add(data)
            else:
                self.db.create_table(table_name, data)

            job.chapter_count += 1
            job.total_chunks += len(chunks)
            job.status = "processing"

            logger.debug(f"Stored chapter {chapter_num} ({len(chunks)} chunks) in LanceDB")
            return True

        except Exception as e:
            logger.error(f"Failed to store chapter in LanceDB: {e}")
            return False

    def _store_chapter_file(
        self,
        job: SummaryJob,
        chapter_num: int,
        chapter_title: str,
        chunks: List[str]
    ) -> bool:
        """Store chapter in file system"""
        try:
            import json

            job_dir = self.db_path / job.job_id
            chapter_file = job_dir / f"chapter_{chapter_num:03d}.json"

            data = {
                "chapter_num": chapter_num,
                "chapter_title": chapter_title,
                "chunks": chunks,
                "chunk_count": len(chunks),
            }

            with open(chapter_file, 'w') as f:
                json.dump(data, f, indent=2)

            job.chapter_count += 1
            job.total_chunks += len(chunks)
            job.status = "processing"

            logger.debug(f"Stored chapter {chapter_num} ({len(chunks)} chunks) in file")
            return True

        except Exception as e:
            logger.error(f"Failed to store chapter in file: {e}")
            return False

    def get_chapter_chunks(self, job_id: str, chapter_num: int) -> List[str]:
        """
        Retrieve chunks for a specific chapter.

        Args:
            job_id: Job identifier
            chapter_num: Chapter number

        Returns:
            List of chunk texts
        """
        if job_id not in self.active_jobs:
            logger.error(f"Job not found: {job_id}")
            return []

        job = self.active_jobs[job_id]

        if self.use_lancedb and self.db:
            return self._get_chunks_lancedb(job, chapter_num)
        else:
            return self._get_chunks_file(job, chapter_num)

    def _get_chunks_lancedb(self, job: SummaryJob, chapter_num: int) -> List[str]:
        """Retrieve chunks from LanceDB"""
        try:
            table_name = job.collection_name
            if table_name not in self.db.table_names():
                return []

            table = self.db.open_table(table_name)
            results = table.search().where(f"chapter_num = {chapter_num}").to_list()

            # Sort by chunk_index
            results.sort(key=lambda x: x.get('chunk_index', 0))

            return [r['content'] for r in results]

        except Exception as e:
            logger.error(f"Failed to get chunks from LanceDB: {e}")
            return []

    def _get_chunks_file(self, job: SummaryJob, chapter_num: int) -> List[str]:
        """Retrieve chunks from file system"""
        try:
            import json

            job_dir = self.db_path / job.job_id
            chapter_file = job_dir / f"chapter_{chapter_num:03d}.json"

            if not chapter_file.exists():
                return []

            with open(chapter_file, 'r') as f:
                data = json.load(f)

            return data.get('chunks', [])

        except Exception as e:
            logger.error(f"Failed to get chunks from file: {e}")
            return []

    def get_all_chapters(self, job_id: str) -> Dict[int, List[str]]:
        """
        Get all chapters for a job.

        Returns:
            Dict mapping chapter_num to list of chunks
        """
        if job_id not in self.active_jobs:
            return {}

        job = self.active_jobs[job_id]
        chapters = {}

        if self.use_lancedb and self.db:
            try:
                table_name = job.collection_name
                if table_name in self.db.table_names():
                    table = self.db.open_table(table_name)
                    results = table.to_pandas()

                    for chapter_num in results['chapter_num'].unique():
                        chapter_data = results[results['chapter_num'] == chapter_num]
                        chapter_data = chapter_data.sort_values('chunk_index')
                        chapters[chapter_num] = chapter_data['content'].tolist()

            except Exception as e:
                logger.error(f"Failed to get all chapters from LanceDB: {e}")
        else:
            # File-based
            job_dir = self.db_path / job.job_id
            for chapter_file in sorted(job_dir.glob("chapter_*.json")):
                import json
                with open(chapter_file, 'r') as f:
                    data = json.load(f)
                chapters[data['chapter_num']] = data['chunks']

        return chapters

    def store_chapter_summary(
        self,
        job_id: str,
        chapter_num: int,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store summary for a chapter"""
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        job_dir = self.db_path / job.job_id

        import json
        summary_file = job_dir / f"summary_{chapter_num:03d}.json"

        data = {
            "chapter_num": chapter_num,
            "summary": summary,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }

        with open(summary_file, 'w') as f:
            json.dump(data, f, indent=2)

        return True

    def get_chapter_summary(self, job_id: str, chapter_num: int) -> Optional[str]:
        """Retrieve summary for a chapter"""
        if job_id not in self.active_jobs:
            return None

        job = self.active_jobs[job_id]
        job_dir = self.db_path / job.job_id
        summary_file = job_dir / f"summary_{chapter_num:03d}.json"

        if not summary_file.exists():
            return None

        import json
        with open(summary_file, 'r') as f:
            data = json.load(f)

        return data.get('summary')

    def get_all_summaries(self, job_id: str) -> Dict[int, str]:
        """Get all chapter summaries for a job"""
        if job_id not in self.active_jobs:
            return {}

        job = self.active_jobs[job_id]
        job_dir = self.db_path / job.job_id
        summaries = {}

        import json
        for summary_file in sorted(job_dir.glob("summary_*.json")):
            with open(summary_file, 'r') as f:
                data = json.load(f)
            summaries[data['chapter_num']] = data['summary']

        return summaries

    def cleanup_job(self, job_id: str) -> bool:
        """
        Clean up a completed job.

        Removes ephemeral collection and job metadata.
        """
        if job_id not in self.active_jobs:
            logger.warning(f"Job not found for cleanup: {job_id}")
            return False

        job = self.active_jobs[job_id]

        try:
            # Clean up LanceDB table
            if self.use_lancedb and self.db:
                table_name = job.collection_name
                if table_name in self.db.table_names():
                    self.db.drop_table(table_name)
                    logger.info(f"Dropped LanceDB table: {table_name}")

            # Clean up file storage
            job_dir = self.db_path / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir)
                logger.info(f"Removed job directory: {job_dir}")

            # Remove from active jobs
            job.status = "cleaned"
            del self.active_jobs[job_id]

            logger.info(f"Cleaned up job: {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup job {job_id}: {e}")
            return False

    def cleanup_old_jobs(self) -> int:
        """
        Clean up jobs older than max_job_age_hours.

        Returns:
            Number of jobs cleaned up
        """
        max_age_seconds = self.max_job_age_hours * 3600
        cleaned = 0

        for job_id in list(self.active_jobs.keys()):
            job = self.active_jobs[job_id]
            if job.age_seconds > max_age_seconds:
                if self.cleanup_job(job_id):
                    cleaned += 1

        if cleaned:
            logger.info(f"Cleaned up {cleaned} old jobs")

        return cleaned

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job"""
        if job_id not in self.active_jobs:
            return None

        job = self.active_jobs[job_id]
        return {
            "job_id": job.job_id,
            "file_path": job.file_path,
            "status": job.status,
            "chapter_count": job.chapter_count,
            "total_chunks": job.total_chunks,
            "age_seconds": job.age_seconds,
            "created_at": job.created_at.isoformat(),
        }

    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active jobs"""
        return [self.get_job_status(job_id) for job_id in self.active_jobs]
