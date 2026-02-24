"""
SQLite-based document store for parent chunks.

Stores parent chunks in SQLite for efficient retrieval by ID after
child matching. This is simpler and more appropriate than putting
large documents in the vector store.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from backend.config import get_settings
from backend.rag.chunking.models import ParentChunk

logger = logging.getLogger(__name__)


class SQLiteDocStore:
    """
    SQLite-based document store for parent chunks.

    Provides:
    - CRUD operations for parent chunks
    - Batch insert for ingestion efficiency
    - Query by video_id for management
    - Statistics and introspection
    """

    def __init__(self, db_path: Path | str | None = None):
        """
        Initialize the document store.

        Args:
            db_path: Path to SQLite database file (defaults to settings)
        """
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else settings.docstore_path

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    start_time_seconds INTEGER DEFAULT 0,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    duration TEXT,
                    duration_seconds INTEGER,
                    video_url TEXT,
                    source TEXT DEFAULT 'vimeo',
                    access_level TEXT DEFAULT 'members_only',
                    created_date TEXT,
                    description TEXT,
                    tags TEXT,  -- JSON array
                    indexed_at TEXT NOT NULL
                )
            """)

            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_video_id
                ON parent_chunks (video_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_category
                ON parent_chunks (category)
            """)

            # Feedback table for user ratings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    session_id TEXT,
                    query TEXT,
                    feedback_type TEXT NOT NULL,
                    parent_ids TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_message_id
                ON feedback (message_id)
            """)

    def add(self, parent: ParentChunk) -> None:
        """
        Add a single parent chunk.

        Args:
            parent: ParentChunk to store
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO parent_chunks (
                    parent_id, video_id, text, token_count,
                    chunk_index, total_chunks, start_char, end_char,
                    start_time_seconds,
                    title, category, duration, duration_seconds,
                    video_url, source, access_level,
                    created_date, description, tags, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                parent.parent_id,
                parent.video_id,
                parent.text,
                parent.token_count,
                parent.chunk_index,
                parent.total_chunks,
                parent.start_char,
                parent.end_char,
                parent.start_time_seconds,
                parent.title,
                parent.category,
                parent.duration,
                parent.duration_seconds,
                parent.video_url,
                parent.source,
                parent.access_level,
                parent.created_date,
                parent.description,
                json.dumps(parent.tags),
                parent.indexed_at.isoformat(),
            ))

    def add_batch(self, parents: list[ParentChunk]) -> int:
        """
        Add multiple parent chunks in a single transaction.

        Args:
            parents: List of ParentChunks to store

        Returns:
            Number of chunks added
        """
        if not parents:
            return 0

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO parent_chunks (
                    parent_id, video_id, text, token_count,
                    chunk_index, total_chunks, start_char, end_char,
                    start_time_seconds,
                    title, category, duration, duration_seconds,
                    video_url, source, access_level,
                    created_date, description, tags, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    p.parent_id,
                    p.video_id,
                    p.text,
                    p.token_count,
                    p.chunk_index,
                    p.total_chunks,
                    p.start_char,
                    p.end_char,
                    p.start_time_seconds,
                    p.title,
                    p.category,
                    p.duration,
                    p.duration_seconds,
                    p.video_url,
                    p.source,
                    p.access_level,
                    p.created_date,
                    p.description,
                    json.dumps(p.tags),
                    p.indexed_at.isoformat(),
                )
                for p in parents
            ])

        return len(parents)

    def get(self, parent_id: str) -> Optional[ParentChunk]:
        """
        Retrieve a parent chunk by ID.

        Args:
            parent_id: The parent chunk ID

        Returns:
            ParentChunk if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM parent_chunks WHERE parent_id = ?
            """, (parent_id,)).fetchone()

            if row is None:
                return None

            return self._row_to_parent(row)

    def get_batch(self, parent_ids: list[str]) -> dict[str, ParentChunk]:
        """
        Retrieve multiple parent chunks by ID.

        Args:
            parent_ids: List of parent IDs

        Returns:
            Dict mapping parent_id to ParentChunk
        """
        if not parent_ids:
            return {}

        placeholders = ",".join("?" * len(parent_ids))
        with self._get_connection() as conn:
            rows = conn.execute(f"""
                SELECT * FROM parent_chunks WHERE parent_id IN ({placeholders})
            """, parent_ids).fetchall()

            return {
                row["parent_id"]: self._row_to_parent(row)
                for row in rows
            }

    def get_by_video_id(self, video_id: str) -> list[ParentChunk]:
        """
        Get all parent chunks for a video.

        Args:
            video_id: The video ID

        Returns:
            List of ParentChunks ordered by chunk_index
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM parent_chunks
                WHERE video_id = ?
                ORDER BY chunk_index
            """, (video_id,)).fetchall()

            return [self._row_to_parent(row) for row in rows]

    def delete_by_video_id(self, video_id: str) -> int:
        """
        Delete all parent chunks for a video.

        Args:
            video_id: The video ID

        Returns:
            Number of chunks deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM parent_chunks WHERE video_id = ?
            """, (video_id,))
            return cursor.rowcount

    def exists(self, parent_id: str) -> bool:
        """Check if a parent chunk exists."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT 1 FROM parent_chunks WHERE parent_id = ? LIMIT 1
            """, (parent_id,)).fetchone()
            return row is not None

    def get_stats(self) -> dict:
        """
        Get statistics about the document store.

        Returns:
            Dict with counts and metadata
        """
        with self._get_connection() as conn:
            # Total chunks
            total = conn.execute(
                "SELECT COUNT(*) FROM parent_chunks"
            ).fetchone()[0]

            # Unique videos
            videos = conn.execute(
                "SELECT COUNT(DISTINCT video_id) FROM parent_chunks"
            ).fetchone()[0]

            # Categories
            categories = conn.execute("""
                SELECT category, COUNT(*) as count
                FROM parent_chunks
                GROUP BY category
                ORDER BY count DESC
            """).fetchall()

            # Total tokens
            total_tokens = conn.execute(
                "SELECT SUM(token_count) FROM parent_chunks"
            ).fetchone()[0] or 0

            return {
                "total_parent_chunks": total,
                "unique_videos": videos,
                "total_tokens": total_tokens,
                "categories": {
                    row["category"]: row["count"]
                    for row in categories
                },
            }

    def clear(self) -> int:
        """
        Clear all data from the store.

        Returns:
            Number of chunks deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM parent_chunks")
            return cursor.rowcount

    def get_random_video_id(
        self,
        category: str | None = None,
        exclude_video_ids: list[str] | None = None,
    ) -> dict | None:
        """
        Pick a random video, optionally filtered by category and exclusions.

        Returns:
            Dict with video_id, title, category, video_url, source
            or None if no matching video exists.
        """
        conditions = []
        params: list = []

        if category:
            conditions.append("category = ?")
            params.append(category)

        if exclude_video_ids:
            placeholders = ",".join("?" * len(exclude_video_ids))
            conditions.append(f"video_id NOT IN ({placeholders})")
            params.extend(exclude_video_ids)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._get_connection() as conn:
            row = conn.execute(
                f"""
                SELECT video_id, title, category, video_url, source
                FROM parent_chunks
                {where}
                GROUP BY video_id, title, category, video_url, source
                ORDER BY RANDOM()
                LIMIT 1
                """,
                params,
            ).fetchone()

            if not row:
                return None

            return {
                "video_id": row["video_id"],
                "title": row["title"],
                "category": row["category"],
                "video_url": row["video_url"],
                "source": row["source"],
            }

    def add_feedback(
        self,
        message_id: str,
        feedback_type: str,
        session_id: str | None = None,
        query: str | None = None,
        parent_ids: list[str] | None = None,
    ) -> str:
        """
        Add user feedback for a message.

        Args:
            message_id: ID of the message being rated
            feedback_type: "up" or "down"
            session_id: Optional session identifier
            query: Optional original query text
            parent_ids: Optional list of parent chunk IDs used

        Returns:
            Generated feedback ID
        """
        import uuid
        feedback_id = str(uuid.uuid4())

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO feedback (
                    id, message_id, session_id, query,
                    feedback_type, parent_ids, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                feedback_id,
                message_id,
                session_id,
                query,
                feedback_type,
                json.dumps(parent_ids) if parent_ids else None,
            ))

        return feedback_id

    def get_feedback_stats(self) -> dict:
        """
        Get feedback statistics.

        Returns:
            Dict with feedback counts and breakdown
        """
        with self._get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM feedback"
            ).fetchone()[0]

            up_count = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'up'"
            ).fetchone()[0]

            down_count = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'down'"
            ).fetchone()[0]

            return {
                "total_feedback": total,
                "thumbs_up": up_count,
                "thumbs_down": down_count,
            }

    def _row_to_parent(self, row: sqlite3.Row) -> ParentChunk:
        """Convert a database row to ParentChunk."""
        from datetime import datetime

        tags = row["tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)
        elif tags is None:
            tags = []

        indexed_at = row["indexed_at"]
        if isinstance(indexed_at, str):
            indexed_at = datetime.fromisoformat(indexed_at)

        return ParentChunk(
            parent_id=row["parent_id"],
            video_id=row["video_id"],
            text=row["text"],
            token_count=row["token_count"],
            chunk_index=row["chunk_index"],
            total_chunks=row["total_chunks"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            start_time_seconds=row["start_time_seconds"] if "start_time_seconds" in row.keys() else 0,
            title=row["title"],
            category=row["category"],
            duration=row["duration"],
            duration_seconds=row["duration_seconds"],
            video_url=row["video_url"],
            source=row["source"],
            access_level=row["access_level"],
            created_date=row["created_date"],
            description=row["description"],
            tags=tags,
            indexed_at=indexed_at,
        )


# Module-level singleton for convenience
_docstore: Optional[SQLiteDocStore] = None


def get_docstore() -> SQLiteDocStore:
    """Get the singleton docstore instance."""
    global _docstore
    if _docstore is None:
        _docstore = SQLiteDocStore()
    return _docstore
