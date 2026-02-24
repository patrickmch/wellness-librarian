"""
Supabase store for parent/child chunks and feedback.

Provides unified storage using Supabase Postgres with pgvector extension
for vector similarity search. Replaces ChromaDB + SQLite architecture.
"""

import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

from backend.config import get_settings
from backend.rag.chunking.models import ParentChunk, ChildChunk

logger = logging.getLogger(__name__)


@dataclass
class ChildSearchResult:
    """Result from child chunk vector search."""
    child_id: str
    parent_id: str
    video_id: str
    text: str
    token_count: int
    child_index: int
    total_children: int
    start_token: int
    end_token: int
    title: str
    category: str
    video_url: str
    source: str
    score: float  # Cosine similarity (1 = identical)
    embedding: Optional[list[float]] = None

    def to_child_chunk(self) -> ChildChunk:
        """Convert to ChildChunk for compatibility."""
        return ChildChunk(
            child_id=self.child_id,
            parent_id=self.parent_id,
            video_id=self.video_id,
            text=self.text,
            token_count=self.token_count,
            child_index=self.child_index,
            total_children=self.total_children,
            start_token=self.start_token,
            end_token=self.end_token,
            title=self.title,
            category=self.category,
            video_url=self.video_url,
            source=self.source,
        )


class SupabaseStore:
    """
    Supabase-backed store for RAG data.

    Provides:
    - Parent chunk storage and retrieval
    - Child chunk storage with vector embeddings
    - Vector similarity search via pgvector
    - Feedback storage for analytics
    """

    def __init__(self, db_url: str | None = None):
        """
        Initialize Supabase store.

        Args:
            db_url: PostgreSQL connection URL (defaults from settings)
        """
        settings = get_settings()
        self.db_url = db_url or settings.supabase_db_url

        if not self.db_url:
            raise ValueError(
                "SUPABASE_DB_URL not configured. "
                "Set it in .env or environment variables."
            )

        # Test connection and ensure schema exists
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[psycopg2.extensions.connection]:
        """Context manager for database connections."""
        conn = psycopg2.connect(self.db_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema if needed."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Parent chunks table
                cur.execute("""
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
                        tags JSONB,
                        indexed_at TIMESTAMPTZ NOT NULL
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_parent_video_id
                    ON parent_chunks(video_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_parent_category
                    ON parent_chunks(category);
                """)

                # Child chunks table with vector embeddings
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS child_chunks (
                        child_id TEXT PRIMARY KEY,
                        parent_id TEXT NOT NULL REFERENCES parent_chunks(parent_id) ON DELETE CASCADE,
                        text TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        child_index INTEGER NOT NULL,
                        total_children INTEGER NOT NULL,
                        start_token INTEGER NOT NULL,
                        end_token INTEGER NOT NULL,
                        embedding vector(1024),
                        video_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        category TEXT NOT NULL,
                        video_url TEXT,
                        source TEXT DEFAULT 'vimeo'
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_child_parent
                    ON child_chunks(parent_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_child_video
                    ON child_chunks(video_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_child_category
                    ON child_chunks(category);
                """)

                # Feedback table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id TEXT PRIMARY KEY,
                        message_id TEXT NOT NULL,
                        session_id TEXT,
                        query TEXT,
                        feedback_type TEXT NOT NULL,
                        parent_ids JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_message
                    ON feedback(message_id);
                """)

        logger.info("Supabase schema initialized")

    def create_vector_index(self, lists: int = 100) -> None:
        """
        Create IVFFlat index for vector search.

        Should be called after bulk data loading for better index quality.
        The 'lists' parameter should be roughly sqrt(num_rows) for optimal performance.

        Args:
            lists: Number of IVF lists (default 100, good for ~10k vectors)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Drop existing index if any
                cur.execute("DROP INDEX IF EXISTS idx_child_embedding;")

                # Create IVFFlat index for cosine similarity
                cur.execute(f"""
                    CREATE INDEX idx_child_embedding
                    ON child_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {lists});
                """)

        logger.info(f"Created IVFFlat vector index with {lists} lists")

    # -------------------------------------------------------------------------
    # Parent Chunk Operations
    # -------------------------------------------------------------------------

    def add_parent(self, parent: ParentChunk) -> None:
        """Add a single parent chunk."""
        self.add_parent_batch([parent])

    def add_parent_batch(self, parents: list[ParentChunk]) -> int:
        """
        Add multiple parent chunks in a batch.

        Args:
            parents: List of ParentChunks to store

        Returns:
            Number of chunks added
        """
        if not parents:
            return 0

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                values = [
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
                        json.dumps(p.tags) if p.tags else None,
                        p.indexed_at.isoformat() if isinstance(p.indexed_at, datetime) else p.indexed_at,
                    )
                    for p in parents
                ]

                execute_values(
                    cur,
                    """
                    INSERT INTO parent_chunks (
                        parent_id, video_id, text, token_count,
                        chunk_index, total_chunks, start_char, end_char,
                        start_time_seconds, title, category, duration,
                        duration_seconds, video_url, source, access_level,
                        created_date, description, tags, indexed_at
                    ) VALUES %s
                    ON CONFLICT (parent_id) DO UPDATE SET
                        video_id = EXCLUDED.video_id,
                        text = EXCLUDED.text,
                        token_count = EXCLUDED.token_count,
                        chunk_index = EXCLUDED.chunk_index,
                        total_chunks = EXCLUDED.total_chunks,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char,
                        start_time_seconds = EXCLUDED.start_time_seconds,
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        duration = EXCLUDED.duration,
                        duration_seconds = EXCLUDED.duration_seconds,
                        video_url = EXCLUDED.video_url,
                        source = EXCLUDED.source,
                        access_level = EXCLUDED.access_level,
                        created_date = EXCLUDED.created_date,
                        description = EXCLUDED.description,
                        tags = EXCLUDED.tags,
                        indexed_at = EXCLUDED.indexed_at
                    """,
                    values,
                )

        return len(parents)

    def get_parent(self, parent_id: str) -> Optional[ParentChunk]:
        """Get a single parent chunk by ID."""
        parents = self.get_parent_batch([parent_id])
        return parents.get(parent_id)

    def get_parent_batch(self, parent_ids: list[str]) -> dict[str, ParentChunk]:
        """
        Get multiple parent chunks by ID.

        Args:
            parent_ids: List of parent IDs to retrieve

        Returns:
            Dict mapping parent_id to ParentChunk
        """
        if not parent_ids:
            return {}

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM parent_chunks
                    WHERE parent_id = ANY(%s)
                    """,
                    (parent_ids,)
                )
                rows = cur.fetchall()

        return {
            row["parent_id"]: self._row_to_parent(row)
            for row in rows
        }

    def get_parents_by_video(self, video_id: str) -> list[ParentChunk]:
        """Get all parent chunks for a video."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM parent_chunks
                    WHERE video_id = %s
                    ORDER BY chunk_index
                    """,
                    (video_id,)
                )
                rows = cur.fetchall()

        return [self._row_to_parent(row) for row in rows]

    def delete_by_video_id(self, video_id: str) -> int:
        """Delete all chunks (parent and child) for a video."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Children are deleted via CASCADE
                cur.execute(
                    "DELETE FROM parent_chunks WHERE video_id = %s",
                    (video_id,)
                )
                return cur.rowcount

    def _row_to_parent(self, row: dict) -> ParentChunk:
        """Convert database row to ParentChunk."""
        tags = row.get("tags")
        if isinstance(tags, str):
            tags = json.loads(tags)
        elif tags is None:
            tags = []

        indexed_at = row.get("indexed_at")
        if isinstance(indexed_at, str):
            indexed_at = datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
        elif indexed_at is None:
            indexed_at = datetime.utcnow()

        return ParentChunk(
            parent_id=row["parent_id"],
            video_id=row["video_id"],
            text=row["text"],
            token_count=row["token_count"],
            chunk_index=row["chunk_index"],
            total_chunks=row["total_chunks"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            start_time_seconds=row.get("start_time_seconds", 0),
            title=row["title"],
            category=row["category"],
            duration=row.get("duration"),
            duration_seconds=row.get("duration_seconds"),
            video_url=row.get("video_url"),
            source=row.get("source", "vimeo"),
            access_level=row.get("access_level", "members_only"),
            created_date=row.get("created_date"),
            description=row.get("description"),
            tags=tags,
            indexed_at=indexed_at,
        )

    # -------------------------------------------------------------------------
    # Child Chunk Operations
    # -------------------------------------------------------------------------

    def add_child_batch(
        self,
        children: list[ChildChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add multiple child chunks with embeddings.

        Args:
            children: List of ChildChunks to store
            embeddings: Corresponding embedding vectors

        Returns:
            Number of chunks added
        """
        if not children or not embeddings:
            return 0

        if len(children) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(children)} children, {len(embeddings)} embeddings"
            )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                values = [
                    (
                        c.child_id,
                        c.parent_id,
                        c.text,
                        c.token_count,
                        c.child_index,
                        c.total_children,
                        c.start_token,
                        c.end_token,
                        str(emb),  # pgvector accepts string format
                        c.video_id,
                        c.title,
                        c.category,
                        c.video_url,
                        c.source,
                    )
                    for c, emb in zip(children, embeddings)
                ]

                execute_values(
                    cur,
                    """
                    INSERT INTO child_chunks (
                        child_id, parent_id, text, token_count,
                        child_index, total_children, start_token, end_token,
                        embedding, video_id, title, category, video_url, source
                    ) VALUES %s
                    ON CONFLICT (child_id) DO UPDATE SET
                        parent_id = EXCLUDED.parent_id,
                        text = EXCLUDED.text,
                        token_count = EXCLUDED.token_count,
                        child_index = EXCLUDED.child_index,
                        total_children = EXCLUDED.total_children,
                        start_token = EXCLUDED.start_token,
                        end_token = EXCLUDED.end_token,
                        embedding = EXCLUDED.embedding,
                        video_id = EXCLUDED.video_id,
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        video_url = EXCLUDED.video_url,
                        source = EXCLUDED.source
                    """,
                    values,
                )

        return len(children)

    def search_children(
        self,
        query_embedding: list[float],
        top_k: int = 30,
        category: str | None = None,
        include_embeddings: bool = False,
    ) -> list[ChildSearchResult]:
        """
        Search child chunks by vector similarity.

        Args:
            query_embedding: Query vector (1024 dimensions for voyage-3)
            top_k: Number of results to return
            category: Optional category filter
            include_embeddings: Whether to return embedding vectors

        Returns:
            List of ChildSearchResult sorted by similarity (highest first)
        """
        embedding_col = ", embedding" if include_embeddings else ""

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if category:
                    cur.execute(
                        f"""
                        SELECT
                            child_id, parent_id, text, token_count,
                            child_index, total_children, start_token, end_token,
                            video_id, title, category, video_url, source,
                            1 - (embedding <=> %s::vector) as score
                            {embedding_col}
                        FROM child_chunks
                        WHERE category = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (str(query_embedding), category, str(query_embedding), top_k)
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT
                            child_id, parent_id, text, token_count,
                            child_index, total_children, start_token, end_token,
                            video_id, title, category, video_url, source,
                            1 - (embedding <=> %s::vector) as score
                            {embedding_col}
                        FROM child_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (str(query_embedding), str(query_embedding), top_k)
                    )

                rows = cur.fetchall()

        results = []
        for row in rows:
            # Parse embedding from string if present
            embedding = row.get("embedding")
            if embedding and isinstance(embedding, str):
                # pgvector returns embeddings as string like "[0.1,0.2,...]"
                embedding = [float(x) for x in embedding.strip("[]").split(",")]

            results.append(ChildSearchResult(
                child_id=row["child_id"],
                parent_id=row["parent_id"],
                video_id=row["video_id"],
                text=row["text"],
                token_count=row["token_count"],
                child_index=row["child_index"],
                total_children=row["total_children"],
                start_token=row["start_token"],
                end_token=row["end_token"],
                title=row["title"],
                category=row["category"],
                video_url=row["video_url"],
                source=row.get("source", "vimeo"),
                score=float(row["score"]),
                embedding=embedding,
            ))
        return results

    def get_child_count(self) -> int:
        """Get total number of child chunks."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM child_chunks")
                return cur.fetchone()[0]

    # -------------------------------------------------------------------------
    # Feedback Operations
    # -------------------------------------------------------------------------

    def add_feedback(
        self,
        message_id: str,
        feedback_type: str,
        session_id: str | None = None,
        query: str | None = None,
        parent_ids: list[str] | None = None,
    ) -> str:
        """
        Store user feedback.

        Args:
            message_id: ID of the message being rated
            feedback_type: "up" or "down"
            session_id: Optional session identifier
            query: Optional original query text
            parent_ids: Optional list of parent chunk IDs used

        Returns:
            Generated feedback ID
        """
        feedback_id = str(uuid.uuid4())

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback (
                        id, message_id, session_id, query,
                        feedback_type, parent_ids, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        feedback_id,
                        message_id,
                        session_id,
                        query,
                        feedback_type,
                        json.dumps(parent_ids) if parent_ids else None,
                    )
                )

        return feedback_id

    def get_feedback_stats(self) -> dict:
        """Get feedback statistics."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM feedback")
                total = cur.fetchone()[0]

                cur.execute(
                    "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'up'"
                )
                up_count = cur.fetchone()[0]

                cur.execute(
                    "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'down'"
                )
                down_count = cur.fetchone()[0]

        return {
            "total_feedback": total,
            "thumbs_up": up_count,
            "thumbs_down": down_count,
        }

    # -------------------------------------------------------------------------
    # Random Video Selection
    # -------------------------------------------------------------------------

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
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                conditions = []
                params: list = []

                if category:
                    conditions.append("category = %s")
                    params.append(category)

                if exclude_video_ids:
                    conditions.append("video_id != ALL(%s)")
                    params.append(exclude_video_ids)

                where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

                cur.execute(
                    f"""
                    SELECT video_id, title, category, video_url, source
                    FROM parent_chunks
                    {where}
                    GROUP BY video_id, title, category, video_url, source
                    ORDER BY RANDOM()
                    LIMIT 1
                    """,
                    params,
                )
                row = cur.fetchone()

                if not row:
                    return None

                return {
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "category": row["category"],
                    "video_url": row["video_url"],
                    "source": row.get("source", "vimeo"),
                }

    # -------------------------------------------------------------------------
    # Statistics and Management
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get store statistics."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Parent counts
                cur.execute("SELECT COUNT(*) FROM parent_chunks")
                parent_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(DISTINCT video_id) FROM parent_chunks")
                video_count = cur.fetchone()[0]

                cur.execute("SELECT COALESCE(SUM(token_count), 0) FROM parent_chunks")
                total_tokens = cur.fetchone()[0]

                # Categories
                cur.execute("""
                    SELECT category, COUNT(*) as count
                    FROM parent_chunks
                    GROUP BY category
                    ORDER BY count DESC
                """)
                categories = {row[0]: row[1] for row in cur.fetchall()}

                # Child count
                cur.execute("SELECT COUNT(*) FROM child_chunks")
                child_count = cur.fetchone()[0]

        return {
            "total_parent_chunks": parent_count,
            "unique_videos": video_count,
            "total_tokens": total_tokens,
            "categories": categories,
            "child_chunks_indexed": child_count,
        }

    def clear_all(self) -> dict:
        """Clear all data from the store."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM feedback")
                feedback_deleted = cur.rowcount

                cur.execute("DELETE FROM child_chunks")
                children_deleted = cur.rowcount

                cur.execute("DELETE FROM parent_chunks")
                parents_deleted = cur.rowcount

        return {
            "parents_deleted": parents_deleted,
            "children_deleted": children_deleted,
            "feedback_deleted": feedback_deleted,
        }


# Module-level singleton
_store: Optional[SupabaseStore] = None


def get_supabase_store() -> SupabaseStore:
    """Get the singleton Supabase store instance."""
    global _store
    if _store is None:
        _store = SupabaseStore()
    return _store
