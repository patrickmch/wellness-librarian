"""
Data models for parent-child chunking.

Parent chunks are larger context windows (500-2000 tokens) stored in the
docstore for retrieval after child matching.

Child chunks are smaller searchable units (250 tokens) indexed in the
vector store with references back to their parent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import md5
from typing import Optional


@dataclass
class ParentChunk:
    """
    A parent chunk represents a larger context window from a transcript.

    Parent chunks are:
    - Stored in SQLite docstore (not in vector store)
    - Retrieved by parent_id after child matching
    - Sized for LLM context comprehension (500-2000 tokens)
    - Can be topic-segmented or fixed-size
    """

    parent_id: str  # Unique identifier
    video_id: str  # Source video ID
    text: str  # Full text content
    token_count: int  # Token count for this chunk

    # Position within video
    chunk_index: int  # 0-indexed position
    total_chunks: int  # Total parent chunks in video
    start_char: int  # Character offset in original
    end_char: int  # End character offset

    # Metadata
    title: str
    category: str
    duration: str  # Formatted duration
    duration_seconds: int
    video_url: str

    # Optional fields with defaults
    start_time_seconds: int = 0  # Video timestamp for deep-linking
    source: str = "vimeo"  # "youtube" or "vimeo"
    access_level: str = "members_only"
    created_date: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Timestamps
    indexed_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def generate_id(cls, video_id: str, chunk_index: int) -> str:
        """Generate deterministic parent_id."""
        content = f"{video_id}::parent::{chunk_index}"
        return md5(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "parent_id": self.parent_id,
            "video_id": self.video_id,
            "text": self.text,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_time_seconds": self.start_time_seconds,
            "title": self.title,
            "category": self.category,
            "duration": self.duration,
            "duration_seconds": self.duration_seconds,
            "video_url": self.video_url,
            "source": self.source,
            "access_level": self.access_level,
            "created_date": self.created_date,
            "description": self.description,
            "tags": self.tags,
            "indexed_at": self.indexed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParentChunk":
        """Create from dictionary."""
        indexed_at = data.get("indexed_at")
        if isinstance(indexed_at, str):
            indexed_at = datetime.fromisoformat(indexed_at)
        elif indexed_at is None:
            indexed_at = datetime.utcnow()

        return cls(
            parent_id=data["parent_id"],
            video_id=data["video_id"],
            text=data["text"],
            token_count=data["token_count"],
            chunk_index=data["chunk_index"],
            total_chunks=data["total_chunks"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            start_time_seconds=data.get("start_time_seconds", 0),
            title=data["title"],
            category=data["category"],
            duration=data["duration"],
            duration_seconds=data["duration_seconds"],
            video_url=data["video_url"],
            source=data.get("source", "vimeo"),
            access_level=data.get("access_level", "members_only"),
            created_date=data.get("created_date"),
            description=data.get("description"),
            tags=data.get("tags", []),
            indexed_at=indexed_at,
        )


@dataclass
class ChildChunk:
    """
    A child chunk is a small searchable unit within a parent chunk.

    Child chunks are:
    - Indexed in ChromaDB vector store
    - Used for fine-grained semantic search
    - Reference their parent for context expansion
    - Sized for search precision (250 tokens with overlap)
    """

    child_id: str  # Unique identifier
    parent_id: str  # Reference to parent chunk
    video_id: str  # Source video ID (denormalized for filtering)
    text: str  # Text content
    token_count: int  # Token count

    # Position within parent
    child_index: int  # 0-indexed position within parent
    total_children: int  # Total children in this parent
    start_token: int  # Token offset within parent
    end_token: int  # End token offset

    # Denormalized metadata for filtering/display (from parent)
    title: str
    category: str
    video_url: str
    source: str = "vimeo"

    @classmethod
    def generate_id(cls, parent_id: str, child_index: int) -> str:
        """Generate deterministic child_id."""
        content = f"{parent_id}::child::{child_index}"
        return md5(content.encode()).hexdigest()

    def to_chroma_metadata(self) -> dict:
        """Convert to ChromaDB metadata format."""
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "video_id": self.video_id,
            "child_index": self.child_index,
            "total_children": self.total_children,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "title": self.title,
            "category": self.category,
            "video_url": self.video_url,
            "source": self.source,
        }

    @classmethod
    def from_chroma_result(cls, doc: str, metadata: dict) -> "ChildChunk":
        """Create from ChromaDB search result."""
        return cls(
            child_id=metadata["child_id"],
            parent_id=metadata["parent_id"],
            video_id=metadata["video_id"],
            text=doc,
            token_count=metadata.get("token_count", 0),
            child_index=metadata["child_index"],
            total_children=metadata["total_children"],
            start_token=metadata["start_token"],
            end_token=metadata["end_token"],
            title=metadata["title"],
            category=metadata["category"],
            video_url=metadata["video_url"],
            source=metadata.get("source", "vimeo"),
        )
