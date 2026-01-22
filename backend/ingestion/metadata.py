"""
Metadata handling for RAG chunks.
Builds rich chunk metadata from VideoMetadata for storage in vector DB.
"""

from dataclasses import dataclass
from typing import Optional

from backend.ingestion.loader import VideoMetadata


@dataclass
class ChunkMetadata:
    """Metadata attached to each chunk in the vector store."""
    # Video identification
    video_id: str  # YouTube or Vimeo ID
    title: str

    # Content context
    category: str
    chunk_index: int
    total_chunks: int

    # Rich metadata
    duration: str
    duration_seconds: int
    video_url: str  # Full URL to video
    created_date: Optional[str]
    description: Optional[str]
    tags: list[str]

    # Source identification
    source: str  # "youtube" or "vimeo"
    access_level: str  # "public" or "members_only"

    # Chunk positioning (for potential timestamp linking)
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        """Convert to dict for ChromaDB metadata storage."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "duration": self.duration,
            "duration_seconds": self.duration_seconds,
            "video_url": self.video_url,
            "created_date": self.created_date or "",
            "description": self.description or "",
            "tags": ",".join(self.tags) if self.tags else "",
            "source": self.source,
            "access_level": self.access_level,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_video_metadata(
        cls,
        video_meta: VideoMetadata,
        chunk_index: int,
        total_chunks: int,
        start_char: int,
        end_char: int,
    ) -> "ChunkMetadata":
        """Create ChunkMetadata from VideoMetadata and chunk info."""
        return cls(
            video_id=video_meta.video_id,
            title=video_meta.title,
            category=video_meta.folder_name,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            duration=video_meta.duration_formatted,
            duration_seconds=video_meta.duration_seconds,
            video_url=video_meta.video_url,
            created_date=video_meta.created_time.date().isoformat() if video_meta.created_time else None,
            description=video_meta.description,
            tags=video_meta.tags,
            source=video_meta.source,
            access_level=video_meta.access_level,
            start_char=start_char,
            end_char=end_char,
        )


def format_source_citation(metadata: dict) -> str:
    """
    Format metadata into a readable source citation.

    Args:
        metadata: Chunk metadata dict from vector store

    Returns:
        Formatted citation string
    """
    title = metadata.get("title", "Unknown")
    category = metadata.get("category", "")
    duration = metadata.get("duration", "")
    video_url = metadata.get("video_url", "")
    source = metadata.get("source", "")

    parts = [f'"{title}"']

    if category:
        parts.append(f"({category})")

    if duration:
        parts.append(f"[{duration}]")

    if source:
        parts.append(f"[{source.title()}]")

    citation = " ".join(parts)

    if video_url:
        citation += f" - {video_url}"

    return citation
