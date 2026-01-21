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
    vimeo_id: str
    title: str

    # Content context
    category: str
    chunk_index: int
    total_chunks: int

    # Rich metadata
    duration: str
    duration_seconds: int
    vimeo_url: str
    created_date: Optional[str]
    description: Optional[str]
    tags: list[str]

    # Chunk positioning (for potential timestamp linking)
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        """Convert to dict for ChromaDB metadata storage."""
        return {
            "vimeo_id": self.vimeo_id,
            "title": self.title,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "duration": self.duration,
            "duration_seconds": self.duration_seconds,
            "vimeo_url": self.vimeo_url,
            "created_date": self.created_date or "",
            "description": self.description or "",
            "tags": ",".join(self.tags) if self.tags else "",
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
            vimeo_id=video_meta.vimeo_id,
            title=video_meta.title,
            category=video_meta.folder_name,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            duration=video_meta.duration_formatted,
            duration_seconds=video_meta.duration_seconds,
            vimeo_url=video_meta.vimeo_url,
            created_date=video_meta.created_time.date().isoformat() if video_meta.created_time else None,
            description=video_meta.description,
            tags=video_meta.tags,
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
    vimeo_url = metadata.get("vimeo_url", "")

    parts = [f'"{title}"']

    if category:
        parts.append(f"({category})")

    if duration:
        parts.append(f"[{duration}]")

    citation = " ".join(parts)

    if vimeo_url:
        citation += f" - {vimeo_url}"

    return citation
