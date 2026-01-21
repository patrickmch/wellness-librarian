"""
Text chunking module for splitting transcripts into embeddable segments.
Uses recursive character splitting with overlap for semantic continuity.
"""

from dataclasses import dataclass
from typing import Iterator
import hashlib

from backend.config import get_settings
from backend.ingestion.loader import TranscriptFile


@dataclass
class TextChunk:
    """A chunk of text with metadata for embedding."""
    text: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    source_id: str  # Vimeo ID
    source_title: str

    @property
    def chunk_id(self) -> str:
        """
        Deterministic ID for deduplication.
        Based on Vimeo ID and chunk index to ensure consistency across runs.
        """
        return hashlib.md5(
            f"{self.source_id}::chunk::{self.chunk_index}".encode()
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.text)


# Separators in order of preference (most semantic to least)
SEPARATORS = [
    "\n\n",     # Paragraph break
    "\n",       # Line break
    ". ",       # Sentence end
    "? ",       # Question end
    "! ",       # Exclamation end
    ", ",       # Clause break
    " ",        # Word break
]


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into chunks using recursive character splitting.

    Attempts to split on semantic boundaries (paragraphs, sentences)
    while respecting chunk size limits.

    Args:
        text: The text to split
        chunk_size: Maximum characters per chunk (default from settings)
        chunk_overlap: Overlap between chunks (default from settings)

    Returns:
        List of text chunks
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    # Clean and normalize text
    text = text.strip()
    text = " ".join(text.split())  # Normalize whitespace

    # If text fits in one chunk, return as-is
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Determine end position
        end = min(start + chunk_size, len(text))

        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Look for the best separator within the chunk
            best_break = end
            for sep in SEPARATORS:
                # Search backwards from end for separator
                break_pos = text.rfind(sep, start + chunk_overlap, end)
                if break_pos != -1:
                    best_break = break_pos + len(sep)
                    break
            end = best_break

        # Extract chunk
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Move start position (with overlap)
        if end >= len(text):
            break
        start = max(start + 1, end - chunk_overlap)

    return chunks


def chunk_transcript(
    transcript: TranscriptFile,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Iterator[TextChunk]:
    """
    Chunk a transcript into TextChunk objects with metadata.

    Args:
        transcript: TranscriptFile with content and metadata
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks

    Yields:
        TextChunk objects with full metadata
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks = split_text(transcript.content, chunk_size, chunk_overlap)
    total_chunks = len(chunks)

    # Track positions for metadata
    current_pos = 0

    for idx, chunk_text in enumerate(chunks):
        # Calculate positions (approximation due to normalization)
        start_pos = current_pos
        end_pos = start_pos + len(chunk_text)

        yield TextChunk(
            text=chunk_text,
            chunk_index=idx,
            total_chunks=total_chunks,
            start_char=start_pos,
            end_char=end_pos,
            source_id=transcript.metadata.vimeo_id,
            source_title=transcript.metadata.title,
        )

        # Update position (accounting for overlap)
        current_pos = end_pos - chunk_overlap


def chunk_text_simple(
    text: str,
    source_id: str,
    source_title: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Iterator[TextChunk]:
    """
    Chunk plain text without TranscriptFile wrapper.

    Useful for adding new content that doesn't come from VTT files.

    Args:
        text: Plain text content
        source_id: Unique identifier for the source
        source_title: Title for the source
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks

    Yields:
        TextChunk objects
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks = split_text(text, chunk_size, chunk_overlap)
    total_chunks = len(chunks)

    current_pos = 0

    for idx, chunk_text in enumerate(chunks):
        start_pos = current_pos
        end_pos = start_pos + len(chunk_text)

        yield TextChunk(
            text=chunk_text,
            chunk_index=idx,
            total_chunks=total_chunks,
            start_char=start_pos,
            end_char=end_pos,
            source_id=source_id,
            source_title=source_title,
        )

        current_pos = end_pos - chunk_overlap


def estimate_chunk_count(total_chars: int, chunk_size: int = 1000, overlap: int = 200) -> int:
    """
    Estimate number of chunks for a given text length.

    Args:
        total_chars: Total character count
        chunk_size: Chunk size
        overlap: Overlap size

    Returns:
        Estimated number of chunks
    """
    if total_chars <= chunk_size:
        return 1

    effective_chunk = chunk_size - overlap
    return 1 + ((total_chars - chunk_size) // effective_chunk) + 1
