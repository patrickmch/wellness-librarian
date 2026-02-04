"""
Parent chunking for transcript documents.

Creates parent chunks from transcripts that are:
- Large enough for LLM context (500-2000 tokens)
- Boundaries at natural text breaks when possible
- Stored in docstore for retrieval after child matching
"""

import logging
from typing import Iterator

import tiktoken

from backend.config import get_settings
from backend.rag.chunking.models import ParentChunk
from backend.ingestion.loader import TranscriptFile

logger = logging.getLogger(__name__)

# Use cl100k_base encoding (GPT-4, text-embedding-3)
_encoder: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Get or initialize the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(get_encoder().encode(text))


def find_timestamp_for_char_position(
    timestamps: list[tuple[float, float, str]],
    char_position: int
) -> int:
    """
    Map character position in transcript to video timestamp.

    The timestamps list contains (start_sec, end_sec, text) tuples from VTT parsing.
    We accumulate character counts to find which timestamp segment contains
    the given character position.

    Args:
        timestamps: List of (start_seconds, end_seconds, text) from VTT
        char_position: Character offset in the joined transcript

    Returns:
        Video timestamp in seconds (rounded down)
    """
    if not timestamps:
        return 0

    cumulative_chars = 0
    for start_sec, end_sec, text in timestamps:
        segment_len = len(text) + 1  # +1 for space between segments
        if cumulative_chars + segment_len >= char_position:
            return int(start_sec)
        cumulative_chars += segment_len

    # Default to last known timestamp
    return int(timestamps[-1][0])


def find_sentence_boundary(text: str, target_pos: int, search_range: int = 200) -> int:
    """
    Find a sentence boundary near target position.

    Searches for sentence-ending punctuation followed by whitespace.

    Args:
        text: The full text
        target_pos: Target character position
        search_range: How far to search from target

    Returns:
        Position after the sentence boundary, or target_pos if none found
    """
    # Search backward first (prefer shorter chunks)
    search_start = max(0, target_pos - search_range)
    search_end = min(len(text), target_pos + search_range)
    search_text = text[search_start:search_end]

    # Look for sentence endings: . or ? or ! followed by space/newline
    best_pos = None
    for i, char in enumerate(search_text):
        if char in ".?!" and i + 1 < len(search_text) and search_text[i + 1] in " \n":
            actual_pos = search_start + i + 1
            # Prefer boundaries before target (shorter chunks)
            if actual_pos <= target_pos:
                best_pos = actual_pos
            elif best_pos is None:
                best_pos = actual_pos
                break  # Take first one after target

    return best_pos if best_pos else target_pos


def chunk_transcript(
    transcript: TranscriptFile,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[ParentChunk]:
    """
    Chunk a transcript into parent chunks.

    Uses fixed-size chunking with sentence boundary awareness.
    For spoken transcripts, this typically works better than
    topic-based segmentation.

    Args:
        transcript: The transcript to chunk
        min_tokens: Minimum tokens per chunk (default from settings)
        max_tokens: Maximum tokens per chunk (default from settings)

    Returns:
        List of ParentChunk objects
    """
    settings = get_settings()
    min_tokens = min_tokens or settings.parent_min_tokens
    max_tokens = max_tokens or settings.parent_max_tokens

    text = transcript.content
    if not text.strip():
        return []

    encoder = get_encoder()
    tokens = encoder.encode(text)
    total_tokens = len(tokens)

    # Get timestamps for deep-linking
    timestamps = getattr(transcript, 'timestamps', [])

    if total_tokens <= max_tokens:
        # Entire transcript fits in one chunk
        return [ParentChunk(
            parent_id=ParentChunk.generate_id(transcript.file_id, 0),
            video_id=transcript.file_id,
            text=text,
            token_count=total_tokens,
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=len(text),
            title=transcript.metadata.title,
            category=transcript.category,
            duration=transcript.metadata.duration_formatted,
            duration_seconds=transcript.metadata.duration_seconds,
            video_url=transcript.metadata.video_url,
            start_time_seconds=0,  # Start of video
            source=transcript.metadata.source,
            access_level=transcript.metadata.access_level,
            created_date=transcript.metadata.created_time.isoformat() if transcript.metadata.created_time else None,
            description=transcript.metadata.description,
            tags=transcript.metadata.tags,
        )]

    # Need to split into multiple chunks
    chunks = []
    chunk_index = 0
    start_char = 0

    while start_char < len(text):
        # Target end position based on max tokens
        # Rough estimate: 4 chars per token on average
        target_end = start_char + (max_tokens * 4)

        if target_end >= len(text):
            # Last chunk - take everything
            chunk_text = text[start_char:]
            end_char = len(text)
        else:
            # Find sentence boundary near target
            end_char = find_sentence_boundary(text, target_end)

            # Verify token count and adjust if needed
            chunk_text = text[start_char:end_char]
            chunk_tokens = count_tokens(chunk_text)

            # If over max, search earlier
            while chunk_tokens > max_tokens and end_char > start_char + 100:
                end_char = find_sentence_boundary(text, end_char - 200)
                chunk_text = text[start_char:end_char]
                chunk_tokens = count_tokens(chunk_text)

            # If under min, extend forward
            while chunk_tokens < min_tokens and end_char < len(text):
                end_char = find_sentence_boundary(text, end_char + 200)
                if end_char >= len(text):
                    end_char = len(text)
                chunk_text = text[start_char:end_char]
                chunk_tokens = count_tokens(chunk_text)

        chunk_text = text[start_char:end_char].strip()
        if not chunk_text:
            break

        chunk_tokens = count_tokens(chunk_text)

        # Calculate video timestamp for this chunk's start position
        start_time = find_timestamp_for_char_position(timestamps, start_char)

        chunks.append(ParentChunk(
            parent_id=ParentChunk.generate_id(transcript.file_id, chunk_index),
            video_id=transcript.file_id,
            text=chunk_text,
            token_count=chunk_tokens,
            chunk_index=chunk_index,
            total_chunks=-1,  # Updated after loop
            start_char=start_char,
            end_char=end_char,
            title=transcript.metadata.title,
            category=transcript.category,
            duration=transcript.metadata.duration_formatted,
            duration_seconds=transcript.metadata.duration_seconds,
            video_url=transcript.metadata.video_url,
            start_time_seconds=start_time,
            source=transcript.metadata.source,
            access_level=transcript.metadata.access_level,
            created_date=transcript.metadata.created_time.isoformat() if transcript.metadata.created_time else None,
            description=transcript.metadata.description,
            tags=transcript.metadata.tags,
        ))

        chunk_index += 1
        start_char = end_char

    # Update total_chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    return chunks


def chunk_text_simple(
    text: str,
    video_id: str,
    title: str,
    category: str,
    video_url: str = "",
    duration: str = "",
    duration_seconds: int = 0,
    source: str = "vimeo",
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[ParentChunk]:
    """
    Chunk plain text into parent chunks.

    Convenience function when you don't have a TranscriptFile.

    Args:
        text: Text to chunk
        video_id: Source video ID
        title: Video title
        category: Content category
        video_url: URL to video
        duration: Formatted duration string
        duration_seconds: Duration in seconds
        source: Content source
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk

    Returns:
        List of ParentChunk objects
    """
    settings = get_settings()
    min_tokens = min_tokens or settings.parent_min_tokens
    max_tokens = max_tokens or settings.parent_max_tokens

    if not text.strip():
        return []

    encoder = get_encoder()
    tokens = encoder.encode(text)
    total_tokens = len(tokens)

    if total_tokens <= max_tokens:
        return [ParentChunk(
            parent_id=ParentChunk.generate_id(video_id, 0),
            video_id=video_id,
            text=text,
            token_count=total_tokens,
            chunk_index=0,
            total_chunks=1,
            start_char=0,
            end_char=len(text),
            title=title,
            category=category,
            duration=duration,
            duration_seconds=duration_seconds,
            video_url=video_url,
            source=source,
        )]

    # Reuse chunking logic with minimal metadata
    chunks = []
    chunk_index = 0
    start_char = 0

    while start_char < len(text):
        target_end = start_char + (max_tokens * 4)

        if target_end >= len(text):
            chunk_text = text[start_char:]
            end_char = len(text)
        else:
            end_char = find_sentence_boundary(text, target_end)
            chunk_text = text[start_char:end_char]
            chunk_tokens = count_tokens(chunk_text)

            while chunk_tokens > max_tokens and end_char > start_char + 100:
                end_char = find_sentence_boundary(text, end_char - 200)
                chunk_text = text[start_char:end_char]
                chunk_tokens = count_tokens(chunk_text)

        chunk_text = text[start_char:end_char].strip()
        if not chunk_text:
            break

        chunk_tokens = count_tokens(chunk_text)

        chunks.append(ParentChunk(
            parent_id=ParentChunk.generate_id(video_id, chunk_index),
            video_id=video_id,
            text=chunk_text,
            token_count=chunk_tokens,
            chunk_index=chunk_index,
            total_chunks=-1,
            start_char=start_char,
            end_char=end_char,
            title=title,
            category=category,
            duration=duration,
            duration_seconds=duration_seconds,
            video_url=video_url,
            source=source,
        ))

        chunk_index += 1
        start_char = end_char

    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    return chunks


def estimate_parent_count(text: str) -> int:
    """
    Estimate how many parent chunks a text will produce.

    Args:
        text: The text to estimate

    Returns:
        Estimated number of parent chunks
    """
    settings = get_settings()
    max_tokens = settings.parent_max_tokens

    total_tokens = count_tokens(text)
    if total_tokens <= max_tokens:
        return 1

    return max(1, total_tokens // max_tokens + 1)
