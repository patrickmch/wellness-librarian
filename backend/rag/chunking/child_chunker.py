"""
Token-based child chunking for parent-child retrieval.

Splits parent chunks into smaller child chunks for fine-grained
semantic search. Uses tiktoken for accurate token counting.
"""

import logging
from typing import Iterator

import tiktoken

from backend.config import get_settings
from backend.rag.chunking.models import ParentChunk, ChildChunk

logger = logging.getLogger(__name__)

# Use cl100k_base encoding (GPT-4, text-embedding-3)
# This is also reasonably close to other model tokenizers
_encoder: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Get or initialize the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for

    Returns:
        Token count
    """
    encoder = get_encoder()
    return len(encoder.encode(text))


def tokens_to_text(tokens: list[int]) -> str:
    """
    Convert token IDs back to text.

    Args:
        tokens: List of token IDs

    Returns:
        Decoded text
    """
    encoder = get_encoder()
    return encoder.decode(tokens)


def split_into_children(
    parent: ParentChunk,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[ChildChunk]:
    """
    Split a parent chunk into child chunks for vector indexing.

    Args:
        parent: The parent chunk to split
        chunk_tokens: Target tokens per child (default from settings)
        overlap_tokens: Token overlap between children (default from settings)

    Returns:
        List of ChildChunk objects
    """
    settings = get_settings()
    chunk_tokens = chunk_tokens or settings.child_chunk_tokens
    overlap_tokens = overlap_tokens or settings.child_chunk_overlap

    encoder = get_encoder()
    tokens = encoder.encode(parent.text)

    if len(tokens) <= chunk_tokens:
        # Parent is small enough to be a single child
        return [ChildChunk(
            child_id=ChildChunk.generate_id(parent.parent_id, 0),
            parent_id=parent.parent_id,
            video_id=parent.video_id,
            text=parent.text,
            token_count=len(tokens),
            child_index=0,
            total_children=1,
            start_token=0,
            end_token=len(tokens),
            title=parent.title,
            category=parent.category,
            video_url=parent.video_url,
            source=parent.source,
        )]

    # Calculate step size (chunk_tokens - overlap_tokens)
    step = chunk_tokens - overlap_tokens

    children = []
    start = 0
    child_index = 0

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_token_ids = tokens[start:end]
        chunk_text = encoder.decode(chunk_token_ids)

        children.append(ChildChunk(
            child_id=ChildChunk.generate_id(parent.parent_id, child_index),
            parent_id=parent.parent_id,
            video_id=parent.video_id,
            text=chunk_text,
            token_count=len(chunk_token_ids),
            child_index=child_index,
            total_children=-1,  # Will be set after loop
            start_token=start,
            end_token=end,
            title=parent.title,
            category=parent.category,
            video_url=parent.video_url,
            source=parent.source,
        ))

        child_index += 1
        start += step

        # Don't create a tiny final chunk
        if start < len(tokens) and len(tokens) - start < overlap_tokens:
            break

    # Update total_children count
    total = len(children)
    for child in children:
        child.total_children = total

    return children


def chunk_parents_to_children(
    parents: list[ParentChunk],
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> Iterator[ChildChunk]:
    """
    Generator that yields child chunks from a list of parents.

    Args:
        parents: List of parent chunks
        chunk_tokens: Target tokens per child
        overlap_tokens: Token overlap between children

    Yields:
        ChildChunk objects
    """
    for parent in parents:
        children = split_into_children(parent, chunk_tokens, overlap_tokens)
        yield from children


def estimate_child_count(parent: ParentChunk) -> int:
    """
    Estimate how many children a parent will produce.

    Useful for progress tracking during ingestion.

    Args:
        parent: The parent chunk

    Returns:
        Estimated number of children
    """
    settings = get_settings()
    chunk_tokens = settings.child_chunk_tokens
    overlap_tokens = settings.child_chunk_overlap

    if parent.token_count <= chunk_tokens:
        return 1

    step = chunk_tokens - overlap_tokens
    return max(1, (parent.token_count - overlap_tokens) // step + 1)
