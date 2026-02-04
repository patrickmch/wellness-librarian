"""
Chunking package for parent-child document splitting.

Provides:
- ParentChunk / ChildChunk data models
- Token-based parent chunking (500-2000 tokens)
- Token-based child chunking (250 tokens with overlap)
"""

from backend.rag.chunking.models import ParentChunk, ChildChunk
from backend.rag.chunking.parent_chunker import (
    chunk_transcript,
    chunk_text_simple,
    count_tokens,
    estimate_parent_count,
)
from backend.rag.chunking.child_chunker import (
    split_into_children,
    chunk_parents_to_children,
    estimate_child_count,
)

__all__ = [
    # Data models
    "ParentChunk",
    "ChildChunk",
    # Parent chunking
    "chunk_transcript",
    "chunk_text_simple",
    "count_tokens",
    "estimate_parent_count",
    # Child chunking
    "split_into_children",
    "chunk_parents_to_children",
    "estimate_child_count",
]
