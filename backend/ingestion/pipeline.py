"""
Ingestion pipeline for processing transcripts into vector store.
Handles batch processing with progress tracking and rate limiting.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from backend.config import get_settings
from backend.ingestion.loader import (
    load_transcripts,
    load_single_transcript,
    TranscriptFile,
    get_transcript_count,
)
from backend.ingestion.metadata import ChunkMetadata
from backend.rag.chunker import chunk_transcript, TextChunk
from backend.rag.embeddings import embed_texts, estimate_embedding_cost
from backend.rag.vectorstore import (
    upsert_documents,
    document_exists,
    delete_by_vimeo_id,
    get_document_count,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""
    transcripts_processed: int
    chunks_created: int
    chunks_skipped: int  # Already existed
    embedding_cost_estimate: float
    duration_seconds: float
    errors: list[str]

    def __str__(self) -> str:
        return (
            f"Ingestion Complete:\n"
            f"  Transcripts: {self.transcripts_processed}\n"
            f"  Chunks created: {self.chunks_created}\n"
            f"  Chunks skipped: {self.chunks_skipped}\n"
            f"  Est. cost: ${self.embedding_cost_estimate:.4f}\n"
            f"  Duration: {self.duration_seconds:.1f}s\n"
            f"  Errors: {len(self.errors)}"
        )


def ingest_transcript(
    transcript: TranscriptFile,
    force: bool = False,
) -> tuple[int, int]:
    """
    Ingest a single transcript into the vector store.

    Args:
        transcript: TranscriptFile to ingest
        force: If True, replace existing chunks

    Returns:
        Tuple of (chunks_created, chunks_skipped)
    """
    # Generate chunks
    chunks = list(chunk_transcript(transcript))
    total_chunks = len(chunks)

    if not chunks:
        logger.warning(f"No chunks generated for {transcript.metadata.title}")
        return 0, 0

    # Check for existing chunks if not forcing
    if not force:
        first_chunk_id = chunks[0].chunk_id
        if document_exists(first_chunk_id):
            logger.debug(f"Skipping {transcript.metadata.title} - already indexed")
            return 0, total_chunks

    # If forcing, delete existing chunks for this video
    if force:
        deleted = delete_by_vimeo_id(transcript.metadata.vimeo_id)
        if deleted:
            logger.debug(f"Deleted {deleted} existing chunks for {transcript.metadata.title}")

    # Prepare data for vector store
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_meta = ChunkMetadata.from_video_metadata(
            video_meta=transcript.metadata,
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )

        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        metadatas.append(chunk_meta.to_dict())

    # Generate embeddings and add to store
    upsert_documents(ids=ids, documents=documents, metadatas=metadatas)

    return total_chunks, 0


def ingest_all(
    source_dir: Path | None = None,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    batch_size: int = 100,
) -> IngestionStats:
    """
    Ingest all transcripts from source directory.

    Args:
        source_dir: Directory containing transcripts (default from settings)
        force: If True, re-index all transcripts
        progress_callback: Optional callback(current, total, message)
        batch_size: Batch size for embedding generation

    Returns:
        IngestionStats with results
    """
    settings = get_settings()
    source_dir = Path(source_dir or settings.transcript_source_dir)

    start_time = time.time()
    errors = []

    # Get total count for progress
    total_transcripts = get_transcript_count(source_dir)
    logger.info(f"Starting ingestion of {total_transcripts} transcripts from {source_dir}")

    # Collect all chunks first for efficient batch embedding
    all_chunks: list[tuple[TextChunk, TranscriptFile]] = []
    transcripts_to_process = []

    for i, transcript in enumerate(load_transcripts(source_dir)):
        if progress_callback:
            progress_callback(i + 1, total_transcripts, f"Loading: {transcript.metadata.title}")

        # Check if already indexed (unless forcing)
        if not force:
            chunks = list(chunk_transcript(transcript))
            if chunks and document_exists(chunks[0].chunk_id):
                logger.debug(f"Skipping {transcript.metadata.title} - already indexed")
                continue

        transcripts_to_process.append(transcript)

    if not transcripts_to_process:
        logger.info("All transcripts already indexed")
        return IngestionStats(
            transcripts_processed=0,
            chunks_created=0,
            chunks_skipped=total_transcripts,
            embedding_cost_estimate=0.0,
            duration_seconds=time.time() - start_time,
            errors=[],
        )

    # Process transcripts and collect chunks
    for transcript in transcripts_to_process:
        try:
            chunks = list(chunk_transcript(transcript))
            for chunk in chunks:
                all_chunks.append((chunk, transcript))
        except Exception as e:
            error_msg = f"Error processing {transcript.metadata.title}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    if not all_chunks:
        return IngestionStats(
            transcripts_processed=len(transcripts_to_process),
            chunks_created=0,
            chunks_skipped=0,
            embedding_cost_estimate=0.0,
            duration_seconds=time.time() - start_time,
            errors=errors,
        )

    # Prepare data for batch embedding
    ids = []
    documents = []
    metadatas = []

    for chunk, transcript in all_chunks:
        chunk_meta = ChunkMetadata.from_video_metadata(
            video_meta=transcript.metadata,
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )

        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        metadatas.append(chunk_meta.to_dict())

    # Estimate cost
    cost_estimate = estimate_embedding_cost(documents)
    logger.info(f"Generating embeddings for {len(documents)} chunks (est. ${cost_estimate:.4f})")

    if progress_callback:
        progress_callback(0, len(documents), "Generating embeddings...")

    # Generate embeddings in batches with progress
    embeddings = embed_texts(documents, batch_size=batch_size)

    if progress_callback:
        progress_callback(len(documents), len(documents), "Storing in vector database...")

    # Add to vector store
    upsert_documents(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        batch_size=batch_size,
    )

    duration = time.time() - start_time

    stats = IngestionStats(
        transcripts_processed=len(transcripts_to_process),
        chunks_created=len(all_chunks),
        chunks_skipped=total_transcripts - len(transcripts_to_process),
        embedding_cost_estimate=cost_estimate,
        duration_seconds=duration,
        errors=errors,
    )

    logger.info(str(stats))
    return stats


def ingest_single(
    filepath: Path,
    source_dir: Path | None = None,
    force: bool = False,
) -> IngestionStats:
    """
    Ingest a single transcript file.

    Args:
        filepath: Path to transcript file
        source_dir: Root directory for metadata lookup
        force: If True, replace existing chunks

    Returns:
        IngestionStats with results
    """
    settings = get_settings()
    source_dir = Path(source_dir or settings.transcript_source_dir)

    start_time = time.time()
    errors = []

    try:
        transcript = load_single_transcript(filepath, source_dir)
        chunks_created, chunks_skipped = ingest_transcript(transcript, force=force)

        # Estimate cost for this transcript
        chunks = list(chunk_transcript(transcript))
        cost_estimate = estimate_embedding_cost([c.text for c in chunks]) if chunks_created > 0 else 0.0

        return IngestionStats(
            transcripts_processed=1,
            chunks_created=chunks_created,
            chunks_skipped=chunks_skipped,
            embedding_cost_estimate=cost_estimate,
            duration_seconds=time.time() - start_time,
            errors=errors,
        )

    except Exception as e:
        error_msg = f"Error ingesting {filepath}: {e}"
        logger.error(error_msg)
        return IngestionStats(
            transcripts_processed=0,
            chunks_created=0,
            chunks_skipped=0,
            embedding_cost_estimate=0.0,
            duration_seconds=time.time() - start_time,
            errors=[error_msg],
        )
