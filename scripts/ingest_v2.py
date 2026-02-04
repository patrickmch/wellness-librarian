#!/usr/bin/env python3
"""
Enhanced pipeline ingestion CLI for parent-child indexing.

Creates the v2 vector store with:
- Token-based parent chunks (500-2000 tokens) in SQLite docstore
- Token-based child chunks (250 tokens) in ChromaDB
- Voyage AI embeddings

Usage:
    python scripts/ingest_v2.py              # Ingest all (skip existing)
    python scripts/ingest_v2.py --force      # Re-ingest all
    python scripts/ingest_v2.py --stats      # Show collection stats
    python scripts/ingest_v2.py --reset      # Reset and re-ingest
    python scripts/ingest_v2.py --estimate   # Estimate costs
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.ingestion.loader import (
    load_transcripts,
    get_transcript_count,
    get_categories,
)
from backend.rag.chunking import (
    chunk_transcript,
    split_into_children,
    count_tokens,
    ParentChunk,
    ChildChunk,
)
from backend.rag.docstore.sqlite_store import SQLiteDocStore
from backend.rag.providers.voyage_provider import VoyageEmbeddingProvider

import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class V2IngestionStats:
    """Statistics from v2 ingestion."""

    transcripts_processed: int = 0
    parents_created: int = 0
    children_created: int = 0
    total_parent_tokens: int = 0
    total_child_tokens: int = 0
    embedding_cost_estimate: float = 0.0
    duration_seconds: float = 0.0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def __str__(self) -> str:
        return (
            f"Transcripts: {self.transcripts_processed}\n"
            f"Parent chunks: {self.parents_created} ({self.total_parent_tokens:,} tokens)\n"
            f"Child chunks: {self.children_created} ({self.total_child_tokens:,} tokens)\n"
            f"Estimated cost: ${self.embedding_cost_estimate:.4f}\n"
            f"Duration: {self.duration_seconds:.1f}s\n"
            f"Errors: {len(self.errors)}"
        )


def get_v2_collection(settings=None) -> chromadb.Collection:
    """Get or create the v2 ChromaDB collection."""
    settings = settings or get_settings()

    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )

    return client.get_or_create_collection(
        name=settings.chroma_collection_v2,
        metadata={"hnsw:space": "cosine"},
    )


def reset_v2_collection(settings=None):
    """Reset the v2 collection and docstore."""
    settings = settings or get_settings()

    # Reset ChromaDB collection
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )

    try:
        client.delete_collection(settings.chroma_collection_v2)
        logger.info(f"Deleted collection: {settings.chroma_collection_v2}")
    except (ValueError, chromadb.errors.NotFoundError):
        logger.info(f"Collection {settings.chroma_collection_v2} doesn't exist")

    # Reset docstore
    docstore = SQLiteDocStore(settings.docstore_path)
    count = docstore.clear()
    logger.info(f"Cleared {count} parents from docstore")


def show_stats():
    """Display v2 collection statistics."""
    settings = get_settings()

    print("\n=== Source Directory ===")
    print(f"Path: {settings.transcript_source_dir}")

    try:
        transcript_count = get_transcript_count(settings.transcript_source_dir)
        categories = get_categories(settings.transcript_source_dir)
        print(f"Transcripts: {transcript_count}")
        print(f"Categories: {len(categories)}")
        for cat in categories:
            print(f"  - {cat}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("\n=== V2 Vector Store (Children) ===")
    try:
        collection = get_v2_collection(settings)
        child_count = collection.count()
        print(f"Child chunks indexed: {child_count}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== DocStore (Parents) ===")
    try:
        docstore = SQLiteDocStore(settings.docstore_path)
        stats = docstore.get_stats()
        print(f"Parent chunks: {stats['total_parent_chunks']}")
        print(f"Unique videos: {stats['unique_videos']}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print("Parents by category:")
        for cat, count in sorted(stats['categories'].items()):
            print(f"  - {cat}: {count}")
    except Exception as e:
        print(f"Error: {e}")


def estimate_costs():
    """Estimate Voyage embedding costs for v2 ingestion."""
    settings = get_settings()

    print("\n=== V2 Cost Estimate ===")
    print("Analyzing transcripts...")

    total_parent_tokens = 0
    total_child_tokens = 0
    parent_count = 0
    child_count = 0

    for transcript in load_transcripts(settings.transcript_source_dir):
        parents = chunk_transcript(transcript)
        for parent in parents:
            parent_count += 1
            total_parent_tokens += parent.token_count

            children = split_into_children(parent)
            for child in children:
                child_count += 1
                total_child_tokens += child.token_count

    # Voyage pricing: $0.06 per 1M tokens for voyage-3
    cost_per_million = 0.06
    embedding_cost = (total_child_tokens / 1_000_000) * cost_per_million

    print(f"\nParent chunks: {parent_count:,}")
    print(f"Parent tokens: {total_parent_tokens:,}")
    print(f"Child chunks: {child_count:,}")
    print(f"Child tokens: {total_child_tokens:,}")
    print(f"\nEstimated Voyage embedding cost: ${embedding_cost:.4f}")
    print("(Only child chunks are embedded; parents are stored as-is)")


def ingest_v2(
    source_dir: Path | None = None,
    force: bool = False,
    progress_callback=None,
) -> V2IngestionStats:
    """
    Ingest transcripts into v2 parent-child structure.

    Args:
        source_dir: Override transcript source directory
        force: Re-ingest even if exists
        progress_callback: Optional progress callback(current, total, message)

    Returns:
        V2IngestionStats with results
    """
    settings = get_settings()
    source_dir = source_dir or settings.transcript_source_dir

    stats = V2IngestionStats()
    start_time = time.time()

    # Initialize components
    docstore = SQLiteDocStore(settings.docstore_path)
    collection = get_v2_collection(settings)

    try:
        embedding_provider = VoyageEmbeddingProvider()
    except ValueError as e:
        stats.errors.append(str(e))
        return stats

    # Load transcripts
    transcripts = list(load_transcripts(source_dir))
    total_transcripts = len(transcripts)

    if progress_callback:
        progress_callback(0, total_transcripts, "Starting...")

    for i, transcript in enumerate(transcripts):
        try:
            # Check if already indexed
            video_id = transcript.file_id
            existing_parents = docstore.get_by_video_id(video_id)

            if existing_parents and not force:
                logger.debug(f"Skipping {transcript.metadata.title} (already indexed)")
                if progress_callback:
                    progress_callback(i + 1, total_transcripts, f"Skipped: {transcript.metadata.title[:40]}")
                continue

            # Delete existing if force re-index
            if existing_parents and force:
                docstore.delete_by_video_id(video_id)
                # Delete children from ChromaDB
                try:
                    existing_children = collection.get(
                        where={"video_id": video_id}
                    )
                    if existing_children["ids"]:
                        collection.delete(ids=existing_children["ids"])
                except Exception:
                    pass

            # Create parent chunks
            parents = chunk_transcript(transcript)

            if not parents:
                logger.warning(f"No parent chunks created for {transcript.metadata.title}")
                continue

            # Store parents in docstore
            docstore.add_batch(parents)
            stats.parents_created += len(parents)
            stats.total_parent_tokens += sum(p.token_count for p in parents)

            # Create and embed child chunks
            all_children: list[ChildChunk] = []
            for parent in parents:
                children = split_into_children(parent)
                all_children.extend(children)

            if not all_children:
                continue

            # Embed children with Voyage
            child_texts = [c.text for c in all_children]
            embeddings = embedding_provider.embed_texts(child_texts, input_type="document")

            # Store children in ChromaDB
            collection.add(
                ids=[c.child_id for c in all_children],
                documents=child_texts,
                embeddings=embeddings,
                metadatas=[c.to_chroma_metadata() for c in all_children],
            )

            stats.children_created += len(all_children)
            stats.total_child_tokens += sum(c.token_count for c in all_children)
            stats.transcripts_processed += 1

            if progress_callback:
                progress_callback(
                    i + 1,
                    total_transcripts,
                    f"Indexed: {transcript.metadata.title[:40]}"
                )

        except Exception as e:
            error_msg = f"Error processing {transcript.metadata.title}: {e}"
            logger.exception(error_msg)
            stats.errors.append(error_msg)

    # Calculate cost estimate
    cost_per_million = 0.06  # Voyage pricing
    stats.embedding_cost_estimate = (stats.total_child_tokens / 1_000_000) * cost_per_million
    stats.duration_seconds = time.time() - start_time

    return stats


def progress_callback(current: int, total: int, message: str):
    """Display progress during ingestion."""
    percent = (current / total * 100) if total > 0 else 0
    bar_width = 40
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {percent:5.1f}% ({current}/{total}) {message[:50]:<50}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest transcripts into v2 parent-child structure"
    )
    parser.add_argument("--force", action="store_true", help="Re-ingest all transcripts")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--reset", action="store_true", help="Reset collection before ingesting")
    parser.add_argument("--estimate", action="store_true", help="Estimate costs without ingesting")
    parser.add_argument("--source-dir", type=str, help="Override transcript source directory")

    args = parser.parse_args()

    settings = get_settings()

    # Check for API key
    if not settings.voyage_api_key and not (args.stats or args.estimate):
        print("Error: VOYAGE_API_KEY not set. Add it to .env file or environment.")
        sys.exit(1)

    if args.stats:
        show_stats()
        return

    if args.estimate:
        estimate_costs()
        return

    if args.reset:
        print("Resetting v2 collection and docstore...")
        reset_v2_collection(settings)
        print("Reset complete.")

    source_dir = Path(args.source_dir) if args.source_dir else None

    print("\n=== Starting V2 Ingestion ===")
    print(f"Source: {source_dir or settings.transcript_source_dir}")
    print(f"Force re-index: {args.force or args.reset}")
    print(f"Embedding model: {settings.voyage_embedding_model}")
    print(f"Parent chunk tokens: {settings.parent_min_tokens}-{settings.parent_max_tokens}")
    print(f"Child chunk tokens: {settings.child_chunk_tokens} (overlap: {settings.child_chunk_overlap})")

    try:
        stats = ingest_v2(
            source_dir=source_dir,
            force=args.force or args.reset,
            progress_callback=progress_callback,
        )

        print("\n\n=== V2 Ingestion Complete ===")
        print(stats)

        if stats.errors:
            print("\nErrors:")
            for error in stats.errors[:10]:  # Limit error output
                print(f"  - {error}")
            if len(stats.errors) > 10:
                print(f"  ... and {len(stats.errors) - 10} more errors")

    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nIngestion failed: {e}")
        logger.exception("Ingestion error")
        sys.exit(1)


if __name__ == "__main__":
    main()
