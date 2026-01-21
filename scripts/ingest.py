#!/usr/bin/env python3
"""
Batch ingestion CLI for wellness transcripts.

Usage:
    python scripts/ingest.py              # Ingest all (skip existing)
    python scripts/ingest.py --force      # Re-ingest all
    python scripts/ingest.py --stats      # Show collection stats
    python scripts/ingest.py --reset      # Reset collection and re-ingest
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.ingestion.pipeline import ingest_all, IngestionStats
from backend.ingestion.loader import get_transcript_count, get_categories
from backend.rag.vectorstore import get_collection_stats, reset_collection, get_document_count
from backend.rag.embeddings import estimate_embedding_cost
from backend.rag.chunker import chunk_transcript
from backend.ingestion.loader import load_transcripts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def show_stats():
    """Display collection statistics."""
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

    print("\n=== Vector Store ===")
    try:
        stats = get_collection_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Unique videos: {stats['unique_videos']}")
        print("Chunks by category:")
        for cat, count in sorted(stats['categories'].items()):
            print(f"  - {cat}: {count}")
    except Exception as e:
        print(f"Error accessing vector store: {e}")


def estimate_costs():
    """Estimate embedding costs before ingestion."""
    settings = get_settings()

    print("\n=== Cost Estimate ===")

    all_texts = []
    for transcript in load_transcripts(settings.transcript_source_dir):
        for chunk in chunk_transcript(transcript):
            all_texts.append(chunk.text)

    cost = estimate_embedding_cost(all_texts)
    print(f"Total chunks: {len(all_texts)}")
    print(f"Estimated embedding cost: ${cost:.4f}")


def progress_callback(current: int, total: int, message: str):
    """Display progress during ingestion."""
    percent = (current / total * 100) if total > 0 else 0
    bar_width = 40
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {percent:5.1f}% ({current}/{total}) {message[:50]:<50}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Ingest wellness transcripts into vector store")
    parser.add_argument("--force", action="store_true", help="Re-ingest all transcripts")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--reset", action="store_true", help="Reset collection before ingesting")
    parser.add_argument("--estimate", action="store_true", help="Estimate costs without ingesting")
    parser.add_argument("--source-dir", type=str, help="Override transcript source directory")

    args = parser.parse_args()

    settings = get_settings()

    # Check for API key
    if not settings.openai_api_key and not (args.stats or args.estimate):
        print("Error: OPENAI_API_KEY not set. Add it to .env file or environment.")
        sys.exit(1)

    if args.stats:
        show_stats()
        return

    if args.estimate:
        estimate_costs()
        return

    if args.reset:
        print("Resetting collection...")
        reset_collection()
        print("Collection reset complete.")

    source_dir = Path(args.source_dir) if args.source_dir else None

    print("\n=== Starting Ingestion ===")
    print(f"Source: {source_dir or settings.transcript_source_dir}")
    print(f"Force re-index: {args.force or args.reset}")

    try:
        stats = ingest_all(
            source_dir=source_dir,
            force=args.force or args.reset,
            progress_callback=progress_callback,
        )

        print("\n\n=== Ingestion Complete ===")
        print(stats)

        if stats.errors:
            print("\nErrors:")
            for error in stats.errors:
                print(f"  - {error}")

    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nIngestion failed: {e}")
        logger.exception("Ingestion error")
        sys.exit(1)


if __name__ == "__main__":
    main()
