#!/usr/bin/env python3
"""
Add a single transcript to the vector store.

Usage:
    python scripts/add_transcript.py /path/to/transcript.vtt
    python scripts/add_transcript.py /path/to/transcript.vtt --force
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.ingestion.pipeline import ingest_single

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Add a single transcript to vector store")
    parser.add_argument("filepath", type=str, help="Path to transcript file (VTT)")
    parser.add_argument("--force", action="store_true", help="Replace if exists")
    parser.add_argument("--source-dir", type=str, help="Override source directory for metadata lookup")

    args = parser.parse_args()

    settings = get_settings()

    if not settings.openai_api_key:
        print("Error: OPENAI_API_KEY not set. Add it to .env file or environment.")
        sys.exit(1)

    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    source_dir = Path(args.source_dir) if args.source_dir else None

    print(f"Ingesting: {filepath}")
    print(f"Force: {args.force}")

    try:
        stats = ingest_single(
            filepath=filepath,
            source_dir=source_dir,
            force=args.force,
        )

        print("\n=== Result ===")
        print(stats)

        if stats.errors:
            print("\nErrors:")
            for error in stats.errors:
                print(f"  - {error}")
            sys.exit(1)

    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
