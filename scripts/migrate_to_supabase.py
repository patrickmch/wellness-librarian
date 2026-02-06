#!/usr/bin/env python3
"""
Migrate data from ChromaDB + SQLite to Supabase.

This script copies existing indexed data to Supabase without re-embedding.
Use this to migrate an existing deployment to Supabase storage.

Usage:
    python scripts/migrate_to_supabase.py              # Migrate all data
    python scripts/migrate_to_supabase.py --dry-run    # Preview without writing
    python scripts/migrate_to_supabase.py --verify     # Verify migration integrity
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.rag.docstore.sqlite_store import SQLiteDocStore
from backend.rag.stores.supabase_store import SupabaseStore
from backend.rag.chunking.models import ChildChunk

import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def get_chroma_collection():
    """Get the ChromaDB v2 collection."""
    settings = get_settings()
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


def migrate_parents(docstore: SQLiteDocStore, supabase_store: SupabaseStore, dry_run: bool = False):
    """Migrate parent chunks from SQLite to Supabase."""
    logger.info("Migrating parent chunks...")

    # Get all parents from SQLite
    with docstore._get_connection() as conn:
        rows = conn.execute("SELECT * FROM parent_chunks").fetchall()

    parents = [docstore._row_to_parent(row) for row in rows]
    total = len(parents)

    if dry_run:
        logger.info(f"[DRY RUN] Would migrate {total} parent chunks")
        return total

    # Batch insert
    for i in range(0, total, BATCH_SIZE):
        batch = parents[i:i + BATCH_SIZE]
        supabase_store.add_parent_batch(batch)
        logger.info(f"Migrated parents: {min(i + BATCH_SIZE, total)}/{total}")

    return total


def migrate_children(collection, supabase_store: SupabaseStore, dry_run: bool = False):
    """Migrate child chunks with embeddings from ChromaDB to Supabase."""
    logger.info("Migrating child chunks with embeddings...")

    # Get total count
    total = collection.count()
    logger.info(f"Total children in ChromaDB: {total}")

    if total == 0:
        logger.warning("No children found in ChromaDB")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Would migrate {total} child chunks")
        return total

    # Fetch all children with embeddings
    # ChromaDB doesn't support pagination, so we get all at once
    results = collection.get(
        include=["documents", "metadatas", "embeddings"],
    )

    child_ids = results["ids"]
    child_docs = results["documents"]
    child_metas = results["metadatas"]
    child_embeddings = results["embeddings"]

    # Process in batches
    migrated = 0
    for i in range(0, total, BATCH_SIZE):
        batch_ids = child_ids[i:i + BATCH_SIZE]
        batch_docs = child_docs[i:i + BATCH_SIZE]
        batch_metas = child_metas[i:i + BATCH_SIZE]
        batch_embeddings = child_embeddings[i:i + BATCH_SIZE]

        # Convert to ChildChunk objects
        children = []
        embeddings = []
        for doc, meta, emb in zip(batch_docs, batch_metas, batch_embeddings):
            child = ChildChunk.from_chroma_result(doc, meta)
            children.append(child)
            # Convert numpy array to Python list for proper serialization
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            embeddings.append(emb)

        # Insert into Supabase
        supabase_store.add_child_batch(children, embeddings)
        migrated += len(children)
        logger.info(f"Migrated children: {migrated}/{total}")

    return migrated


def migrate_feedback(docstore: SQLiteDocStore, supabase_store: SupabaseStore, dry_run: bool = False):
    """Migrate feedback from SQLite to Supabase."""
    logger.info("Migrating feedback...")

    import json

    with docstore._get_connection() as conn:
        rows = conn.execute("SELECT * FROM feedback").fetchall()

    total = len(rows)

    if total == 0:
        logger.info("No feedback to migrate")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Would migrate {total} feedback entries")
        return total

    # Insert feedback one by one (typically small volume)
    for row in rows:
        parent_ids = row["parent_ids"]
        if isinstance(parent_ids, str):
            parent_ids = json.loads(parent_ids)

        supabase_store.add_feedback(
            message_id=row["message_id"],
            feedback_type=row["feedback_type"],
            session_id=row["session_id"],
            query=row["query"],
            parent_ids=parent_ids,
        )

    logger.info(f"Migrated {total} feedback entries")
    return total


def verify_migration(docstore: SQLiteDocStore, collection, supabase_store: SupabaseStore):
    """Verify that migration was successful."""
    logger.info("Verifying migration...")

    # Get source counts
    sqlite_stats = docstore.get_stats()
    chroma_count = collection.count()
    sqlite_feedback = docstore.get_feedback_stats()

    # Get destination counts
    supabase_stats = supabase_store.get_stats()
    supabase_feedback = supabase_store.get_feedback_stats()

    print("\n=== Migration Verification ===")
    print("\nParent Chunks:")
    print(f"  SQLite:   {sqlite_stats['total_parent_chunks']}")
    print(f"  Supabase: {supabase_stats['total_parent_chunks']}")

    print("\nChild Chunks:")
    print(f"  ChromaDB: {chroma_count}")
    print(f"  Supabase: {supabase_stats['child_chunks_indexed']}")

    print("\nFeedback:")
    print(f"  SQLite:   {sqlite_feedback['total_feedback']}")
    print(f"  Supabase: {supabase_feedback['total_feedback']}")

    # Check for mismatches
    issues = []
    if sqlite_stats['total_parent_chunks'] != supabase_stats['total_parent_chunks']:
        issues.append("Parent chunk count mismatch")
    if chroma_count != supabase_stats['child_chunks_indexed']:
        issues.append("Child chunk count mismatch")
    if sqlite_feedback['total_feedback'] != supabase_feedback['total_feedback']:
        issues.append("Feedback count mismatch")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\nAll counts match! Migration verified.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate data from ChromaDB + SQLite to Supabase"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--verify", action="store_true", help="Verify migration integrity")
    parser.add_argument("--skip-parents", action="store_true", help="Skip parent migration")
    parser.add_argument("--skip-children", action="store_true", help="Skip child migration")
    parser.add_argument("--skip-feedback", action="store_true", help="Skip feedback migration")
    parser.add_argument("--create-index", action="store_true", help="Create vector index after migration")

    args = parser.parse_args()

    settings = get_settings()

    # Check for Supabase config
    if not settings.supabase_db_url:
        print("Error: SUPABASE_DB_URL not set. Add it to .env file or environment.")
        sys.exit(1)

    # Initialize stores
    docstore = SQLiteDocStore(settings.docstore_path)
    collection = get_chroma_collection()
    supabase_store = SupabaseStore()

    if args.verify:
        success = verify_migration(docstore, collection, supabase_store)
        sys.exit(0 if success else 1)

    print("\n=== Starting Migration to Supabase ===")
    if args.dry_run:
        print("[DRY RUN MODE - No data will be written]\n")

    # Migrate parents
    if not args.skip_parents:
        parent_count = migrate_parents(docstore, supabase_store, dry_run=args.dry_run)
        print(f"Parents migrated: {parent_count}")
    else:
        print("Skipping parent migration")

    # Migrate children
    if not args.skip_children:
        child_count = migrate_children(collection, supabase_store, dry_run=args.dry_run)
        print(f"Children migrated: {child_count}")
    else:
        print("Skipping child migration")

    # Migrate feedback
    if not args.skip_feedback:
        feedback_count = migrate_feedback(docstore, supabase_store, dry_run=args.dry_run)
        print(f"Feedback migrated: {feedback_count}")
    else:
        print("Skipping feedback migration")

    # Create vector index
    if args.create_index and not args.dry_run:
        import math
        stats = supabase_store.get_stats()
        child_count = stats.get("child_chunks_indexed", 0)
        lists = max(10, int(math.sqrt(child_count)))
        print(f"\nCreating IVFFlat vector index with {lists} lists...")
        supabase_store.create_vector_index(lists=lists)
        print("Vector index created.")

    print("\n=== Migration Complete ===")

    if not args.dry_run:
        print("\nRun with --verify to check migration integrity")
        print("Set STORE_BACKEND=supabase in your environment to use Supabase")


if __name__ == "__main__":
    main()
