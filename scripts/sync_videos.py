#!/usr/bin/env python3
"""
Video sync CLI for incremental video ingestion.

Discovers new videos from Vimeo and YouTube, downloads transcripts,
and ingests them into the RAG pipeline.

Usage:
    python scripts/sync_videos.py              # Full sync (discover + download + ingest)
    python scripts/sync_videos.py --discover-only   # Just discover, don't process
    python scripts/sync_videos.py --platform vimeo  # Single platform
    python scripts/sync_videos.py --force           # Re-sync all (ignore tracking)
    python scripts/sync_videos.py --status          # Show sync stats
    python scripts/sync_videos.py --dry-run         # Preview without changes
    python scripts/sync_videos.py --ingest-only     # Just ingest pending videos
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.sync.orchestrator import SyncOrchestrator, SyncStats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def progress_callback(current: int, total: int, message: str):
    """Display progress during sync."""
    percent = (current / total * 100) if total > 0 else 0
    bar_width = 40
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {percent:5.1f}% ({current}/{total}) {message[:50]:<50}", end="", flush=True)


def show_status():
    """Display current sync status."""
    print("\n=== Video Sync Status ===\n")

    try:
        orchestrator = SyncOrchestrator()
        status = orchestrator.get_status()

        if "error" in status:
            print(f"Error: {status['error']}")
            return

        print(f"Total tracked videos: {status.get('total', 0)}")
        print(f"\nBy status:")
        print(f"  Pending:        {status.get('pending', 0)}")
        print(f"  Ingested:       {status.get('ingested', 0)}")
        print(f"  No transcript:  {status.get('no_transcript', 0)}")
        print(f"  Failed:         {status.get('failed', 0)}")
        print(f"  Skipped:        {status.get('skipped', 0)}")

        by_platform = status.get("by_platform", {})
        if by_platform:
            print(f"\nBy platform:")
            for platform, count in by_platform.items():
                print(f"  {platform.capitalize()}: {count}")

    except Exception as e:
        print(f"Error getting status: {e}")
        print("\nMake sure SUPABASE_URL and SUPABASE_KEY are set in .env")


def show_config():
    """Display current sync configuration."""
    settings = get_settings()

    print("\n=== Sync Configuration ===\n")
    print(f"Sync enabled: {settings.sync_enabled}")
    print(f"Transcript directory: {settings.sync_transcript_dir}")
    print(f"\nVimeo:")
    print(f"  Access token: {'***' + settings.vimeo_access_token[-4:] if settings.vimeo_access_token else 'Not set'}")
    print(f"\nYouTube:")
    print(f"  Channel URL: {settings.youtube_channel_url or 'Not set'}")
    print(f"  Playlist IDs: {settings.youtube_playlist_list or 'Not set'}")
    print(f"\nSupabase:")
    print(f"  URL: {settings.supabase_url or 'Not set'}")
    print(f"  Key: {'***' + settings.supabase_key[-4:] if settings.supabase_key else 'Not set'}")


def run_sync(
    platform: str = None,
    discover_only: bool = False,
    ingest_only: bool = False,
    force: bool = False,
    dry_run: bool = False,
    limit: int = None,
):
    """Run the sync process."""
    settings = get_settings()

    print("\n" + "=" * 60)
    print("Video Sync")
    print("=" * 60)

    mode_parts = []
    if discover_only:
        mode_parts.append("DISCOVER ONLY")
    if ingest_only:
        mode_parts.append("INGEST ONLY")
    if dry_run:
        mode_parts.append("DRY RUN")
    if force:
        mode_parts.append("FORCE")

    print(f"Mode: {' | '.join(mode_parts) if mode_parts else 'FULL SYNC'}")
    if platform:
        print(f"Platform: {platform}")
    if limit:
        print(f"Limit: {limit} videos per source")
    print(f"Transcript dir: {settings.sync_transcript_dir}")
    print()

    try:
        orchestrator = SyncOrchestrator()

        if ingest_only:
            # Just ingest pending videos
            print("Ingesting pending videos...")
            stats = orchestrator.ingest_pending(
                force=force,
                progress_callback=progress_callback,
            )
        else:
            # Full sync or discovery
            stats = orchestrator.sync(
                platform=platform,
                discover_only=discover_only,
                force=force,
                dry_run=dry_run,
                limit=limit,
                progress_callback=progress_callback,
            )

        print("\n\n" + "=" * 60)
        print("SYNC COMPLETE")
        print("=" * 60)
        print(stats)

        if stats.errors:
            print("\nErrors:")
            for error in stats.errors[:10]:
                print(f"  - {error}")
            if len(stats.errors) > 10:
                print(f"  ... and {len(stats.errors) - 10} more errors")

        if dry_run:
            print("\n[DRY RUN] No changes were made")

    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        print("\nMake sure required environment variables are set:")
        print("  - SUPABASE_URL and SUPABASE_KEY (for tracking)")
        print("  - VIMEO_ACCESS_TOKEN (for Vimeo sync)")
        print("  - YOUTUBE_CHANNEL_URL or YOUTUBE_PLAYLIST_IDS (for YouTube sync)")
        sys.exit(1)

    except Exception as e:
        print(f"\nSync failed: {e}")
        logger.exception("Sync error")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Incremental video sync from Vimeo and YouTube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/sync_videos.py                    # Full sync
  python scripts/sync_videos.py --discover-only    # Just find new videos
  python scripts/sync_videos.py --platform vimeo   # Vimeo only
  python scripts/sync_videos.py --dry-run          # Preview changes
  python scripts/sync_videos.py --status           # Show current status
  python scripts/sync_videos.py --force            # Re-sync everything
  python scripts/sync_videos.py --ingest-only      # Ingest downloaded transcripts
        """,
    )

    parser.add_argument(
        "--platform",
        choices=["vimeo", "youtube"],
        help="Limit sync to a single platform",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover videos, don't download or ingest",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest pending videos (skip discovery/download)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-sync all videos (ignore existing tracking)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without making changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of videos per source (for testing)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current sync status and exit",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.status:
        show_status()
        return

    if args.config:
        show_config()
        return

    if args.discover_only and args.ingest_only:
        print("Error: Cannot use --discover-only and --ingest-only together")
        sys.exit(1)

    run_sync(
        platform=args.platform,
        discover_only=args.discover_only,
        ingest_only=args.ingest_only,
        force=args.force,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
