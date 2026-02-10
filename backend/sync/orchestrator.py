"""
Sync orchestrator for incremental video ingestion.

Coordinates:
1. Video discovery from Vimeo and YouTube
2. Comparison with already-synced videos (Supabase tracking)
3. Transcript download for new videos
4. Metadata generation for ingestion pipeline
5. RAG ingestion via ingest_v2
6. Status updates in Supabase
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal, Union

from backend.config import get_settings
from backend.sync.video_sync_store import VideoSyncStore, VideoSyncRecord
from backend.sync.vimeo_downloader import VimeoDownloader, VimeoVideo
from backend.sync.youtube_downloader import YouTubeDownloader, YouTubeVideo

logger = logging.getLogger(__name__)

VideoType = Union[VimeoVideo, YouTubeVideo]
Platform = Literal["vimeo", "youtube"]


@dataclass
class SyncStats:
    """Statistics from a sync run."""

    discovered_vimeo: int = 0
    discovered_youtube: int = 0
    already_synced: int = 0
    new_videos: int = 0
    transcripts_downloaded: int = 0
    no_transcript: int = 0
    ingested: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Discovered: {self.discovered_vimeo} Vimeo, {self.discovered_youtube} YouTube\n"
            f"Already synced: {self.already_synced}\n"
            f"New videos: {self.new_videos}\n"
            f"Transcripts downloaded: {self.transcripts_downloaded}\n"
            f"No transcript available: {self.no_transcript}\n"
            f"Ingested to RAG: {self.ingested}\n"
            f"Failed: {self.failed}\n"
            f"Errors: {len(self.errors)}"
        )


class SyncOrchestrator:
    """
    Coordinates incremental video sync from Vimeo and YouTube.

    Usage:
        orchestrator = SyncOrchestrator()

        # Full sync (discover + download + ingest new videos)
        stats = orchestrator.sync()

        # Discovery only (no downloads or ingestion)
        stats = orchestrator.sync(discover_only=True)

        # Single platform
        stats = orchestrator.sync(platform="vimeo")

        # Force re-sync all (ignore tracking)
        stats = orchestrator.sync(force=True)
    """

    def __init__(
        self,
        transcript_dir: Optional[Path] = None,
        store: Optional[VideoSyncStore] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            transcript_dir: Directory to store downloaded transcripts
            store: VideoSyncStore instance (creates one if not provided)
        """
        settings = get_settings()
        self.transcript_dir = transcript_dir or settings.sync_transcript_dir
        self.transcript_dir = Path(self.transcript_dir)

        # Initialize store (may raise if Supabase not configured)
        self.store = store

        # Downloaders are lazy-initialized
        self._vimeo_downloader: Optional[VimeoDownloader] = None
        self._youtube_downloader: Optional[YouTubeDownloader] = None

    @property
    def vimeo_downloader(self) -> Optional[VimeoDownloader]:
        """Lazy-load Vimeo downloader."""
        if self._vimeo_downloader is None:
            try:
                self._vimeo_downloader = VimeoDownloader()
            except ValueError as e:
                logger.warning(f"Vimeo downloader not available: {e}")
        return self._vimeo_downloader

    @property
    def youtube_downloader(self) -> YouTubeDownloader:
        """Lazy-load YouTube downloader."""
        if self._youtube_downloader is None:
            self._youtube_downloader = YouTubeDownloader()
        return self._youtube_downloader

    def _ensure_store(self) -> VideoSyncStore:
        """Ensure store is initialized."""
        if self.store is None:
            self.store = VideoSyncStore()
        return self.store

    def _video_to_record(self, video: VideoType) -> VideoSyncRecord:
        """Convert a video object to a sync record."""
        return VideoSyncRecord(
            video_id=video.video_id,
            platform=video.platform,
            title=video.title,
            folder_name=video.folder_name,
            duration_seconds=getattr(video, "duration_seconds", None),
            video_url=video.video_url,
            last_seen_at=datetime.now(timezone.utc),
            status="pending",
            metadata=video.to_metadata_dict(),
        )

    def discover(
        self,
        platform: Optional[Platform] = None,
        limit: Optional[int] = None,
    ) -> tuple[list[VideoType], SyncStats]:
        """
        Discover videos from configured sources.

        Args:
            platform: Limit to specific platform ("vimeo" or "youtube")
            limit: Maximum videos per source (for testing)

        Returns:
            Tuple of (discovered videos, stats)
        """
        stats = SyncStats()
        videos: list[VideoType] = []

        # Discover Vimeo videos
        if platform in (None, "vimeo") and self.vimeo_downloader:
            logger.info("Discovering Vimeo videos...")
            try:
                for video in self.vimeo_downloader.discover_all(limit=limit):
                    videos.append(video)
                    stats.discovered_vimeo += 1
            except Exception as e:
                error_msg = f"Vimeo discovery error: {e}"
                logger.exception(error_msg)
                stats.errors.append(error_msg)

        # Discover YouTube videos
        if platform in (None, "youtube"):
            logger.info("Discovering YouTube videos...")
            try:
                for video in self.youtube_downloader.discover_all(limit=limit):
                    videos.append(video)
                    stats.discovered_youtube += 1
            except Exception as e:
                error_msg = f"YouTube discovery error: {e}"
                logger.exception(error_msg)
                stats.errors.append(error_msg)

        logger.info(
            f"Discovery complete: {stats.discovered_vimeo} Vimeo, "
            f"{stats.discovered_youtube} YouTube"
        )
        return videos, stats

    def sync(
        self,
        platform: Optional[Platform] = None,
        discover_only: bool = False,
        force: bool = False,
        dry_run: bool = False,
        limit: Optional[int] = None,
        progress_callback=None,
    ) -> SyncStats:
        """
        Run incremental sync.

        Args:
            platform: Limit to specific platform
            discover_only: Only discover videos, don't download or ingest
            force: Re-sync all videos (ignore existing tracking)
            dry_run: Preview what would happen without making changes
            limit: Maximum videos per source (for testing)
            progress_callback: Optional callback(current, total, message)

        Returns:
            SyncStats with results
        """
        settings = get_settings()

        if not settings.sync_enabled and not force:
            logger.warning("Sync is disabled (SYNC_ENABLED=false)")
            return SyncStats()

        # Step 1: Discover videos
        videos, stats = self.discover(platform=platform, limit=limit)

        if not videos:
            logger.info("No videos discovered")
            return stats

        # Step 2: Get already-synced IDs (unless forcing)
        synced_ids: set[str] = set()
        if not force:
            try:
                store = self._ensure_store()
                synced_ids = store.get_all_ids(platform=platform)
                logger.info(f"Found {len(synced_ids)} already-synced videos")
            except Exception as e:
                logger.warning(f"Could not check synced IDs (continuing anyway): {e}")

        # Step 3: Filter to new videos
        new_videos = [v for v in videos if v.video_id not in synced_ids]
        stats.already_synced = len(videos) - len(new_videos)
        stats.new_videos = len(new_videos)

        logger.info(f"New videos to process: {len(new_videos)}")

        if discover_only:
            # Just update tracking with discovered videos
            if not dry_run:
                self._update_tracking(videos, stats)
            return stats

        if not new_videos:
            logger.info("No new videos to process")
            return stats

        # Step 4: Process new videos
        total = len(new_videos)
        for i, video in enumerate(new_videos):
            if progress_callback:
                progress_callback(i + 1, total, f"Processing: {video.title[:40]}")

            try:
                self._process_video(video, stats, dry_run=dry_run)
            except Exception as e:
                error_msg = f"Error processing {video.title}: {e}"
                logger.exception(error_msg)
                stats.errors.append(error_msg)
                stats.failed += 1

                if not dry_run:
                    try:
                        store = self._ensure_store()
                        store.mark_failed(video.video_id, str(e))
                    except Exception:
                        pass

        # Step 5: Generate metadata.json for ingestion
        if not dry_run and stats.transcripts_downloaded > 0:
            self._generate_metadata()

        return stats

    def _process_video(
        self, video: VideoType, stats: SyncStats, dry_run: bool = False
    ) -> bool:
        """
        Process a single video: download transcript and track in Supabase.

        Args:
            video: The video to process
            stats: Stats object to update
            dry_run: If True, don't actually download

        Returns:
            True if successfully processed
        """
        # Check if video has transcript
        has_transcript = getattr(video, "has_transcript", True)

        if not has_transcript:
            logger.info(f"No transcript: {video.title}")
            stats.no_transcript += 1

            if not dry_run:
                record = self._video_to_record(video)
                record.status = "no_transcript"
                store = self._ensure_store()
                store.upsert(record)

            return False

        # Download transcript
        if dry_run:
            logger.info(f"[DRY RUN] Would download: {video.title}")
            stats.transcripts_downloaded += 1
            return True

        transcript_path = None
        if video.platform == "vimeo" and self.vimeo_downloader:
            transcript_path = self.vimeo_downloader.download_transcript(
                video, self.transcript_dir
            )
        elif video.platform == "youtube":
            transcript_path = self.youtube_downloader.download_transcript(
                video, self.transcript_dir
            )

        if transcript_path:
            stats.transcripts_downloaded += 1

            # Track in Supabase
            record = self._video_to_record(video)
            record.status = "pending"  # Will be 'ingested' after RAG ingestion
            record.transcript_file = str(transcript_path.relative_to(self.transcript_dir))
            store = self._ensure_store()
            store.upsert(record)

            return True
        else:
            stats.no_transcript += 1

            # Track as no_transcript
            record = self._video_to_record(video)
            record.status = "no_transcript"
            store = self._ensure_store()
            store.upsert(record)

            return False

    def _update_tracking(self, videos: list[VideoType], stats: SyncStats):
        """Update tracking table with discovered videos."""
        store = self._ensure_store()
        records = [self._video_to_record(v) for v in videos]

        # Batch upsert
        count = store.upsert_batch(records)
        logger.info(f"Updated tracking for {count} videos")

    def _generate_metadata(self):
        """Generate metadata.json files for ingestion pipeline."""
        # Generate separate metadata files for vimeo and youtube
        for platform in ["vimeo", "youtube"]:
            platform_dir = self.transcript_dir / platform
            if not platform_dir.exists():
                continue

            # Collect metadata from tracking
            try:
                store = self._ensure_store()
                records = store.get_by_status("pending", platform=platform)
                records.extend(store.get_by_status("ingested", platform=platform))
            except Exception as e:
                logger.warning(f"Could not load records for {platform}: {e}")
                continue

            if not records:
                continue

            # Build metadata.json
            videos = []
            for record in records:
                if record.metadata:
                    video_meta = dict(record.metadata)
                    video_meta["has_transcript"] = record.transcript_file is not None
                    video_meta["transcript_file"] = record.transcript_file
                    videos.append(video_meta)

            manifest = {
                "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source": platform,
                "total_videos": len(videos),
                "videos_with_transcripts": sum(1 for v in videos if v.get("has_transcript")),
                "videos": videos,
            }

            metadata_path = platform_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Generated {metadata_path} with {len(videos)} videos")

    def ingest_pending(
        self,
        force: bool = False,
        progress_callback=None,
    ) -> SyncStats:
        """
        Ingest pending videos into RAG pipeline.

        This runs the ingest_v2 pipeline for videos that have been
        downloaded but not yet ingested.

        Args:
            force: Re-ingest even if already ingested
            progress_callback: Optional callback(current, total, message)

        Returns:
            SyncStats with ingestion results
        """
        # Import here to avoid circular imports
        from scripts.ingest_v2 import ingest_v2, V2IngestionStats

        stats = SyncStats()

        # Get pending videos
        store = self._ensure_store()
        pending = store.get_pending()

        if not pending:
            logger.info("No pending videos to ingest")
            return stats

        logger.info(f"Found {len(pending)} pending videos to ingest")

        # Run ingestion for each platform's transcript directory
        for platform in ["vimeo", "youtube"]:
            platform_dir = self.transcript_dir / platform
            if not platform_dir.exists():
                continue

            metadata_path = platform_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            logger.info(f"Ingesting {platform} transcripts from {platform_dir}")

            try:
                ingest_stats: V2IngestionStats = ingest_v2(
                    source_dir=platform_dir,
                    force=force,
                    progress_callback=progress_callback,
                )

                stats.ingested += ingest_stats.transcripts_processed

                # Mark successfully ingested videos
                if ingest_stats.transcripts_processed > 0:
                    platform_pending = [p for p in pending if p.platform == platform]
                    for record in platform_pending:
                        store.mark_ingested(record.video_id, record.transcript_file)

                if ingest_stats.errors:
                    stats.errors.extend(ingest_stats.errors)

            except Exception as e:
                error_msg = f"Ingestion error for {platform}: {e}"
                logger.exception(error_msg)
                stats.errors.append(error_msg)

        return stats

    def get_status(self) -> dict:
        """Get current sync status."""
        try:
            store = self._ensure_store()
            return store.get_stats()
        except Exception as e:
            return {"error": str(e)}
