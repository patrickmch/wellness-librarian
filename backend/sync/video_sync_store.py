"""
Supabase CRUD operations for video_sync table.

Tracks which videos have been discovered and ingested for incremental sync.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
import logging

from supabase import create_client, Client

from backend.config import get_settings

logger = logging.getLogger(__name__)

VideoStatus = Literal["pending", "ingested", "failed", "no_transcript", "skipped"]
Platform = Literal["vimeo", "youtube"]


@dataclass
class VideoSyncRecord:
    """A video sync tracking record."""

    video_id: str
    platform: Platform
    title: Optional[str] = None
    transcript_file: Optional[str] = None
    folder_name: Optional[str] = None
    duration_seconds: Optional[int] = None
    video_url: Optional[str] = None
    ingested_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    status: VideoStatus = "pending"
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Supabase upsert."""
        data = {
            "video_id": self.video_id,
            "platform": self.platform,
            "title": self.title,
            "transcript_file": self.transcript_file,
            "folder_name": self.folder_name,
            "duration_seconds": self.duration_seconds,
            "video_url": self.video_url,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }
        if self.ingested_at:
            data["ingested_at"] = self.ingested_at.isoformat()
        if self.last_seen_at:
            data["last_seen_at"] = self.last_seen_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "VideoSyncRecord":
        """Create from Supabase row."""
        return cls(
            video_id=data["video_id"],
            platform=data["platform"],
            title=data.get("title"),
            transcript_file=data.get("transcript_file"),
            folder_name=data.get("folder_name"),
            duration_seconds=data.get("duration_seconds"),
            video_url=data.get("video_url"),
            ingested_at=_parse_datetime(data.get("ingested_at")),
            last_seen_at=_parse_datetime(data.get("last_seen_at")),
            status=data.get("status", "pending"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string."""
    if not value:
        return None
    try:
        # Handle both Z suffix and +00:00
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


class VideoSyncStore:
    """
    Supabase-backed store for video sync tracking.

    Provides CRUD operations for the video_sync table to enable
    incremental sync (only process new videos).
    """

    TABLE_NAME = "video_sync"

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize the store.

        Args:
            supabase_url: Supabase project URL (or from settings)
            supabase_key: Supabase anon/service key (or from settings)
        """
        settings = get_settings()
        self.url = supabase_url or settings.supabase_url
        self.key = supabase_key or settings.supabase_key

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials required. Set SUPABASE_URL and SUPABASE_KEY in .env"
            )

        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """Lazy-load Supabase client."""
        if self._client is None:
            self._client = create_client(self.url, self.key)
        return self._client

    def upsert(self, record: VideoSyncRecord) -> VideoSyncRecord:
        """
        Insert or update a video sync record.

        Args:
            record: The record to upsert

        Returns:
            The upserted record with timestamps populated
        """
        data = record.to_dict()
        result = (
            self.client.table(self.TABLE_NAME)
            .upsert(data, on_conflict="video_id")
            .execute()
        )
        if result.data:
            return VideoSyncRecord.from_dict(result.data[0])
        return record

    def upsert_batch(self, records: list[VideoSyncRecord]) -> int:
        """
        Batch upsert multiple records.

        Args:
            records: List of records to upsert

        Returns:
            Number of records upserted
        """
        if not records:
            return 0

        data = [r.to_dict() for r in records]
        result = (
            self.client.table(self.TABLE_NAME)
            .upsert(data, on_conflict="video_id")
            .execute()
        )
        return len(result.data) if result.data else 0

    def get(self, video_id: str) -> Optional[VideoSyncRecord]:
        """
        Get a single record by video ID.

        Args:
            video_id: The video ID to look up

        Returns:
            The record if found, None otherwise
        """
        result = (
            self.client.table(self.TABLE_NAME)
            .select("*")
            .eq("video_id", video_id)
            .execute()
        )
        if result.data:
            return VideoSyncRecord.from_dict(result.data[0])
        return None

    def get_all_ids(self, platform: Optional[Platform] = None) -> set[str]:
        """
        Get all tracked video IDs.

        Args:
            platform: Optional filter by platform

        Returns:
            Set of video IDs
        """
        query = self.client.table(self.TABLE_NAME).select("video_id")
        if platform:
            query = query.eq("platform", platform)

        result = query.execute()
        return {row["video_id"] for row in result.data} if result.data else set()

    def get_by_status(
        self, status: VideoStatus, platform: Optional[Platform] = None
    ) -> list[VideoSyncRecord]:
        """
        Get all records with a specific status.

        Args:
            status: The status to filter by
            platform: Optional platform filter

        Returns:
            List of matching records
        """
        query = self.client.table(self.TABLE_NAME).select("*").eq("status", status)
        if platform:
            query = query.eq("platform", platform)

        result = query.execute()
        return [VideoSyncRecord.from_dict(row) for row in result.data] if result.data else []

    def get_pending(self, platform: Optional[Platform] = None) -> list[VideoSyncRecord]:
        """Get all videos pending ingestion."""
        return self.get_by_status("pending", platform)

    def mark_ingested(
        self, video_id: str, transcript_file: Optional[str] = None
    ) -> Optional[VideoSyncRecord]:
        """
        Mark a video as successfully ingested.

        Args:
            video_id: The video ID
            transcript_file: Path to the transcript file

        Returns:
            Updated record
        """
        data = {
            "status": "ingested",
            "ingested_at": datetime.utcnow().isoformat(),
        }
        if transcript_file:
            data["transcript_file"] = transcript_file

        result = (
            self.client.table(self.TABLE_NAME)
            .update(data)
            .eq("video_id", video_id)
            .execute()
        )
        if result.data:
            return VideoSyncRecord.from_dict(result.data[0])
        return None

    def mark_failed(self, video_id: str, error_message: str) -> Optional[VideoSyncRecord]:
        """
        Mark a video as failed to process.

        Args:
            video_id: The video ID
            error_message: Description of the error

        Returns:
            Updated record
        """
        result = (
            self.client.table(self.TABLE_NAME)
            .update({"status": "failed", "error_message": error_message})
            .eq("video_id", video_id)
            .execute()
        )
        if result.data:
            return VideoSyncRecord.from_dict(result.data[0])
        return None

    def mark_no_transcript(self, video_id: str) -> Optional[VideoSyncRecord]:
        """Mark a video as having no transcript available."""
        result = (
            self.client.table(self.TABLE_NAME)
            .update({"status": "no_transcript"})
            .eq("video_id", video_id)
            .execute()
        )
        if result.data:
            return VideoSyncRecord.from_dict(result.data[0])
        return None

    def update_last_seen(self, video_ids: list[str]) -> int:
        """
        Update last_seen_at for multiple videos.

        Args:
            video_ids: List of video IDs to update

        Returns:
            Number of records updated
        """
        if not video_ids:
            return 0

        now = datetime.utcnow().isoformat()
        # Supabase doesn't support bulk update with IN clause directly,
        # so we do individual updates (could be optimized with RPC)
        count = 0
        for video_id in video_ids:
            result = (
                self.client.table(self.TABLE_NAME)
                .update({"last_seen_at": now})
                .eq("video_id", video_id)
                .execute()
            )
            if result.data:
                count += 1
        return count

    def get_stats(self, platform: Optional[Platform] = None) -> dict:
        """
        Get sync statistics.

        Args:
            platform: Optional platform filter

        Returns:
            Dictionary with counts by status
        """
        query = self.client.table(self.TABLE_NAME).select("status, platform")
        if platform:
            query = query.eq("platform", platform)

        result = query.execute()

        stats = {
            "total": 0,
            "pending": 0,
            "ingested": 0,
            "failed": 0,
            "no_transcript": 0,
            "skipped": 0,
            "by_platform": {"vimeo": 0, "youtube": 0},
        }

        if result.data:
            for row in result.data:
                stats["total"] += 1
                status = row.get("status", "pending")
                if status in stats:
                    stats[status] += 1
                plat = row.get("platform")
                if plat in stats["by_platform"]:
                    stats["by_platform"][plat] += 1

        return stats

    def reset_failed(self, platform: Optional[Platform] = None) -> int:
        """
        Reset failed videos back to pending for retry.

        Args:
            platform: Optional platform filter

        Returns:
            Number of records reset
        """
        query = self.client.table(self.TABLE_NAME).update({"status": "pending", "error_message": None})
        query = query.eq("status", "failed")
        if platform:
            query = query.eq("platform", platform)

        result = query.execute()
        return len(result.data) if result.data else 0

    def delete(self, video_id: str) -> bool:
        """
        Delete a sync record.

        Args:
            video_id: The video ID to delete

        Returns:
            True if deleted
        """
        result = (
            self.client.table(self.TABLE_NAME)
            .delete()
            .eq("video_id", video_id)
            .execute()
        )
        return bool(result.data)

    def clear_all(self, platform: Optional[Platform] = None) -> int:
        """
        Delete all sync records (for testing/reset).

        Args:
            platform: Optional platform filter

        Returns:
            Number of records deleted
        """
        query = self.client.table(self.TABLE_NAME).delete()
        if platform:
            query = query.eq("platform", platform)
        else:
            # Supabase requires a filter for delete, use a tautology
            query = query.neq("video_id", "")

        result = query.execute()
        return len(result.data) if result.data else 0
