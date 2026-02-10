"""
Video sync package for incremental video ingestion.

Handles discovery and downloading of video transcripts from Vimeo and YouTube,
with tracking in Supabase to enable incremental sync (only process new videos).
"""

from backend.sync.video_sync_store import VideoSyncStore, VideoSyncRecord
from backend.sync.vimeo_downloader import VimeoDownloader, VimeoVideo
from backend.sync.youtube_downloader import YouTubeDownloader, YouTubeVideo
from backend.sync.orchestrator import SyncOrchestrator

__all__ = [
    "VideoSyncStore",
    "VideoSyncRecord",
    "VimeoDownloader",
    "VimeoVideo",
    "YouTubeDownloader",
    "YouTubeVideo",
    "SyncOrchestrator",
]
