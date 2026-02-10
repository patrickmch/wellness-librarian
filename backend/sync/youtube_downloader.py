"""
YouTube video discovery and transcript downloader.

Adapted from ~/code/scripts/youtube_transcripts.py for use in the sync pipeline.
Uses yt-dlp for video enumeration and transcript download (no API key needed).
"""

import re
import json
import subprocess
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime

from backend.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class YouTubeVideo:
    """A discovered YouTube video with metadata."""

    video_id: str  # YouTube video ID (11 chars)
    title: str
    description: Optional[str]
    duration_seconds: int
    upload_date: Optional[str]  # YYYY-MM-DD format
    video_url: str
    channel: str
    channel_id: Optional[str]
    folder_name: str  # Playlist name or channel name for categorization
    tags: list[str] = field(default_factory=list)
    view_count: Optional[int] = None
    has_transcript: bool = True  # Assume available, check during download

    @property
    def platform(self) -> str:
        return "youtube"

    def to_metadata_dict(self) -> dict:
        """Convert to metadata.json format for ingestion."""
        # Ensure "free_resource" tag for YouTube content
        tags = list(self.tags)
        if "free_resource" not in tags:
            tags.append("free_resource")

        return {
            "youtube_id": self.video_id,
            "youtube_url": self.video_url,
            "title": self.title,
            "description": self.description or "",
            "duration_seconds": self.duration_seconds,
            "upload_date": self.upload_date,
            "channel": self.channel,
            "channel_id": self.channel_id,
            "folder_path": self.folder_name,
            "folder_name": self.folder_name,
            "tags": tags,
            "has_transcript": self.has_transcript,
            "source": "youtube",
            "access_level": "free_resource",  # YouTube content is public
        }


class YouTubeDownloader:
    """
    Downloads video metadata and transcripts from YouTube.

    Uses yt-dlp for all operations (no API key required).

    Supports two modes:
    1. discover_channel/discover_playlist: Enumerate videos without downloading
    2. download_transcript(): Download VTT for a specific video
    """

    def __init__(self, yt_dlp_path: Optional[str] = None):
        """
        Initialize the downloader.

        Args:
            yt_dlp_path: Path to yt-dlp executable (auto-detected if not provided)
        """
        self.yt_dlp_path = yt_dlp_path or self._find_yt_dlp()

        if not self._check_yt_dlp():
            raise RuntimeError(
                "yt-dlp not found. Install with: brew install yt-dlp or pip install yt-dlp"
            )

    def _find_yt_dlp(self) -> str:
        """Find yt-dlp executable."""
        # Check wellness-librarian venv first
        settings = get_settings()
        project_root = Path(settings.chroma_persist_dir).parent.parent
        venv_yt_dlp = project_root / "venv" / "bin" / "yt-dlp"
        if venv_yt_dlp.exists():
            return str(venv_yt_dlp)
        # Fall back to system
        return "yt-dlp"

    def _check_yt_dlp(self) -> bool:
        """Check if yt-dlp is installed."""
        try:
            result = subprocess.run(
                [self.yt_dlp_path, "--version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _sanitize_filename(self, name: str) -> str:
        """Make a string safe for use as filename."""
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        if len(sanitized) > 150:
            sanitized = sanitized[:150]
        return sanitized or "untitled"

    def _format_date(self, date_str: Optional[str]) -> Optional[str]:
        """Convert YYYYMMDD to YYYY-MM-DD."""
        if date_str and len(date_str) == 8:
            try:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            except (ValueError, IndexError):
                pass
        return date_str

    def _get_video_list(self, url: str) -> list[dict]:
        """Get video list from URL using yt-dlp flat playlist."""
        cmd = [
            self.yt_dlp_path,
            "--dump-json",
            "--flat-playlist",
            "--no-warnings",
            url,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"yt-dlp error: {result.stderr[:200]}")
                return []

            videos = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        videos.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return videos

        except subprocess.TimeoutExpired:
            logger.error("yt-dlp timed out")
            return []
        except Exception as e:
            logger.error(f"Error running yt-dlp: {e}")
            return []

    def _get_full_video_info(self, video_id: str) -> Optional[dict]:
        """Get full metadata for a single video."""
        cmd = [
            self.yt_dlp_path,
            "--dump-json",
            "--no-warnings",
            f"https://www.youtube.com/watch?v={video_id}",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
        except Exception as e:
            logger.warning(f"Could not get full info for {video_id}: {e}")

        return None

    def _extract_video(self, video: dict, folder_name: str) -> YouTubeVideo:
        """Extract YouTubeVideo from yt-dlp output."""
        # Handle both flat playlist format and full video format
        video_id = video.get("id") or video.get("url", "").split("=")[-1]

        # Extract tags
        tags = video.get("tags", []) or []
        if isinstance(tags, str):
            tags = [tags]

        return YouTubeVideo(
            video_id=video_id,
            title=video.get("title") or video.get("fulltitle", "Untitled"),
            description=video.get("description"),
            duration_seconds=int(video.get("duration", 0) or 0),
            upload_date=self._format_date(video.get("upload_date")),
            video_url=f"https://www.youtube.com/watch?v={video_id}",
            channel=video.get("channel") or video.get("uploader", ""),
            channel_id=video.get("channel_id") or video.get("uploader_id"),
            folder_name=folder_name,
            tags=tags,
            view_count=video.get("view_count"),
        )

    def discover_channel(
        self, channel_url: str, limit: Optional[int] = None
    ) -> Iterator[YouTubeVideo]:
        """
        Discover all videos from a YouTube channel.

        Args:
            channel_url: YouTube channel URL (e.g., https://www.youtube.com/@ChannelName)
            limit: Maximum number of videos to return

        Yields:
            YouTubeVideo objects for each discovered video
        """
        logger.info(f"Discovering videos from channel: {channel_url}")

        videos = self._get_video_list(channel_url)
        logger.info(f"Found {len(videos)} videos")

        # Try to extract channel name for folder
        folder_name = "YouTube"
        if videos and videos[0].get("channel"):
            folder_name = videos[0]["channel"]

        for i, video in enumerate(videos):
            if limit and i >= limit:
                return

            # Get full video info if we only have flat playlist data
            video_id = video.get("id") or video.get("url", "").split("=")[-1]
            if "description" not in video and video_id:
                full_info = self._get_full_video_info(video_id)
                if full_info:
                    video = full_info

            yield self._extract_video(video, folder_name)

    def discover_playlist(
        self, playlist_id: str, folder_name: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterator[YouTubeVideo]:
        """
        Discover all videos from a YouTube playlist.

        Args:
            playlist_id: YouTube playlist ID (PLxxxxxx)
            folder_name: Override folder name (defaults to playlist title)
            limit: Maximum number of videos to return

        Yields:
            YouTubeVideo objects for each discovered video
        """
        url = f"https://www.youtube.com/playlist?list={playlist_id}"
        logger.info(f"Discovering videos from playlist: {playlist_id}")

        videos = self._get_video_list(url)
        logger.info(f"Found {len(videos)} videos in playlist")

        # Use provided folder name or try to extract from first video
        actual_folder = folder_name or "YouTube Playlist"
        if not folder_name and videos:
            # Try to get playlist title from first video
            video_id = videos[0].get("id") or videos[0].get("url", "").split("=")[-1]
            full_info = self._get_full_video_info(video_id) if video_id else None
            if full_info and full_info.get("playlist_title"):
                actual_folder = full_info["playlist_title"]

        for i, video in enumerate(videos):
            if limit and i >= limit:
                return

            # Get full video info if needed
            video_id = video.get("id") or video.get("url", "").split("=")[-1]
            if "description" not in video and video_id:
                full_info = self._get_full_video_info(video_id)
                if full_info:
                    video = full_info

            yield self._extract_video(video, actual_folder)

    def discover_all(
        self,
        channel_url: Optional[str] = None,
        playlist_ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[YouTubeVideo]:
        """
        Discover videos from configured sources.

        Args:
            channel_url: YouTube channel URL (or from settings)
            playlist_ids: List of playlist IDs (or from settings)
            limit: Maximum videos per source

        Yields:
            YouTubeVideo objects from all configured sources
        """
        settings = get_settings()

        # Use provided or settings values
        channel_url = channel_url or settings.youtube_channel_url
        playlist_ids = playlist_ids or settings.youtube_playlist_list

        seen_ids = set()  # Avoid duplicates across sources

        # Discover from channel
        if channel_url:
            for video in self.discover_channel(channel_url, limit):
                if video.video_id not in seen_ids:
                    seen_ids.add(video.video_id)
                    yield video

        # Discover from playlists
        for playlist_id in playlist_ids:
            for video in self.discover_playlist(playlist_id, limit=limit):
                if video.video_id not in seen_ids:
                    seen_ids.add(video.video_id)
                    yield video

    def download_transcript(
        self, video: YouTubeVideo, output_dir: Path
    ) -> Optional[Path]:
        """
        Download the transcript for a video.

        Args:
            video: The YouTubeVideo to download transcript for
            output_dir: Directory to save the transcript

        Returns:
            Path to the downloaded VTT file, or None if failed/unavailable
        """
        safe_title = self._sanitize_filename(video.title)
        output_base = output_dir / "youtube" / video.folder_name / safe_title

        # Ensure directory exists
        output_base.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.yt_dlp_path,
            "--write-subs",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--skip-download",
            "--no-warnings",
            "-o", str(output_base),
            f"https://www.youtube.com/watch?v={video.video_id}",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Check for downloaded subtitle files
            possible_files = [
                output_base.with_suffix(".en.vtt"),
                output_base.with_suffix(".en-US.vtt"),
                output_base.parent / f"{output_base.stem}.en.vtt",
                output_base.parent / f"{output_base.stem}.en-US.vtt",
            ]

            for sub_file in possible_files:
                if sub_file.exists():
                    logger.info(f"Downloaded: {sub_file.name}")
                    return sub_file

            # Also check for any .vtt file with the video name
            for vtt_file in output_base.parent.glob(f"{output_base.stem}*.vtt"):
                logger.info(f"Downloaded: {vtt_file.name}")
                return vtt_file

            logger.warning(f"No transcript available for: {video.title}")
            return None

        except subprocess.TimeoutExpired:
            logger.error(f"Transcript download timed out for: {video.title}")
            return None
        except Exception as e:
            logger.error(f"Error downloading transcript for {video.title}: {e}")
            return None

    def get_transcript_relative_path(self, video: YouTubeVideo) -> str:
        """Get the relative path where a transcript would be stored."""
        safe_title = self._sanitize_filename(video.title)
        return f"youtube/{video.folder_name}/{safe_title}.en.vtt"
