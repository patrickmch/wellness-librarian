"""
Vimeo video discovery and transcript downloader.

Adapted from ~/code/scripts/vimeo_transcripts.py for use in the sync pipeline.
Separates discovery (API enumeration) from download (transcript fetch).
"""

import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime, timezone

import requests

from backend.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class VimeoVideo:
    """A discovered Vimeo video with metadata."""

    video_id: str  # Numeric Vimeo ID (extracted from URI)
    uri: str  # Full Vimeo URI like /videos/123456
    title: str
    description: Optional[str]
    duration_seconds: int
    created_time: Optional[datetime]
    video_url: str
    folder_path: str  # Hierarchical folder path
    folder_name: str  # Immediate parent folder name
    tags: list[str] = field(default_factory=list)
    language: Optional[str] = None
    has_transcript: bool = False
    transcript_link: Optional[str] = None  # Direct download URL if available

    @property
    def platform(self) -> str:
        return "vimeo"

    def to_metadata_dict(self) -> dict:
        """Convert to metadata.json format for ingestion."""
        return {
            "vimeo_id": self.video_id,
            "vimeo_uri": self.uri,
            "vimeo_url": self.video_url,
            "title": self.title,
            "description": self.description or "",
            "duration_seconds": self.duration_seconds,
            "created_time": self.created_time.isoformat() if self.created_time else None,
            "folder_path": self.folder_path,
            "folder_name": self.folder_name,
            "tags": self.tags,
            "language": self.language,
            "has_transcript": self.has_transcript,
            "source": "vimeo",
            "access_level": "members_only",  # Vimeo content is members-only
        }


class VimeoDownloader:
    """
    Downloads video metadata and transcripts from Vimeo.

    Supports two modes:
    1. discover_all(): Enumerate all videos without downloading
    2. download_transcript(): Download VTT for a specific video
    """

    BASE_URL = "https://api.vimeo.com"

    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize the downloader.

        Args:
            access_token: Vimeo API access token (or from settings)
        """
        settings = get_settings()
        self.access_token = access_token or settings.vimeo_access_token

        if not self.access_token:
            raise ValueError(
                "Vimeo access token required. Set VIMEO_ACCESS_TOKEN in .env"
            )

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.vimeo.*+json;version=3.4",
        })

        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[int] = None

    def _make_request(
        self, endpoint: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        """Make an API request with rate limit handling."""
        url = f"{self.BASE_URL}{endpoint}" if endpoint.startswith("/") else endpoint

        try:
            response = self.session.get(url, params=params, timeout=30)

            # Update rate limit info
            self._rate_limit_remaining = response.headers.get("X-RateLimit-Remaining")
            self._rate_limit_reset = response.headers.get("X-RateLimit-Reset")

            # Handle rate limiting
            if response.status_code == 429:
                reset_time = int(self._rate_limit_reset or 60)
                logger.warning(f"Rate limited. Waiting {reset_time} seconds...")
                time.sleep(reset_time)
                return self._make_request(endpoint, params)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return None

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def _paginate(self, endpoint: str, params: Optional[dict] = None) -> list[dict]:
        """Fetch all pages from a paginated endpoint."""
        all_items = []
        params = params or {}
        params["per_page"] = 100  # Max allowed

        while endpoint:
            data = self._make_request(endpoint, params)
            if not data:
                break

            items = data.get("data", [])
            all_items.extend(items)

            # Get next page URL
            paging = data.get("paging", {})
            endpoint = paging.get("next")
            params = None  # Next URL includes params

        return all_items

    def _sanitize_filename(self, name: str) -> str:
        """Make a string safe for use as filename."""
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        return sanitized or "untitled"

    def _get_folders(self) -> list[dict]:
        """Retrieve all folders/projects from the account."""
        logger.info("Fetching Vimeo folders...")
        folders = self._paginate("/me/projects")
        logger.info(f"Found {len(folders)} folders")
        return folders

    def _build_folder_map(self, folders: list[dict]) -> dict[str, dict]:
        """Build a mapping of folder URIs to their info and paths."""
        folder_map = {}

        # First pass: create basic entries
        for folder in folders:
            uri = folder.get("uri", "")
            name = folder.get("name", "Untitled")
            parent = folder.get("parent_folder", {})
            parent_uri = parent.get("uri") if parent else None

            folder_map[uri] = {
                "name": self._sanitize_filename(name),
                "parent_uri": parent_uri,
                "path": None,  # Will be computed
            }

        # Second pass: compute full paths
        def get_path(uri: str) -> Path:
            if uri not in folder_map:
                return Path()

            info = folder_map[uri]
            if info["path"] is not None:
                return info["path"]

            if info["parent_uri"] and info["parent_uri"] in folder_map:
                parent_path = get_path(info["parent_uri"])
                info["path"] = parent_path / info["name"]
            else:
                info["path"] = Path(info["name"])

            return info["path"]

        for uri in folder_map:
            get_path(uri)

        return folder_map

    def _get_video_texttracks(self, video_uri: str) -> list[dict]:
        """Fetch available transcripts/captions for a video."""
        data = self._make_request(f"{video_uri}/texttracks")
        if data:
            return data.get("data", [])
        return []

    def _extract_video(
        self, video: dict, folder_path: Path, folder_name: str
    ) -> VimeoVideo:
        """Extract VimeoVideo from API response."""
        uri = video.get("uri", "")
        video_id = uri.split("/")[-1] if uri else ""

        # Extract tags
        tags = []
        for tag in video.get("tags", []) or []:
            if isinstance(tag, dict):
                tags.append(tag.get("name", ""))
            elif isinstance(tag, str):
                tags.append(tag)

        # Parse created_time
        created_time = None
        time_str = video.get("created_time")
        if time_str:
            try:
                created_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Check for transcript
        texttracks = self._get_video_texttracks(uri)
        transcript_link = None
        has_transcript = False

        if texttracks:
            # Prefer 'captions' type, then any
            transcript = None
            for track in texttracks:
                if track.get("type") == "captions":
                    transcript = track
                    break
            if not transcript:
                transcript = texttracks[0]

            if transcript and transcript.get("link"):
                has_transcript = True
                transcript_link = transcript.get("link")

        return VimeoVideo(
            video_id=video_id,
            uri=uri,
            title=video.get("name", "Untitled"),
            description=video.get("description"),
            duration_seconds=int(video.get("duration", 0)),
            created_time=created_time,
            video_url=video.get("link", ""),
            folder_path=str(folder_path) if folder_path else "",
            folder_name=folder_name,
            tags=tags,
            language=video.get("language"),
            has_transcript=has_transcript,
            transcript_link=transcript_link,
        )

    def discover_all(self, limit: Optional[int] = None) -> Iterator[VimeoVideo]:
        """
        Discover all videos in the Vimeo account.

        This method enumerates all folders and videos without downloading
        transcripts. Use download_transcript() to fetch individual transcripts.

        Args:
            limit: Maximum number of videos to return (for testing)

        Yields:
            VimeoVideo objects for each discovered video
        """
        folders = self._get_folders()
        folder_map = self._build_folder_map(folders)

        videos_yielded = 0

        # Process videos in each folder
        for folder in folders:
            folder_uri = folder.get("uri", "")
            folder_name = folder.get("name", "Untitled")
            folder_path = folder_map.get(folder_uri, {}).get("path", Path(folder_name))

            logger.info(f"Processing folder: {folder_name}")
            videos = self._paginate(f"{folder_uri}/videos")

            for video in videos:
                if limit and videos_yielded >= limit:
                    return

                yield self._extract_video(video, folder_path, folder_name)
                videos_yielded += 1

        # Process root videos (not in any folder)
        logger.info("Processing root videos (no folder)")
        root_videos = self._paginate("/me/videos")

        # Get URIs of videos that are in folders
        folder_video_uris = set()
        for folder in folders:
            folder_uri = folder.get("uri", "")
            videos = self._paginate(f"{folder_uri}/videos")
            for v in videos:
                folder_video_uris.add(v.get("uri"))

        for video in root_videos:
            if video.get("uri") in folder_video_uris:
                continue  # Skip videos already processed in folders

            if limit and videos_yielded >= limit:
                return

            yield self._extract_video(video, Path(), "(root)")
            videos_yielded += 1

    def download_transcript(
        self, video: VimeoVideo, output_dir: Path
    ) -> Optional[Path]:
        """
        Download the transcript for a video.

        Args:
            video: The VimeoVideo with transcript_link
            output_dir: Directory to save the transcript

        Returns:
            Path to the downloaded VTT file, or None if failed
        """
        if not video.has_transcript or not video.transcript_link:
            logger.warning(f"No transcript available for: {video.title}")
            return None

        # Build output path: output_dir/folder_path/title.vtt
        safe_title = self._sanitize_filename(video.title)
        if video.folder_path:
            relative_path = Path(video.folder_path) / f"{safe_title}.vtt"
        else:
            relative_path = Path(f"{safe_title}.vtt")

        output_path = output_dir / "vimeo" / relative_path

        try:
            # Transcript URLs don't need auth header
            response = requests.get(video.transcript_link, timeout=30)
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(response.text, encoding="utf-8")
                logger.info(f"Downloaded: {output_path.name}")
                return output_path
            else:
                logger.error(
                    f"Download failed ({response.status_code}): {video.title}"
                )
                return None

        except Exception as e:
            logger.error(f"Download error for {video.title}: {e}")
            return None

    def get_transcript_relative_path(self, video: VimeoVideo) -> str:
        """Get the relative path where a transcript would be stored."""
        safe_title = self._sanitize_filename(video.title)
        if video.folder_path:
            return f"vimeo/{video.folder_path}/{safe_title}.vtt"
        return f"vimeo/{safe_title}.vtt"
