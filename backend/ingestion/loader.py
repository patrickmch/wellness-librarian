"""
File loader for VTT transcripts with metadata.json integration.
Parses WebVTT format and joins with rich Vimeo metadata.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Rich metadata for a video from metadata.json."""
    vimeo_id: str
    title: str
    description: Optional[str]
    duration_seconds: int
    created_time: Optional[datetime]
    vimeo_url: str
    folder_path: str
    folder_name: str
    tags: list[str]
    language: Optional[str]
    transcript_file: str

    @property
    def duration_formatted(self) -> str:
        """Format duration as HH:MM:SS or MM:SS."""
        mins, secs = divmod(self.duration_seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage in vector DB metadata."""
        return {
            "vimeo_id": self.vimeo_id,
            "title": self.title,
            "description": self.description,
            "duration_seconds": self.duration_seconds,
            "duration": self.duration_formatted,
            "created_date": self.created_time.date().isoformat() if self.created_time else None,
            "vimeo_url": self.vimeo_url,
            "category": self.folder_name,
            "tags": self.tags,
            "language": self.language,
        }


@dataclass
class TranscriptFile:
    """Loaded transcript with content and metadata."""
    path: Path
    content: str  # Plain text extracted from VTT
    metadata: VideoMetadata
    timestamps: list[tuple[float, float, str]] = field(default_factory=list)  # (start, end, text)

    @property
    def file_id(self) -> str:
        """Unique identifier using Vimeo ID."""
        return self.metadata.vimeo_id

    @property
    def category(self) -> str:
        """Category from folder name."""
        return self.metadata.folder_name


def parse_vtt_timestamp(ts: str) -> float:
    """
    Parse VTT timestamp to seconds.

    Args:
        ts: Timestamp string like "00:01:23.456" or "01:23.456"

    Returns:
        Seconds as float
    """
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        hours, mins, secs = parts
        return int(hours) * 3600 + int(mins) * 60 + float(secs)
    elif len(parts) == 2:
        mins, secs = parts
        return int(mins) * 60 + float(secs)
    return 0.0


def parse_vtt(content: str) -> tuple[str, list[tuple[float, float, str]]]:
    """
    Parse WebVTT content to extract plain text and timestamps.

    Args:
        content: Raw VTT file content

    Returns:
        Tuple of (plain_text, timestamps_list)
        timestamps_list contains (start_seconds, end_seconds, text)
    """
    lines = content.strip().split("\n")
    text_parts = []
    timestamps = []

    # Skip WEBVTT header
    i = 0
    while i < len(lines) and not lines[i].strip().startswith("WEBVTT"):
        i += 1
    i += 1  # Skip WEBVTT line

    # Pattern for timestamp line: 00:00:10.400 --> 00:00:12.000
    timestamp_pattern = re.compile(r"(\d+:)?(\d+:\d+[.,]\d+)\s*-->\s*(\d+:)?(\d+:\d+[.,]\d+)")

    current_start = 0.0
    current_end = 0.0

    while i < len(lines):
        line = lines[i].strip()

        # Skip cue numbers (just digits)
        if line.isdigit():
            i += 1
            continue

        # Check for timestamp line
        match = timestamp_pattern.match(line)
        if match:
            # Parse start time
            start_str = (match.group(1) or "") + match.group(2)
            end_str = (match.group(3) or "") + match.group(4)
            current_start = parse_vtt_timestamp(start_str)
            current_end = parse_vtt_timestamp(end_str)
            i += 1
            continue

        # Empty line indicates end of cue
        if not line:
            i += 1
            continue

        # This is caption text
        # Remove VTT formatting tags like <c> or positioning
        clean_text = re.sub(r"<[^>]+>", "", line)
        clean_text = clean_text.strip()

        if clean_text:
            text_parts.append(clean_text)
            timestamps.append((current_start, current_end, clean_text))

        i += 1

    # Join text with spaces, clean up multiple spaces
    plain_text = " ".join(text_parts)
    plain_text = re.sub(r"\s+", " ", plain_text).strip()

    return plain_text, timestamps


def load_metadata(source_dir: Path) -> dict[str, VideoMetadata]:
    """
    Load metadata.json and create lookup by transcript_file path.

    Args:
        source_dir: Root directory containing metadata.json

    Returns:
        Dict mapping transcript_file paths to VideoMetadata objects
    """
    metadata_path = source_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {source_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata_map = {}
    for video in data.get("videos", []):
        if not video.get("has_transcript"):
            continue

        transcript_file = video.get("transcript_file")
        if not transcript_file:
            continue

        # Parse created_time
        created_time = None
        if video.get("created_time"):
            try:
                created_time = datetime.fromisoformat(video["created_time"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        metadata_map[transcript_file] = VideoMetadata(
            vimeo_id=video.get("vimeo_id", ""),
            title=video.get("title", ""),
            description=video.get("description"),
            duration_seconds=video.get("duration_seconds", 0),
            created_time=created_time,
            vimeo_url=video.get("vimeo_url", ""),
            folder_path=video.get("folder_path", ""),
            folder_name=video.get("folder_name", "Uncategorized"),
            tags=video.get("tags", []),
            language=video.get("language"),
            transcript_file=transcript_file,
        )

    logger.info(f"Loaded metadata for {len(metadata_map)} videos")
    return metadata_map


def load_transcripts(source_dir: Path) -> Iterator[TranscriptFile]:
    """
    Load all VTT transcript files with metadata.

    Yields TranscriptFile objects with content and rich metadata.

    Args:
        source_dir: Root directory containing metadata.json and VTT files

    Yields:
        TranscriptFile objects for each transcript
    """
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {source_dir}")

    # Load metadata lookup
    metadata_map = load_metadata(source_dir)

    # Process each transcript referenced in metadata
    for transcript_path, metadata in sorted(metadata_map.items()):
        full_path = source_dir / transcript_path

        if not full_path.exists():
            logger.warning(f"Transcript file not found: {full_path}")
            continue

        try:
            vtt_content = full_path.read_text(encoding="utf-8", errors="replace")
            plain_text, timestamps = parse_vtt(vtt_content)

            if not plain_text.strip():
                logger.warning(f"Empty transcript: {transcript_path}")
                continue

            yield TranscriptFile(
                path=full_path,
                content=plain_text,
                metadata=metadata,
                timestamps=timestamps,
            )

        except Exception as e:
            logger.error(f"Error loading {transcript_path}: {e}")
            continue


def load_single_transcript(filepath: Path, source_dir: Path) -> TranscriptFile:
    """
    Load a single transcript file with metadata.

    Args:
        filepath: Path to the VTT file
        source_dir: Root directory for metadata lookup

    Returns:
        TranscriptFile object
    """
    filepath = Path(filepath)
    source_dir = Path(source_dir)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Get relative path for metadata lookup
    try:
        relative_path = str(filepath.relative_to(source_dir))
    except ValueError:
        relative_path = filepath.name

    # Load metadata
    metadata_map = load_metadata(source_dir)
    metadata = metadata_map.get(relative_path)

    if not metadata:
        # Create minimal metadata if not found in JSON
        metadata = VideoMetadata(
            vimeo_id=filepath.stem,
            title=filepath.stem,
            description=None,
            duration_seconds=0,
            created_time=None,
            vimeo_url="",
            folder_path=filepath.parent.name,
            folder_name=filepath.parent.name,
            tags=[],
            language="en",
            transcript_file=relative_path,
        )

    vtt_content = filepath.read_text(encoding="utf-8", errors="replace")
    plain_text, timestamps = parse_vtt(vtt_content)

    return TranscriptFile(
        path=filepath,
        content=plain_text,
        metadata=metadata,
        timestamps=timestamps,
    )


def get_categories(source_dir: Path) -> list[str]:
    """
    Get list of all categories from metadata.

    Args:
        source_dir: Root directory containing metadata.json

    Returns:
        Sorted list of unique category names
    """
    metadata_map = load_metadata(source_dir)
    categories = {m.folder_name for m in metadata_map.values()}
    return sorted(categories)


def get_transcript_count(source_dir: Path) -> int:
    """Get count of transcripts from metadata."""
    metadata_map = load_metadata(source_dir)
    return len(metadata_map)
