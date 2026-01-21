"""
Configuration management using Pydantic Settings.
Loads from environment variables with sensible defaults for development.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    admin_api_key: str = "dev-admin-key"  # For protected endpoints

    # Paths
    transcript_source_dir: Path = Path.home() / "Documents/wellness_evolution_community/vimeo_transcripts_rag"
    chroma_persist_dir: Path = Path(__file__).parent.parent / "data/chroma_db"

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Chunking settings
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 200  # characters

    # Retrieval settings
    default_top_k: int = 8  # Number of chunks to retrieve
    similarity_threshold: float = 0.3  # Minimum similarity score

    # LLM settings
    claude_model: str = "claude-sonnet-4-20250514"
    max_response_tokens: int = 1024

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @property
    def chroma_collection_name(self) -> str:
        return "wellness_transcripts"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
