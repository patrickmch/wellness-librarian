"""
Configuration management using Pydantic Settings.
Loads from environment variables with sensible defaults for development.
"""

from pathlib import Path
from functools import lru_cache
from typing import Literal
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
    voyage_api_key: str = ""  # For Voyage AI embeddings/reranking
    admin_api_key: str = "dev-admin-key"  # For protected endpoints

    # Paths
    transcript_source_dir: Path = Path.home() / "Documents/wellness_evolution_community/vimeo_transcripts_rag"
    chroma_persist_dir: Path = Path(__file__).parent.parent / "data/chroma_db"
    docstore_path: Path = Path(__file__).parent.parent / "data/docstore.sqlite"

    # Pipeline selection
    rag_pipeline: Literal["legacy", "enhanced", "ab_test"] = "legacy"
    ab_test_ratio: float = 0.5  # Ratio of requests to route to enhanced pipeline

    # Embedding settings (legacy/OpenAI)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Voyage AI embedding settings
    voyage_embedding_model: str = "voyage-3"
    voyage_embedding_dimensions: int = 1024

    # Chunking settings (legacy - character-based)
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 200  # characters

    # Parent-child chunking settings (enhanced pipeline)
    child_chunk_tokens: int = 250  # Token-based child chunks
    child_chunk_overlap: int = 50  # Token overlap between children
    parent_min_tokens: int = 500  # Minimum tokens for a parent chunk
    parent_max_tokens: int = 2000  # Maximum tokens for a parent chunk

    # Retrieval settings
    default_top_k: int = 8  # Number of chunks to retrieve
    similarity_threshold: float = 0.3  # Minimum similarity score

    # Enhanced retrieval settings
    child_top_k: int = 30  # Search this many children, then expand to parents
    enable_reranking: bool = True  # Use Voyage rerank-2
    rerank_top_n: int = 8  # Return this many after reranking

    # Diversity settings
    max_chunks_per_video: int = 1  # Limit chunks from single video (was 2)
    enable_mmr: bool = True  # Maximal Marginal Relevance
    mmr_lambda: float = 0.5  # Balance relevance (1.0) vs diversity (0.0)

    # LLM settings
    claude_model: str = "claude-sonnet-4-20250514"
    max_response_tokens: int = 1024

    # Critic/verification settings
    enable_critic: bool = True  # Verify responses against sources
    critic_model: str = "claude-3-5-haiku-20241022"  # Fast model for verification

    # Supabase settings (for pgvector storage)
    supabase_url: str = ""  # e.g., https://xxx.supabase.co
    supabase_key: str = ""  # anon/service key
    supabase_db_url: str = ""  # Direct postgres connection for bulk ops

    # Storage backend: "sqlite" for local, "supabase" for production
    store_backend: Literal["sqlite", "supabase"] = "sqlite"

    # Video sync settings
    vimeo_access_token: str = ""  # Vimeo API access token
    youtube_channel_url: str = ""  # YouTube channel URL for sync (e.g., https://www.youtube.com/@ChannelName)
    youtube_playlist_ids: str = ""  # Comma-separated playlist IDs (optional, alternative to channel)
    sync_enabled: bool = True  # Enable/disable video sync
    sync_transcript_dir: Path = Path(__file__).parent.parent / "data/transcripts"  # Where to store downloaded VTTs

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @property
    def youtube_playlist_list(self) -> list[str]:
        """Parse comma-separated playlist IDs into list."""
        if not self.youtube_playlist_ids:
            return []
        return [p.strip() for p in self.youtube_playlist_ids.split(",") if p.strip()]

    @property
    def chroma_collection_name(self) -> str:
        """Collection name for legacy pipeline."""
        return "wellness_transcripts"

    @property
    def chroma_collection_v2(self) -> str:
        """Collection name for enhanced pipeline (parent-child)."""
        return "wellness_transcripts_v2"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
