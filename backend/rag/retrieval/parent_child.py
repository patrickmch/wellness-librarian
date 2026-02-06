"""
Parent-child retriever for enhanced RAG.

Implements the parent-child retrieval strategy:
1. Search small child chunks for precision
2. Expand to parent chunks for context
3. Apply diversity filtering
4. Rerank for final ordering

Supports two storage backends:
- SQLite + ChromaDB (legacy, for local development)
- Supabase with pgvector (production, unified storage)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from backend.config import get_settings
from backend.rag.chunking.models import ParentChunk, ChildChunk
from backend.rag.providers.voyage_provider import VoyageEmbeddingProvider
from backend.rag.reranking.voyage_reranker import VoyageReranker
from backend.rag.retrieval.diversity import apply_diversity_filters

logger = logging.getLogger(__name__)


@dataclass
class ParentRetrievalResult:
    """A retrieved parent chunk with score and metadata."""

    parent: ParentChunk
    score: float  # Final score after reranking
    matched_children: list[str] = field(default_factory=list)  # Child IDs that matched

    @property
    def text(self) -> str:
        return self.parent.text

    @property
    def title(self) -> str:
        return self.parent.title

    @property
    def category(self) -> str:
        return self.parent.category

    @property
    def video_url(self) -> str:
        return self.parent.video_url

    @property
    def video_id(self) -> str:
        return self.parent.video_id


@dataclass
class ParentRetrievalResponse:
    """Collection of parent retrieval results."""

    query: str
    results: list[ParentRetrievalResult]
    total_results: int
    children_searched: int  # How many children were searched
    parents_expanded: int  # How many unique parents

    def get_context_text(self, max_chunks: int = 8) -> str:
        """Format results as context for LLM."""
        if not self.results:
            return "No relevant content found."

        context_parts = []
        for i, result in enumerate(self.results[:max_chunks]):
            context_parts.append(
                f"[Source {i+1}: {result.title} ({result.category})]\n{result.text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_sources(self) -> list[dict]:
        """Get sources with metadata for each result (not deduplicated).

        Returns one source per result to enable per-citation deep-linking.
        Each source includes timestamp and excerpt for glass box compliance.
        """
        sources = []

        for result in self.results:
            # Truncate excerpt for display (500 chars max)
            excerpt = result.text
            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "..."

            sources.append({
                "title": result.title,
                "category": result.category,
                "video_url": result.video_url,
                "duration": result.parent.duration,
                "video_id": result.video_id,
                "source": result.parent.source,
                "start_time_seconds": result.parent.start_time_seconds,
                "excerpt": excerpt,
            })

        return sources


class ParentChildRetriever:
    """
    Parent-child retriever for enhanced RAG.

    Strategy:
    1. Embed query with Voyage (query mode)
    2. Search for top-N child chunks (ChromaDB or Supabase)
    3. Apply per-video diversity filter to children
    4. Expand to unique parent chunks
    5. Rerank parents with Voyage rerank-2
    6. Return top-K parents with context

    This approach provides:
    - Precision: Small child chunks match specific queries
    - Context: Parent chunks provide full comprehension
    - Diversity: Per-video limits prevent single-source domination
    - Quality: Reranking improves final relevance ordering

    Supports two backends:
    - "sqlite": ChromaDB for children, SQLite for parents (local dev)
    - "supabase": Supabase pgvector for all data (production)
    """

    def __init__(
        self,
        embedding_provider: VoyageEmbeddingProvider | None = None,
        reranker: VoyageReranker | None = None,
        backend: str | None = None,
    ):
        """
        Initialize the parent-child retriever.

        Args:
            embedding_provider: Voyage embedding provider (default: create new)
            reranker: Voyage reranker (default: create new)
            backend: "sqlite" or "supabase" (default from settings.store_backend)
        """
        settings = get_settings()

        # Initialize components (lazy if possible)
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._backend = backend or settings.store_backend

        # Backend-specific initialization (lazy)
        self._supabase_store = None
        self._docstore = None
        self._chroma_client = None
        self._collection = None
        self._collection_name = settings.chroma_collection_v2

    def _get_embedding_provider(self) -> VoyageEmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = VoyageEmbeddingProvider()
        return self._embedding_provider

    def _get_reranker(self) -> VoyageReranker:
        """Get or create reranker."""
        if self._reranker is None:
            self._reranker = VoyageReranker()
        return self._reranker

    def _get_supabase_store(self):
        """Get or create Supabase store."""
        if self._supabase_store is None:
            from backend.rag.stores.supabase_store import get_supabase_store
            self._supabase_store = get_supabase_store()
        return self._supabase_store

    def _get_docstore(self):
        """Get or create SQLite docstore (for sqlite backend)."""
        if self._docstore is None:
            from backend.rag.docstore.sqlite_store import get_docstore
            self._docstore = get_docstore()
        return self._docstore

    def _get_collection(self):
        """Get or create ChromaDB collection (for sqlite backend)."""
        if self._collection is None:
            import chromadb
            settings = get_settings()
            self._chroma_client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def retrieve(
        self,
        query: str,
        child_top_k: int | None = None,
        final_top_k: int | None = None,
        category: str | None = None,
        min_score: float | None = None,
        enable_reranking: bool | None = None,
    ) -> ParentRetrievalResponse:
        """
        Retrieve parent chunks using parent-child strategy.

        Args:
            query: Search query
            child_top_k: Number of children to search (default from settings)
            final_top_k: Final number of parents to return (default from settings)
            category: Optional category filter
            min_score: Minimum similarity score (0-1)
            enable_reranking: Whether to rerank results (default from settings)

        Returns:
            ParentRetrievalResponse with parent chunks
        """
        settings = get_settings()
        child_top_k = child_top_k or settings.child_top_k
        final_top_k = final_top_k or settings.rerank_top_n or settings.default_top_k
        min_score = min_score or settings.similarity_threshold
        enable_reranking = enable_reranking if enable_reranking is not None else settings.enable_reranking

        # Step 1: Embed query
        embedding_provider = self._get_embedding_provider()
        query_embedding = embedding_provider.embed_query(query)

        # Step 2: Search children (backend-specific)
        if self._backend == "supabase":
            children_with_scores, child_count = self._search_children_supabase(
                query_embedding, child_top_k, category, min_score
            )
        else:
            children_with_scores, child_count = self._search_children_chroma(
                query_embedding, child_top_k, category, min_score
            )

        if not children_with_scores:
            return ParentRetrievalResponse(
                query=query,
                results=[],
                total_results=0,
                children_searched=child_count,
                parents_expanded=0,
            )

        # Step 3: Apply diversity filter to children
        # Extract embeddings for MMR (if available)
        has_embeddings = children_with_scores[0][2] is not None
        filtered_children = apply_diversity_filters(
            children_with_scores,
            query_embedding=query_embedding,
            video_id_getter=lambda x: x[0].video_id,
            embedding_getter=lambda x: x[2] if x[2] is not None else query_embedding,
            use_mmr=settings.enable_mmr and has_embeddings,
        )

        # Step 4: Expand to unique parents (backend-specific)
        parent_ids = list(set(child[0].parent_id for child in filtered_children))

        if self._backend == "supabase":
            store = self._get_supabase_store()
            parents = store.get_parent_batch(parent_ids)
        else:
            docstore = self._get_docstore()
            parents = docstore.get_batch(parent_ids)

        if not parents:
            logger.warning(f"No parents found for IDs: {parent_ids[:5]}...")
            return ParentRetrievalResponse(
                query=query,
                results=[],
                total_results=0,
                children_searched=child_count,
                parents_expanded=0,
            )

        # Track which children matched which parent
        parent_children_map: dict[str, list[str]] = {}
        parent_best_score: dict[str, float] = {}
        for child, score, _ in filtered_children:
            pid = child.parent_id
            if pid not in parent_children_map:
                parent_children_map[pid] = []
                parent_best_score[pid] = 0.0
            parent_children_map[pid].append(child.child_id)
            parent_best_score[pid] = max(parent_best_score[pid], score)

        # Step 5: Rerank parents
        if enable_reranking and len(parents) > 1:
            reranker = self._get_reranker()
            parent_list = list(parents.values())
            parent_texts = [p.text for p in parent_list]

            rerank_results = reranker.rerank(
                query=query,
                documents=parent_texts,
                top_n=final_top_k,
            )

            # Build final results
            final_results = [
                ParentRetrievalResult(
                    parent=parent_list[r.index],
                    score=r.score,
                    matched_children=parent_children_map.get(parent_list[r.index].parent_id, []),
                )
                for r in rerank_results
            ]
        else:
            # No reranking - use child-based scores
            sorted_parents = sorted(
                parents.values(),
                key=lambda p: parent_best_score.get(p.parent_id, 0),
                reverse=True,
            )[:final_top_k]

            final_results = [
                ParentRetrievalResult(
                    parent=p,
                    score=parent_best_score.get(p.parent_id, 0),
                    matched_children=parent_children_map.get(p.parent_id, []),
                )
                for p in sorted_parents
            ]

        return ParentRetrievalResponse(
            query=query,
            results=final_results,
            total_results=len(final_results),
            children_searched=child_count,
            parents_expanded=len(parents),
        )

    def _search_children_supabase(
        self,
        query_embedding: list[float],
        top_k: int,
        category: str | None,
        min_score: float,
    ) -> tuple[list[tuple[ChildChunk, float, list[float] | None]], int]:
        """Search children using Supabase pgvector."""
        store = self._get_supabase_store()
        settings = get_settings()

        results = store.search_children(
            query_embedding=query_embedding,
            top_k=top_k,
            category=category,
            include_embeddings=settings.enable_mmr,
        )

        children_with_scores = []
        for r in results:
            if r.score >= min_score:
                child = r.to_child_chunk()
                children_with_scores.append((child, r.score, r.embedding))

        return children_with_scores, len(results)

    def _search_children_chroma(
        self,
        query_embedding: list[float],
        top_k: int,
        category: str | None,
        min_score: float,
    ) -> tuple[list[tuple[ChildChunk, float, list[float] | None]], int]:
        """Search children using ChromaDB."""
        collection = self._get_collection()

        where_filter = None
        if category:
            where_filter = {"category": category}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        # Handle empty results
        if not results["ids"] or not results["ids"][0]:
            return [], 0

        child_ids = results["ids"][0]
        child_docs = results["documents"][0]
        child_metas = results["metadatas"][0]
        child_distances = results["distances"][0]
        child_embeddings = results["embeddings"][0] if results.get("embeddings") is not None else None

        children_with_scores = []
        for i, (doc, meta, dist) in enumerate(zip(child_docs, child_metas, child_distances)):
            score = 1.0 - dist  # Convert distance to similarity
            if score >= min_score:
                child = ChildChunk.from_chroma_result(doc, meta)
                embedding = child_embeddings[i] if child_embeddings is not None else None
                children_with_scores.append((child, score, embedding))

        return children_with_scores, len(child_ids)

    def get_stats(self) -> dict:
        """Get retriever statistics."""
        settings = get_settings()

        if self._backend == "supabase":
            try:
                store = self._get_supabase_store()
                store_stats = store.get_stats()
                child_count = store_stats.get("child_chunks_indexed", 0)
            except Exception:
                store_stats = {}
                child_count = 0
        else:
            try:
                collection = self._get_collection()
                child_count = collection.count()
            except Exception:
                child_count = 0

            try:
                docstore = self._get_docstore()
                store_stats = docstore.get_stats()
            except Exception:
                store_stats = {}

        return {
            "retriever": "parent_child",
            "backend": self._backend,
            "embedding_provider": "voyage",
            "embedding_model": settings.voyage_embedding_model,
            "reranking_enabled": settings.enable_reranking,
            "diversity_filtering": True,
            "max_per_video": settings.max_chunks_per_video,
            "mmr_enabled": settings.enable_mmr,
            "mmr_lambda": settings.mmr_lambda,
            "child_chunks_indexed": child_count,
            **store_stats,
        }


# Convenience function
def get_parent_child_retriever() -> ParentChildRetriever:
    """Get a ParentChildRetriever instance."""
    return ParentChildRetriever()
