"""
ChromaDB vector store operations.
Handles collection management, document storage, and similarity search.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings
from backend.rag.embeddings import embed_text, embed_texts

logger = logging.getLogger(__name__)

# Module-level client cache
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Get or create ChromaDB persistent client.

    Returns:
        ChromaDB PersistentClient instance
    """
    global _client

    if _client is None:
        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        _client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        logger.info(f"ChromaDB client initialized at {persist_dir}")

    return _client


def get_collection() -> chromadb.Collection:
    """
    Get or create the wellness transcripts collection.

    Uses OpenAI embeddings via our custom embedding function.

    Returns:
        ChromaDB Collection instance
    """
    global _collection

    if _collection is None:
        settings = get_settings()
        client = get_chroma_client()

        # Get or create collection
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={
                "description": "Wellness Evolution Community video transcripts",
                "hnsw:space": "cosine",  # Use cosine similarity
            },
        )
        logger.info(f"Collection '{settings.chroma_collection_name}' ready with {_collection.count()} documents")

    return _collection


def add_documents(
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict],
    embeddings: Optional[Sequence[list[float]]] = None,
    batch_size: int = 100,
) -> int:
    """
    Add documents to the collection with embeddings.

    If embeddings are not provided, they will be generated automatically.

    Args:
        ids: Unique IDs for each document
        documents: Text content of each document
        metadatas: Metadata dict for each document
        embeddings: Pre-computed embeddings (optional)
        batch_size: Batch size for adding to ChromaDB

    Returns:
        Number of documents added
    """
    collection = get_collection()

    # Generate embeddings if not provided
    if embeddings is None:
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = embed_texts(documents, batch_size=batch_size)

    # Add in batches
    total_added = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = list(ids[i:i + batch_size])
        batch_docs = list(documents[i:i + batch_size])
        batch_metas = list(metadatas[i:i + batch_size])
        batch_embs = list(embeddings[i:i + batch_size])

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embs,
        )
        total_added += len(batch_ids)
        logger.debug(f"Added batch {i // batch_size + 1}, total: {total_added}")

    return total_added


def upsert_documents(
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict],
    embeddings: Optional[Sequence[list[float]]] = None,
    batch_size: int = 100,
) -> int:
    """
    Upsert documents (add or update if exists).

    Args:
        ids: Unique IDs for each document
        documents: Text content of each document
        metadatas: Metadata dict for each document
        embeddings: Pre-computed embeddings (optional)
        batch_size: Batch size for operations

    Returns:
        Number of documents upserted
    """
    collection = get_collection()

    # Generate embeddings if not provided
    if embeddings is None:
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = embed_texts(documents, batch_size=batch_size)

    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = list(ids[i:i + batch_size])
        batch_docs = list(documents[i:i + batch_size])
        batch_metas = list(metadatas[i:i + batch_size])
        batch_embs = list(embeddings[i:i + batch_size])

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embs,
        )
        total_upserted += len(batch_ids)

    return total_upserted


def search(
    query: str,
    n_results: int = 8,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None,
) -> dict:
    """
    Search for similar documents.

    Args:
        query: Search query text
        n_results: Number of results to return
        where: Metadata filter (e.g., {"category": "Meditations"})
        where_document: Document content filter

    Returns:
        Dict with ids, documents, metadatas, distances
    """
    settings = get_settings()
    collection = get_collection()

    # Generate query embedding
    query_embedding = embed_text(query)

    # Execute search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        where_document=where_document,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten results (query returns nested lists)
    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def search_with_embeddings(
    query_embedding: list[float],
    n_results: int = 8,
    where: Optional[dict] = None,
) -> dict:
    """
    Search using pre-computed query embedding.

    Args:
        query_embedding: Pre-computed embedding vector
        n_results: Number of results to return
        where: Metadata filter

    Returns:
        Dict with ids, documents, metadatas, distances
    """
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def get_document_count() -> int:
    """Get total number of documents in collection."""
    collection = get_collection()
    return collection.count()


def document_exists(doc_id: str) -> bool:
    """Check if a document ID exists in the collection."""
    collection = get_collection()
    result = collection.get(ids=[doc_id], include=[])
    return len(result["ids"]) > 0


def get_documents_by_video_id(video_id: str) -> dict:
    """
    Get all chunks for a specific video.

    Args:
        video_id: YouTube or Vimeo video ID

    Returns:
        Dict with ids, documents, metadatas
    """
    collection = get_collection()
    results = collection.get(
        where={"video_id": video_id},
        include=["documents", "metadatas"],
    )
    return results


def delete_by_video_id(video_id: str) -> int:
    """
    Delete all chunks for a specific video.

    Args:
        video_id: YouTube or Vimeo video ID

    Returns:
        Number of documents deleted
    """
    collection = get_collection()

    # Get existing documents for this video
    existing = collection.get(
        where={"video_id": video_id},
        include=[],
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])

    return 0


def reset_collection() -> None:
    """
    Delete and recreate the collection.
    WARNING: This deletes all data!
    """
    global _collection

    settings = get_settings()
    client = get_chroma_client()

    try:
        client.delete_collection(settings.chroma_collection_name)
        logger.warning(f"Deleted collection '{settings.chroma_collection_name}'")
    except ValueError:
        pass  # Collection doesn't exist

    _collection = None
    get_collection()  # Recreate


def get_collection_stats() -> dict:
    """Get statistics about the collection."""
    collection = get_collection()
    total_chunks = collection.count()

    # Get all documents to analyze categories and sources
    sample = collection.get(
        limit=total_chunks,  # Get all documents for accurate stats
        include=["metadatas"],
    )

    categories = {}
    sources = {}
    videos = set()

    for meta in sample.get("metadatas", []):
        cat = meta.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1

        source = meta.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

        videos.add(meta.get("video_id"))

    return {
        "total_chunks": collection.count(),
        "unique_videos": len(videos),
        "categories": categories,
        "sources": sources,
    }
