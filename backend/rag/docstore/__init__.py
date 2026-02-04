"""
Document store package for persistent parent chunk storage.

Provides SQLite-based storage for parent chunks that are retrieved
by ID after child matching in the vector store.
"""

from backend.rag.docstore.sqlite_store import SQLiteDocStore

__all__ = [
    "SQLiteDocStore",
]
