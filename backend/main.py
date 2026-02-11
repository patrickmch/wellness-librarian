"""
FastAPI application entry point for the Wellness Librarian.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import get_settings
from backend.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    settings = get_settings()
    logger.info(f"Starting Wellness Librarian")
    logger.info(f"Store backend: {settings.store_backend}")
    logger.info(f"RAG pipeline: {settings.rag_pipeline}")

    # Initialize store connection on startup
    if settings.store_backend == "supabase":
        from backend.rag.stores.supabase_store import get_supabase_store
        store = get_supabase_store()
        stats = store.get_stats()
        chunk_count = stats.get("total_parent_chunks", 0)
        logger.info(f"Supabase store ready with {chunk_count} parent chunks")
    else:
        from backend.rag.vectorstore import get_collection
        collection = get_collection()
        logger.info(f"ChromaDB ready with {collection.count()} chunks")

    yield

    # Shutdown
    logger.info("Shutting down Wellness Librarian")


# Create FastAPI app
app = FastAPI(
    title="Wellness Librarian",
    description="A RAG-powered assistant for the Wellness Evolution Community video library",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")

# Serve frontend static files
try:
    from pathlib import Path
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/")
        async def serve_frontend():
            """Serve the frontend application."""
            return FileResponse(str(frontend_dir / "index.html"))
except Exception as e:
    logger.warning(f"Could not mount frontend: {e}")


def main():
    """Run the application with uvicorn."""
    import uvicorn
    settings = get_settings()

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
