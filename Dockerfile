# Wellness Librarian - Production Dockerfile
#
# Supports two storage backends:
# - SQLite + ChromaDB (local development, requires volume mount)
# - Supabase with pgvector (production, no volume needed)

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install system dependencies (including curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Create data directory for SQLite/ChromaDB fallback (local development)
# Production with Supabase doesn't need seeded data, but the directory must exist
RUN mkdir -p ./data/chroma_db

# Expose port (Railway will override with $PORT)
EXPOSE ${PORT}

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Run the application directly (no init script needed with Supabase)
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
