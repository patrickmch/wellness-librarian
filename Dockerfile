# Wellness Librarian - Production Dockerfile

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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Copy data to SEED directory (not the mount point)
# This allows volume seeding on first deploy
COPY data/ ./data-seed/

# Create empty data directory (volume will mount here)
RUN mkdir -p ./data

# Make init script executable
RUN chmod +x ./scripts/init-data.sh

# Expose port (Railway will override with $PORT)
EXPOSE ${PORT}

# Health check using curl (more reliable in containers)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Use init script as entrypoint (seeds volume if empty, then runs CMD)
ENTRYPOINT ["./scripts/init-data.sh"]

# Default command (init script handles PORT variable)
CMD []
