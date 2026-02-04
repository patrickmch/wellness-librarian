#!/bin/bash
# Initialize data directory from seed data if volume is empty
#
# This script handles Railway volume seeding:
# 1. Seed data is baked into the image at /app/data-seed/
# 2. Volume mounts at /app/data/ (empty on first deploy)
# 3. This script copies seed data to volume if empty
# 4. Subsequent deploys find existing data and skip copying

set -e

DATA_DIR="/app/data"
SEED_DIR="/app/data-seed"

echo "=== Data Initialization ==="
echo "Data directory: $DATA_DIR"
echo "Seed directory: $SEED_DIR"

# Check if data directory is empty or missing critical files
if [ ! -f "$DATA_DIR/docstore.sqlite" ] || [ ! -d "$DATA_DIR/chroma_db" ]; then
    echo "Data directory is empty or incomplete. Seeding from backup..."

    if [ -d "$SEED_DIR" ]; then
        echo "Copying seed data to volume..."
        cp -r "$SEED_DIR"/* "$DATA_DIR"/
        echo "Seed data copied successfully!"
        ls -la "$DATA_DIR"
    else
        echo "ERROR: Seed directory not found at $SEED_DIR"
        echo "The application may not work correctly."
    fi
else
    echo "Data directory already populated. Skipping seed."
    echo "Contents:"
    ls -la "$DATA_DIR"
fi

echo "=== Starting Application ==="
# If arguments provided, run them; otherwise run default command
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
fi
