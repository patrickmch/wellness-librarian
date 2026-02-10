-- Video Sync Tracking Table
-- Tracks which videos have been discovered, downloaded, and ingested from Vimeo/YouTube

CREATE TABLE IF NOT EXISTS video_sync (
    video_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL CHECK (platform IN ('vimeo', 'youtube')),
    title TEXT,
    transcript_file TEXT,           -- Relative path to VTT file
    folder_name TEXT,               -- Category/folder from source
    duration_seconds INTEGER,
    video_url TEXT,
    ingested_at TIMESTAMPTZ,        -- When embedded into ChromaDB
    last_seen_at TIMESTAMPTZ,       -- Last time seen in platform API
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'ingested', 'failed', 'no_transcript', 'skipped')),
    error_message TEXT,             -- Error details if status = 'failed'
    metadata JSONB DEFAULT '{}',    -- Additional platform-specific metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_video_sync_status ON video_sync(status);
CREATE INDEX IF NOT EXISTS idx_video_sync_platform ON video_sync(platform);
CREATE INDEX IF NOT EXISTS idx_video_sync_platform_status ON video_sync(platform, status);
CREATE INDEX IF NOT EXISTS idx_video_sync_last_seen ON video_sync(last_seen_at);

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_video_sync_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_video_sync_updated_at ON video_sync;
CREATE TRIGGER trigger_video_sync_updated_at
    BEFORE UPDATE ON video_sync
    FOR EACH ROW
    EXECUTE FUNCTION update_video_sync_updated_at();

-- Helpful view for sync statistics
CREATE OR REPLACE VIEW video_sync_stats AS
SELECT
    platform,
    status,
    COUNT(*) as count,
    MAX(last_seen_at) as most_recent_seen,
    MAX(ingested_at) as most_recent_ingested
FROM video_sync
GROUP BY platform, status
ORDER BY platform, status;

COMMENT ON TABLE video_sync IS 'Tracks video discovery and ingestion status for incremental sync';
COMMENT ON COLUMN video_sync.video_id IS 'Platform-specific video ID (Vimeo URI path or YouTube video ID)';
COMMENT ON COLUMN video_sync.status IS 'pending=discovered, ingested=in RAG, failed=error during processing, no_transcript=no captions available, skipped=manually excluded';
