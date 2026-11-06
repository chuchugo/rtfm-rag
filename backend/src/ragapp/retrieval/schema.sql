CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    title TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per /api/chat call. Backs both live monitoring (latency, refusal
-- rate, thumbs up/down) and offline eval (retrieved_chunks feeds Recall@K /
-- MRR / context precision once scored against a labeled question set).
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    -- [{"chunk_id": ..., "document_id": ..., "filename": ..., "page": ..., "rank": 0}, ...]
    -- in retrieval rank order (rank 0 = top result).
    retrieved_chunks JSONB NOT NULL,
    -- [{"filename": ..., "page": ...}, ...] chunks the answer text actually cited.
    cited_sources JSONB NOT NULL,
    refused BOOLEAN NOT NULL DEFAULT false,
    -- retrieval_latency_ms = embed_latency_ms + search_latency_ms, kept as its
    -- own column since it's what most monitoring queries actually want.
    retrieval_latency_ms INT NOT NULL,
    embed_latency_ms INT,
    search_latency_ms INT,
    generation_latency_ms INT NOT NULL,
    -- 1 = thumbs up, -1 = thumbs down, NULL = no feedback given.
    feedback SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS embed_latency_ms INT;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS search_latency_ms INT;

CREATE INDEX IF NOT EXISTS query_logs_created_at_idx ON query_logs (created_at);
