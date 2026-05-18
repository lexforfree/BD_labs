CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS listings (
    id          BIGSERIAL PRIMARY KEY,
    listing_id  TEXT NOT NULL,
    name        TEXT,
    description TEXT,
    city        TEXT,
    price       NUMERIC,
    bedrooms    SMALLINT,
    bathrooms   NUMERIC,
    latitude    DOUBLE PRECISION,
    longitude   DOUBLE PRECISION,
    embedding   vector(384)
);

-- HNSW index for fast approximate nearest-neighbor search
-- Built after data load via index_all.py (CREATE INDEX takes time on large tables)
