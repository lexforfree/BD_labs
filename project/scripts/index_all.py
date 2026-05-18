"""
Index listings + embeddings into pgvector, Qdrant, and Milvus.

Usage:
  python index_all.py [--limit N]   # --limit for testing with smaller subset

Reads:
  data/processed/listings.csv
  data/processed/embeddings.npy
  data/processed/embeddings_ids.txt

Writes:
  results/index_stats.json  (indexing times per DB)
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, PointStruct, VectorParams,
                                   HnswConfigDiff, OptimizersConfigDiff)
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      MilvusClient, connections, utility)
from tqdm import tqdm

LISTINGS_PATH = "/data/processed/listings.csv"
EMB_PATH      = "/data/processed/embeddings.npy"
IDS_PATH      = "/data/processed/embeddings_ids.txt"
RESULTS_PATH  = "/results/index_stats.json"

DIM = 384
BATCH = 1000

PG_DSN      = os.environ.get("PG_DSN", "postgresql://pguser:pgpassword@postgres:5432/listings")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(limit: int | None):
    print("Loading embeddings and metadata...", flush=True)
    embeddings = np.load(EMB_PATH)

    with open(IDS_PATH) as fh:
        emb_ids = [line.strip() for line in fh if line.strip()]

    id_to_idx = {lid: i for i, lid in enumerate(emb_ids)}

    rows = []
    with open(LISTINGS_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            idx = id_to_idx.get(row["id"])
            if idx is None:
                continue
            rows.append({
                "listing_id": row["id"],
                "name":        row["name"][:200],
                "description": row["description"][:500],
                "city":        row["city"],
                "price":       float(row["price"]) if row["price"] else 0.0,
                "bedrooms":    int(row["bedrooms"]) if row["bedrooms"] else 0,
                "bathrooms":   float(row["bathrooms"]) if row["bathrooms"] else 0.0,
                "latitude":    float(row["latitude"]),
                "longitude":   float(row["longitude"]),
                "embedding":   embeddings[idx],
            })
            if limit and len(rows) >= limit:
                break

    print(f"Loaded {len(rows):,} records")
    return rows


def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── pgvector ──────────────────────────────────────────────────────────────────

def index_pgvector(rows: list) -> float:
    print("\n── pgvector ──────────────────────────────────")
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()

    cur.execute("TRUNCATE listings RESTART IDENTITY;")
    conn.commit()

    t0 = time.time()
    for batch in tqdm(list(batched(rows, BATCH)), desc="pgvector insert"):
        execute_values(cur, """
            INSERT INTO listings
              (listing_id, name, description, city, price, bedrooms, bathrooms,
               latitude, longitude, embedding)
            VALUES %s
        """, [(
            r["listing_id"], r["name"], r["description"], r["city"],
            r["price"], r["bedrooms"], r["bathrooms"],
            r["latitude"], r["longitude"],
            r["embedding"].tolist(),
        ) for r in batch])
        conn.commit()

    print("Building HNSW index...", flush=True)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS listings_embedding_hnsw
        ON listings USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    conn.commit()
    elapsed = time.time() - t0

    cur.close()
    conn.close()
    print(f"pgvector: {elapsed:.1f}s  ({len(rows)/elapsed:.0f} rec/s)")
    return elapsed


# ── Qdrant ────────────────────────────────────────────────────────────────────

def index_qdrant(rows: list) -> float:
    print("\n── Qdrant ────────────────────────────────────")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists("listings"):
        client.delete_collection("listings")

    client.create_collection(
        collection_name="listings",
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=64),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
    )

    t0 = time.time()
    for i, batch in enumerate(tqdm(list(batched(rows, BATCH)), desc="qdrant upsert")):
        points = [
            PointStruct(
                id=i * BATCH + j,
                vector=r["embedding"].tolist(),
                payload={k: r[k] for k in
                         ("listing_id","name","description","city","price","bedrooms",
                          "bathrooms","latitude","longitude")},
            )
            for j, r in enumerate(batch)
        ]
        client.upsert(collection_name="listings", points=points)

    elapsed = time.time() - t0
    print(f"Qdrant: {elapsed:.1f}s  ({len(rows)/elapsed:.0f} rec/s)")
    return elapsed


# ── Milvus ────────────────────────────────────────────────────────────────────

def index_milvus(rows: list) -> float:
    print("\n── Milvus ────────────────────────────────────")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection("listings"):
        utility.drop_collection("listings")

    schema = CollectionSchema(fields=[
        FieldSchema("pk",          DataType.INT64,   is_primary=True, auto_id=True),
        FieldSchema("listing_id",  DataType.VARCHAR, max_length=64),
        FieldSchema("name",        DataType.VARCHAR, max_length=256),
        FieldSchema("description", DataType.VARCHAR, max_length=1024),
        FieldSchema("city",        DataType.VARCHAR, max_length=64),
        FieldSchema("price",       DataType.FLOAT),
        FieldSchema("bedrooms",    DataType.INT16),
        FieldSchema("bathrooms",   DataType.FLOAT),
        FieldSchema("latitude",    DataType.DOUBLE),
        FieldSchema("longitude",   DataType.DOUBLE),
        FieldSchema("embedding",   DataType.FLOAT_VECTOR, dim=DIM),
    ], description="Airbnb listings")

    col = Collection("listings", schema)

    t0 = time.time()
    for batch in tqdm(list(batched(rows, BATCH)), desc="milvus insert"):
        col.insert([
            [r["listing_id"]  for r in batch],
            [r["name"]        for r in batch],
            [r["description"] for r in batch],
            [r["city"]        for r in batch],
            [r["price"]       for r in batch],
            [r["bedrooms"]    for r in batch],
            [r["bathrooms"]   for r in batch],
            [r["latitude"]    for r in batch],
            [r["longitude"]   for r in batch],
            [r["embedding"].tolist() for r in batch],
        ])

    col.create_index("embedding", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 64},
    })
    col.load()
    elapsed = time.time() - t0

    connections.disconnect("default")
    print(f"Milvus: {elapsed:.1f}s  ({len(rows)/elapsed:.0f} rec/s)")
    return elapsed


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Index only first N records (for testing)")
    args = parser.parse_args()

    rows = load_data(args.limit)
    n = len(rows)

    stats = {"n_records": n}
    stats["pgvector_index_s"] = index_pgvector(rows)
    stats["qdrant_index_s"]   = index_qdrant(rows)
    stats["milvus_index_s"]   = index_milvus(rows)

    os.makedirs("/results", exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\n{'─'*50}")
    print(f"Records indexed: {n:,}")
    print(f"pgvector : {stats['pgvector_index_s']:.1f}s")
    print(f"Qdrant   : {stats['qdrant_index_s']:.1f}s")
    print(f"Milvus   : {stats['milvus_index_s']:.1f}s")
    print(f"\nStats saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
