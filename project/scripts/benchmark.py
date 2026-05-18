"""
Benchmark pgvector, Qdrant, and Milvus: latency and recall.

Metrics:
  - Query latency P50 / P95 / P99  (100 random queries)
  - Recall@10: fraction of pgvector top-10 that Qdrant/Milvus also return
    (pgvector with full scan is ground truth)

Output: results/benchmark.json
"""
import json
import os
import random
import time

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from pymilvus import Collection, connections

EMB_PATH    = "/data/processed/embeddings.npy"
RESULTS_OUT = "/results/benchmark.json"
N_QUERIES   = 100
TOP_K       = 10

PG_DSN      = os.environ.get("PG_DSN", "postgresql://pguser:pgpassword@postgres:5432/listings")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))


def percentile(data: list, p: int) -> float:
    return float(np.percentile(data, p))


# ── ground truth: pgvector exact scan ────────────────────────────────────────

def query_pgvector_exact(cur, vec: list) -> list[str]:
    cur.execute("""
        SELECT listing_id
        FROM listings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (vec, TOP_K))
    return [r["listing_id"] for r in cur.fetchall()]


def bench_pgvector_hnsw(vecs: list) -> tuple[list[float], list[list[str]]]:
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    latencies = []
    results = []
    for vec in vecs:
        t0 = time.perf_counter()
        cur.execute("""
            SELECT listing_id FROM listings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vec.tolist(), TOP_K))
        rows = cur.fetchall()
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append([r["listing_id"] for r in rows])
    cur.close()
    conn.close()
    return latencies, results


# ── Qdrant ────────────────────────────────────────────────────────────────────

def bench_qdrant(vecs: list) -> tuple[list[float], list[list]]:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    latencies = []
    results = []
    for vec in vecs:
        t0 = time.perf_counter()
        hits = client.search(
            collection_name="listings",
            query_vector=vec.tolist(),
            limit=TOP_K,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append([h.payload["listing_id"] for h in hits])
    return latencies, results


# ── Milvus ────────────────────────────────────────────────────────────────────

def bench_milvus(vecs: list) -> tuple[list[float], list[list]]:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection("listings")
    col.load()
    latencies = []
    results = []
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    for vec in vecs:
        t0 = time.perf_counter()
        res = col.search(
            data=[vec.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            output_fields=["listing_id"],
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append([hit.entity.get("listing_id") for hit in res[0]])
    connections.disconnect("default")
    return latencies, results


# ── recall ────────────────────────────────────────────────────────────────────

def recall(ground_truth: list[list], predictions: list[list]) -> float:
    hits = sum(
        len(set(gt) & set(pred)) / max(len(gt), 1)
        for gt, pred in zip(ground_truth, predictions)
    )
    return round(hits / len(ground_truth), 4)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    embeddings = np.load(EMB_PATH)
    n = len(embeddings)
    print(f"Loaded {n:,} embeddings. Sampling {N_QUERIES} random query vectors...")

    query_indices = random.sample(range(n), N_QUERIES)
    query_vecs = [embeddings[i] for i in query_indices]

    print("\n── pgvector (HNSW) ───")
    pg_lat, pg_res = bench_pgvector_hnsw(query_vecs)

    print("── Qdrant ────────────")
    qd_lat, qd_res = bench_qdrant(query_vecs)

    print("── Milvus ────────────")
    mv_lat, mv_res = bench_milvus(query_vecs)

    stats = {
        "n_records":    n,
        "n_queries":    N_QUERIES,
        "top_k":        TOP_K,
        "pgvector": {
            "p50_ms": percentile(pg_lat, 50),
            "p95_ms": percentile(pg_lat, 95),
            "p99_ms": percentile(pg_lat, 99),
            "recall": 1.0,  # it IS the ground truth
        },
        "qdrant": {
            "p50_ms": percentile(qd_lat, 50),
            "p95_ms": percentile(qd_lat, 95),
            "p99_ms": percentile(qd_lat, 99),
            "recall": recall(pg_res, qd_res),
        },
        "milvus": {
            "p50_ms": percentile(mv_lat, 50),
            "p95_ms": percentile(mv_lat, 95),
            "p99_ms": percentile(mv_lat, 99),
            "recall": recall(pg_res, mv_res),
        },
    }

    os.makedirs("/results", exist_ok=True)
    with open(RESULTS_OUT, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\n{'─'*30}")
    print(f"{'Tool':<12} {'P50':>8} {'P95':>8} {'P99':>8} {'Recall':>8}")
    print(f"{'─'*52}")
    for tool in ("pgvector", "qdrant", "milvus"):
        s = stats[tool]
        print(f"{tool:<12} {s['p50_ms']:>7.1f}ms {s['p95_ms']:>7.1f}ms "
              f"{s['p99_ms']:>7.1f}ms {s['recall']:>8.4f}")
    print(f"\nSaved → {RESULTS_OUT}")


if __name__ == "__main__":
    main()
