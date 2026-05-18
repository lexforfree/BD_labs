"""
Vector DB Demo Dashboard — Flask app.

Routes:
  GET  /                   — main page (map + search)
  POST /api/search         — semantic search across all 3 DBs
  GET  /api/benchmark      — benchmark results JSON
  GET  /api/sample         — random 2000 listings for map background
"""
import csv
import json
import os
import random
import re
import time

import numpy as np
import psycopg2
from flask import Flask, jsonify, render_template, request
from psycopg2.extras import RealDictCursor
from pymilvus import Collection, connections
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

PG_DSN      = os.environ.get("PG_DSN", "postgresql://pguser:pgpassword@postgres:5432/listings")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")
LISTINGS_CSV = "/data/processed/listings.csv"

DEFAULT_TOP_K = 20
MAX_TOP_K = 200
_model = None
_sample_cache = None
_listing_cache = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> list[float]:
    vec = get_model().encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def parse_top_k(value) -> int:
    try:
        return min(max(int(value), 1), MAX_TOP_K)
    except (TypeError, ValueError):
        return DEFAULT_TOP_K


def ann_search_depth(top_k: int) -> int:
    return min(max(top_k * 2, 64), 1000)


def clean_description(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def load_listing_cache() -> dict[str, dict]:
    global _listing_cache
    if _listing_cache is not None:
        return _listing_cache

    cache = {}
    if not os.path.exists(LISTINGS_CSV):
        _listing_cache = cache
        return cache

    with open(LISTINGS_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            listing_id = row.get("id", "")
            if not listing_id:
                continue
            cache[listing_id] = {
                "listing_id": listing_id,
                "name": row.get("name", ""),
                "description": clean_description(row.get("description", "")),
                "city": row.get("city", ""),
                "price": float(row["price"]) if row.get("price") else 0.0,
                "bedrooms": int(float(row["bedrooms"])) if row.get("bedrooms") else 0,
                "bathrooms": float(row["bathrooms"]) if row.get("bathrooms") else 0.0,
                "latitude": float(row["latitude"]) if row.get("latitude") else None,
                "longitude": float(row["longitude"]) if row.get("longitude") else None,
                "text": row.get("text", ""),
                "url": f"https://www.airbnb.com/rooms/{listing_id}",
            }

    _listing_cache = cache
    return cache


def enrich_rows(rows: list[dict]) -> list[dict]:
    listings = load_listing_cache()
    enriched = []
    for row in rows:
        listing_id = str(row.get("listing_id", ""))
        merged = {**row, **listings.get(listing_id, {})}
        if "score" in row:
            merged["score"] = row["score"]
        if "description" not in merged:
            merged["description"] = ""
        if "url" not in merged and listing_id:
            merged["url"] = f"https://www.airbnb.com/rooms/{listing_id}"
        enriched.append(merged)
    return enriched


# ── search backends ───────────────────────────────────────────────────────────

def search_pgvector(vec: list, top_k: int) -> tuple[list, float]:
    t0 = time.perf_counter()
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SET LOCAL hnsw.ef_search = %s", (ann_search_depth(top_k),))
    cur.execute("""
        SELECT listing_id, name, description, city, price, bedrooms, bathrooms,
               latitude, longitude,
               1 - (embedding <=> %s::vector) AS score
        FROM listings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (vec, vec, top_k))
    rows = enrich_rows([dict(r) for r in cur.fetchall()])
    latency_ms = (time.perf_counter() - t0) * 1000
    cur.close()
    conn.close()
    return rows, latency_ms


def search_qdrant(vec: list, top_k: int) -> tuple[list, float]:
    t0 = time.perf_counter()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    hits = client.search(
        collection_name="listings",
        query_vector=vec,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=ann_search_depth(top_k)),
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    rows = enrich_rows([
        {**h.payload, "score": round(h.score, 4)}
        for h in hits
    ])
    return rows, latency_ms


def search_milvus(vec: list, top_k: int) -> tuple[list, float]:
    t0 = time.perf_counter()
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection("listings")
    col.load()
    res = col.search(
        data=[vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": ann_search_depth(top_k)}},
        limit=top_k,
        output_fields=["listing_id", "name", "city", "price",
                       "bedrooms", "bathrooms", "latitude", "longitude"],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    rows = enrich_rows([
        {**{f: hit.entity.get(f) for f in
            ("listing_id","name","city","price","bedrooms","bathrooms",
             "latitude","longitude")},
         "score": round(hit.distance, 4)}
        for hit in res[0]
    ])
    connections.disconnect("default")
    return rows, latency_ms


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    query = (data or {}).get("query", "").strip()
    db = (data or {}).get("db", "all")
    dbs = (data or {}).get("dbs")
    top_k = parse_top_k((data or {}).get("top_k"))

    if not query:
        return jsonify({"error": "empty query"}), 400

    vec = embed(query)
    selected = set(dbs if isinstance(dbs, list) else [])
    if not selected:
        selected = {"pgvector", "qdrant", "milvus"} if db == "all" else {db}

    results = {}
    if "pgvector" in selected:
        rows, ms = search_pgvector(vec, top_k)
        results["pgvector"] = {"results": rows, "latency_ms": round(ms, 2), "top_k": top_k}
    if "qdrant" in selected:
        rows, ms = search_qdrant(vec, top_k)
        results["qdrant"] = {"results": rows, "latency_ms": round(ms, 2), "top_k": top_k}
    if "milvus" in selected:
        rows, ms = search_milvus(vec, top_k)
        results["milvus"] = {"results": rows, "latency_ms": round(ms, 2), "top_k": top_k}

    return jsonify(results)


@app.route("/api/benchmark")
def benchmark():
    path = os.path.join(RESULTS_DIR, "benchmark.json")
    if not os.path.exists(path):
        return jsonify({"error": "Run benchmark.py first"}), 404
    with open(path) as fh:
        return jsonify(json.load(fh))


@app.route("/api/sample")
def sample():
    global _sample_cache
    if _sample_cache is not None:
        return jsonify(_sample_cache)

    if not os.path.exists(LISTINGS_CSV):
        return jsonify([])

    all_rows = []
    with open(LISTINGS_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                all_rows.append({
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "name": row["name"][:60],
                    "city": row["city"],
                    "price": row["price"],
                })
            except (ValueError, KeyError):
                pass

    sample = random.sample(all_rows, min(2000, len(all_rows)))
    _sample_cache = sample
    return jsonify(sample)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
