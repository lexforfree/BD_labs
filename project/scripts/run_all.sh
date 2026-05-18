#!/bin/bash
# Full pipeline: download → preprocess → embed → index → benchmark
# Run from the host machine (docker-compose must be up).
set -e

step() { echo ""; echo "══════════════════════════════════════"; echo " $1"; echo "══════════════════════════════════════"; }

step "1/5 — Download Airbnb data"
bash "$(dirname "$0")/download_data.sh"

step "2/5 — Preprocess (merge cities)"
docker exec runner python3 /scripts/preprocess.py /data/raw /data/processed/listings.csv

step "3/5 — Generate embeddings (~30 min on CPU)"
docker exec runner python3 /scripts/generate_embeddings.py \
    /data/processed/listings.csv /data/processed/embeddings.npy

step "4/5 — Index into pgvector + Qdrant + Milvus"
docker exec runner python3 /scripts/index_all.py

step "5/5 — Benchmark"
docker exec runner python3 /scripts/benchmark.py

echo ""
echo "Done! Open http://localhost:5050"
