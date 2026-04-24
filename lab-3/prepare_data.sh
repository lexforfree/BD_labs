#!/usr/bin/env bash
# Prepare data for the mini search engine.
# Run this script once before docker compose up.
# Data is saved to ./data/ and mounted into containers.

set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="$(pwd)/data"
VENV=".venv"
PY="$VENV/bin/python"
mkdir -p "$DATA_DIR"

echo "=== Setting up Python environment ==="
if [ ! -f "$PY" ]; then
    uv venv "$VENV" --python 3.11
fi

uv pip install --python "$PY" \
    "datasets>=2.18.0" \
    "huggingface_hub>=0.20.0" \
    "networkx==3.3" \
    "scikit-learn==1.4.2" \
    "sentence-transformers>=2.7.0" \
    "numpy>=1.26" \
    "tqdm>=4.66" \
    "requests>=2.31"

export DATA_DIR

echo ""
echo "=== Step 1: Download article texts (streaming) ==="
"$PY" indexer/download_text.py

echo ""
echo "=== Step 2: Build inverted index + TF-IDF similarity graph ==="
"$PY" indexer/build_index.py

echo ""
echo "=== Step 3: Pre-compute BERT embeddings ==="
"$PY" indexer/precompute_bert.py

echo ""
echo "=== Step 4: PageRank via MapReduce ==="
"$PY" indexer/pagerank/mapreduce_pagerank.py

echo ""
echo "=== Step 5: PageRank via Pregel ==="
"$PY" indexer/pagerank/pregel_pagerank.py

echo ""
echo "=== Done! Data is ready in ./data/ ==="
echo "    Now run: docker compose up web"
