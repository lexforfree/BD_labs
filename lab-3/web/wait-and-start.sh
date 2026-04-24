#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data}"

for f in "$DATA_DIR/articles.jsonl" "$DATA_DIR/index.pkl" "$DATA_DIR/graph.pkl"; do
    if [ ! -f "$f" ]; then
        echo "[web] ERROR: $f not found."
        echo "[web] Run ./prepare_data.sh first, then docker compose up."
        exit 1
    fi
done

echo "[web] Data files OK. Starting Flask..."
exec flask run
