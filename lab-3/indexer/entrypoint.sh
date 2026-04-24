#!/bin/bash
set -e

echo "=== Step 1: Download article texts ==="
python download_text.py

echo "=== Step 2: Download link graph ==="
python download_links.py

echo "=== Step 3: Build inverted index and graph ==="
python build_index.py

echo "=== Step 4: PageRank via MapReduce ==="
python pagerank/mapreduce_pagerank.py

echo "=== Step 5: PageRank via Pregel ==="
python pagerank/pregel_pagerank.py

echo "=== Indexing complete ==="
