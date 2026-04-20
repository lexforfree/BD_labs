#!/bin/bash
# Preprocess raw NFL CSVs and upload merged file to HDFS.
# Runs inside the namenode container.
set -e

RAW_DIR="/data/raw"
PROCESSED_FILE="/data/processed/nfl_all.csv"
HDFS_DIR="/data/nfl/processed"

echo "=== Step 1: Preprocess CSV files ==="
python3 /scripts/preprocess.py "$RAW_DIR" "$PROCESSED_FILE"

echo ""
echo "=== Step 2: Create HDFS directory ==="
hdfs dfs -mkdir -p "$HDFS_DIR"
hdfs dfs -rm -f "${HDFS_DIR}/nfl_all.csv" 2>/dev/null || true

echo ""
echo "=== Step 3: Upload to HDFS ==="
hdfs dfs -put "$PROCESSED_FILE" "${HDFS_DIR}/nfl_all.csv"

echo ""
echo "=== Verification ==="
hdfs dfs -ls "$HDFS_DIR"
hdfs dfs -du -s -h "$HDFS_DIR"
echo "Upload complete."
