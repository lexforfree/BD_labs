#!/bin/bash
# run_all.sh — full benchmark pipeline (run from the host machine).
# Requires docker-compose to be up and healthy.
set -e

print_header() { echo ""; echo "══════════════════════════════════════"; echo " $1"; echo "══════════════════════════════════════"; }

print_header "1/4 — Upload data to HDFS"
docker exec namenode bash /scripts/upload_to_hdfs.sh

print_header "2/4 — Hadoop MapReduce"
docker exec namenode bash /scripts/run_mr.sh

print_header "3/4 — Hive"
docker exec hiveserver2 bash /scripts/run_hive.sh

print_header "4/4 — Apache Spark"
docker exec spark bash /scripts/run_spark.sh

print_header "Generating report"
docker exec visualization python3 /app/plot_results.py

echo ""
echo "✓ Pipeline complete. Open http://localhost:5050"
echo "  Timing results: lab-2/results/timing_comparison.json"
