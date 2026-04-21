#!/bin/bash
# Run PySpark Win Probability job.
# Runs inside the spark container.
set -e

SPARK_JOB="/opt/spark-jobs/win_probability.py"
TIMING_FILE="/results/timing_comparison.json"

echo "=== Spark Win Probability Job ==="
echo "Master: spark://spark:7077"

START_MS=$(python3 -c "import time; print(int(time.time() * 1000))")
echo "Started at $(date)"

/opt/spark/bin/spark-submit \
    --master spark://spark:7077 \
    --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
    --conf spark.hadoop.dfs.client.use.datanode.hostname=true \
    "$SPARK_JOB"

END_MS=$(python3 -c "import time; print(int(time.time() * 1000))")
ELAPSED_MS=$((END_MS - START_MS))
echo "Finished in ${ELAPSED_MS} ms"
