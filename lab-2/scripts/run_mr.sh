#!/bin/bash
# Run Hadoop Streaming MapReduce Win Probability job.
# Runs inside the namenode container.
set -e

HDFS_INPUT="/data/nfl/processed/nfl_all.csv"
HDFS_OUTPUT="/output/win_probability_mr"
RESULTS_FILE="/results/mr_result.json"
TIMING_FILE="/results/timing_comparison.json"
STREAMING_JAR="/opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar"

echo "=== MapReduce Win Probability Job ==="
echo "Input:  hdfs://${HDFS_INPUT}"
echo "Output: hdfs://${HDFS_OUTPUT}"

# Remove previous output
hdfs dfs -rm -r -f "$HDFS_OUTPUT" 2>/dev/null || true

# Upload streaming scripts so YARN task containers can access them
hdfs dfs -mkdir -p /scripts
hdfs dfs -put -f /mapreduce/mapper.py  /scripts/mapper.py
hdfs dfs -put -f /mapreduce/reducer.py /scripts/reducer.py

START_MS=$(python3 -c "import time; print(int(time.time() * 1000))")
echo "Started at $(date)"

hadoop jar "$STREAMING_JAR" \
    -files "hdfs:///scripts/mapper.py,hdfs:///scripts/reducer.py" \
    -mapper  "python3 mapper.py" \
    -reducer "python3 reducer.py" \
    -input  "$HDFS_INPUT" \
    -output "$HDFS_OUTPUT"

END_MS=$(python3 -c "import time; print(int(time.time() * 1000))")
ELAPSED_MS=$((END_MS - START_MS))
echo "Finished in ${ELAPSED_MS} ms"

# Collect result JSON lines from HDFS into a single array
mkdir -p /results
hdfs dfs -cat "${HDFS_OUTPUT}/part-*" | python3 - <<'PYEOF'
import sys, json
rows = []
for line in sys.stdin:
    line = line.strip()
    if line:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
with open("/results/mr_result.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"Saved {len(rows)} rows to /results/mr_result.json")
PYEOF

# Save timing
python3 - <<PYEOF
import json, os
timing_file = "${TIMING_FILE}"
data = {}
if os.path.exists(timing_file):
    try:
        with open(timing_file) as f:
            data = json.load(f)
    except Exception:
        pass
data["mapreduce_ms"] = ${ELAPSED_MS}
with open(timing_file, "w") as f:
    json.dump(data, f, indent=2)
print("Timing saved: MapReduce = ${ELAPSED_MS} ms")
PYEOF
