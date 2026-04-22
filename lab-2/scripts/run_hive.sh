#!/bin/bash
# Run Hive Win Probability query and save results as JSON.
# Runs inside the hiveserver2 container.
set -e

HDFS_OUTPUT="hdfs://namenode:9000/output/hive_win_prob"
RESULTS_FILE="/results/hive_result.json"
TIMING_FILE="/results/timing_comparison.json"

echo "=== Hive Win Probability Query ==="

# Create schema (external table) if not already done
beeline -u "jdbc:hive2://localhost:10000" \
    --silent=true \
    -f /hive/schema.hql \
    -n root -p root

# Remove previous query output
hadoop fs -rm -r -f "$HDFS_OUTPUT" 2>/dev/null || true

START_MS=$(date +%s%3N)
echo "Started at $(date)"

# Run the analytical query (writes TSV to HDFS)
beeline -u "jdbc:hive2://localhost:10000" \
    --silent=true \
    -f /hive/query.hql

END_MS=$(date +%s%3N)
ELAPSED_MS=$((END_MS - START_MS))
echo "Finished in ${ELAPSED_MS} ms"

# Fetch TSV output from HDFS and convert to JSON
mkdir -p /results
hdfs dfs -cat "${HDFS_OUTPUT}/*" > /tmp/hive_output.tsv
python3 - <<'PYEOF'
import json

COLUMNS = ["qtr", "down", "score_diff_bucket", "field_bucket", "wins", "total", "win_probability"]
rows = []
with open("/tmp/hive_output.tsv") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < len(COLUMNS):
            continue
        try:
            rows.append({
                "qtr":              int(parts[0]),
                "down":             int(parts[1]),
                "score_diff_bucket": parts[2],
                "field_bucket":     parts[3],
                "wins":             int(parts[4]),
                "total":            int(parts[5]),
                "win_probability":  float(parts[6]),
            })
        except (ValueError, IndexError):
            pass
with open("/results/hive_result.json", "w") as f:
    json.dump(rows, f, indent=2)
print("Saved {} rows to /results/hive_result.json".format(len(rows)))
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
data["hive_ms"] = ${ELAPSED_MS}
with open(timing_file, "w") as f:
    json.dump(data, f, indent=2)
print("Timing saved: Hive = ${ELAPSED_MS} ms")
PYEOF
