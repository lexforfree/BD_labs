"""
NFL Win Probability analysis using Apache Spark.

Groups plays by (qtr, down, score_diff_bucket, field_bucket)
and computes empirical win probability = wins / total.
"""
import json
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, round as spark_round, sum as spark_sum, when

HDFS_INPUT = "hdfs://namenode:9000/data/nfl/processed/nfl_all.csv"
RESULTS_FILE = "/results/spark_result.json"
TIMING_FILE = "/results/timing_comparison.json"
MIN_SAMPLE = 10


def score_diff_bucket(col_name: str):
    return (
        when(col(col_name) <= -14, "le-14")
        .when(col(col_name) <= -7, "-13to-7")
        .when(col(col_name) <= -1, "-6to-1")
        .when(col(col_name) == 0, "0")
        .when(col(col_name) <= 6, "1to6")
        .when(col(col_name) <= 13, "7to13")
        .otherwise("ge14")
    )


def field_bucket(col_name: str):
    return (
        when(col(col_name) <= 25, "0-25")
        .when(col(col_name) <= 50, "26-50")
        .when(col(col_name) <= 75, "51-75")
        .otherwise("76-100")
    )


spark = SparkSession.builder \
    .appName("NFL Win Probability") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

start_time = time.time()

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(HDFS_INPUT)
)

# Filter to regulation plays only
df = df.filter(
    col("qtr").between(1, 4)
    & col("down").between(1, 4)
    & col("win").isNotNull()
)

# Bucketing
df = df.withColumn("score_diff_bucket", score_diff_bucket("score_differential")) \
       .withColumn("field_bucket", field_bucket("yardline_100"))

# Aggregate
result_df = (
    df.groupBy("qtr", "down", "score_diff_bucket", "field_bucket")
    .agg(
        spark_sum("win").alias("wins"),
        count("*").alias("total"),
    )
    .filter(col("total") >= MIN_SAMPLE)
    .withColumn("win_probability", spark_round(col("wins") / col("total"), 4))
    # Bayesian estimate: posterior mean under Beta(alpha, beta) prior.
    # Beta(2,2) encodes weak belief that win prob ~0.5; smooths sparse buckets toward 0.5.
    # Formula: (wins + alpha) / (total + alpha + beta)
    .withColumn("win_prob_bayes", spark_round(
        (col("wins") + lit(2.0)) / (col("total") + lit(4.0)), 4
    ))
    .orderBy("qtr", "down", "score_diff_bucket", "field_bucket")
)

elapsed_ms = int((time.time() - start_time) * 1000)

rows = result_df.collect()
output = []
for row in rows:
    output.append(
        {
            "qtr": int(row["qtr"]),
            "down": int(row["down"]),
            "score_diff_bucket": row["score_diff_bucket"],
            "field_bucket": row["field_bucket"],
            "wins": int(row["wins"]),
            "total": int(row["total"]),
            "win_probability": float(row["win_probability"]),
            "win_prob_bayes": float(row["win_prob_bayes"]),
        }
    )

os.makedirs("/results", exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    json.dump(output, f, indent=2)

# Update timing comparison
timing: dict = {}
if os.path.exists(TIMING_FILE):
    try:
        with open(TIMING_FILE) as f:
            timing = json.load(f)
    except Exception:
        pass
timing["spark_ms"] = elapsed_ms
with open(TIMING_FILE, "w") as f:
    json.dump(timing, f, indent=2)

print(f"Spark job completed in {elapsed_ms} ms")
print(f"Result rows: {len(output)}")

spark.stop()
