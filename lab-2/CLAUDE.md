# Lab 2 — Big Data: NFL Win Probability

## Quick start

```bash
# 1. Start the cluster (~2-3 min for all services to become healthy)
docker-compose up -d --build

# 2. Download dataset (one-time, ~3 GB total)
bash scripts/download_data.sh

# 3. Run full benchmark pipeline
bash scripts/run_all.sh

# 4. Open dashboard
open http://localhost:5050
```

## Service URLs

| Service           | URL                      |
|-------------------|--------------------------|
| HDFS NameNode UI  | http://localhost:9870    |
| YARN ResourceMgr  | http://localhost:8088    |
| MR History Server | http://localhost:8188    |
| Spark Master UI   | http://localhost:8080    |
| HiveServer2 UI    | http://localhost:10002   |
| Airflow           | http://localhost:8081  (admin/admin) |
| Dashboard         | http://localhost:5050    |

## Dataset

NFL Play-by-Play, nflverse 2014–2023.  
Downloaded via `scripts/download_data.sh`.

## Research question

> How does a team's win probability change during a game based on
> (quarter, down, yards to go, field position, score differential)?

Computed as empirical frequency: `wins / total plays` per bucket,
which equals the posterior mean under a uniform Beta(1,1) prior —
the Bayesian baseline.

## Key files

```
mapreduce/mapper.py      # Hadoop Streaming mapper
mapreduce/reducer.py     # Hadoop Streaming reducer
hive/schema.hql          # Hive external table
hive/query.hql           # Hive aggregation query
spark/win_probability.py # PySpark job
airflow/dags/nfl_pipeline.py  # Airflow DAG (weekly schedule)
visualization/app.py     # Flask dashboard + REST API
```

## Rules

- No pandas, no scikit-learn anywhere in the pipeline.
- All distributed jobs must write timing to `results/timing_comparison.json`.
- Python version: 3.12 (visualization container).
