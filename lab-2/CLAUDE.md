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

| Service           | URL                                  |
|-------------------|--------------------------------------|
| HDFS NameNode UI  | http://localhost:9870                |
| YARN ResourceMgr  | http://localhost:8088                |
| MR History Server | http://localhost:8188                |
| Spark Master UI   | http://localhost:8080                |
| HiveServer2 UI    | http://localhost:10002               |
| Airflow           | http://localhost:8081  (admin/admin) |
| Dashboard         | http://localhost:5050                |

## Dataset

NFL Play-by-Play, nflverse 2014–2023.
Downloaded via `scripts/download_data.sh`.
~404k plays after preprocessing (6.7 MB CSV on HDFS).

## Research question

> How does a team's win probability change during a game based on
> (quarter, down, field position, score differential)?

Bucketing:
- **score_diff**: le-14 | -13to-7 | -6to-1 | 0 | 1to6 | 7to13 | ge14
- **field_bucket**: 0-25 | 26-50 | 51-75 | 76-100 (yardline_100)
- Minimum 10 plays per bucket (`MIN_SAMPLE = 10`)
- 447 buckets total

### Estimates

| Name | Formula | Notes |
|------|---------|-------|
| MLE | `wins / total` | Unreliable for sparse buckets |
| Bayes Beta(2,2) | `(wins + 2) / (total + 4)` | Posterior mean; prior = mild belief toward 0.5 |

Both are stored in `spark_result.json` as `win_probability` and `win_prob_bayes`.

## Key files

```
mapreduce/mapper.py           # Hadoop Streaming mapper
mapreduce/reducer.py          # Hadoop Streaming reducer
hive/schema.hql               # Hive external table
hive/query.hql                # Hive aggregation query
spark/win_probability.py      # PySpark job (MLE + Bayesian estimate)
airflow/dags/nfl_pipeline.py  # Airflow DAG (weekly schedule)
visualization/app.py          # Flask dashboard + REST API
visualization/templates/index.html  # Dashboard UI
```

## Results (last run)

| Tool       | Time     | Rows |
|------------|----------|------|
| MapReduce  | ~35 s    | 447  |
| Hive       | ~60 s    | 447  |
| Spark      | ~5.5 s   | 447  |

Spark is ~7× faster than MapReduce, ~11× faster than Hive.
Results stored in `results/mr_result.json`, `results/hive_result.json`, `results/spark_result.json`.

## Known gotchas

- `pipe | python3 - <<'PYEOF'` — heredoc wins stdin, pipe data is discarded.
  Fixed in `run_mr.sh` and `run_hive.sh`: dump HDFS output to `/tmp/*.txt` first.
- `hiveserver2/Dockerfile` installs python3 on top of `bde2020/hive:2.3.2-postgresql-metastore`
  (Debian Stretch EOL — uses archive.debian.org with `--allow-unauthenticated`).
- PostgreSQL must be version 13 (not 14+): old Hive JDBC driver does not support SCRAM-SHA-256.
- Airflow `BashOperator` bash_commands ending in `.sh` need a trailing space to avoid
  `TemplateNotFound` (template_ext includes `.sh`).
- After rebuilding `visualization/`, run `docker compose build visualization && docker compose up -d visualization`
  (restart alone does not pick up template changes — they are baked into the image).

## Rules

- No pandas, no scikit-learn anywhere in the pipeline.
- All distributed jobs must write timing to `results/timing_comparison.json`.
- Python version: 3.12 (visualization container).
