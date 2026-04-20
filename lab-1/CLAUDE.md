# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

University Big Data course — 3 laboratory works on distributed computing.
Lab texts: `labs_text.md`. Each lab lives in its own folder (`lab-1/`, `lab-2/`, `lab-3/`).

## Technology stack

- **Runtime**: Docker Compose (all services containerized)
- **Workers**: R language
- **Coordinator**: Python
- **Messaging**: NATS — all inter-service communication uses NATS subjects

## Architecture pattern

Every lab uses the same topology:

```
coordinator (Python) → publishes tasks → [NATS] → workers (R, queue group "workers")
coordinator          ← receives results ← [NATS] ←
```

- **coordinator** — orchestrates computation, benchmarks, prints results
- **worker** — stateless R process; subscribes to queue group `"workers"`, replies via NATS request-reply
- **nats** — `nats:latest`, the only shared bus

## Lab 1 — MapReduce matrix multiplication + linear regression

Goal: compare numpy vs distributed MapReduce on matrix operations.

```bash
cd lab-1

# after code change
docker compose build coordinator worker                   


docker compose up -d --scale worker=4                    # start R workers
docker compose run --rm coordinator python matmul.py     # matrix multiplication
docker compose run --rm coordinator python linreg.py     # linear regression
```

Each script reads sizes from stdin, generates random matrices, runs both numpy and MapReduce, then prints timing and correctness comparison. If size > `NUMPY_MAX_SIZE=15_000`, numpy is skipped (memory guard for 8 GB RAM).

Shared module `mapreduce.py` — `matmul_mapreduce(nc, A, B)` distributes one NATS task per output cell.  
Task format: `{i, j, row_a: [...], col_b: [...]}` → reply: `{i, j, val: float}`

R worker uses a minimal raw-socket NATS client (only `jsonlite` required, no extra packages).
