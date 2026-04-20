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

## Running a lab

```bash
# Start workers (scale as needed):
cd lab-N
docker compose up -d --scale worker=4

# Run coordinator interactively:
docker compose run --rm coordinator
```

## Architecture pattern

Every lab uses the same topology:

```
coordinator (Python) → publishes tasks → [NATS] → workers (R, queue group)
coordinator          ← receives results ← [NATS] ←
```

- **coordinator** — orchestrates computation, benchmarks, prints results
- **worker** — stateless R process; subscribes to queue group `"workers"`, replies via NATS request-reply
- **nats** — `nats:latest`, the only shared bus

## Lab 1 — MapReduce matrix multiplication + linear regression

Goal: compare numpy vs distributed MapReduce on matrix operations.

- Coordinator reads matrix size from stdin, generates random matrices
- If size ≤ `NUMPY_MAX_SIZE=15_000`: runs numpy benchmark first
- Always runs MapReduce benchmark (tasks distributed to R workers via NATS)
- Prints timing comparison and correctness check

Task format: `{i, j, row_a: [...], col_b: [...]}` → reply: `{i, j, val: float}`

R worker uses a minimal raw-socket NATS client (no external R packages beyond `jsonlite`).
