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
- **worker** — stateless R process; subscribes to queue group `"workers"`, replies via NATS request-reply; reconnects automatically on connection loss
- **nats** — `nats:latest`, the only shared bus

## Lab 1 — MapReduce matrix multiplication + linear regression ✓

Goal: compare numpy vs distributed MapReduce on matrix operations.

```bash
cd lab-1

# after code change
docker compose build coordinator worker

docker compose up -d --scale worker=4                    # start R workers
docker compose run --rm coordinator python matmul.py     # matrix multiplication
docker compose run --rm coordinator python linreg.py     # linear regression
```

Each script reads sizes from stdin, generates random matrices, runs both numpy and MapReduce, then prints timing and correctness comparison. If size > `NUMPY_MAX_SIZE=15_000`, numpy is skipped (memory guard).

### mapreduce.py — distributed operations

All functions use `_CONCURRENCY=512` semaphore (limits in-flight NATS requests) and `_CHUNK=256` batch size (limits live coroutine count). `.tolist()` is called only when the semaphore is acquired — no full matrix copies upfront.

| Function | Description |
|---|---|
| `matmul_mapreduce(nc, A, B)` | A @ B — one cell per task, reply: `{i, j, val}` |
| `transpose_mapreduce(nc, A)` | A^T — one row per task, reply: `{row_idx, row}` |
| `solve_mapreduce(nc, A, b)` | solve Ax=b — single task, reply: `{x}` |
| `ata_aty_mapreduce(nc, row_gen, M, K)` | A^T·A and A^T·Y by streaming rows — full matrix A never in RAM; uses `_CONCURRENCY_ATA=16`, `_CHUNK_ATA=16` (large responses ~K²×20 bytes each) |

### worker.R — task types

| `type` | Computation |
|---|---|
| `matmul` | dot product `sum(row_a * col_b)` |
| `transpose` | echo row back (coordinator repositions it) |
| `solve` | `solve(A, b)` via R |
| `ata_row` | `outer(row_a, row_a)` + `row_a * y` (partial A^T·A and A^T·Y contributions) |

### linreg.py — memory strategy

For **small matrices** (`max(M,K) ≤ 15000`): A and Y are generated in full (needed for numpy benchmark). `row_gen` wraps them.

For **large matrices**: A is never stored. `row_gen(i)` uses `np.random.default_rng([7, i])` to generate row i on demand — at most `_CONCURRENCY_ATA=16` rows in memory at any moment.
