"""MapReduce matrix operations over NATS: matmul, transpose, solve, ata+aty."""

import asyncio
import json
from typing import Callable, Tuple
import numpy as np
import nats as nats_lib

# Max NATS requests in flight for small-payload operations (matmul, transpose).
_CONCURRENCY = 512
# Rows per asyncio.gather batch — limits the number of live coroutine objects.
_CHUNK = 256

# ata_row responses carry a K×K matrix (~K²×20 bytes of JSON).
# Keep concurrency low to avoid flooding NATS buffers with huge messages.
_CONCURRENCY_ATA = 16
_CHUNK_ATA = 16


# ── matmul ───────────────────────────────────────────────────────────────────

async def _compute_cell(sem: asyncio.Semaphore, nc: nats_lib.NATS,
                        i: int, j: int, A: np.ndarray, B: np.ndarray) -> tuple:
    async with sem:
        # .tolist() is called only when the semaphore is acquired so at most
        # _CONCURRENCY rows/cols are converted to Python lists at any time.
        payload = json.dumps({
            "type": "matmul", "i": i, "j": j,
            "row_a": A[i, :].tolist(),
            "col_b": B[:, j].tolist(),
        }).encode()
        response = await nc.request("tasks.compute", payload, timeout=60)
    data = json.loads(response.data)
    return data["i"], data["j"], data["val"]


async def matmul_mapreduce(nc: nats_lib.NATS,
                           A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute A @ B — rows processed in _CHUNK batches, _CONCURRENCY tasks in flight."""
    M, _ = A.shape
    _, N = B.shape
    sem = asyncio.Semaphore(_CONCURRENCY)
    C = np.zeros((M, N))

    for row_start in range(0, M, _CHUNK):
        row_end = min(row_start + _CHUNK, M)
        tasks = [
            _compute_cell(sem, nc, i, j, A, B)
            for i in range(row_start, row_end)
            for j in range(N)
        ]
        for i, j, val in await asyncio.gather(*tasks):
            C[i, j] = val

    return C


# ── transpose ────────────────────────────────────────────────────────────────

async def _transpose_row(sem: asyncio.Semaphore, nc: nats_lib.NATS,
                         row_idx: int, A: np.ndarray) -> tuple:
    async with sem:
        payload = json.dumps({
            "type": "transpose",
            "row_idx": row_idx,
            "row": A[row_idx, :].tolist(),
        }).encode()
        response = await nc.request("tasks.compute", payload, timeout=60)
    data = json.loads(response.data)
    return data["row_idx"], data["row"]


async def transpose_mapreduce(nc: nats_lib.NATS, A: np.ndarray) -> np.ndarray:
    """Compute A^T — rows processed in _CHUNK batches, _CONCURRENCY tasks in flight."""
    M, K = A.shape
    sem = asyncio.Semaphore(_CONCURRENCY)
    At = np.zeros((K, M))

    for row_start in range(0, M, _CHUNK):
        row_end = min(row_start + _CHUNK, M)
        tasks = [_transpose_row(sem, nc, i, A) for i in range(row_start, row_end)]
        for row_idx, row in await asyncio.gather(*tasks):
            At[:, row_idx] = row

    return At


# ── solve ─────────────────────────────────────────────────────────────────────

async def solve_mapreduce(nc: nats_lib.NATS,
                          A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b on a worker (A is the small K×K system, fits in one payload)."""
    payload = json.dumps({"type": "solve", "A": A.tolist(), "b": b.tolist()}).encode()
    response = await nc.request("tasks.compute", payload, timeout=60)
    data = json.loads(response.data)
    return np.array(data["x"]).reshape(-1, 1)


# ── ata + aty (streaming — full A is never held in memory) ───────────────────

async def _ata_row(sem: asyncio.Semaphore, nc: nats_lib.NATS,
                   i: int, row_gen: Callable) -> Tuple[np.ndarray, np.ndarray]:
    async with sem:
        # row_gen is called only when the semaphore is acquired, so at most
        # _CONCURRENCY rows exist as Python lists at any moment.
        row_a, y_i = row_gen(i)
        payload = json.dumps({"type": "ata_row", "row_a": row_a, "y": y_i}).encode()
        response = await nc.request("tasks.compute", payload, timeout=60)
    data = json.loads(response.data)
    return np.array(data["ata_partial"]), np.array(data["aty_partial"])


async def ata_aty_mapreduce(
    nc: nats_lib.NATS,
    row_gen: Callable[[int], Tuple[list, float]],
    M: int,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute A^T*A (K×K) and A^T*Y (K×1) by streaming rows through workers.
    row_gen(i) must return (row_a: list[float], y_i: float) for row i.
    The full matrix A is never stored — only _CONCURRENCY rows exist at once.
    """
    sem = asyncio.Semaphore(_CONCURRENCY_ATA)
    AtA = np.zeros((K, K))
    AtY = np.zeros((K, 1))

    for batch_start in range(0, M, _CHUNK_ATA):
        batch_end = min(batch_start + _CHUNK_ATA, M)
        tasks = [_ata_row(sem, nc, i, row_gen) for i in range(batch_start, batch_end)]
        for ata_p, aty_p in await asyncio.gather(*tasks):
            AtA += ata_p.reshape(K, K)
            AtY += aty_p.reshape(K, 1)

    return AtA, AtY
