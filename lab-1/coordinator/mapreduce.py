"""MapReduce matrix multiplication over NATS."""

import asyncio
import json
import numpy as np
import nats as nats_lib


async def _compute_cell(nc: nats_lib.NATS, i: int, j: int, row_a: list, col_b: list) -> tuple:
    """Send one dot-product task to a worker and await the result."""
    payload = json.dumps({"i": i, "j": j, "row_a": row_a, "col_b": col_b}).encode()
    response = await nc.request("tasks.compute", payload, timeout=60)
    data = json.loads(response.data)
    return data["i"], data["j"], data["val"]


async def matmul_mapreduce(nc: nats_lib.NATS, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute A @ B using MapReduce over NATS.
    Each output cell C[i,j] becomes one task distributed to R workers.
    All tasks are issued concurrently via asyncio.gather.
    """
    M, K = A.shape
    _, N = B.shape

    # Pre-convert to Python lists once to avoid repeated serialisation overhead
    A_rows = [A[i, :].tolist() for i in range(M)]
    B_cols = [B[:, j].tolist() for j in range(N)]

    tasks = [
        _compute_cell(nc, i, j, A_rows[i], B_cols[j])
        for i in range(M)
        for j in range(N)
    ]

    results = await asyncio.gather(*tasks)

    C = np.zeros((M, N))
    for i, j, val in results:
        C[i][j] = val
    return C
