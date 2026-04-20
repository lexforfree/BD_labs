"""
Lab 1, part 2 — Linear regression benchmark: numpy vs MapReduce.

Usage:
    docker compose run --rm coordinator python linreg.py

Solves AX = Y via normal equations: X = (A'A)^{-1} A'Y.

For small matrices (max(M,K) <= NUMPY_MAX_SIZE):
  A is generated in full — used for both the numpy benchmark and MapReduce.

For large matrices:
  A is never stored. Rows are generated on demand inside ata_aty_mapreduce,
  so at most _CONCURRENCY rows exist as Python lists at any moment.
"""

import asyncio
import os
import time

import numpy as np
import nats

from mapreduce import ata_aty_mapreduce, solve_mapreduce

NUMPY_MAX_SIZE = 15_000


def read_positive_int(prompt: str) -> int:
    while True:
        try:
            val = int(input(prompt))
            if val > 0:
                return val
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input, try again.")


async def main() -> None:
    nats_url = os.environ.get("NATS_URL", "nats://nats:4222")
    nc = await nats.connect(nats_url)
    await asyncio.sleep(1)

    print("Linear Regression:\tsolve AX = Y")
    print("Method:\t\tnormal equations X = (A'A)^{-1} A'Y")

    M = read_positive_int("Enter number of samples M: ")
    K = read_positive_int("Enter number of features K: ")

    rng    = np.random.default_rng(7)
    X_true = rng.standard_normal((K, 1))

    use_numpy = max(M, K) <= NUMPY_MAX_SIZE

    if use_numpy:
        A = rng.standard_normal((M, K))
        Y = A @ X_true + rng.standard_normal((M, 1)) * 0.01

        t0 = time.perf_counter()
        X_numpy, *_ = np.linalg.lstsq(A, Y, rcond=None)
        t_np = time.perf_counter() - t0
        print(f"[numpy]\t\tlstsq {M}×{K}  →  {t_np:.4f}s")

        def row_gen(i: int):
            return A[i, :].tolist(), float(Y[i, 0])
    else:
        X_numpy = None
        print(f"[numpy]\t\tskipped — matrix exceeds NUMPY_MAX_SIZE={NUMPY_MAX_SIZE}")

        # Rows generated on demand, independently seeded — A never lives in RAM.
        def row_gen(i: int):
            row_rng = np.random.default_rng([7, i])
            row_a = row_rng.standard_normal(K)
            y_i   = float(row_a @ X_true.ravel() + row_rng.standard_normal() * 0.01)
            return row_a.tolist(), y_i

    # --- MapReduce ---
    print(f"[MapReduce]\tstep 1/2 — stream {M:,} rows  →  A'A ({K}x{K}) + A'Y ({K}x1)...")
    t0 = time.perf_counter()
    AtA, AtY = await ata_aty_mapreduce(nc, row_gen, M, K)

    print(f"[MapReduce]\tstep 2/2 — solve (A'A)x = A'Y  ({K}x{K} system)...")
    X_mr = await solve_mapreduce(nc, AtA, AtY)
    t_mr = time.perf_counter() - t0
    print(f"[MapReduce]\tdone\t\t{t_mr:.4f}s")

    # --- comparison ---
    if X_numpy is not None:
        match    = np.allclose(X_numpy, X_mr, atol=1e-6)
        max_diff = float(np.max(np.abs(X_numpy - X_mr)))
        print(f"Match:\t\t{'YES' if match else 'NO'}\t\tmax_diff={max_diff:.2e}")
        ratio = t_mr / t_np
        if ratio >= 1:
            print(f"Speedup:\tnumpy is {ratio:.1f}x faster than MapReduce")
        else:
            print(f"Speedup:\tMapReduce is {1/ratio:.1f}x faster than numpy")

    await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
